from tools.data import Logger
import subprocess, select, time
from copy import deepcopy
import json
import sys
import collections
import atexit
import threading

class ProcessManager:
    """
    Manage multiple processes.
    """
    def __init__(self, config, query=1):
        """
        Initialize ProcessManager.
        Query every `query` seconds.
        """
        assert("cmds" in config.data.keys())
        assert(type(config["cmds"]) == list)
        assert("vars" in config.data.keys())
        assert(type(config["vars"]) == dict)
        for var in config["vars"].keys():
            assert(type(config["vars"][var]) == list)
        assert("logger" in config.data.keys())
        assert(type(config["logger"]) == Logger)
        self.cmds = config["cmds"]
        self.vars = config["vars"]
        self.logger = config["logger"]
        self.query = query
        atexit.register(self.cleanup)

    def cleanup(self):
        """
        Clean up things.
        """
        if hasattr(self, "processes"):
            for key in self.processes.keys():
                self.processes[key].kill()
        print("Cleaning up!")

    def get_commands(self, silent=True):
        """
        Show the various commands that are going to be executed.
        """
        all_cmds = [[cmd, {}] for cmd in self.cmds]
        for var, values in self.vars.items():
            new_cmds = []
            for value in values:
                for ci in range(len(all_cmds)):
                    # assert(all_cmds[ci][0].count("[%s]" % var) >= 1)
                    new_cmd = all_cmds[ci][0].replace("[%s]" % var, str(value))
                    new_dict = deepcopy(all_cmds[ci][1])
                    assert("var" not in new_dict.keys())
                    new_dict[var] = value
                    new_cmds += [[new_cmd, new_dict]]
            all_cmds = new_cmds
        if not silent:
            json.dumps(all_cmds)
        return all_cmds
    
    def read_output(self, pdesc, whichfd):
        while self.loops[pdesc]:
            for line in whichfd.readline().decode("utf-8").split("\n"):
                if line.strip() != "":
                    self.outputs[pdesc].append(line.strip())

    def launch(self):
        self.processes = {}
        self.outputs = {}
        self.threadouts = {}
        self.threaderrs = {}
        self.loops = {}
        process_names = self.get_commands()
        for pcmd, pdict in process_names:
            pdesc = "".join(["[%s=%s]"%(key,value) for key, value in pdict.items()])
            self.processes[pdesc] = subprocess.Popen(pcmd.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.outputs[pdesc] = collections.deque(maxlen=100)
            self.loops[pdesc] = True
            self.threadouts[pdesc] = threading.Thread(target=self.read_output, args=(pdesc, self.processes[pdesc].stdout))
            self.threaderrs[pdesc] = threading.Thread(target=self.read_output, args=(pdesc, self.processes[pdesc].stderr))
            self.threadouts[pdesc].start()
            self.threaderrs[pdesc].start()

    def check(self):
        assert(hasattr(self, "processes"))
        metrics = {}
        finished = 0
        for key, value in self.processes.items():
            if len(self.outputs[key]) > 0:
                metrics[key] = self.outputs[key][-1]
            else:
                metrics[key] = "[STARTING]"
            if value.poll() != None:
                metrics[key] += "[END:%d]" % value.poll()
                self.loops[key] = False
                finished += 1
        self.logger.update(metrics)
        if finished == len(self.processes):
            print("Finished!")
            sys.exit(0)

