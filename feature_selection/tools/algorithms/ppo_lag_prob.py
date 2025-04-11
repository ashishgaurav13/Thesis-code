from tools.base import Policy
from tools.safe_rl.pg.network import mlp_actor_critic, placeholders, \
    placeholders_from_spaces, count_vars
from tools.safe_rl.pg.utils import values_as_sorted_list
import tensorflow as tf
from tools.safe_rl.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from tools.safe_rl.utils.mpi_tools import mpi_fork, proc_id, num_procs, mpi_sum
from tools.safe_rl.pg.buffer import CPOBuffer
from tools.safe_rl.pg.agents import PPOAgent
import numpy as np
from tools.utils import combine_dicts
import gym
from tools.safe_rl.utils.logx import EpochLogger
import torch
import tools, os, basic
import tqdm
from .ppo_lag import loadmodel, save, PPOPolicyWithCost

class PPOLagProb:
    """
    OpenAI PPO Lagrange implementation
    with probabilistic gradients
    """

    def __init__(self, config):
        self.config = config
        self.f = lambda x: 0.5 - 0.5 * x / tf.sqrt(x*x + basic.small_eps)
        tf.compat.v1.reset_default_graph()
        tf.random.set_seed(config["seed"])
        self.config["agent"] = PPOAgent(**dict(
            clip_ratio=self.config["ppo_clip_param"],
            reward_penalized=False,
            objective_penalized=True,
            learn_penalty=True,
            penalty_param_loss=True
        ))
        self.policy = PPOPolicyWithCost(config)
        self.savelogger = EpochLogger()
        self.config["surr_cost_rescale_ph"] = tf.compat.v1.placeholder(tf.float32, shape=())
        self.config["cur_cost_ph"] = tf.compat.v1.placeholder(tf.float32, shape=())
        obs_shape = self.config["env"].observation_space.shape
        act_shape = self.config["env"].action_space.shape
        gamma = cost_gamma = config["discount_factor"]
        lam = cost_lam = 0.97 if "gae_lambda" not in config.keys() else config["gae_lambda"]
        steps_per_epoch = 4000 if "steps_per_epoch" not in config.keys() else config["steps_per_epoch"]
        self.config["steps_per_epoch"] = steps_per_epoch
        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.config["local_steps_per_epoch"] = local_steps_per_epoch
        pi_info_shapes = {k: v.shape.as_list()[1:] for k,v in self.config["pi_info_phs"].items()}
        self.config["buf"] = CPOBuffer(local_steps_per_epoch,
            obs_shape, 
            act_shape, 
            pi_info_shapes, 
            gamma, 
            lam,
            cost_gamma,
            cost_lam)
        if self.config["agent"].use_penalty:
            with tf.compat.v1.variable_scope('penalty'):
                # param_init = np.log(penalty_init)
                penalty_init=1. if "penalty_init" not in self.config.keys() else self.config["penalty_init"]
                param_init = np.log(max(np.exp(penalty_init)-1, 1e-8))
                self.config["penalty_param"] = tf.compat.v1.get_variable('penalty_param',
                                            initializer=float(param_init),
                                            trainable=self.config["agent"].learn_penalty,
                                            dtype=tf.float32)
            # penalty = tf.exp(penalty_param)
            self.config["penalty"] = tf.nn.softplus(self.config["penalty_param"])
        if self.config["agent"].learn_penalty:
            if self.config["agent"].penalty_param_loss:
                # penalty_loss = -self.config["penalty_param"] * (self.config["cur_cost_ph"] - self.config["beta"])
                penalty_loss = -self.config["penalty_param"] * (self.config["delta"] - self.config["cur_cost_ph"])
            else:
                # ...
                penalty_loss = -self.config["penalty"] * (self.config["delta"] - self.config["cur_cost_ph"])
            penalty_lr = 5e-2 if "penalty_lr" not in self.config.keys() else self.config["penalty_lr"]
            self.config["train_penalty"] = MpiAdamOptimizer(learning_rate=penalty_lr).minimize(penalty_loss)
        self.config["ratio"] = tf.exp(self.config["logp"] - self.config["logp_old_ph"])
        if self.config["agent"].clipped_adv:
            self.config["min_adv"] = tf.where(self.config["adv_ph"]>0, 
                            (1+self.config["agent"].clip_ratio)*self.config["adv_ph"], 
                            (1-self.config["agent"].clip_ratio)*self.config["adv_ph"]
                            )
            self.config["surr_adv"] = tf.reduce_mean(tf.minimum(self.config["ratio"] * self.config["adv_ph"], self.config["min_adv"]))
        else:
            self.config["surr_adv"] = tf.reduce_mean(self.config["ratio"] * self.config["adv_ph"])

        # compute surrogate cost in a different way
        s_ph = self.config["x_ph"]
        a_diff = self.config["pi"] # this is an act_n vector (continuous) or single value between 0 and act_n-1 (discrete)
        logpi = self.config["logp_pi"]
        cret = self.config["cret_ph"]
        loss1_term = tf.stop_gradient(cret) * tf.math.reduce_sum(logpi)
        t = cret-basic.beta
        factor = basic.small_eps / ((tf.sqrt(t*t + basic.small_eps))**3)
        loss2_term = tf.stop_gradient(factor) * cret
        # feasibility_loss = -1. * (basic.delta > tf.stop_gradient(cret)) * (loss1_term-0.5*loss2_term)
        # self.config["feasibility_loss"] = feasibility_loss
        # feasibility_lr = 1e-3
        # self.config["train_feasibility"] = MpiAdamOptimizer(learning_rate=feasibility_lr_lr).minimize(self.config["feasibility_loss"])

        self.config["surr_cost"] = tf.reduce_mean(self.config["ratio"] * self.config["cadv_ph"])

        ent_reg = self.config["ppo_entropy_coef"]
        self.config["pi_objective"] = self.config["surr_adv"] + ent_reg * self.config["ent"]

        # compute surrogate cost in a different way
        if self.config["agent"].objective_penalized:
            self.config["pi_objective"] -= self.config["penalty"] * self.config["surr_cost"]
            self.config["pi_objective"] /= (1 + self.config["penalty"])

        self.config["pi_loss"] = -self.config["pi_objective"]
        self.config["train_pi"] = MpiAdamOptimizer(learning_rate=self.config["agent"].pi_lr).minimize(self.config["pi_loss"])
        self.config["training_package"] = dict(train_pi=self.config["train_pi"]) #, train_feasibility=self.config["train_feasibility"])
        self.config["training_package"].update(dict(pi_loss=self.config["pi_loss"], 
            surr_cost=self.config["surr_cost"],
            d_kl=self.config["d_kl"], 
            target_kl=0.01 if "target_kl" not in self.config.keys() else self.config["target_kl"],
            cost_lim=self.config["beta"]))
        self.config["agent"].prepare_update(self.config["training_package"])
        self.config["v_loss"] = tf.reduce_mean((self.config["ret_ph"] - self.config["v"])**2)
        self.config["vc_loss"] = tf.reduce_mean((self.config["cret_ph"] - self.config["vc"])**2)
        self.config["total_value_loss"] = self.config["v_loss"] + self.config["vc_loss"]
        vf_lr=1e-3 if "vf_lr" not in self.config.keys() else self.config["vf_lr"]
        self.config["train_vf"] = MpiAdamOptimizer(learning_rate=vf_lr).minimize(self.config["total_value_loss"])
        self.config["sess"] = tf.compat.v1.Session()
        self.config["sess"].run(tf.compat.v1.global_variables_initializer())
        self.config["sess"].run(sync_all_params())
        self.config["saver"] = tf.compat.v1.train.Saver()
        self.config["agent"].prepare_session(self.config["sess"])
        self.epoch = 0
    
    # pass array of episode costs and lengths
    def update(self, epcosts, eplens):
        c = [self.f(epcost - self.config["beta"]) for epcost in epcosts]
        c2 = 0
        for item in c:
            c2 += item
        c2 /= len(c)
        c = c2
        meaneplen = 0
        for eplen in eplens:
            meaneplen += eplen
        meaneplen /= len(eplens)
        inputs = {k:v for k,v in zip(self.config["buf_phs"], self.config["buf"].get())}
        inputs[self.config["surr_cost_rescale_ph"]] = meaneplen
        inputs[self.config["cur_cost_ph"]] = c
        measures = dict(LossPi=self.config["pi_loss"],
                        SurrCost=self.config["surr_cost"],
                        LossV=self.config["v_loss"],
                        Entropy=self.config["ent"])
        measures['LossVC'] = self.config["vc_loss"]
        if self.config["agent"].use_penalty:
            measures['Penalty'] = self.config["penalty"]
        pre_update_measures = self.config["sess"].run(measures, feed_dict=inputs)
        # self.config["logger"].update(pre_update_measures)
        if self.config["agent"].learn_penalty:
            self.config["sess"].run(self.config["train_penalty"], feed_dict={self.config["cur_cost_ph"]: cur_cost})
        self.config["agent"].update_pi(inputs)
        vf_iters = 80 if "vf_iters" not in self.config.keys() else self.config["vf_iters"]
        for _ in range(vf_iters):
            self.config["sess"].run(self.config["train_vf"], feed_dict=inputs)
        del measures['Entropy']
        measures['KL'] = self.config["d_kl"]
        post_update_measures = self.config["sess"].run(measures, feed_dict=inputs)
        deltas = dict()
        for k in post_update_measures:
            if k in pre_update_measures:
                deltas['Delta'+k] = post_update_measures[k] - pre_update_measures[k]
        # self.config["logger"].update(dict(KL=post_update_measures['KL'], **deltas))
        return combine_dicts(pre_update_measures,post_update_measures)

    def train(self, evaluator=None, no_mix=False, **kwargs):
        """
        Train one epoch of PPO, and evaluate if needed.
        """
        # Collect some episodes
        metrics = {}
        self.state_action_cost_data = []
        self.max_cost_reached_data = []

        if "forward_only" not in kwargs.keys():
            kwargs["forward_only"] = False

        o, r, d, c, ep_ret, ep_cost, ep_len = self.config["env"].reset(), 0, False, 0, 0, 0, 0
        cur_penalty = 0
        cum_cost = 0
        edcv = 0
        discount = 1
        beta = self.config["beta"]

        if self.config["agent"].use_penalty:
            cur_penalty = self.config["sess"].run(self.config["penalty"])

        for t in range(self.config["local_steps_per_epoch"]):
            
            # Get outputs from policy
            get_action_outs = self.config["sess"].run(self.config["get_action_ops"], 
                                        feed_dict={self.config["x_ph"]: o[np.newaxis]})
            a = self.policy.act(o)
            v_t = get_action_outs['v']
            vc_t = get_action_outs.get('vc', 0)  # Agent may not use cost value func
            logp_t = get_action_outs['logp_pi']
            pi_info_t = get_action_outs['pi_info']

            # Step in environment
            step_data = self.config["env"].step(a)
            if type(step_data) == dict:
                o2, r, d, info = step_data["next_state"], step_data["reward"], step_data["done"], step_data["info"]
            else:
                o2, r, d, info = step_data

            # Include penalty on cost
            c = self.config["cost"]((o, a))
            c = float(c)

            # Track cumulative cost over training
            cum_cost += c
            edcv += discount * c
            discount *= self.config["discount_factor"]

            # save and log
            if not kwargs["forward_only"]:
                self.config["buf"].store(o, a, r, v_t, c, vc_t, logp_t, pi_info_t)

            o = o2
            ep_ret += r
            ep_cost += c
            ep_len += 1

            terminal = d or (ep_len == self.config["time_limit"])
            if terminal or (t==self.config["local_steps_per_epoch"]-1):

                # If trajectory didn't reach terminal state, bootstrap value target(s)
                if d and not(ep_len == self.config["time_limit"]):
                    # Note: we do not count env time out as true terminal state
                    last_val, last_cval = 0, 0
                else:
                    feed_dict={self.config["x_ph"]: o[np.newaxis]}
                    last_val, last_cval = self.config["sess"].run([self.config["v"], self.config["vc"]], feed_dict=feed_dict)
                self.config["buf"].finish_path(last_val, last_cval)

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    if "avg_env_reward" not in metrics:
                        metrics["avg_env_reward"] = []
                    if "avg_env_ep_len" not in metrics:
                        metrics["avg_env_ep_len"] = []
                    if "avg_env_edcv" not in metrics:
                        metrics["avg_env_edcv"] = []
                    metrics["avg_env_reward"] += [ep_ret]
                    metrics["avg_env_ep_len"] += [ep_len]
                    metrics["avg_env_edcv"] += [edcv]

                # Reset environment
                if "max_cost_reached" not in metrics:
                    metrics["max_cost_reached"] = []
                metrics["max_cost_reached"] += [edcv >= beta]
                o, r, d, c, ep_ret, ep_len, ep_cost = self.config["env"].reset(), 0, False, 0, 0, 0, 0
                edcv = 0
                discount = 1

        # metrics["avg_env_reward"] = np.mean(metrics["avg_env_reward"])
        # metrics["avg_env_ep_len"] = np.mean(metrics["avg_env_ep_len"])
        metrics["avg_env_edcv"] = np.mean(metrics["avg_env_edcv"])
        metrics["max_cost_reached"] = np.mean(metrics["max_cost_reached"])
        metrics["avg_cost"] = cum_cost/self.config["local_steps_per_epoch"]
        ## TODO
        # metrics["p_c_tau_leq_beta"]
        if kwargs["forward_only"]:
            return metrics
        new_metrics = self.update(metrics["avg_env_reward"], metrics["avg_env_ep_len"])
        metrics["avg_env_reward"] = np.mean(metrics["avg_env_reward"]) # 
        metrics["avg_env_ep_len"] = np.mean(metrics["avg_env_ep_len"]) # 
        metrics = combine_dicts(metrics, new_metrics)

        # Evaluate
        if evaluator != None:
            eval_metrics = evaluator.evaluate(self.policy)
            metrics = combine_dicts(metrics, eval_metrics)
        self.epoch += 1
        if not no_mix and \
            "mix_save_epoch" in self.config.keys() and \
            self.epoch % self.config["mix_save_epoch"] == 0:
            if not os.path.exists("runs"):
                os.mkdir("runs")
            fname = "runs/seed%d-itr%d-%s"%(self.config["seed"],self.epoch,tools.utils.timestamp())
            save(self.config["sess"], self.config["saver"], fname)
            if "past_pi_weights" not in self.config.keys():
                self.config["past_pi_weights"] = []
            self.config["past_pi_weights"] = self.config["past_pi_weights"] + \
                [fname] 
            print("Intermediate save at epoch=%d" % self.epoch)
            if "flow" in self.config.keys():
                dataset = self.config["env"].trajectory_dataset(self.policy, 
                    self.config["expert_episodes"], weights=None, is_torch_policy=self.config["is_torch_policy"])
                aS = self.config["vector_state_reduction"](dataset.S)
                aA = self.config["vector_action_reduction"](dataset.A)
                aSA = self.config["vector_input_format"](aS, aA)
                sims = []
                for i in range(aSA.shape[0]):
                    tSA = aSA[i].view(-1, self.config["i"])[torch.nonzero(
                        dataset.M[i].view(-1)).view(-1)]
                    tp = -self.config["flow"].log_probs(tSA)
                    tpf = (tp > self.config["expert_nll"][0]+\
                        self.config["expert_nll"][1]).float().mean()
                    sims += [tpf.item()]
                am = np.mean(sims)
                if "past_pi_dissimilarities" not in self.config.keys():
                    self.config["past_pi_dissimilarities"] = []
                self.config["past_pi_dissimilarities"] = \
                    self.config["past_pi_dissimilarities"] +\
                    [am]
        return metrics