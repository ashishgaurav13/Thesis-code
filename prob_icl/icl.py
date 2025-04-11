import basic
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import tools
tools.utils.nowarnings()
import utils
import torch
import matplotlib.pyplot as plt
import tqdm
import atexit
import argparse
import glob
import datetime
import gym

parser = argparse.ArgumentParser()
parser.add_argument("-o", type=str, required=False, default="demo", help="Output directory (overwrite)")
parser.add_argument("-expert_dir", type=str, required=False, default="", help="Expert data directory")
parser.add_argument("-seed", type=int, required=False, default=0, help="Training seed")
parser.add_argument("-env", type=str, required=True, help="Environment")
parser.add_argument("-beta", type=float, required=False, default=-1, help="ICL beta")
parser.add_argument("-expert_only", action='store_true', default=False, help="Terminate after generating expert data")
parser.add_argument("-ppo_lag", action='store_true', default=False, help='use OpenAI PPO Lag')
args = parser.parse_args()

if args.ppo_lag:
    basic.use_ppo_lag = True
else:
    basic.use_ppo_lag = False

basic.true_constraint_function = utils.true_constraint_function
basic.env_name = args.env
basic.output_dir = args.o
if basic.env_name not in ['highd', 'exid']:
    if args.expert_dir != "":
        basic.expert_data_file = "%s/expert_data.pt" % args.expert_dir
    else:
        basic.expert_data_file = "%s/expert_data.pt" % basic.output_dir
if not os.path.exists(basic.output_dir):
    os.mkdir(basic.output_dir)
if basic.CLEAR_FILES:
    for f in glob.glob('%s/*' % basic.output_dir):
        os.remove(f)
basic.seed = args.seed   

atexit.register(utils.end_logging)
utils.start_logging("%s/0_log.txt" % basic.output_dir)
print("Started ... %s" % (datetime.datetime.utcnow()))
if basic.env_name not in ['highd', 'exid']:
    print("Expert data file: %s" % basic.expert_data_file)
utils.seed_fn(basic.seed)

basic.env = utils.create_env()
basic.obs_n = basic.env.observation_space.shape[0]
if type(basic.env.action_space) == gym.spaces.Discrete:
    basic.act_n = basic.env.action_space.n
else:
    basic.act_n = basic.env.action_space.shape[0]
basic.env.seed(basic.seed)

if args.beta != -1:
    basic.beta = args.beta
print("Beta: %g" % (basic.beta))

saved_chkpts = glob.glob("%s/iter_*.pt" % basic.output_dir)

if len(saved_chkpts) > 0:

    iter_numbers = [int(item.split("_")[-1].split(".")[0]) for item in saved_chkpts]
    max_iter_number = max(iter_numbers)
    restore_file = "%s/iter_%d.pt" % (basic.output_dir, max_iter_number)
    print("Restoring to ... %s" % restore_file)
    basic.constraint_nn, basic.constraint_opt, \
        basic.policy_mixture, basic.weights_mixture = torch.load(restore_file)
    if basic.env_name not in ['highd', 'exid']:
        basic.expert_data = utils.generate_expert_data(basic.env)
    basic.agent_data = None

    if args.expert_only:
        print("Generated expert data, exit")
        exit(0)

    flow_data = utils.convert_to_flow_data(basic.expert_data)
    flow_config = tools.data.Configuration({
        "normalize_flow_inputs": True,
        "minibatch_size": basic.minibatch_size,
        "learning_rate": basic.learning_rate,
    })
    flow_config["t"].device = basic.device
    basic.flow = tools.functions.create_flow(flow_config, flow_data, "realnvp", basic.constraint_fn_input_dim)
    for flowepoch in range(basic.flow_iters):
        metrics = basic.flow.train()
        print(metrics)
    nll = -basic.flow.log_probs(flow_data).detach().cpu()
    basic.expert_nll = (nll.mean(), nll.std())

    iterations = list(range(max_iter_number+1, basic.n_iters))

else:

    basic.constraint_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.constraint_fn_input_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, 1), torch.nn.Sigmoid(),
    ).to(basic.device)
    basic.constraint_opt = torch.optim.Adam(basic.constraint_nn.parameters(), lr=basic.learning_rate)
    utils.visualize_constraint(utils.current_constraint_function, "%s/0_initial_constraint.png" % basic.output_dir)

    if basic.env_name not in ['highd', 'exid']:
        basic.expert_data = utils.generate_expert_data(basic.env)
    basic.agent_data = None
    utils.visualize_accrual(basic.expert_data, "%s/0_expert_accrual.png" % basic.output_dir)

    if args.expert_only:
        print("Generated expert data, exit")
        exit(0)

    flow_data = utils.convert_to_flow_data(basic.expert_data)
    flow_config = tools.data.Configuration({
        "normalize_flow_inputs": True,
        "minibatch_size": basic.minibatch_size,
        "learning_rate": basic.learning_rate,
    })
    flow_config["t"].device = basic.device
    basic.flow = tools.functions.create_flow(flow_config, flow_data, "realnvp", basic.constraint_fn_input_dim)
    for flowepoch in range(basic.flow_iters):
        metrics = basic.flow.train()
        print(metrics)
    nll = -basic.flow.log_probs(flow_data).detach().cpu()
    basic.expert_nll = (nll.mean(), nll.std())

    basic.policy_mixture, basic.weights_mixture = [], []
    utils.show_metrics(p=False)
    iterations = list(range(basic.n_iters))

print("Loading finished ... %s" % (datetime.datetime.utcnow()))

for itr in iterations:
    print("ICL iteration %d ... %s" % (itr, datetime.datetime.utcnow()))
    value_nn, policy_nn = utils.make_nn()
    ppolag_config = None
    if basic.use_ppo_lag:
        ppolag_config = utils.ppo_lag(\
            basic.ppo_iters,
            basic.env,
            utils.current_constraint_function,
            utils.condition,
            utils.command
        )
        policy_nn = ppolag_config["policy"]
    elif basic.use_gae:
        utils.ppo_penalty_gae(\
            basic.ppo_iters, 
            basic.env, 
            policy_nn, 
            value_nn, 
            utils.current_constraint_function, 
            utils.condition, 
            utils.command
        )
    else:
        utils.ppo_penalty(\
            basic.ppo_iters, 
            basic.env, 
            policy_nn, 
            value_nn, 
            utils.current_constraint_function, 
            utils.condition,
            utils.command
        )
    basic.agent_data = utils.collect_trajectories(\
        len(basic.expert_data), 
        basic.env, 
        policy_nn, 
        utils.current_constraint_function\
    )
    mixture_data = utils.collect_trajectories_mixture(\
        len(basic.expert_data), 
        basic.env, 
        basic.policy_mixture, 
        basic.weights_mixture, 
        utils.current_constraint_function,
        ppolag_config=ppolag_config if basic.use_ppo_lag else None\
    )
    if basic.env_name not in ['highd', 'exid']:
        fig, ax = plt.subplots(1, 4, figsize=(12, 3))
        utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax[0])
        ax[0].set_title("True constraint")
        utils.visualize_constraint(utils.current_constraint_function, fig=fig, ax=ax[1])
        ax[1].set_title("Learned constraint (itr = %d)" % itr)
        utils.visualize_accrual(basic.expert_data, fig=fig, ax=ax[2])
        ax[2].set_title("Expert accrual")
        utils.visualize_accrual(basic.agent_data, fig=fig, ax=ax[3])
        ax[3].set_title("Agent accrual")
    else:
        fig, ax = plt.subplots(1, 3, figsize=(9, 3))
        utils.visualize_constraint(utils.current_constraint_function, fig=fig, ax=ax[0])
        ax[0].set_title("Learned constraint (itr = %d)" % itr)
        utils.visualize_accrual(basic.expert_data, fig=fig, ax=ax[1])
        ax[1].set_title("Expert accrual")
        utils.visualize_accrual(basic.agent_data, fig=fig, ax=ax[2])
        ax[2].set_title("Agent accrual")
    fig.tight_layout()
    fig.savefig("%s/%d_iteration.png" % (basic.output_dir, itr))
    mixture_data_constraint_returns = utils.compute_current_constraint_value_trajectory(basic.constraint_nn, mixture_data)
    for _ in tqdm.tqdm(range(basic.ca_iters)):
        utils.constraint_function_adjustment(\
            basic.ca_iters, 
            basic.constraint_nn, 
            basic.constraint_opt,
            basic.expert_data,
            mixture_data\
        )
    utils.show_metrics(p=False)
    savefile = "%s/iter_%d.pt" % (basic.output_dir, itr)
    torch.save([basic.constraint_nn, basic.constraint_opt, \
        basic.policy_mixture, basic.weights_mixture], savefile)
    print("ICL iteration complete ... saved chkpt %s" % savefile)
    print("ICL iteration complete ... %s" % (datetime.datetime.utcnow()))

print("Final results ... %s" % (datetime.datetime.utcnow()))
ppolag_config = None
if basic.use_ppo_lag:
    ppolag_config = utils.ppo_lag( \
        basic.ppo_iters,
        basic.env,
        utils.current_constraint_function,
        utils.condition,
        utils.command
    )
    policy_nn = ppolag_config["policy"]
elif basic.use_gae:
    value_nn, policy_nn = utils.make_nn()
    utils.ppo_penalty_gae(basic.ppo_iters, basic.env, policy_nn, value_nn, utils.current_constraint_function)
else:
    value_nn, policy_nn = utils.make_nn()
    utils.ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, utils.current_constraint_function)
basic.agent_data = utils.collect_trajectories(\
    len(basic.expert_data),
    basic.env, 
    policy_nn, 
    utils.current_constraint_function
)
if basic.env_name not in ['highd', 'exid']:
    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    utils.visualize_constraint(basic.true_constraint_function, fig=fig, ax=ax[0])
    ax[0].set_title("True constraint")
    utils.visualize_constraint(utils.current_constraint_function, fig=fig, ax=ax[1])
    ax[1].set_title("Final constraint")
    utils.visualize_accrual(basic.expert_data, fig=fig, ax=ax[2])
    ax[2].set_title("Expert accrual")
    utils.visualize_accrual(basic.agent_data, fig=fig, ax=ax[3])
    ax[3].set_title("Final accrual")
else:
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    utils.visualize_constraint(utils.current_constraint_function, fig=fig, ax=ax[0])
    ax[0].set_title("Final constraint")
    utils.visualize_accrual(basic.expert_data, fig=fig, ax=ax[1])
    ax[1].set_title("Expert accrual")
    utils.visualize_accrual(basic.agent_data, fig=fig, ax=ax[2])
    ax[2].set_title("Final accrual")
fig.tight_layout()
fig.savefig("%s/%d_final_iteration.png" % (basic.output_dir, basic.n_iters))
utils.show_metrics(p=False)
if basic.continuous_actions == True: # for continuous actions, saving the state_dict is better since Gaussian2 class is not serializable
    if basic.use_ppo_lag: # tf policy
         utils.save(ppolag_config["sess"], ppolag_config["saver"], "%s/final_policy" % (basic.output_dir))
         torch.save([basic.agent_data], "%s/final_policy.pt" % (basic.output_dir))
    else:
        torch.save([basic.agent_data, value_nn.state_dict(), policy_nn.state_dict()], "%s/final_policy.pt" % (basic.output_dir))
else:
    if basic.use_ppo_lag:
        utils.save(ppolag_config["sess"], ppolag_config["saver"], "%s/final_policy" % (basic.output_dir))
        torch.save([basic.agent_data], "%s/final_policy.pt" % (basic.output_dir))
    else:
        torch.save([basic.agent_data, value_nn, policy_nn], "%s/final_policy.pt" % (basic.output_dir))
print("Final results saved ... %s" % (datetime.datetime.utcnow()))

print("Finished ... %s" % (datetime.datetime.utcnow()))
