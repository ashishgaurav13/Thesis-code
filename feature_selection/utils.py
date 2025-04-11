import sys, os
import copy
import logging
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tools
from functools import partial
from tqdm import tqdm, trange
import basic
import ot
from scipy.spatial.distance import cdist
import gym
from gym import ActionWrapper
from gym.spaces import Box
from gym.utils.step_api_compatibility import step_api_compatibility
from typing import Any, Callable
from gym import RewardWrapper
import time
from collections import deque
from typing import Optional
import tensorflow as tf














# Logging



class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            try:
                f.flush()
            except:
                pass

def start_logging(filename):
    if not os.path.exists(filename):
        basic.log_file = open(filename, 'w')
    else:
        basic.log_file = open(filename, 'a')
    sys.stdout = Tee(sys.stdout, basic.log_file)
    print(" ".join("\""+arg+"\"" if " " in arg else arg for arg in sys.argv))

def end_logging():
    basic.log_file.close()





# MINE







def mine(data_x, data_y, K=1000, viz=False):
    n = data_x.shape[0]
    x_dim = data_x.shape[1]
    y_dim = data_y.shape[1]
    T = torch.nn.Sequential(
        torch.nn.Linear(x_dim+y_dim, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1)
    )
    opt = torch.optim.Adam(T.parameters(), lr=3e-4)
    i_last = None
    if viz:
        r = trange(K)
    else:
        r = trange(K)
    for k in r:
        T.zero_grad()
        opt.zero_grad()
        data_xy = torch.cat([data_x, data_y], dim=1)
        data_x_y = torch.cat([data_x, data_y[torch.randperm(n)]], dim=1)
        i = torch.mean(T(data_xy))-torch.log(torch.exp(T(data_x_y)).mean())
        (-i).backward()
        opt.step()
        i_last = i.item()
        if i_last < 0:
            i_last = 0.
    return i_last









# load save tf model






def loadmodel(session, saver, checkpoint_dir):
    session.run(tf.compat.v1.global_variables_initializer())
    ckpt = tf.compat.v1.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(session, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

def save(session, saver, checkpoint_dir, ppolag_config=None):
    d = os.path.join(checkpoint_dir, "model")
    cnt = 0
    while not os.path.exists(d+".meta") and cnt <= 10:
        saver.save(session, d)
        cnt += 1
    if cnt >= 10:
        print("Unable to save??? ", d)
        print("sess", ppolag_config["sess"], "saver", ppolag_config["saver"])
    else:
        print("saved", d, os.path.exists(d+".meta"))










# wrappers









def add_vector_episode_statistics( 
    info: dict, episode_info: dict, num_envs: int, env_num: int
):
    """Add episode statistics.
    Add statistics coming from the vectorized environment.
    Args:
        info (dict): info dict of the environment.
        episode_info (dict): episode statistics data.
        num_envs (int): number of environments.
        env_num (int): env number of the vectorized environments.
    Returns:
        info (dict): the input info dict with the episode statistics.
    """
    info["episode"] = info.get("episode", {})
    info["_episode"] = info.get("_episode", np.zeros(num_envs, dtype=bool))
    info["_episode"][env_num] = True
    for k in episode_info.keys():
        info_array = info["episode"].get(k, np.zeros(num_envs))
        info_array[env_num] = episode_info[k]
        info["episode"][k] = info_array
    return info

class RecordEpisodeStatistics(gym.Wrapper):
    """This wrapper will keep track of cumulative rewards and episode lengths.
    At the end of an episode, the statistics of the episode will be added to ``info``
    using the key ``episode``. If using a vectorized environment also the key
    ``_episode`` is used which indicates whether the env at the respective index has
    the episode statistics.
    After the completion of an episode, ``info`` will look like this::
        >>> info = {
        ...     ...
        ...     "episode": {
        ...         "r": "<cumulative reward>",
        ...         "l": "<episode length>",
        ...         "t": "<elapsed time since instantiation of wrapper>"
        ...     },
        ... }
    For a vectorized environments the output will be in the form of::
        >>> infos = {
        ...     ...
        ...     "episode": {
        ...         "r": "<array of cumulative reward>",
        ...         "l": "<array of episode length>",
        ...         "t": "<array of elapsed time since instantiation of wrapper>"
        ...     },
        ...     "_episode": "<boolean array of length num-envs>"
        ... }
    Moreover, the most recent rewards and episode lengths are stored in buffers that can be accessed via
    :attr:`wrapped_env.return_queue` and :attr:`wrapped_env.length_queue` respectively.
    Attributes:
        return_queue: The cumulative rewards of the last ``deque_size``-many episodes
        length_queue: The lengths of the last ``deque_size``-many episodes
    """

    def __init__(self, env: gym.Env, deque_size: int = 100, new_step_api: bool = False):
        """This wrapper will keep track of cumulative rewards and episode lengths.
        Args:
            env (Env): The environment to apply the wrapper
            deque_size: The size of the buffers :attr:`return_queue` and :attr:`length_queue`
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env, new_step_api)
        self.num_envs = getattr(env, "num_envs", 1)
        self.t0 = time.perf_counter()
        self.episode_count = 0
        self.episode_returns: Optional[np.ndarray] = None
        self.episode_lengths: Optional[np.ndarray] = None
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)
        self.is_vector_env = getattr(env, "is_vector_env", False)

    def reset(self, **kwargs):
        """Resets the environment using kwargs and resets the episode returns and lengths."""
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def seed(self, s=None):
        self.env.seed(s)
    
    def step(self, action):
        """Steps through the environment, recording the episode statistics."""
        (
            observations,
            rewards,
            terminateds,
            truncateds,
            infos,
        ) = step_api_compatibility(self.env.step(action), True, self.is_vector_env)
        assert isinstance(
            infos, dict
        ), f"`info` dtype is {type(infos)} while supported dtype is `dict`. This may be due to usage of other wrappers in the wrong order."
        self.episode_returns += rewards
        self.episode_lengths += 1
        if not self.is_vector_env:
            terminateds = [terminateds]
            truncateds = [truncateds]
        terminateds = list(terminateds)
        truncateds = list(truncateds)
        for i in range(len(terminateds)):
            if terminateds[i] or truncateds[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "episode": {
                        "r": episode_return,
                        "l": episode_length,
                        "t": round(time.perf_counter() - self.t0, 6),
                    }
                }
                if self.is_vector_env:
                    infos = add_vector_episode_statistics(
                        infos, episode_info["episode"], self.num_envs, i
                    )
                else:
                    infos = {**infos, **episode_info}
                self.return_queue.append(episode_return)
                self.length_queue.append(episode_length)
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
        return step_api_compatibility(
            (
                observations,
                rewards,
                terminateds if self.is_vector_env else terminateds[0],
                truncateds if self.is_vector_env else truncateds[0],
                infos,
            ),
            self.new_step_api,
            self.is_vector_env,
        )

class ClipAction(ActionWrapper):
    """Clip the continuous action within the valid :class:`Box` observation space bound.

    Example:
        >>> import gym
        >>> env = gym.make('Bipedal-Walker-v3')
        >>> env = ClipAction(env)
        >>> env.action_space
        Box(-1.0, 1.0, (4,), float32)
        >>> env.step(np.array([5.0, 2.0, -10.0, 0.0]))
        # Executes the action np.array([1.0, 1.0, -1.0, 0]) in the base environment
    """

    def __init__(self, env: gym.Env):
        """A wrapper for clipping continuous actions within the valid bound.

        Args:
            env: The environment to apply the wrapper
        """
        assert isinstance(env.action_space, Box)
        super().__init__(env, new_step_api=True)

    def action(self, action):
        """Clips the action within the valid bounds.

        Args:
            action: The action to clip

        Returns:
            The clipped action
        """
        return np.clip(action, self.action_space.low, self.action_space.high)

# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
    mean, var, count, batch_mean, batch_var, batch_count
):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

    Note:
        The normalization depends on past trajectories and observations will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8, new_step_api: bool = False):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.

        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env, new_step_api)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, terminateds, truncateds, infos = step_api_compatibility(
            self.env.step(action), True, self.is_vector_env
        )
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return step_api_compatibility(
            (obs, rews, terminateds, truncateds, infos),
            self.new_step_api,
            self.is_vector_env,
        )

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        if kwargs.get("return_info", False):
            obs, info = self.env.reset(**kwargs)

            if self.is_vector_env:
                return self.normalize(obs), info
            else:
                return self.normalize(np.array([obs]))[0], info
        else:
            obs = self.env.reset(**kwargs)

            if self.is_vector_env:
                return self.normalize(obs)
            else:
                return self.normalize(np.array([obs]))[0]

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)

    def unnormalize(self, obs):
        return obs * np.sqrt(self.obs_rms.var + self.epsilon) + self.obs_rms.mean


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
        new_step_api: bool = False,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
            new_step_api (bool): Whether the wrapper's step method outputs two booleans (new API) or one boolean (old API)
        """
        super().__init__(env, new_step_api)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, terminateds, truncateds, infos = step_api_compatibility(
            self.env.step(action), True, self.is_vector_env
        )
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            dones = terminateds or truncateds
        else:
            dones = np.bitwise_or(terminateds, truncateds)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return step_api_compatibility(
            (obs, rews, terminateds, truncateds, infos),
            self.new_step_api,
            self.is_vector_env,
        )

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)

class TransformObservation(gym.ObservationWrapper):
    """Transform the observation via an arbitrary function :attr:`f`.

    The function :attr:`f` should be defined on the observation space of the base environment, ``env``, and should, ideally, return values in the same space.

    If the transformation you wish to apply to observations returns values in a *different* space, you should subclass :class:`ObservationWrapper`, implement the transformation, and set the new observation space accordingly. If you were to use this wrapper instead, the observation space would be set incorrectly.

    Example:
        >>> import gym
        >>> import numpy as np
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformObservation(env, lambda obs: obs + 0.1*np.random.randn(*obs.shape))
        >>> env.reset()
        array([-0.08319338,  0.04635121, -0.07394746,  0.20877492])
    """

    def __init__(self, env: gym.Env, f: Callable[[Any], Any]):
        """Initialize the :class:`TransformObservation` wrapper with an environment and a transform function :param:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the observation
        """
        super().__init__(env, new_step_api=True)
        assert callable(f)
        self.f = f

    def observation(self, observation):
        """Transforms the observations with callable :attr:`f`.

        Args:
            observation: The observation to transform

        Returns:
            The transformed observation
        """
        return self.f(observation)

    def unnormalize(self, obs):
        return self.env.unnormalize(obs)

class TransformReward(RewardWrapper):
    """Transform the reward via an arbitrary function.

    Warning:
        If the base environment specifies a reward range which is not invariant under :attr:`f`, the :attr:`reward_range` of the wrapped environment will be incorrect.

    Example:
        >>> import gym
        >>> env = gym.make('CartPole-v1')
        >>> env = TransformReward(env, lambda r: 0.01*r)
        >>> env.reset()
        >>> observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        >>> reward
        0.01
    """

    def __init__(self, env: gym.Env, f: Callable[[float], float]):
        """Initialize the :class:`TransformReward` wrapper with an environment and reward transform function :param:`f`.

        Args:
            env: The environment to apply the wrapper
            f: A function that transforms the reward
        """
        super().__init__(env, new_step_api=True)
        assert callable(f)
        self.f = f

    def reward(self, reward):
        """Transforms the reward using callable :attr:`f`.

        Args:
            reward: The reward to transform

        Returns:
            The transformed reward
        """
        return self.f(reward)





















# data generation



def generate_data(filename, c, only_success=False):
    print(filename, os.path.exists(filename), os.getcwd())
    if os.path.exists(filename):
        data = torch.load(filename)
        # print("Loaded expert data")
    else:
        # value_nn, policy_nn = make_nn()
        if basic.use_ppo_lag:
            ppolag_config = ppo_lag(basic.ppo_iters, basic.env, c)
            policy_nn = ppolag_config["policy"]
        else:
            value_nn, policy_nn = make_nn()
            ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, c)
        # ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, c)
        data = collect_trajectories(basic.n_trajs, basic.env, policy_nn, c, only_success=only_success)
        torch.save(data, filename)
    return data







# PPO Penalty



def seed_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





def gridworld_imshow(m, fig, ax):
    m = np.array(m).squeeze()
    assert len(m.shape) == 2
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(m, cmap="gray")
    # im.set_clim(0, 1)
    ax.set_xticks(np.arange(m.shape[0]))
    ax.set_yticks(np.arange(m.shape[1]))
    cbar = fig.colorbar(im, cax=cax)



def visualize_accrual(data, fig=None, ax=None):
    accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
    for S, A in data:
        for s, a in zip(S, A):
            x, y = s
            accrual[a][x][y] += 1
    accrual = np.mean(accrual, axis=0)
    accrual /= np.max(accrual)
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    gridworld_imshow(accrual, fig, ax)





def visualize_constraint(constraint_fn, savefig=None, fig=None, ax=None):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    grid_for_action = []
    for a in np.arange(basic.n_actions):
        grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
        for x in np.arange(basic.gridworld_dim):
            for y in np.arange(basic.gridworld_dim):
                grid[x, y] = constraint_fn(([x, y], a))
        grid_for_action += [grid]
    average_grid = np.mean(grid_for_action, axis=0)
    gridworld_imshow(average_grid, fig, ax)






def make_nn():
    value_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, 1),
    ).to(basic.device)
    policy_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.act_n),
    ).to(basic.device)
    return value_nn, policy_nn





# def play_episode(env, policy_nn, constraint_fn):
#     S, A, R, C = [], [], [], []
#     S += [env.reset()]
#     done = False
#     while not done:
#         probs = torch.nn.Softmax(dim=-1)(policy_nn(torch.tensor(S[-1], device=basic.device, dtype=torch.float))).view(-1)
#         action = np.random.choice(basic.act_n, p=probs.cpu().detach().numpy())
#         A += [action]
#         next_state, reward, done, info = env.step(action)
#         C += [constraint_fn((S[-1], action))]
#         S += [next_state]
#         R += [reward]
#     return S, A, R, C







def play_episode(env, policy_nn, constraint_fn, get_r=False, render=False, **kwargs):
    S, A, R, C = [], [], [], []
    S += [env.reset()]
    if render:
        env.render(**kwargs)
    done = False
    Normal = tools.utils.FixedNormal(0., 1.)
    if get_r:
        info_R = None
    while not done:
        if type(policy_nn) != tools.algorithms.PPOPolicyWithCost:
            if not basic.continuous_actions:
                probs = torch.nn.Softmax(dim=-1)(policy_nn(torch.tensor(S[-1], device=basic.device, dtype=torch.float))).view(-1)
                action = np.random.choice(basic.act_n, p=probs.cpu().detach().numpy())
            else:
                try:
                    Normal.update(*policy_nn(torch.tensor(S[-1], device=basic.device, dtype=torch.float)))
                    action = Normal.sample().view(-1).detach().cpu().numpy()
                except:
                    print("nan error")
                    print("state: ", S[-1])
                    print("model weights: ", policy_nn.state_dict())
                    print("forward pass: ", policy_nn(torch.tensor(S[-1], device=basic.device, dtype=torch.float)))
                    if hasattr(basic, "Epoch_info"):
                        torch.save(basic.Epoch_info, "epoch_info_debug.pt")
                        for idx, item in enumerate(basic.Epoch_info[::-1]):
                            print("Debug info for iter current-%d" % idx)
                            print(item)
                    exit(0)
        else:
            action = policy_nn.act(S[-1])
            if not basic.continuous_actions:
                action = int(action)
        A += [action]
        next_state, reward, done, info = env.step(action)
        if render:
            env.render(**kwargs)
        if get_r:
            if "episode" in info:
                # print(info_R)
                info_R = info["episode"]["r"]
        C += [constraint_fn((S[-1], action))]
        if 'cost' in info.keys():
            C[-1] += info['cost']
        S += [next_state]
        R += [reward]
    if get_r:
        return S, A, R, C, info_R
    return S, A, R, C








def discount(x, invert=False):
    n = len(x)
    g = 0
    d = []
    for i in range(n):
        if not invert:
            g = x[n - 1 - i] + basic.discount_factor * g
            d = [g] + d
        else:
            g = g + x[i] * (basic.discount_factor**i)
            d = d + [g]
    return d






class ReplayBuffer:
    def __init__(self, N):
        self.N = N
        self.S = torch.zeros((self.N, basic.obs_n), dtype=torch.float, device=basic.device)
        self.A = torch.zeros((self.N), dtype=torch.long, device=basic.device)
        self.G = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.log_probs = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.i = 0
        self.filled = 0

    def add(self, S, A, G, log_probs):
        M = S.shape[0]
        self.filled = min(self.filled + M, self.N)
        assert M <= self.N
        for j in range(M):
            self.S[self.i] = S[j, :]
            self.A[self.i] = A[j]
            self.G[self.i] = G[j]
            self.log_probs[self.i] = log_probs[j]
            self.i = (self.i + 1) % self.N

    def sample(self, n):
        global device
        minibatch = random.sample(range(self.filled), min(n, self.filled))
        S, A, G, log_probs = [], [], [], []
        for mbi in minibatch:
            s, a, g, lp = self.S[mbi], self.A[mbi], self.G[mbi], self.log_probs[mbi]
            S += [s]
            A += [a]
            G += [g]
            log_probs += [lp]
        return torch.stack(S), torch.stack(A), torch.stack(G), torch.stack(log_probs)








def ppo_lag(n_epochs, env, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None):
    tf.compat.v1.disable_eager_execution()
    config = dict(
        env=basic.env,
        hidden_size=basic.hidden_dim,
        seed=basic.seed,
        ppo_clip_param=basic.clip_param,
        discount_factor=basic.discount_factor,
        gae_lambda=basic.gae_lambda,
        steps_per_epoch=basic.max_steps_per_epoch,
        beta=basic.beta,
        ppo_entropy_coef=basic.entropy_coef,
        cost=constraint_fn,
        time_limit=basic.time_limit,
        delta=basic.delta,
        vf_lr=basic.learning_rate,
        pi_lr=basic.learning_rate,
        penalty_lr=basic.learning_rate*100,
        vf_iters=basic.ppo_subepochs,
    )
    ppolag = tools.algorithms.PPOLag(config)
    R = []
    MCR = []
    C = []
    best_R = -float('inf')
    best_policy_params = ''
    for epoch in range(n_epochs):
        # if basic.beta_start != None:
        #     curr_beta = basic.beta_start+(basic.beta-basic.beta_start)*epoch/(n_epochs-1)
        #     config["beta"] = curr_beta
        #     print("Curr beta: %.2f" % config["beta"])
        metrics = ppolag.train(no_mix=True, forward_only=False)
        R += [metrics["avg_env_reward"]]
        C += [metrics["avg_env_edcv"]]
        MCR += [metrics["max_cost_reached"]]
        r_win = sum(R[-basic.window_size:])/(len(R[-basic.window_size:])+1e-3)
        c_win = sum(C[-basic.window_size:]) / (len(C[-basic.window_size:]) + 1e-3)
        mcr_win = sum(MCR[-basic.window_size:])/(len(MCR[-basic.window_size:])+1e-3)
        print("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f\tG_window = %.2f\tGc_window = %.2f\tMCR_window = %.2f" % \
              (epoch+1, metrics["avg_env_reward"], metrics["avg_env_edcv"], metrics["max_cost_reached"], r_win, c_win, mcr_win))
        if basic.use_early_stopping:
            if R[-1] > best_R and C[-1] < basic.beta: # C[-1] < curr_beta:
                old_R = copy.copy(best_R)
                best_R = R[-1]
                if not os.path.exists("tmp"):
                    os.mkdir("tmp")
                best_policy_params = "tmp/seed%d-itr%d-%s-best" % (basic.seed, epoch, tools.utils.timestamp())
                save(ppolag.config["sess"], ppolag.config["saver"], best_policy_params, ppolag_config=ppolag.config)
                print("updated best policy, old_R: %.2f, new_R: %.2f" % (old_R, best_R))
        if additional_fn_condition != None and additional_fn_epoch_end != None:
            if additional_fn_condition(epoch, ppolag.policy) == True:
                # if best_policy_params != '':
                #     curr_policy_params = "tmp/seed%d-itr%d-%s-curr" % (basic.seed, epoch, tools.utils.timestamp())
                #     save(ppolag.config["sess"], ppolag.config["saver"], curr_policy_params)
                #     loadmodel(ppolag.config["sess"], ppolag.config["saver"], best_policy_params)
                additional_fn_epoch_end(epoch, ppolag.policy, ppolag_config=ppolag.config)
                # if best_policy_params != '':
                #     loadmodel(ppolag.config["sess"], ppolag.config["saver"], curr_policy_params)
    if basic.use_early_stopping:
        if best_R > -float('inf'):
            print("Using best policy_nn params due to early stopping")
            loadmodel(ppolag.config["sess"], ppolag.config["saver"], best_policy_params)
    return ppolag.config






def ppo_penalty(n_epochs, env, policy_nn, value_nn, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None):
    # Additional function to be run when additional function condition is met
    # Additional function takes in current policy_nn and current iteration number
    buffer = ReplayBuffer(basic.replay_buffer_size)
    value_opt = torch.optim.Adam(value_nn.parameters(), lr=basic.learning_rate)
    policy_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate)
    pbar = tqdm(total=n_epochs)
    for epoch in range(n_epochs):
        S_e, A_e, G_e, Gc_e, Indices = [], [], [], [], []
        G0_e, Gc0_e = [], []
        max_cost_reached = 0.0
        max_cost_reached_n = 0
        S_e_buf, A_e_buf, G_e_buf = [], [], []
        for episode in range(basic.episodes_per_epoch):
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
            start_index = len(A_e)
            S_e += S[:-1]  # ignore last state
            A_e += A
            G_e += discount(R)
            G0_e += [float(discount(R)[0])]
            Gc_e += discount(C)
            Gc0_e += [float(discount(C)[0])]
            end_index = len(A_e)
            Indices += [(start_index, end_index)]
            # Only add those experiences to replay buffer which are before the constraint violation
            inverted_discount = torch.tensor(discount(C, invert=True), dtype=torch.float, device=basic.device)
            good_until = len(inverted_discount)
            for idx, item in enumerate(inverted_discount):
                if item >= basic.beta:
                    good_until = idx + 1
                    break
            S_e_buf += S[:-1][:good_until]
            A_e_buf += A[:good_until]
            G_e_buf += discount(R[:good_until])
            if Gc0_e[-1] >= basic.beta:
                max_cost_reached += 1
            max_cost_reached_n += 1
        pbar.set_description("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f"
            % (epoch, np.mean(G0_e), np.mean(Gc0_e), float(max_cost_reached / max_cost_reached_n)))
        S_e = torch.tensor(S_e, dtype=torch.float, device=basic.device)
        A_e = torch.tensor(A_e, dtype=torch.long, device=basic.device)
        G_e = torch.tensor(G_e, dtype=torch.float, device=basic.device)
        Gc_e = torch.tensor(Gc_e, dtype=torch.float, device=basic.device)
        S_e_buf = torch.tensor(S_e_buf, dtype=torch.float, device=basic.device)
        A_e_buf = torch.tensor(A_e_buf, dtype=torch.long, device=basic.device)
        G_e_buf = torch.tensor(G_e_buf, dtype=torch.float, device=basic.device)
        log_probs_e_buf = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e_buf)).gather(1, A_e_buf.view(-1, 1)).view(-1)
        buffer.add(S_e_buf, A_e_buf, G_e_buf, log_probs_e_buf.detach())
        feasibility_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate_feasibility)
        # Penalty update on complete trajectories
        for start_index, end_index in Indices:
            Gc = Gc_e[start_index:end_index].view(-1)
            log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_e[start_index:end_index].view(-1, 1)).view(-1)
            feasibility_opt.zero_grad()
            feasibility_loss = (Gc[0] >= basic.beta) * ((Gc * log_probs).sum())
            feasibility_loss.backward()
            feasibility_opt.step()
        # Policy and value update from replay buffer
        for subepoch in range(basic.ppo_subepochs):
            S, A, G, old_log_probs = buffer.sample(basic.minibatch_size)
            log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S)).gather(1, A.view(-1, 1)).view(-1)
            value_opt.zero_grad()
            value_loss = (G - value_nn(S)).pow(2).mean()
            value_loss.backward()
            value_opt.step()
            policy_opt.zero_grad()
            advantages = G - value_nn(S)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - basic.clip_param, 1 + basic.clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            probs_all = torch.nn.Softmax(dim=-1)(policy_nn(S))
            log_probs_all = torch.nn.LogSoftmax(dim=-1)(policy_nn(S))
            entropy = -(probs_all * log_probs_all).sum(1).mean()
            policy_loss -= basic.entropy_coef * entropy
            policy_loss.backward()
            policy_opt.step()
        # Run additional function at epoch end if condition is met
        # Just in case we need it for later! (and we will)
        if additional_fn_condition != None and additional_fn_epoch_end != None:
            if additional_fn_condition(epoch, policy_nn) == True:
                additional_fn_epoch_end(epoch, policy_nn)
        pbar.update(1)




def collect_trajectories(n, env, policy_nn, constraint_fn, only_success=False):
    data = []
    for traj in tqdm(range(n)):
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        while (not (discount(C)[0] <= basic.beta)) and only_success:
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data








# normalizing flow



def flow_get_expert_nll(expert_data):
    flow_data = convert_to_flow_data(expert_data)
    print(flow_data.shape)
    flow_config = tools.data.Configuration({
        "normalize_flow_inputs": True,
        "minibatch_size": basic.minibatch_size,
        "learning_rate": basic.learning_rate,
    })
    flow_config["t"].device = basic.device
    basic.flow = tools.functions.create_flow(flow_config, flow_data, "realnvp", basic.input_dim)
    for flowepoch in range(basic.flow_iters):
        metrics = basic.flow.train()
        print(metrics)
    nll = -basic.flow.log_probs(flow_data)
    expert_nll = (nll.mean(), nll.std())
    return basic.flow, expert_nll




def convert_to_flow_data(data):
    flow_data = []
    for S, A in data:
        for s, a in zip(S, A):
            sa = basic.sa_func(s, a)
            if basic.cs != None:
                sa2 = []
                for csii in basic.cs:
                    sa2 += [sa[csii]]
                sa = sa2
            sa_tensor = torch.tensor(sa, dtype=torch.float, device=basic.device)
            flow_data += [sa_tensor.cpu().numpy()]  # Change this depending on your constraint_nn input
    flow_data = torch.tensor(flow_data, dtype=torch.float, device=basic.device).view(-1, basic.input_dim)# .squeeze()
    return flow_data




def dissimilarity_wrt_expert(data, mean=True):
    expert_nll_mean, expert_nll_std = basic.expert_nll
    sims = []
    for S, A in data:
        traj_data = []
        for s, a in zip(S, A):
            sa = basic.sa_func(s, a)
            if basic.cs != None:
                sa2 = []
                for csii in basic.cs:
                    sa2 += [sa[csii]]
                sa = sa2
            traj_data += [sa]
        # traj_data = [basic.sa_func(s, a) for s, a in zip(S, A)]
        traj_data = torch.tensor(traj_data, dtype=torch.float, device=basic.device).view(-1, basic.input_dim)
        traj_nll = -basic.flow.log_probs(traj_data).detach()
        if np.isnan(traj_nll).any() or np.isnan(expert_nll_mean.detach().numpy()) or np.isnan(expert_nll_std.detach().numpy()):
            sims += [np.random.randint(2)]
        else:
            sims += [(traj_nll > expert_nll_mean + expert_nll_std).float().mean()]
    if mean:
        return np.mean(sims)
    return torch.tensor(sims, dtype=torch.float, device=basic.device)









# constraint function adjustment



# def collect_trajectories_mixture(n, env, policy_mixture, weights_mixture, constraint_fn):
#     data = []
#     value_nn, policy_nn = make_nn()
#     normalized_weights_mixture = np.copy(weights_mixture) / np.sum(weights_mixture)
#     m = len(weights_mixture)
#     for traj in tqdm(range(n)):
#         chosen_policy_idx = np.random.choice(m, p=normalized_weights_mixture)
#         policy_nn.load_state_dict(policy_mixture[chosen_policy_idx])
#         S, A, R, C = play_episode(env, policy_nn, constraint_fn)
#         data += [[S[:-1], A]]
#     return data


def collect_trajectories_mixture(n, env, policy_mixture, weights_mixture, constraint_fn, ppolag_config=None):
    data = []
    value_nn, policy_nn = make_nn()
    normalized_weights_mixture = np.copy(weights_mixture) / np.sum(weights_mixture)
    m = len(weights_mixture)
    for traj in tqdm(range(n)):
        chosen_policy_idx = np.random.choice(m, p=normalized_weights_mixture)
        if ppolag_config == None:
            policy_nn.load_state_dict(policy_mixture[chosen_policy_idx])
        else:
            loadmodel(ppolag_config["sess"], ppolag_config["saver"], policy_mixture[chosen_policy_idx])
            policy_nn = ppolag_config["policy"]
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data



def compute_current_constraint_value_trajectory(constraint_nn, data):
    Gc0 = []
    for S, A in data:
        input_to_constraint_nn = []
        for s, a in zip(S, A):
            sa = basic.sa_func(s, a)
            if basic.cs != None:
                sa2 = []
                for csii in basic.cs:
                    sa2 += [sa[csii]]
                sa = sa2
            sa_tensor = torch.tensor(sa, dtype=torch.float, device=basic.device)
            input_to_constraint_nn += [sa_tensor.cpu().numpy()]  # Change this depending on your constraint_nn input
        input_to_constraint_nn = torch.tensor(input_to_constraint_nn, dtype=torch.float, device=basic.device).view(-1, basic.input_dim)
        constraint_values = constraint_nn(input_to_constraint_nn).view(-1)
        Gc0 += [discount(constraint_values)[0]]
    return torch.stack(Gc0)





def constraint_function_adjustment(n, constraint_nn, constraint_opt, expert_data, agent_data):
    losses = []
    for _ in range(n):
        basic.constraint_opt.zero_grad()
        per_traj_dissimilarity = dissimilarity_wrt_expert(agent_data, mean=False)
        per_traj_dissimilarity = per_traj_dissimilarity / per_traj_dissimilarity.sum()
        agent_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, agent_data)
        expert_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, expert_data)
        loss1 = -(agent_data_constraint_returns * per_traj_dissimilarity).sum()
        if 'gridworld' in basic.env_name:
            loss2 = ((expert_data_constraint_returns >= basic.beta).float() * (expert_data_constraint_returns - basic.beta)).mean()        
        else:
            loss2 = ((expert_data_constraint_returns.mean() >= basic.beta).float()) * ((expert_data_constraint_returns - basic.beta).mean())
        # loss2 = ((expert_data_constraint_returns >= basic.beta).float() * (expert_data_constraint_returns - basic.beta)).mean()
        loss = loss1 + basic.alpha * loss2
        loss.backward()
        constraint_opt.step()
        losses += [loss.item()]
    return np.mean(losses)





def condition(epoch, policy_nn):
    if (epoch + 1) % basic.policy_add_to_mix_every == 0:
        return True
    return False



# def command(epoch, policy_nn):
#     basic.agent_data = collect_trajectories(len(basic.expert_data), basic.env, policy_nn, current_constraint_function)
#     basic.policy_mixture += [copy.deepcopy(policy_nn.state_dict())]
#     basic.weights_mixture += [dissimilarity_wrt_expert(basic.agent_data)]
#     print("Added policy with dissimilarity = %.2f" % basic.weights_mixture[-1])

def command(epoch, policy_nn, ppolag_config=None):
    agent_data = collect_trajectories(len(basic.expert_data), basic.env, policy_nn, current_constraint_function)
    if ppolag_config != None:
        if not os.path.exists("tmp"):
            os.mkdir("tmp")
        fname = "tmp/seed%d-itr%d-%s" % (basic.seed, epoch, tools.utils.timestamp())
        save(ppolag_config["sess"], ppolag_config["saver"], fname, ppolag_config=ppolag_config)
        basic.policy_mixture += [fname]
    else:
        basic.policy_mixture += [copy.deepcopy(policy_nn.state_dict())]
    basic.weights_mixture += [dissimilarity_wrt_expert(agent_data)]
    print("Added policy with dissimilarity = %.2f" % basic.weights_mixture[-1])




# constraint related



def current_constraint_function(sa):
    s, a = sa
    sa = basic.sa_func(s, a)
    if basic.cs != None:
        sa2 = []
        for csii in basic.cs:
            sa2 += [sa[csii]]
        sa = sa2
    sa_tensor = torch.tensor(sa, device=basic.device, dtype=torch.float)
    return basic.constraint_nn(sa_tensor)






def create_constraint_nn(input_dim, hidden_dim):
    constraint_nn = torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, 1), torch.nn.Sigmoid(),
    )
    return constraint_nn










# icl



def icl(expert_data, N=5, save_dir=".", outer_itr=0):
    if not os.path.exists(f"{save_dir}/{outer_itr}"):
        os.mkdir(f"{save_dir}/{outer_itr}")
    basic.expert_data = expert_data
    basic.flow, basic.expert_nll = flow_get_expert_nll(basic.expert_data)
    basic.constraint_nn = create_constraint_nn(basic.input_dim, basic.hidden_dim)
    basic.constraint_opt = torch.optim.Adam(basic.constraint_nn.parameters(), lr=basic.learning_rate)
    basic.policy_mixture, basic.weights_mixture = [], []
    expert_satisfaction = (compute_current_constraint_value_trajectory(basic.constraint_nn, basic.expert_data) <= basic.beta).float().mean()
    print("Expert satisfaction = %.2f" % expert_satisfaction)
    # nad_value = None
    for itr in range(N):
        print(f"ICL iteration: {itr}")
        ppolag_config = None
        if basic.use_ppo_lag:
            ppolag_config = ppo_lag(basic.ppo_iters, basic.env, current_constraint_function, condition, command)
            policy_nn = ppolag_config["policy"]
        else:
            value_nn, policy_nn = make_nn()
            ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, current_constraint_function, condition, command)
        basic.agent_data = collect_trajectories(len(basic.expert_data), basic.env, policy_nn, current_constraint_function)
        torch.save([basic.constraint_nn.state_dict(), basic.agent_data], f"{save_dir}/{outer_itr}/{itr}.pt")
        mixture_data = collect_trajectories_mixture(len(basic.expert_data), basic.env, basic.policy_mixture, basic.weights_mixture, current_constraint_function,
            ppolag_config=ppolag_config if basic.use_ppo_lag else None)
        mixture_data_constraint_returns = compute_current_constraint_value_trajectory(basic.constraint_nn, mixture_data)
        for _ in tqdm(range(basic.ca_iters)):
            constraint_function_adjustment(basic.ca_iters, basic.constraint_nn, basic.constraint_opt, basic.expert_data, mixture_data)
        expert_satisfaction = (compute_current_constraint_value_trajectory(basic.constraint_nn, basic.expert_data) <= basic.beta).float().mean()
        print("Expert satisfaction = %.2f" % expert_satisfaction)
        # nad_value = nad(basic.expert_data, basic.agent_data)
        # print("NAD = %.2f" % nad_value)
    print("Final policy")
    if basic.use_ppo_lag:
        ppolag_config = ppo_lag(basic.ppo_iters, basic.env, current_constraint_function, condition, command)
        policy_nn = ppolag_config["policy"]
    else:
        value_nn, policy_nn = make_nn()
        ppo_penalty(basic.ppo_iters, basic.env, policy_nn, value_nn, current_constraint_function, condition, command)
    basic.agent_data = collect_trajectories(len(basic.expert_data), basic.env, policy_nn, current_constraint_function)
    torch.save([basic.constraint_nn.state_dict(), basic.agent_data], f"{save_dir}/{outer_itr}/final.pt")
    return basic.constraint_nn, basic.agent_data













# Accrual






def wasserstein_distance2d(u, v, p='cityblock'):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape and len(u.shape) == 2)
    dim1, dim2 = u.shape
    assert(p in ['euclidean', 'cityblock'])
    coords = np.zeros((dim1*dim2, 2)).astype('float')
    for i in range(dim1):
        for j in range(dim2):
            coords[i*dim2+j, :] = [i, j]
    d = cdist(coords, coords, p)
    u /= u.sum()
    v /= v.sum()
    return ot.emd2(u.flatten(), v.flatten(), d)







def nad(expert_data, data):
    if basic.env_name == 'gridworld':
        accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in data:
            for s, a in zip(S, A):
                # print(s,a)
                x, y = s
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        expert_accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in expert_data:
            for s, a in zip(S, A):
                # print(s,a)
                x, y = s
                expert_accrual[a][x][y] += 1
        expert_accrual = np.mean(expert_accrual, axis=0)
        expert_accrual /= (np.max(expert_accrual)+1e-6)
        return wasserstein_distance2d(expert_accrual, accrual)
    elif basic.env_name == 'cartpole':
        rng = np.arange(-2.4, 2.4+0.1, 0.1)
        accrual = np.zeros((2, len(rng)))
        for S, A in data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 48)
                accrual[int(a)][bin_nbr] += 1
        accrual[0, :] /= (accrual[0, :].max()+1e-6)
        accrual[1, :] /= (accrual[1, :].max()+1e-6)
        expert_accrual = np.zeros((2, len(rng)))
        for S, A in expert_data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 48)
                expert_accrual[int(a)][bin_nbr] += 1
        expert_accrual[0, :] /= (expert_accrual[0, :].max()+1e-6)
        expert_accrual[1, :] /= (expert_accrual[1, :].max()+1e-6)
        return 0.5*(wasserstein_distance2d(expert_accrual[0, :].reshape(1, -1), accrual[0, :].reshape(1, -1))+\
            wasserstein_distance2d(expert_accrual[1, :].reshape(1, -1), accrual[1, :].reshape(1, -1)))
    elif basic.env_name == 'highd':
        expert_accrual_vel = np.zeros((81))
        expert_accrual_c2c = np.zeros((201))
        for S, A in expert_data:
            for s, a in zip(S, A):
                if 0 <= s[-1]*2.5*basic.state_scaling[-1] <= 200:
                    v = s[2]*basic.state_scaling[2]
                    c2c = s[-1]*2.5*basic.state_scaling[-1]
                    expert_accrual_vel[int(np.round(v))] += 1
                    expert_accrual_c2c[int(np.round(c2c))] += 1
        expert_accrual_vel /= np.sum(expert_accrual_vel)
        expert_accrual_c2c /= np.sum(expert_accrual_c2c)
        accrual_vel = np.zeros((81))
        accrual_c2c = np.zeros((201))
        for S, A in data:
            for s, a in zip(S, A):
                if 0 <= s[-1]*2.5*basic.state_scaling[-1] <= 200:
                    v = s[2]*basic.state_scaling[2]
                    c2c = s[-1]*2.5*basic.state_scaling[-1]
                    accrual_vel[int(np.round(v))] += 1
                    accrual_c2c[int(np.round(c2c))] += 1
        accrual_vel /= np.sum(accrual_vel)
        accrual_c2c /= np.sum(accrual_c2c)
        return 0.5*(wasserstein_distance2d(expert_accrual_vel.reshape(1, -1), accrual_vel.reshape(1, -1))+\
            wasserstein_distance2d(expert_accrual_c2c.reshape(1, -1), accrual_c2c.reshape(1, -1)))