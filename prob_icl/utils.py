import basic
import copy
import logging
import random
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch
import tqdm
import tools
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.spatial.distance import cdist
import ot
import gc
import gym
from shapely.geometry import LineString, Point
import mpl_toolkits.axes_grid1.inset_locator as mpl_il
from matplotlib.patches import Polygon as Polygon2
import decimal
import time
from collections import deque
from typing import Optional
from gym.utils.step_api_compatibility import step_api_compatibility
from gym import ActionWrapper
from gym.spaces import Box
from typing import Any, Callable
from gym import RewardWrapper
import tensorflow as tf
from mpl_toolkits import axes_grid1

# def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
#     """Add a vertical color bar to an image plot."""
#     divider = axes_grid1.make_axes_locatable(im.axes)
#     width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
#     pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
#     current_ax = plt.gca()
#     cax = divider.append_axes("right", size=width, pad=pad)
#     plt.sca(current_ax)
#     return im.axes.figure.colorbar(im, cax=cax, **kwargs)

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


# https://stackoverflow.com/questions/11325019/how-to-output-to-the-console-and-file
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

def seed_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    # torch.cuda.memory._record_memory_history(enabled=True)

def human_time(c):
    days = c // 86400
    hours = c // 3600 % 24
    minutes = c // 60 % 60
    seconds = int(c % 60)
    if days == 0:
        return "%g:%g:%g" % (hours, minutes, seconds)
    return "%g days, %g:%g:%g" % (days, hours, minutes, seconds)

# unnormalize_mujoco only triggered for mujoco
def true_constraint_function(sa, unnormalize_mujoco=True):
    if basic.env_name == 'gridworld':
        s, a = sa
        x, y = s[0], s[1]
        u = [(ui, uj) for ui in [3] for uj in [0, 1, 2, 3]]
        if (x, y) in u:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    elif basic.env_name == 'gridworld2':
        # modified environment to test empirical beta(1-delta)
        s, a = sa
        x, y = s[0], s[1]
        u = [(ui, uj) for ui in [2,3,4] for uj in [0, 1, 2, 3]]
        if (x, y) in u:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    elif basic.env_name == 'cartpole':
        s, a = sa
        # if s[0] <= 0.2:
        # if (s[0] > 1. and a == 1) or (s[0] < -1. and a == 0):
        # if (s[0] <= -1.75 and a == 0) or (-1 <= s[0] <= 1) or (s[0] >= 1.75 and a == 1):
        if (s[0] < -1 or s[0] > 1):
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    elif 'mujoco' in basic.env_name:
        s, a = sa
        if unnormalize_mujoco:
            s = basic.env.unnormalize(s)
        if s[0] <= -1:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()
    elif basic.env_name == 'highd':
        print("True constraint function not available - highd")
        exit(0)
    elif basic.env_name == 'exid':
        print("True constraint function not available - exid")
        exit(0)
    else:
        print('Bad env_name (true_constraint_function)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def constraint_mse(true_constraint_fn, constraint_fn):
    if true_constraint_fn == None or constraint_fn == None:
        return None
    if 'gridworld' in basic.env_name:
        grid_for_action, true_grid_for_action = [], []
        for a in np.arange(basic.n_actions):
            grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            true_grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            for x in np.arange(basic.gridworld_dim):
                for y in np.arange(basic.gridworld_dim):
                    grid[x, y] = constraint_fn(([x, y], a)).item()
                    true_grid[x, y] = true_constraint_fn(([x, y], a)).item()
            grid_for_action += [grid]
            true_grid_for_action += [true_grid]
        average_grid = np.mean(grid_for_action, axis=0)
        true_average_grid = np.mean(true_grid_for_action, axis=0)
        return mse(true_average_grid, average_grid)
    elif basic.env_name == 'cartpole':
        a0, a1 = [], []
        true_a0, true_a1 = [], []
        for x in np.arange(-2.4, 2.4+0.1, 0.1):
            a0 += [constraint_fn(([x], 0)).item()]
            a1 += [constraint_fn(([x], 1)).item()]
            true_a0 += [true_constraint_fn(([x], 0)).item()]
            true_a1 += [true_constraint_fn(([x], 1)).item()]
        true_vals = np.array([true_a0, true_a1])
        vals = np.array([a0, a1])
        return mse(true_vals, vals)
    elif 'mujoco' in basic.env_name:
        a = []
        true_a = []
        for x in np.arange(-5, 5+0.1, 0.1):
            a += [constraint_fn(([x], np.zeros(8)), unnormalize_mujoco=False).item()]
            true_a += [true_constraint_fn(([x], np.zeros(8)), unnormalize_mujoco=False).item()]
        true_vals = np.array([true_a])
        vals = np.array([a])
        return mse(true_vals, vals)
    elif basic.env_name == 'highd':
        return None
    elif basic.env_name == 'exid':
        return None
    else:
        print('Bad env_name (CMSE)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def accrual_comparison(expert_data, data):
    if expert_data == None or data == None:
        return None
    if 'gridworld' in basic.env_name:
        accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in data:
            for s, a in zip(S, A):
                x, y = s
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        expert_accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in expert_data:
            for s, a in zip(S, A):
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
    elif 'mujoco' in basic.env_name:
        rng = np.arange(-5, 5+0.1, 0.1)
        accrual = np.zeros((1, len(rng)))
        for S, A in data:
            for s in S:
                if -5 <= s[0] <= 5:
                    bin_nbr = np.clip(int(np.floor((s[0]+5+0.1/2)/0.1)), 0, 100)
                    accrual[0][bin_nbr] += 1
        accrual[0, :] /= (accrual[0, :].max()+1e-6)
        expert_accrual = np.zeros((1, len(rng)))
        for S, A in expert_data:
            for s in S:
                bin_nbr = np.clip(int(np.floor((s[0]+5+0.1/2)/0.1)), 0, 100)
                expert_accrual[0][bin_nbr] += 1
        expert_accrual[0, :] /= (expert_accrual[0, :].max()+1e-6)
        return wasserstein_distance2d(expert_accrual[0, :].reshape(1, -1), accrual[0, :].reshape(1, -1))
    elif basic.env_name == 'highd': # TODO
        return None
    elif basic.env_name == 'exid': # TODO
        return None
    else:
        print('Bad env_name (NAD)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

def show_metrics(p=False):
    if p:
        expert_satisfaction = \
            compute_current_constraint_value_trajectory(\
                basic.constraint_nn, 
                basic.expert_data,
                p=p\
            ).mean()
        print("Expert => P_aprx(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (expert_satisfaction, basic.beta, basic.delta))
        expert_cdf = compute_cdf(basic.constraint_nn, basic.expert_data)
        agent_cdf = compute_cdf(basic.constraint_nn, basic.agent_data)
        if expert_cdf != None:
            print("Expert => P_empr(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (expert_cdf, basic.beta, basic.delta))
        else:
            print("Expert => P_empr(C<=beta) = None, beta = %.2f, delta = %.2f" % (basic.beta, basic.delta))
        if agent_cdf != None:
            print("Agent => P_empr(C<=beta) = %.2f, beta = %.2f, delta = %.2f" % (agent_cdf, basic.beta, basic.delta))
        else:
            print("Agent => P_empr(C<=beta) = None, beta = %.2f, delta = %.2f" % (basic.beta, basic.delta))
    else:
        expert_satisfaction = \
            (compute_current_constraint_value_trajectory(\
                basic.constraint_nn, 
                basic.expert_data,
                p=p\
            ) <= basic.beta).float().mean()
        print("Expert => ExpSat = %.2f, beta = %.2f" % (expert_satisfaction, basic.beta))
        expert_satisfaction2 = \
        compute_current_constraint_value_trajectory(\
            basic.constraint_nn, 
            basic.expert_data,
            p=True\
        ).mean()
        print("Expert => P_aprx(C<=beta) = %.2f, beta = %.2f" % (expert_satisfaction2, basic.beta))
        expert_cdf = compute_cdf(basic.constraint_nn, basic.expert_data)
        agent_cdf = compute_cdf(basic.constraint_nn, basic.agent_data)
        if expert_cdf != None:
            print("Expert => P_empr(C<=beta) = %.2f, beta = %.2f" % (expert_cdf, basic.beta))
        else:
            print("Expert => P_empr(C<=beta) = None, beta = %.2f" % (basic.beta))
        if agent_cdf != None:
            print("Agent => P_empr(C<=beta) = %.2f, beta = %.2f" % (agent_cdf, basic.beta))
        else:
            print("Agent => P_empr(C<=beta) = None, beta = %.2f" % (basic.beta))
    cmse = constraint_mse(true_constraint_function, current_constraint_function)
    if cmse == None:
        print("CMSE: None")
    else:
        print("CMSE: %.2f" % cmse)
    nad = accrual_comparison(basic.expert_data, basic.agent_data)
    if nad == None:
        print("NAD: None")
    else:
        print("NAD: %.2f" % nad)

class HighDStateScaling(tools.base.Environment):
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        step_data = self.env.step(action)
        if type(step_data) == dict:
            observation, reward, done, info = step_data["next_state"], step_data["reward"],\
                step_data["done"], step_data["info"]
        else:
            observation, reward, done, info = step_data
        observation = [observation[i]/basic.state_scaling[i] for i in range(len(observation))]
        if type(step_data) == dict:
            return {
                "next_state": observation, 
                "reward": reward, 
                "done": done, 
                "info": info
            }
        else:
            return observation, reward, done, info

    def seed(self, s=None):
        return self.env.seed(s=s)

    @property
    def state(self):
        return [self.env.state[i]/basic.state_scaling[i] for i in range(len(basic.state_scaling))]

    def reset(self, **kwargs):
        s = self.env.reset(**kwargs)
        assert(s is not tuple)
        s = [s[i]/basic.state_scaling[i] for i in range(len(s))]
        return s
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)

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

def create_env():
    if 'gridworld' in basic.env_name:
        basic.gridworld_dim = 7
        basic.n_actions = 8
        basic.constraint_fn_input_dim = 2
        basic.beta = 0.99
        basic.discount_factor = 1.0
        basic.ppo_iters = 500
        basic.policy_add_to_mix_every = 250
        r = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
        r[6, 0] = 1.0
        t = [(6, 0)]
        if basic.env_name == 'gridworld2':
            # modified environment to test empirical beta+log(1-delta)
            u = [(ui, uj) for ui in [2, 3, 4] for uj in [0, 1, 2, 3]] 
        else:
            u = [(ui, uj) for ui in [3] for uj in [0, 1, 2, 3]] 
        s = [(ui, uj) for ui in [0, 1, 2] for uj in [0, 1]]
        env = tools.environments.GridworldEnvironment(
            start_states=s,
            t=t,
            r=r,
            unsafe_states=u,
            stay_action=False,  # no action to "stay in current cell"
        )
        env = tools.environments.TimeLimit(env, 50)
        env = tools.environments.FollowGymAPI(env)
    elif basic.env_name == 'cartpole':
        basic.n_iters = 15 # 5 for PPO Lag
        basic.hidden_dim = 64
        basic.minibatch_size = 64
        basic.constraint_fn_input_dim = 2
        basic.beta = 20
        basic.max_steps_per_epoch = 4000
        # basic.ppo_subepochs = 100 - uncomment for PPO Lag
        basic.time_limit = 200
        basic.use_gae = True
        basic.use_early_stopping = True
        basic.learning_rate = 5e-4 # only for test time, ppo pen gae
        basic.learning_rate_feasibility = basic.learning_rate * 100 # only for test time, ppo pen gae
        basic.discount_factor = 0.99
        basic.ppo_iters = 300
        basic.policy_add_to_mix_every = 300
        env = tools.environments.GymEnvironment(
            "CustomCartPole", 
            start_pos=[[-2, 2]], # [[-2.4, -1.15], [1.15, 2.4]]
        )
        env = tools.environments.TimeLimit(env, 200)
        env = tools.environments.FollowGymAPI(env)
    elif 'mujoco' in basic.env_name:
        basic.hidden_dim = 64
        basic.episodes_per_epoch = 100
        basic.continuous_actions = True
        basic.constraint_fn_input_dim = 1
        basic.n_iters = 10
        basic.flow_iters = 20
        basic.discount_factor = 0.99
        # basic.learning_rate = 5e-4 - uncomment for PPO Lag
        basic.max_steps_per_epoch = 4000
        basic.ppo_subepochs = 25 # 80 - PPO Lag
        basic.minibatch_size = 64
        basic.clip_param = 0.2
        basic.entropy_coef = 0
        basic.gae_lambda = 0.97
        basic.replay_buffer_size = 50000 # 10000 - PPO Lag
        basic.beta = 15
        basic.use_gae = True
        basic.use_early_stopping = True
        # basic.learning_rate = 5e-4 # only for test time, ppo pen gae
        # basic.learning_rate_feasibility = basic.learning_rate * 10 # only for test time, ppo pen gae
        if 'ant' in basic.env_name:
            env = tools.environments.GymEnvironment('AntWall-v0') # or use HCWithPos-v0
            basic.ppo_iters = 100
            basic.policy_add_to_mix_every = 50 # 50 for PPO Lag
            basic.beta = 15
            basic.alpha = 1 # comment for PPO Lag
            basic.ca_iters = 10 # comment for PPO Lag
        elif 'hc' in basic.env_name:
            env = tools.environments.GymEnvironment('HCWithPos-v0')
            basic.ppo_iters = 100 # 50 for ppo pen gae # 50 for PPO Lag
            basic.policy_add_to_mix_every = 20 # 25 for ppo pen gae # 50 for PPO Lag
            basic.beta = 15
            basic.learning_rate = 5e-4 # only for test time, ppo pen gae
            basic.learning_rate_feasibility = basic.learning_rate * 100 # only for test time, ppo pen gae
            # basic.n_iters = 9
            # basic.hidden_dim = 32
            basic.max_steps_per_epoch = 8000 # comment for PPO Lag
            # basic.n_iters = 5
            basic.clip_param = 0.2
            basic.entropy_coef = 0
            basic.gae_lambda = 0.97
            # basic.learning_rate_feasibility = basic.lrf
            basic.alpha = 1 # comment for PPO Lag
            basic.ca_iters = 5 # comment for PPO Lag
        else:
            print("Mujoco environment not supported")
            exit(0)
        # basic.ppo_iters = 1
        # basic.policy_add_to_mix_every = 1
        # basic.ca_iters = 1
        # basic.flow_iters = 1
        env = tools.environments.FollowGymAPI(env)
        env = RecordEpisodeStatistics(env)
        if 'ant' in basic.env_name:
            env = tools.environments.TimeLimit(env, 500)
            basic.time_limit = 500
        elif 'hc' in basic.env_name:
            env = tools.environments.TimeLimit(env, 500)
            basic.time_limit = 500
        else:
            print("Mujoco environment not supported")
            exit(0)
        env = ClipAction(env)
        # env = NormalizeReward(env, gamma=basic.discount_factor)
        # env = TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    elif basic.env_name == 'highd':
        basic.episodes_per_epoch = 20
        basic.max_steps_per_epoch = 20000
        basic.time_limit = 1000
        # basic.flow_iters = 1
        basic.use_gae = True
        basic.use_early_stopping = True
        basic.ppo_subepochs = 25
        basic.clip_param = 0.1
        basic.entropy_coef = 0.01
        basic.beta = 0.1
        basic.ppo_iters = 50
        basic.policy_add_to_mix_every = 25
        basic.constraint_fn_input_dim = 2
        basic.continuous_actions = True
        basic.discount_factor = 1.0
        basic.n_iters = 3
        basic.expert_data = torch.load("data/expert_data_highd.pt")
        basic.state_scaling = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        # basic.state_scaling = [50, 3, 5, 2, 1, 1, 1, 1, 1, 1, 1, 1, 100, 100]
        # basic.state_scaling = [500, 30, 50, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000]
        # basic.state_scaling[-1] = 200
        new_data = []
        for S, A in basic.expert_data:
            new_S, new_A = [], []
            for s, a in zip(S, A):
                new_s = [s[i]/basic.state_scaling[i] for i in range(len(s))]
                new_S += [new_s]
                new_A += [a]
            new_data += [[new_S, new_A]]
        basic.expert_data = new_data
        env = tools.environments.HighDSampleEnvironmentWrapper(discrete=False)
        env = HighDStateScaling(env)
        env = tools.environments.TimeLimit(env, 1000)
        env = tools.environments.FollowGymAPI(env)
    elif basic.env_name == 'exid':
        basic.extra_kwargs = {
            "lead_value": 15./100,
            "rear_value": -25./100,
            "overlap": 0,
        }
        LANE_CHANGE_EXPERT_DATA = "data/expert_data_exid.pt"
        LANE_CHANGE_DATA = "data/lane_changes_data_exid.pt"
        LANE_CHANGE_IDS = "data/lane_changes_exid.pt"
        basic.expert_data, basic.bad_ids = torch.load(LANE_CHANGE_EXPERT_DATA)
        basic.lane_change_info = torch.load(LANE_CHANGE_DATA)
        basic.lane_change_ids = torch.load(LANE_CHANGE_IDS)
        basic.episodes_per_epoch = 5
        basic.ppo_subepochs = 20
        basic.learning_rate = 1e-3
        basic.learning_rate_feasibility = 1e-4
        basic.clip_param = 0.3
        basic.entropy_coef = 0.001
        basic.beta = 10
        basic.ppo_iters = 100
        basic.policy_add_to_mix_every = 50
        basic.constraint_fn_input_dim = 5
        basic.continuous_actions = True
        basic.discount_factor = 1.0
        basic.n_iters = 5
        env = ExidEnv(bad_ids=basic.bad_ids)
        env = tools.environments.FollowGymAPI(env)
    else:
        print('Bad env_name (create_env)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    # basic.ppo_iters = 2
    # basic.policy_add_to_mix_every = 1
    # basic.ca_iters = 1
    # basic.flow_iters = 1
    # basic.n_iters = 3
    return env

def gridworld_imshow(m, fig, ax):
    m = np.array(m).squeeze()
    assert len(m.shape) == 2
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    m = (m-np.min(m))/(np.max(m)-np.min(m)+1e-3)
    im = ax.imshow(m, cmap="gray")
    # im.set_clim(0, 1)
    ax.set_xticks(np.arange(m.shape[0]))
    ax.set_yticks(np.arange(m.shape[1]))
    cbar = fig.colorbar(im, cax=cax)

def visualize_constraint(constraint_fn, savefig=None, fig=None, ax=None):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    if 'gridworld' in basic.env_name:
        grid_for_action = []
        for a in np.arange(basic.n_actions):
            grid = np.zeros((basic.gridworld_dim, basic.gridworld_dim))
            for x in np.arange(basic.gridworld_dim):
                for y in np.arange(basic.gridworld_dim):
                    grid[x, y] = constraint_fn(([x, y], a))
            grid_for_action += [grid]
        average_grid = np.mean(grid_for_action, axis=0)
        gridworld_imshow(average_grid, fig, ax)
    elif basic.env_name == 'cartpole':
        a0, a1 = [], []
        for x in np.arange(-2.4, 2.4+0.1, 0.1):
            a0 += [constraint_fn(([x], 0)).item()]
            a1 += [constraint_fn(([x], 1)).item()]
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), a0, label="a=0", color='blue')
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), a1, label="a=1", color='red')
        ax.legend(loc='best')
        ax.set_ylim(0-0.05, 1+0.05)
        ax.margins(0.05, 0.05)
    elif 'mujoco' in basic.env_name:
        a = []
        for x in np.arange(-5, 5+0.1, 0.1):
            a += [constraint_fn(([x], np.zeros(8)), unnormalize_mujoco=False).item()]
        ax.plot(np.arange(-5, 5+0.1, 0.1), a, color='pink')
        ax.set_ylim(0-0.05, 1+0.05)
        ax.margins(0.05, 0.05)
    elif basic.env_name == 'highd':
        grid = np.zeros((41, 81))
        for vel in np.arange(0, 41, 1):
            for c2c in np.arange(0, 81, 1):
                # actually state space has many features but we need only 3rd and last feature
                grid[vel][c2c] = constraint_fn(([0, 0, vel/basic.state_scaling[2], c2c/basic.state_scaling[-1]], [0, 0]))
        grid = grid.T
        im = ax.imshow(grid, cmap="gray", origin='lower', extent=[0,40,0,200], aspect=0.2)
        im.set_clim(0, 1)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel("Gap (m)")
        ax.set_xlabel("Ego velocity (m/s)")
    elif basic.env_name == 'exid':
        extra_kwargs = basic.extra_kwargs
        assert("lead_value" in extra_kwargs.keys())
        assert("rear_value" in extra_kwargs.keys())
        assert("overlap" in extra_kwargs.keys())
        lead_value = extra_kwargs["lead_value"]
        rear_value = extra_kwargs["rear_value"]
        overlap = extra_kwargs["overlap"]
        g = np.zeros((61, 81))
        for sgn_dist in np.arange(-2., 2.+0.05, 0.05):
            for a in np.arange(-3, 3+0.1, 0.1):
                g[int(a*10+30)][int((sgn_dist*4+8.)*5)] = float(constraint_fn(([sgn_dist, lead_value, rear_value, overlap], [a])).detach())
        im = ax.imshow(np.flipud(g), cmap="gray", extent = [-3, 3, -10, 10], aspect='auto')
        im.set_clim(0, 1)
        ax.set_ylim([-10, 10])
        ax.set_xlim([-3, 3])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        xp, yp = [], []
        for S, A in basic.expert_data:
            for s, a in zip(S, A):
                # if lead_value-2.5 <= s[1]*100 <= lead_value+2.5 and \
                # rear_value-2.5 <= s[2]*100 <= rear_value+2.5 and \
                # overlap == s[3] and \
                if -10 <= s[0]*5 <= 10 and \
                -3 <= a[0] <= 3:
                    xp += [a[0]]
                    yp += [s[0]*5]
        ax.scatter(xp, yp, s=1, color='red')
        # add_colorbar(im)
        # ax.set_title("L=%s, R=%s, O=%s" % 
        #                         (
        #                             str(lead_value)+" m" if lead_value != 1 else "None",
        #                             str(rear_value)+" m" if rear_value != -1 else "None",
        #                             "yes" if overlap == 1 else "no",
        #                         ), fontsize=10, color="brown")
        ax.set_ylabel("signed distance to target lane (m)")
        ax.set_xlabel("lateral velocity action (m/s)")
    else:
        print('Bad env_name (viz_constraint)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    if savefig != None:
        fig.tight_layout()
        fig.savefig(savefig)

def visualize_accrual(data, savefig=None, fig=None, ax=None):
    if fig == None and ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    if 'gridworld' in basic.env_name:
        accrual = np.array([np.zeros((basic.gridworld_dim, basic.gridworld_dim)) for _ in range(basic.n_actions)])
        for S, A in data:
            for s, a in zip(S, A):
                x, y = s
                accrual[a][x][y] += 1
        accrual = np.mean(accrual, axis=0)
        accrual /= (np.max(accrual)+1e-6)
        gridworld_imshow(accrual, fig, ax)
    elif basic.env_name == 'cartpole':
        accrual = np.zeros_like(np.arange(-2.4, 2.4+0.1, 0.1))
        for S, A in data:
            for s, a in zip(S, A):
                bin_nbr = np.clip(int(np.floor((s[0]+2.4+0.1/2)/0.1)), 0, 49)
                accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax.plot(np.arange(-2.4, 2.4+0.1, 0.1), accrual, color='green')
        ax.margins(0.05, 0.05)
    elif 'mujoco' in basic.env_name:
        accrual = np.zeros_like(np.arange(-5, 5+0.1, 0.1))
        for S, A in data:
            for s in S:
                if -5 <= s[0] <= 5:
                    bin_nbr = np.clip(int(np.floor((s[0]+5+0.1/2)/0.1)), 0, 100)
                    accrual[bin_nbr] += 1
        accrual /= (accrual.max()+1e-6)
        ax.plot(np.arange(-5, 5+0.1, 0.1), accrual, color='green')
        ax.margins(0.05, 0.05)
    elif basic.env_name == 'highd':
        vel, c2c = [], []
        for S, A in data:
            for s, a in zip(S, A):
                if 0 <= s[-1]*2.5*basic.state_scaling[-1] <= 200:
                    vel += [s[2]*basic.state_scaling[2]]
                    c2c += [s[-1]*2.5*basic.state_scaling[-1]]
        ax.scatter(vel, c2c, s=1, color='red')
        ax.set_xlim([0,40])
        ax.set_ylim([0,200])
        ax.set_ylabel("Gap (m)")
        ax.set_xlabel("Ego velocity (m/s)")
    elif basic.env_name == 'exid':
        extra_kwargs = basic.extra_kwargs
        assert("lead_value" in extra_kwargs.keys())
        assert("rear_value" in extra_kwargs.keys())
        assert("overlap" in extra_kwargs.keys())
        lead_value = extra_kwargs["lead_value"]
        rear_value = extra_kwargs["rear_value"]
        overlap = extra_kwargs["overlap"]
        # fig, ax = plt.subplots(1, 1, figsize=(3, 4))
        xp, yp = [], []
        for S, A in data:
            for s, a in zip(S, A):
                # if lead_value-2.5/100. <= s[1] <= lead_value+2.5/100 and \
                #     rear_value-2.5/100 <= s[2] <= rear_value+2.5/100 and \
                #     overlap == s[3]:
                xp += [a[0]]
                yp += [s[0]*5]
        ax.scatter(xp, yp, s=1, color='red')
        # ax.set_title("L=%s, R=%s, O=%s" % 
        #                         (
        #                             str(lead_value)+" m" if lead_value != 1 else "None",
        #                             str(rear_value)+" m" if rear_value != -1 else "None",
        #                             "yes" if overlap == 1 else "no",
        #                         ), fontsize=10, color="brown")
        ax.set_ylabel("signed distance to target lane (m)")
        ax.set_xlabel("lateral velocity action (m/s)")
    else:
        print('Bad env_name (Viz_accrual)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)
    if savefig != None:
        fig.tight_layout()
        fig.savefig(savefig)

# unnormalize_mujoco only triggered for mujoco environment
def current_constraint_function(sa, unnormalize_mujoco=True):
    s, a = sa
    if 'gridworld' in basic.env_name:
        return basic.constraint_nn(torch.tensor(s, device=basic.device, dtype=torch.float)).detach().cpu()  # Change this depending on the constraint_nn input
    elif basic.env_name == 'cartpole':
        return basic.constraint_nn(torch.tensor([s[0], a], device=basic.device, dtype=torch.float)).detach().cpu()
    elif 'mujoco' in basic.env_name:
        new_s = s
        if unnormalize_mujoco:
            new_s = basic.env.unnormalize(new_s)
        return basic.constraint_nn(torch.tensor([new_s[0]], device=basic.device, dtype=torch.float)).detach().cpu()
    elif basic.env_name == 'highd':
        return basic.constraint_nn(torch.tensor([s[2], s[-1]], device=basic.device, dtype=torch.float)).detach().cpu()
    elif basic.env_name == 'exid':
        # return basic.constraint_nn(torch.tensor([s[0], a[0]], device=basic.device, dtype=torch.float)).detach().cpu()
        return basic.constraint_nn(torch.tensor([*s, *a], device=basic.device, dtype=torch.float)).detach().cpu()
    else:
        print('Bad env_name (current_constraint_function)')
        print('Allowed: %s' % basic.env_names_allowed)
        exit(0)

class Gaussian3(torch.jit.ScriptModule):
    def __init__(self, i, o, h=64):
        super().__init__()
        self.actor_mean = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.Tanh(),
            torch.nn.Linear(h, h), torch.nn.Tanh(),
            torch.nn.Linear(h, o)
        )
        self.actor_logstd = torch.nn.Parameter(torch.zeros(o))
    @torch.jit.script_method
    def forward(self, x):
        action_mean = self.actor_mean(x)
        if len(action_mean.shape) == 2:
            action_logstd = self.actor_logstd.repeat(action_mean.shape[0], 1)
        else:
            action_logstd = self.actor_logstd
        action_std = torch.exp(action_logstd)
        return action_mean, action_std

def make_nn():
    value_nn = torch.nn.Sequential(
        torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
        torch.nn.Linear(basic.hidden_dim, 1),
    ).to(basic.device)
    if not basic.continuous_actions:
        policy_nn = torch.nn.Sequential(
            torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(basic.hidden_dim, basic.act_n),
        ).to(basic.device)
    else:
        # if 'mujoco' in basic.env_name:
        #     policy_nn = torch.nn.Sequential(
        #         torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.Tanh(),
        #         torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.Tanh(),
        #         Gaussian3(basic.hidden_dim, basic.act_n),
        #     ).to(basic.device)
        #     policy_nn = torch.jit.script(policy_nn)
        #     print("cleanrl policy_nn")
        # else:
        policy_nn = torch.nn.Sequential(
            torch.nn.Linear(basic.obs_n, basic.hidden_dim), torch.nn.Tanh(),
            torch.nn.Linear(basic.hidden_dim, basic.hidden_dim), torch.nn.Tanh(),
            tools.utils.Gaussian2(basic.hidden_dim, basic.act_n),
        ).to(basic.device)
    return value_nn, policy_nn

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
        if not basic.continuous_actions:
            self.A = torch.zeros((self.N), dtype=torch.long, device=basic.device)
        else:
            self.A = torch.zeros((self.N, basic.act_n), dtype=torch.float, device=basic.device)
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
        minibatch = random.sample(range(self.filled), min(n, self.filled))
        S, A, G, log_probs = [], [], [], []
        for mbi in minibatch:
            s, a, g, lp = self.S[mbi], self.A[mbi], self.G[mbi], self.log_probs[mbi]
            S += [s]
            A += [a]
            G += [g]
            log_probs += [lp]
        return torch.stack(S), torch.stack(A), torch.stack(G), torch.stack(log_probs)

class ReplayBufferGAE:
    def __init__(self, N):
        self.N = N
        self.S = torch.zeros((self.N, basic.obs_n), dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            self.A = torch.zeros((self.N), dtype=torch.long, device=basic.device)
        else:
            self.A = torch.zeros((self.N, basic.act_n), dtype=torch.float, device=basic.device)
        self.G = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.Adv = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.V = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.log_probs = torch.zeros((self.N), dtype=torch.float, device=basic.device)
        self.i = 0
        self.filled = 0

    def add(self, S, A, G, log_probs, Adv, V):
        M = S.shape[0]
        self.filled = min(self.filled + M, self.N)
        assert M <= self.N
        for j in range(M):
            self.S[self.i] = S[j, :]
            self.A[self.i] = A[j]
            self.G[self.i] = G[j]
            self.log_probs[self.i] = log_probs[j]
            self.Adv[self.i] = Adv[j]
            self.V[self.i] = V[j]
            self.i = (self.i + 1) % self.N

    def sample(self, n):
        minibatch = random.sample(range(self.filled), min(n, self.filled))
        S, A, G, log_probs, Adv, V = [], [], [], [], [], []
        for mbi in minibatch:
            s, a, g, lp, adv, v = self.S[mbi], self.A[mbi], self.G[mbi], self.log_probs[mbi], self.Adv[mbi], self.V[mbi]
            S += [s]
            A += [a]
            G += [g]
            log_probs += [lp]
            Adv += [adv]
            V += [v]
        return torch.stack(S), torch.stack(A), torch.stack(G), torch.stack(log_probs), torch.stack(Adv), torch.stack(V)


def sigmoid(x, c):
    return 1./(1.+torch.exp(-c*x))

def sample_gumbel(n, k):
    unif = torch.distributions.Uniform(0,1).sample((n, k))
    g = -torch.log(-torch.log(unif)).to(basic.device)
    return g

def sample_gumbel_softmax(pi, n, temperature):
    k = len(pi)
    g = sample_gumbel(n, k)
    h = (g + torch.log(pi))/temperature
    h_max = h.max(dim=1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

# p=True means probabilistic constraint
def ppo_penalty(n_epochs, env, policy_nn, value_nn, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None, p=False):
    # Additional function to be run when additional function condition is met
    # Additional function takes in current policy_nn and current iteration number
    buffer = ReplayBuffer(basic.replay_buffer_size)
    value_opt = torch.optim.Adam(value_nn.parameters(), lr=basic.learning_rate)
    policy_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate)
    Normal = tools.utils.FixedNormal(0., 1.)
    basic.Epoch_info = []
    best_R = -float('inf')
    best_policy_params = copy.deepcopy(policy_nn.state_dict())
    for epoch in range(n_epochs):
        S_e, A_e, G_e, Gc_e, Indices = [], [], [], [], []
        G0_e, Gc0_e = [], []
        max_cost_reached = 0.0
        max_cost_reached_n = 0
        S_e_buf, A_e_buf, G_e_buf = [], [], []
        basic.Epoch_info.append({"Episode_data": []})
        for episode in range(basic.episodes_per_epoch):
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
            basic.Epoch_info[-1]["Episode_data"] += [(S,A,R,C)]
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
        # print([(np.round(item, 2), item >= basic.beta) for item in Gc0_e])
        # print(max_cost_reached, max_cost_reached_n, basic.beta)
        print("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f"
            % (epoch, np.mean(G0_e), np.mean(Gc0_e), float(max_cost_reached / max_cost_reached_n)))
        if basic.use_early_stopping:
            if p == True:
                if np.mean(G0_e) >= best_R and np.mean([float(item < basic.beta) for item in Gc0_e]) >= basic.delta:
                    old_R = copy.copy(best_R)
                    best_R = np.mean(G0_e)
                    best_policy_params = copy.deepcopy(policy_nn.state_dict())
                    print("updated best policy, old_R: %.2f, new_R: %.2f" % (old_R, best_R))
            else:
                if np.mean(G0_e) >= best_R and np.mean(Gc0_e) < basic.beta:
                    old_R = copy.copy(best_R)
                    best_R = np.mean(G0_e)
                    best_policy_params = copy.deepcopy(policy_nn.state_dict())
                    print("updated best policy, old_R: %.2f, new_R: %.2f" % (old_R, best_R))
        S_e = torch.tensor(S_e, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            A_e = torch.tensor(A_e, dtype=torch.long, device=basic.device)
        else:
            A_e = torch.tensor(A_e, dtype=torch.float, device=basic.device)
        G_e = torch.tensor(G_e, dtype=torch.float, device=basic.device)
        Gc_e = torch.tensor(Gc_e, dtype=torch.float, device=basic.device)
        S_e_buf = torch.tensor(S_e_buf, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            A_e_buf = torch.tensor(A_e_buf, dtype=torch.long, device=basic.device)
        else:
            A_e_buf = torch.tensor(A_e_buf, dtype=torch.float, device=basic.device)
        G_e_buf = torch.tensor(G_e_buf, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            log_probs_e_buf = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e_buf)).gather(1, A_e_buf.view(-1, 1)).view(-1)
        else:
            Normal.update(*policy_nn(S_e_buf))
            log_probs_e_buf = Normal.log_probs(A_e_buf).view(-1)
        buffer.add(S_e_buf, A_e_buf, G_e_buf, log_probs_e_buf.detach())
        feasibility_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate_feasibility)
        # Penalty update on complete trajectories
        if p:
            Gc0s = []
            loss1_terms = []
            loss2_terms = []
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if basic.ppo_sigmoid:
                    Gc0s += [sigmoid(Gc[0] - basic.beta, -basic.sigmoid_lambda)]
                else:
                    t = Gc[0] - basic.beta
                    Gc0s += [0.5 - 0.5 * t / torch.sqrt(t*t + basic.small_eps)]
                if not basic.continuous_actions:
                    A_logits = torch.nn.Softmax(dim=-1)(policy_nn(S_e[start_index:end_index])) # batch x act_dim
                    A_diff = torch.stack([sample_gumbel_softmax(pi = A_logits[t, :], n = 1, temperature = 0.01) for t in range(0, end_index-start_index)]).squeeze(1)
                else:
                    Normal.update(*policy_nn(S_e[start_index:end_index]))
                    A_diff = Normal.rsample()
                if 'gridworld' in basic.env_name:
                    SA_diff = S_e[start_index:end_index] # Change this depending on constraint_nn's input
                elif 'mujoco' in basic.env_name:
                    SA_diff = basic.env.unnormalize(S_e[start_index:end_index, :])[:, :1] # Change this depending on constraint_nn's input
                elif basic.env_name == 'highd':
                    SA_diff = torch.cat([S_e[start_index:end_index, 2:3], S_e[start_index:end_index, -1:]], dim=-1) # Change this depending on constraint_nn's input
                elif basic.env_name == 'cartpole':
                    SA_diff = torch.cat([S_e[start_index:end_index, :1], (A_diff @ torch.arange(basic.act_n).float().to(basic.device)).view(-1, 1)], dim=-1)
                elif basic.env_name == 'exid':
                    SA_diff = torch.cat([S_e[start_index:end_index, :], (A_diff).view(-1, 1)], dim=-1)
                    # SA_diff = torch.cat([S_e[start_index:end_index, :1], (A_diff).view(-1, 1)], dim=-1)
                else:
                    print("Bad env_name (ppo_penalty)")
                    print('Allowed: %s' % basic.env_names_allowed)
                    exit(0)
                Costs = basic.constraint_nn(SA_diff).view(-1)
                Gc0_diff = discount(Costs)[0]
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_diff.detach().argmax(dim=-1).long().view(-1, 1)).view(-1)
                    loss1_terms += [Gc0s[-1].detach() * log_probs.sum()]
                else:
                    # Normal.update(*policy_nn(S_e[start_index:end_index]))
                    log_probs = Normal.log_probs(A_diff.detach()).view(-1)
                    loss1_terms += [Gc0s[-1].detach() * log_probs.sum()]
                if basic.ppo_sigmoid:
                    loss2_terms += [Gc0s[-1].detach() * (1-Gc0s[-1]).detach() * Gc0_diff]
                else:
                    t = Gc[0] - basic.beta
                    factor = basic.small_eps / ((torch.sqrt(t*t + basic.small_eps))**3)
                    loss2_terms += [factor.detach() * Gc0_diff]
            prob_c_tau_leq_beta = torch.mean(torch.stack(Gc0s)).detach()
            loss1 = torch.mean(torch.stack(loss1_terms))
            loss2 = torch.mean(torch.stack(loss2_terms))
            feasibility_opt.zero_grad()
            if basic.ppo_sigmoid:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - basic.sigmoid_lambda * loss2)
            else:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - 0.5 * loss2)
            basic.Epoch_info[-1]["feasibility_loss_prob"] = [loss1, loss2, feasibility_loss]
            feasibility_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
            feasibility_opt.step()
        else:
            feasibility_losses = []
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_e[start_index:end_index].view(-1, 1)).view(-1)
                else:
                    Normal.update(*policy_nn(S_e[start_index:end_index]))
                    log_probs = Normal.log_probs(A_e[start_index:end_index]).view(-1)
                feasibility_opt.zero_grad()
                feasibility_loss = (Gc[0] >= basic.beta) * ((Gc * log_probs).sum())
                feasibility_losses += [feasibility_loss]
                feasibility_loss.backward()
                if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                    torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
                feasibility_opt.step()
            basic.Epoch_info[-1]["feasibility_losses_icl"] = feasibility_losses
        # Policy and value update from replay buffer
        policy_losses = []
        for subepoch in range(basic.ppo_subepochs):
            S, A, G, old_log_probs = buffer.sample(basic.minibatch_size)
            if not basic.continuous_actions:
                log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S)).gather(1, A.view(-1, 1)).view(-1)
            else:
                Normal.update(*policy_nn(S))
                log_probs = Normal.log_probs(A).view(-1)
            value_opt.zero_grad()
            value_loss = (G - value_nn(S)).pow(2).mean()
            value_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(value_nn.parameters(), 1)
            value_opt.step()
            policy_opt.zero_grad()
            advantages = G - value_nn(S)
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
               advantages = (advantages-advantages.mean())/(advantages.std()+1e-5)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - basic.clip_param, 1 + basic.clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            if not basic.continuous_actions:
                probs_all = torch.nn.Softmax(dim=-1)(policy_nn(S))
                log_probs_all = torch.nn.LogSoftmax(dim=-1)(policy_nn(S))
                entropy = -(probs_all * log_probs_all).sum(1).mean()
            else:
                entropy = Normal.entropy().mean()
            policy_loss -= basic.entropy_coef * entropy
            policy_losses += [(log_probs, value_loss, advantages, ratio, entropy, policy_loss)]
            policy_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
            policy_opt.step()
        basic.Epoch_info[-1]["policy_losses"] = policy_losses
        if len(basic.Epoch_info) >= basic.backup_len:
            basic.Epoch_info = basic.Epoch_info[-basic.backup_len:]
        # Run additional function at epoch end if condition is met
        # Just in case we need it for later! (and we will)
        if additional_fn_condition != None and additional_fn_epoch_end != None:
            if additional_fn_condition(epoch, policy_nn) == True:
                additional_fn_epoch_end(epoch, policy_nn)
    if basic.use_early_stopping:
        if best_R > -float('inf'):
            print("Using best policy_nn params due to early stopping")
            policy_nn.load_state_dict(best_policy_params)

def ppo_penalty_gae(n_epochs, env, policy_nn, value_nn, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None, p=False):
    buffer = ReplayBufferGAE(basic.replay_buffer_size)
    value_opt = torch.optim.Adam(value_nn.parameters(), lr=basic.learning_rate, eps=1e-5) # eps Hack
    policy_opt = torch.optim.Adam(policy_nn.parameters(), lr=basic.learning_rate, eps=1e-5) # eps Hack
    # combined_opt = torch.optim.Adam(list(policy_nn.parameters())+list(value_nn.parameters()), lr=basic.learning_rate, eps=1e-5) # eps Hack
    Normal = tools.utils.FixedNormal(0., 1.)
    basic.Epoch_info = []
    G0_all = []
    Gc0_all = []
    MCR_all = []
    best_R = -float('inf')
    best_policy_params = copy.deepcopy(policy_nn.state_dict())
    best_p = 0.
    best_c = float('inf')
    iters_since_best = 0
    init_time = time.time()
    # print("Total epochs ", n_epochs)
    for epoch in range(n_epochs):
        # anneal lr (cleanrl)
        # frac = 1.0 - (epoch) / n_epochs
        # lrnow = frac * basic.learning_rate
        # value_opt.param_groups[0]["lr"] = lrnow
        # policy_opt.param_groups[0]["lr"] = lrnow
        # combined_opt.param_groups[0]["lr"] = lrnow
        # ppo update
        S_e, A_e, G_e, Gc_e, Indices = [], [], [], [], []
        G0_e, Gc0_e = [], []
        G0_e_reported = []
        max_cost_reached = 0.0
        max_cost_reached_n = 0
        S_e_buf, A_e_buf, G_e_buf = [], [], []
        R_e_buf = []
        last_state = None
        last_done = 1
        D_e_buf = []
        basic.Epoch_info.append({"Episode_data": []})
        steps_so_far = 0
        episodes_done = 0
        # print("Epoch number ", epoch)
        for episode in range(basic.episodes_per_epoch):
            if hasattr(basic, "max_steps_per_epoch") and steps_so_far >= basic.max_steps_per_epoch:
                break
            S, A, R, C, InfoR = play_episode(env, policy_nn, constraint_fn, get_r = True)
            episodes_done += 1
            steps_so_far += len(R)
            basic.Epoch_info[-1]["Episode_data"] += [(S,A,R,C)]
            start_index = len(A_e)
            S_e += S[:-1]  # ignore last state
            A_e += A
            G_e += discount(R)
            G0_e += [float(discount(R)[0])]
            if 'mujoco' in basic.env_name:
                G0_all += [InfoR] # [G0_e[-1]]
                G0_e_reported += [InfoR]
            else:
                G0_all += [G0_e[-1]]
                G0_e_reported += [G0_e[-1]]
            Gc_e += discount(C)
            Gc0_e += [float(discount(C)[0])]
            Gc0_all += [Gc0_e[-1]]
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
            last_state = S[:good_until+1][-1]
            A_e_buf += A[:good_until]
            G_e_buf += discount(R[:good_until])
            R_e_buf += R[:good_until]
            dones = [0 for _ in range(good_until)]
            dones[-1] = 1
            D_e_buf += dones
            if Gc0_e[-1] >= basic.beta:
                max_cost_reached += 1
                MCR_all += [1]
            else:
                MCR_all += [0]
            max_cost_reached_n += 1
        # print("Collected ", episodes_done, "episodes")
        if basic.show_window_stats:
            print("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f\tG_window = %.2f\tGc_window = %.2f\tMCR_window = %.2f"
                % (epoch, np.mean(G0_e_reported), np.mean(Gc0_e), float(max_cost_reached / max_cost_reached_n), 
                np.mean(G0_all[-basic.window_size:]), np.mean(Gc0_all[-basic.window_size:]), np.mean(MCR_all[-basic.window_size:])))
        else:
            print("Epoch %d:\tG_avg = %.2f\tGc_avg = %.2f\tMaxCostReached = %.2f"
                % (epoch, np.mean(G0_e_reported), np.mean(Gc0_e), float(max_cost_reached / max_cost_reached_n)))
        if basic.use_early_stopping:
            if p == True:
                pr = np.mean([float(item < basic.beta) for item in Gc0_e])
                if np.mean(G0_e_reported) >= best_R and pr >= basic.delta: # and pr >= best_p:
                    old_R = copy.copy(best_R)
                    best_R = np.mean(G0_e_reported)
                    best_p = pr
                    best_policy_params = copy.deepcopy(policy_nn.state_dict())
                    print("updated best policy, old_R: %.2f, new_R: %.2f" % (old_R, best_R))
                    print("best_p: %.2f" % best_p)
            else:
                if np.mean(G0_e_reported) >= best_R and np.mean(Gc0_e) < basic.beta:
                    old_R = copy.copy(best_R)
                    old_c = copy.copy(best_c)
                    best_R = np.mean(G0_e_reported)
                    best_c = np.mean(Gc0_e)
                    best_policy_params = copy.deepcopy(policy_nn.state_dict())
                    print("updated best policy, old_R: %.2f, new_R: %.2f, old_c: %.2f, new_c: %.2f" % (old_R, best_R, old_c, best_c))
        S_e = torch.tensor(S_e, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            A_e = torch.tensor(A_e, dtype=torch.long, device=basic.device)
        else:
            A_e = torch.tensor(A_e, dtype=torch.float, device=basic.device)
        G_e = torch.tensor(G_e, dtype=torch.float, device=basic.device)
        Gc_e = torch.tensor(Gc_e, dtype=torch.float, device=basic.device)
        S_e_buf = torch.tensor(S_e_buf, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            A_e_buf = torch.tensor(A_e_buf, dtype=torch.long, device=basic.device)
        else:
            A_e_buf = torch.tensor(A_e_buf, dtype=torch.float, device=basic.device)
        G_e_buf = torch.tensor(G_e_buf, dtype=torch.float, device=basic.device)
        if not basic.continuous_actions:
            log_probs_e_buf = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e_buf)).gather(1, A_e_buf.view(-1, 1)).view(-1)
        else:
            Normal.update(*policy_nn(S_e_buf))
            log_probs_e_buf = Normal.log_probs(A_e_buf).view(-1)
        values_e_buf = value_nn(S_e_buf).view(-1)
        R_e_buf = torch.tensor(R_e_buf, dtype=torch.float, device=basic.device)
        # GAE computation
        next_value = value_nn(torch.tensor(last_state, dtype=torch.float, device=basic.device)).view(-1)
        adv_e_buf = torch.zeros_like(R_e_buf).to(basic.device)
        lastgaelam = 0
        n_steps = len(R_e_buf)
        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                nextnonterminal = 1.0 - last_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - D_e_buf[t + 1]
                nextvalues = values_e_buf[t + 1]
            delta = R_e_buf[t] + basic.discount_factor * nextvalues * nextnonterminal - values_e_buf[t]
            adv_e_buf[t] = lastgaelam = delta + basic.discount_factor * basic.gae_lambda * nextnonterminal * lastgaelam
        G_gae_e_buf = adv_e_buf + values_e_buf
        buffer.add(S_e_buf, A_e_buf, G_gae_e_buf.detach(), log_probs_e_buf.detach(), adv_e_buf.detach(), values_e_buf.detach())
        # print("Completed GAE and replay buffer part")
        feasibility_opt = torch.optim.SGD(policy_nn.parameters(), lr=basic.learning_rate_feasibility)
        # Penalty update on complete trajectories
        if p:
            Gc0s = []
            loss1_terms = []
            loss2_terms = []
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if basic.ppo_sigmoid:
                    Gc0s += [sigmoid(Gc[0] - basic.beta, -basic.sigmoid_lambda)]
                else:
                    t = Gc[0] - basic.beta
                    Gc0s += [0.5 - 0.5 * t / torch.sqrt(t*t + basic.small_eps)]
                if not basic.continuous_actions:
                    A_logits = torch.nn.Softmax(dim=-1)(policy_nn(S_e[start_index:end_index])) # batch x act_dim
                    A_diff = torch.stack([sample_gumbel_softmax(pi = A_logits[t, :], n = 1, temperature = 0.01) for t in range(0, end_index-start_index)]).squeeze(1)
                else:
                    Normal.update(*policy_nn(S_e[start_index:end_index]))
                    A_diff = Normal.rsample()
                if 'gridworld' in basic.env_name:
                    SA_diff = S_e[start_index:end_index] # Change this depending on constraint_nn's input
                elif 'mujoco' in basic.env_name:
                    SA_diff = basic.env.unnormalize(S_e[start_index:end_index, :].cpu()).to(basic.device).float()[:, :1] # Change this depending on constraint_nn's input
                elif basic.env_name == 'highd':
                    SA_diff = torch.cat([S_e[start_index:end_index, 2:3], S_e[start_index:end_index, -1:]], dim=-1) # Change this depending on constraint_nn's input
                elif basic.env_name == 'cartpole':
                    SA_diff = torch.cat([S_e[start_index:end_index, :1], (A_diff @ torch.arange(basic.act_n).float().to(basic.device)).view(-1, 1)], dim=-1)
                elif basic.env_name == 'exid':
                    SA_diff = torch.cat([S_e[start_index:end_index, :], (A_diff).view(-1, 1)], dim=-1)
                    # SA_diff = torch.cat([S_e[start_index:end_index, :1], (A_diff).view(-1, 1)], dim=-1)
                else:
                    print("Bad env_name (ppo_penalty_gae)")
                    print('Allowed: %s' % basic.env_names_allowed)
                    exit(0)
                Costs = basic.constraint_nn(SA_diff).view(-1)
                Gc0_diff = discount(Costs)[0]
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_diff.detach().argmax(dim=-1).long().view(-1, 1)).view(-1)
                    loss1_terms += [Gc0s[-1].detach() * log_probs.sum()]
                else:
                    # Normal.update(*policy_nn(S_e[start_index:end_index]))
                    log_probs = Normal.log_probs(A_diff.detach()).view(-1)
                    loss1_terms += [Gc0s[-1].detach() * log_probs.sum()]
                if basic.ppo_sigmoid:
                    loss2_terms += [Gc0s[-1].detach() * (1-Gc0s[-1]).detach() * Gc0_diff]
                else:
                    t = Gc[0] - basic.beta
                    factor = basic.small_eps / ((torch.sqrt(t*t + basic.small_eps))**3)
                    loss2_terms += [factor.detach() * Gc0_diff]
            prob_c_tau_leq_beta = torch.mean(torch.stack(Gc0s)).detach()
            loss1 = torch.mean(torch.stack(loss1_terms))
            loss2 = torch.mean(torch.stack(loss2_terms))
            feasibility_opt.zero_grad()
            if basic.ppo_sigmoid:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - basic.sigmoid_lambda * loss2)
            else:
                feasibility_loss = -1. * (basic.delta > prob_c_tau_leq_beta) * (loss1 - 0.5 * loss2)
            basic.Epoch_info[-1]["feasibility_loss_prob"] = [loss1, loss2, feasibility_loss]
            feasibility_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
            feasibility_opt.step()
        else:
            feasibility_losses = []
            for start_index, end_index in Indices:
                Gc = Gc_e[start_index:end_index].view(-1)
                if not basic.continuous_actions:
                    log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S_e[start_index:end_index])).gather(1, A_e[start_index:end_index].view(-1, 1)).view(-1)
                else:
                    Normal.update(*policy_nn(S_e[start_index:end_index]))
                    log_probs = Normal.log_probs(A_e[start_index:end_index]).view(-1)
                feasibility_opt.zero_grad()
                feasibility_loss = (Gc[0] >= basic.beta) * ((Gc * log_probs).sum())
                feasibility_losses += [feasibility_loss]
                feasibility_loss.backward()
                if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                    torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1)
                feasibility_opt.step()
            basic.Epoch_info[-1]["feasibility_losses_icl"] = feasibility_losses
        # print("Finished feasibility")
        # Policy and value update from replay buffer
        policy_losses = []
        for subepoch in range(basic.ppo_subepochs):
            S, A, G, old_log_probs, Adv, V = buffer.sample(basic.minibatch_size)
            if not basic.continuous_actions:
                log_probs = torch.nn.LogSoftmax(dim=-1)(policy_nn(S)).gather(1, A.view(-1, 1)).view(-1)
            else:
                Normal.update(*policy_nn(S))
                log_probs = Normal.log_probs(A).view(-1)
            value_opt.zero_grad()
            # combined_opt.zero_grad()
            values = value_nn(S)
            value_loss_unclipped = (values - G).pow(2)
            # values_clipped = V + torch.clamp(values - V, -basic.clip_param, basic.clip_param)
            # value_loss_clipped = (values_clipped - G).pow(2)
            # value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            value_loss = value_loss_unclipped.mean()
            value_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(value_nn.parameters(), 1)
            value_opt.step()
            policy_opt.zero_grad()
            ## advantages = G - value_nn(S)
            advantages = Adv
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
               advantages = (advantages-advantages.mean())/(advantages.std()+1e-8) # cleanrl advantage normalization
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - basic.clip_param, 1 + basic.clip_param)
            policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            if not basic.continuous_actions:
                probs_all = torch.nn.Softmax(dim=-1)(policy_nn(S))
                log_probs_all = torch.nn.LogSoftmax(dim=-1)(policy_nn(S))
                entropy = -(probs_all * log_probs_all).sum(1).mean()
            else:
                entropy = Normal.entropy().mean()
            policy_loss -= basic.entropy_coef * entropy
            # entropy_loss = basic.entropy_coef * entropy
            # overall_loss = policy_loss + 0.5 * value_loss - entropy_loss
            policy_losses += [(log_probs, value_loss, advantages, ratio, entropy, policy_loss)]
            policy_loss.backward()
            # overall_loss.backward()
            if 'mujoco' in basic.env_name or basic.env_name in ['highd']:
                torch.nn.utils.clip_grad_norm_(policy_nn.parameters(), 1) # cleanrl gradient clipping
            policy_opt.step()
            # combined_opt.step()
        basic.Epoch_info[-1]["policy_losses"] = policy_losses
        # print("PPO trained")
        if len(basic.Epoch_info) >= basic.backup_len:
            basic.Epoch_info = basic.Epoch_info[-basic.backup_len:]
        # Run additional function at epoch end if condition is met
        # Just in case we need it for later! (and we will)
        if additional_fn_condition != None and additional_fn_epoch_end != None:
            if additional_fn_condition(epoch, policy_nn) == True:
                additional_fn_epoch_end(epoch, policy_nn)
    if basic.use_early_stopping:
        if best_R > -float('inf'):
            print("Using best policy_nn params due to early stopping")
            policy_nn.load_state_dict(best_policy_params)

def ppo_lag(n_epochs, env, constraint_fn, additional_fn_condition=None, additional_fn_epoch_end=None):
    tf.compat.v1.disable_eager_execution()
    config = dict(
        env=env,
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

def collect_trajectories(n, env, policy_nn, constraint_fn, only_success=False):
    data = []
    for traj in tqdm.tqdm(range(n)):
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        while (not (discount(C)[0] <= basic.beta)) and only_success:
            S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data

def convert_to_flow_data(data):
    flow_data = []
    for S, A in data:
        for s, a in zip(S, A):
            if 'gridworld' in basic.env_name:
                flow_data += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                flow_data += [[s[0], a]]
            elif 'mujoco' in basic.env_name:
                new_s = basic.env.unnormalize(s)
                flow_data += [[new_s[0]]]
            elif basic.env_name == 'highd':
                flow_data += [[s[2], s[-1]]]
            elif basic.env_name == 'exid':
                flow_data += [[*s, *a]]
                # flow_data += [[s[0], a[0]]]
            else:
                print("Bad env_name (convert_to_flow_data)")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
    flow_data = torch.tensor(flow_data, dtype=torch.float, device=basic.device)
    return flow_data

def dissimilarity_wrt_expert(data, mean=True):
    expert_nll_mean, expert_nll_std = basic.expert_nll
    sims = []
    for S, A in data:
        traj_data = []
        for s, a in zip(S, A):
            if 'gridworld' in basic.env_name:
                traj_data += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                traj_data += [[s[0], a]]
            elif 'mujoco' in basic.env_name:
                new_s = basic.env.unnormalize(s)
                traj_data += [[new_s[0]]]
            elif basic.env_name == 'highd':
                traj_data += [[s[2], s[-1]]]
            elif basic.env_name == 'exid':
                traj_data += [[*s, *a]]
                # traj_data += [[s[0], a[0]]]
            else:
                print("Bad env_name (dissimilarity_wrt_expert)")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        traj_data = torch.tensor(traj_data, dtype=torch.float, device=basic.device)
        traj_nll = -basic.flow.log_probs(traj_data).detach().cpu()
        sims += [(traj_nll > expert_nll_mean + expert_nll_std).float().mean().cpu()]
    if mean:
        return np.mean(np.array(sims)+1e-3)
    return torch.tensor(sims, dtype=torch.float, device=basic.device)

def collect_trajectories_mixture(n, env, policy_mixture, weights_mixture, constraint_fn, ppolag_config=None):
    data = []
    value_nn, policy_nn = make_nn()
    normalized_weights_mixture = np.copy(weights_mixture) / np.sum(weights_mixture)
    m = len(weights_mixture)
    for traj in tqdm.tqdm(range(n)):
        chosen_policy_idx = np.random.choice(m, p=normalized_weights_mixture)
        if ppolag_config == None:
            policy_nn.load_state_dict(policy_mixture[chosen_policy_idx])
        else:
            loadmodel(ppolag_config["sess"], ppolag_config["saver"], policy_mixture[chosen_policy_idx])
            policy_nn = ppolag_config["policy"]
        S, A, R, C = play_episode(env, policy_nn, constraint_fn)
        data += [[S[:-1], A]]
    return data

def compute_cdf(constraint_nn, data):
    if data == None:
        return None
    Gc0 = []
    for S, A in data:
        input_to_constraint_nn = []
        for s, a in zip(S, A):
            if 'gridworld' in basic.env_name:
                input_to_constraint_nn += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                input_to_constraint_nn += [[s[0], a]]
            elif 'mujoco' in basic.env_name:
                new_s = basic.env.unnormalize(s)
                input_to_constraint_nn += [[new_s[0]]]
            elif basic.env_name == 'highd':
                input_to_constraint_nn += [[s[2], s[-1]]]
            elif basic.env_name == 'exid':
                input_to_constraint_nn += [[*s, *a]]
                # input_to_constraint_nn += [[s[0], a[0]]]
            else:
                print("Bad env_name (compute_cdf)")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        input_to_constraint_nn = torch.tensor(input_to_constraint_nn, dtype=torch.float, device=basic.device)
        constraint_values = constraint_nn(input_to_constraint_nn).view(-1)
        Gc0 += [float(discount(constraint_values)[0] <= basic.beta)]
    return np.mean(Gc0)

def compute_current_constraint_value_trajectory(constraint_nn, data, p=False):
    Gc0 = []
    for S, A in data:
        input_to_constraint_nn = []
        for s, a in zip(S, A):
            if 'gridworld' in basic.env_name:
                input_to_constraint_nn += [s]  # Change this depending on your constraint_nn input
            elif basic.env_name == 'cartpole':
                input_to_constraint_nn += [[s[0], a]]
            elif 'mujoco' in basic.env_name:
                new_s = basic.env.unnormalize(s)
                input_to_constraint_nn += [[new_s[0]]]
            elif basic.env_name == 'highd':
                input_to_constraint_nn += [[s[2], s[-1]]]
            elif basic.env_name == 'exid':
                input_to_constraint_nn += [[*s, *a]]
                # input_to_constraint_nn += [[s[0], a[0]]]
            else:
                print("Bad env_name (compute_current_constraint_value_traj)")
                print('Allowed: %s' % basic.env_names_allowed)
                exit(0)
        input_to_constraint_nn = torch.tensor(input_to_constraint_nn, dtype=torch.float, device=basic.device)
        constraint_values = constraint_nn(input_to_constraint_nn).view(-1)
        if p:
            # Gc0 += [sigmoid(discount(constraint_values)[0] - basic.beta, -basic.sigmoid_lambda)]
            t = discount(constraint_values)[0] - basic.beta
            Gc0 += [0.5 - 0.5 * t / torch.sqrt(t*t + basic.small_eps)]
        else:
            Gc0 += [discount(constraint_values)[0]]
    return torch.stack(Gc0)

def constraint_function_adjustment(n, constraint_nn, constraint_opt, expert_data, agent_data, p=False):
    losses = []
    for _ in range(n):
        constraint_opt.zero_grad()
        per_traj_dissimilarity = dissimilarity_wrt_expert(agent_data, mean=False)
        per_traj_dissimilarity = per_traj_dissimilarity / per_traj_dissimilarity.sum()
        agent_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, agent_data, p=p)
        expert_data_constraint_returns = compute_current_constraint_value_trajectory(constraint_nn, expert_data, p=p)
        if p:
            # print(agent_data_constraint_returns.shape, per_traj_dissimilarity.shape)
            # print(per_traj_dissimilarity, torch.sum(agent_data_constraint_returns))
            loss1 = (agent_data_constraint_returns * per_traj_dissimilarity).sum() # no minus sign
        else:
            loss1 = -(agent_data_constraint_returns * per_traj_dissimilarity).sum()
        if p:
            # print(expert_data_constraint_returns.shape)
            # print(torch.sum(expert_data_constraint_returns))
            loss2 = -(basic.delta > torch.mean(expert_data_constraint_returns).detach()).float() * (torch.mean(expert_data_constraint_returns))
        else:
            if 'gridworld' in basic.env_name:
                loss2 = ((expert_data_constraint_returns >= basic.beta).float() * (expert_data_constraint_returns - basic.beta)).mean()        
            else:
                loss2 = ((expert_data_constraint_returns.mean() >= basic.beta).float()) * ((expert_data_constraint_returns - basic.beta).mean())
        loss = loss1 + basic.alpha * loss2
        loss.backward()
        constraint_opt.step()
        losses += [loss.item()]
    return np.mean(losses)

def condition(epoch, policy_nn):
    if (epoch + 1) % basic.policy_add_to_mix_every == 0:
        return True
    return False

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

def generate_expert_data(env, only_success=False, p=False):  # use only_success if the policy isn't great and you just want to get optimal trajectories
    if os.path.exists(basic.expert_data_file):
        return torch.load(basic.expert_data_file)
    value_nn, policy_nn = make_nn()
    if 'mujoco' in basic.env_name:
        only_success = True
        print("Using successful trajectories")
    if basic.use_gae:
        func = ppo_penalty_gae
    else:
        func = ppo_penalty
    if p:
        func(basic.ppo_iters, env, policy_nn, value_nn, basic.true_constraint_function, p=p)        
    else:
        func(basic.ppo_iters, env, policy_nn, value_nn, basic.true_constraint_function)
    expert_data = collect_trajectories(basic.n_trajs, basic.env, policy_nn, basic.true_constraint_function, only_success=only_success)
    torch.save(expert_data, basic.expert_data_file)
    return expert_data

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

def mse(u, v):
    u = np.array(u)
    v = np.array(v)
    assert(u.shape == v.shape)
    return np.mean(np.power(u-v, 2))

def mem():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
        # print(torch.cuda.list_gpu_processes())
        return "%.2f GB free" % (int(torch.cuda.mem_get_info()[0])/(1024*1024*1024))
    else:
        return "Using CPU"

# Exid environment functions

def euc_dist(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def contains(poly, p):
    """
    Is point p in polygon poly?
    """
    nvert = len(poly)
    i, j, c = 0, nvert-1, False
    while i < nvert:
        if ( ((poly[i][1]>p[1]) != (poly[j][1]>p[1])) and \
            (p[0] < (poly[j][0]-poly[i][0]) * (p[1]-poly[i][1]) / (poly[j][1]-poly[i][1]) + poly[i][0]) ):
               c = not c
        j = i
        i += 1
    return c

def localize(osm, x, y):
    """
    localize a point to get which lanelet id it is in
    """
    for key, val in osm.polys.items():
        if contains(val, (x, y)):
            return key
    return None

def find_l_w(bbox):
    """
    Find the length and width of a rectangular bounding box given the coordinates
    """
    ax, ay = bbox[0, :]
    bx, by = bbox[1, :]
    cx, cy = bbox[2, :]
    ab = euc_dist(ax, ay, bx, by)
    bc = euc_dist(bx, by, cx, cy)
    ca = euc_dist(cx, cy, ax, ay)
    if ab >= bc and ab >= ca:
        l = max(bc, ca)
        w = min(bc, ca)
    elif bc >= ab and bc >= ca:
        l = max(ab, ca)
        w = min(ab, ca)
    else:
        l = max(ab, bc)
        w = min(ab, bc)
    return l, w

def find_closest_node(lanelet_centers, x, y):
    """
    Find the first point in the midpoints of a lanelet to another point (x, y)
    (Fix: not the closest node)
    Also return the midpoints
    """
    points = []
    for node in lanelet_centers:
        points += [(node.x, node.y)]
    return lanelet_centers[0], LineString(points)

def find_segment(ipline, x, y):
    """
    Find the part of a ipline that is the closest to a point (new_point)
    """
    all_dist = []
    for i in range(len(ipline.coords)-1):
        mid1 = (ipline.coords[i][0]*1./2+ipline.coords[i+1][0]*1./2, ipline.coords[i][1]*1./2+ipline.coords[i+1][1]*1./2)
        mid2 = (ipline.coords[i][0]*2./3+ipline.coords[i+1][0]*1./3, ipline.coords[i][1]*2./3+ipline.coords[i+1][1]*1./3)
        mid3 = (ipline.coords[i][0]*1./3+ipline.coords[i+1][0]*2./3, ipline.coords[i][1]*1./3+ipline.coords[i+1][1]*2./3)
        all_dist += [(ipline.coords[i], ipline.coords[i+1], ((mid1[0] - x)**2 + (mid1[1] - y)**2)**0.5)]
        all_dist += [(ipline.coords[i], ipline.coords[i+1], ((mid2[0] - x)**2 + (mid2[1] - y)**2)**0.5)]
        all_dist += [(ipline.coords[i], ipline.coords[i+1], ((mid3[0] - x)**2 + (mid3[1] - y)**2)**0.5)]
    all_dist = sorted(all_dist, key=lambda item: item[-1])
    return [all_dist[0][0], all_dist[0][1]]

def lonlat_ipline(x, y, ipline):
    """
    Return the longitudinal and lateral position of (x, y) w.r.t a line segment
    """
    lon = 0
    lat = None
    rightside = ipline.buffer(100, single_sided=True)
    rightside_coords = list(rightside.exterior.coords)
    if rightside_coords[-1] == rightside_coords[0]:
        rightside_coords = rightside_coords[:-1]
    side = -1 if contains(rightside_coords, (x, y)) else 1
    segpt1, segpt2 = find_segment(ipline, x, y)
    for i in range(len(ipline.coords) - 1):
        coord1, coord2 = ipline.coords[i], ipline.coords[i + 1]
        ax, ay = coord1[0], coord1[1]
        bx, by = coord2[0], coord2[1]
        cx, cy = x, y
        is_closest = segpt1 in [coord1, coord2] and segpt2 in [coord1, coord2]
        if is_closest:  # acute:
            line = LineString([coord1, coord2])
            lon += line.project(Point(x, y))
            lat = side * line.distance(Point(x, y))
            return lon, lat
        else:
            lon += ((ax - bx)**2 + (ay - by)**2)**0.5
    return None, None

def find_point(obj, i, action=None, only_lat=True):
    """
    Take the action given by the environment
    only_lat=True when action has 1 feature --- lateral velocity
    """
    # interpolate to longitudinal position
    new_point = obj.ipline.interpolate(obj.lon_dist)
    # find relevant segment
    segpt1, segpt2 = find_segment(obj.ipline, new_point.x, new_point.y)
    # construct segment vector
    lon_vec_length = ((segpt1[0] - segpt2[0]) ** 2 + (segpt1[1] - segpt2[1]) ** 2) ** 0.5
    dx = (segpt2[0] - segpt1[0]) / lon_vec_length
    dy = (segpt2[1] - segpt1[1]) / lon_vec_length
    # actual displacement from center points data
    dc = (obj.centers[obj.veh_id][i + 1, 0] - obj.centers[obj.veh_id][i, 0], obj.centers[obj.veh_id][i + 1, 1] - obj.centers[obj.veh_id][i, 1])
    # find longitudinal and lateral component of this displacement
    dc_lon = dc[0] * dx + dc[1] * dy
    dc_lat = dc[0] * dy - dc[1] * dx
    reference_vlon = None
    # if action=None, the environment must replay the expert data
    if action == None:
        if only_lat:
            obj.info = {
                "action_lat": dc_lat / obj.dt,
            }
        else:
            if hasattr(obj, "reference_vlon"):
                reference_vlon = obj.reference_vlon
            else:
                reference_vlon = dc_lon / obj.dt
            obj.info = {
                "action_lat": dc_lat / obj.dt,
                "action_lon": dc_lon / obj.dt - reference_vlon,
            }
    # if action has 2 features --- lateral velocity and extra longitudinal velocity
    elif not only_lat:
        action_lat, action_lon = float(action[0]), float(action[1])
        dc_lat = action_lat * obj.dt
        dc_lon = (obj.reference_vlon + action_lon) * obj.dt
    # if action has 1 feature --- lateral velocity
    elif only_lat:
        action_lat = float(action)
        dc_lat = action_lat * obj.dt
    # compute new coordinates
    new_x = obj.dr[0] + new_point.x + dy * obj.lat_dist
    new_y = obj.dr[1] + new_point.y - dx * obj.lat_dist
    # return current x/y coordinates and extra lateral and longitudinal movement
    return new_x, new_y, dc_lat, dc_lon, reference_vlon

def find_target_lead_rear(obj, new_x, new_y, i):
    """
    Find the longitudinal relative position of lead and rear vehicle
    """
    # my longitudinal and lateral position w.r.t. start lane
    my_lon, my_lat = lonlat_ipline(new_x, new_y, obj.ipline)
    # which lanelet id is ego in, currently?
    my_id = localize(obj.osm, new_x, new_y)
    # init target lead and rear values 
    target_lead, target_rear = float('inf'), float('inf')
    # which vehicles have longitudinal overlap with ego?
    # alongside_veh_ids = convert_to_int_items(obj.alongside[obj.veh_id][i])
    overlap = False
    # loop through vehicles in the scene
    for other_veh_id, other_idx in obj.frame_dict[obj.frames[obj.veh_id][i]]:
        if other_veh_id != obj.veh_id:
            # where is the other vehicle?
            other_veh_lanelet_id = localize(obj.osm, obj.centers[other_veh_id][other_idx, 0], obj.centers[other_veh_id][other_idx, 1])
            # longitudinal and lateral position of the other vehicle?
            veh_lon, veh_lat = lonlat_ipline(obj.centers[other_veh_id][other_idx, 0], obj.centers[other_veh_id][other_idx, 1], obj.ipline)
            # other vehicle l, w
            other_l, other_w = find_l_w(obj.bboxes[other_veh_id][other_idx])
            if other_veh_lanelet_id == obj.end_lanelet_id:
                if veh_lon > my_lon and veh_lon != 0. and my_lon != 0.:
                    if veh_lon - my_lon > other_l/2. + obj.my_l/2.:
                        target_lead = min(target_lead, veh_lon - my_lon - (other_l/2. + obj.my_l/2.))
                    else:
                        overlap = True
                if my_lon > veh_lon and veh_lon != 0. and my_lon != 0.:
                    if my_lon - veh_lon > other_l/2. + obj.my_l/2.:
                        target_rear = min(target_rear, my_lon - veh_lon - (other_l/2. + obj.my_l/2.))
                    else:
                        overlap = True
    if target_lead == float('inf'):
        target_lead = 100. # 200.
    if target_rear == float('inf'):
        target_rear = -100 # 200.
    else:
        target_rear *= -1
    if overlap:
        target_lead = 0.
        target_rear = 0.
    return target_lead, target_rear, overlap

class ExidEnv(tools.base.Environment):
    """
    * Action space: 1 feature
        * lateral velocity
    * State space: 4 features
        * signed distance to target lane
        * lon position of lead vehicle in target lane
        * lon position of rear vehicle in target lane
        * longitudinal overlap in target lane
    """
    def __init__(self, given_num=None, bad_ids=[], constraints=[], debug=False, overlap_debug=False):
        self.debug = debug
        self.overlap_debug = overlap_debug
        # constraints to overlay
        self.constraints = constraints
        # from meta file
        self.dt = 1 / 25.0
        # ids to avoid if any are given
        self.bad_ids = bad_ids
        # if given, use this lane change index only
        self.given_num = given_num
        # define action and observation space
        high = float("inf")
        self.action_space = gym.spaces.Box(-high, high, shape=(1,), dtype=np.float32)
        # self.observation_space = gym.spaces.Box(-high, high, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-high, high, shape=(4,), dtype=np.float32)
        # use preprocessed data
        self.data = basic.lane_change_info
        self.veh_ids = basic.lane_change_ids
        self.veh_ids_n = len(self.veh_ids)
        self.V = None
    @property
    def state(self):
        return self.curr_state    
    def seed(self, s=None):
        random.seed(s)
        np.random.seed(s)
    def reset(self):
        # start of a new scenario, delete any previous plotting information
        if hasattr(self, "display_bboxes"):
            del self.display_bboxes
        # if there is an overlap (or no overlap), what is the corresponding constraint value?
        # store this data, and average/print later
        if self.overlap_debug:
            self.overlap_constraint_values = []
            self.non_overlap_constraint_values = []
        # choose the lane change index
        if self.given_num != None:
            idx = self.given_num
        else:
            idx = np.random.randint(self.veh_ids_n)
            while idx in self.bad_ids:
                idx = np.random.randint(self.veh_ids_n)
        # load the lane change information
        self.num, self.veh_id, self.start_lanelet_id, self.end_lanelet_id, self.start_frame, self.end_frame, \
            right_or_left ,_, relevant = self.veh_ids[idx]
        self.max_length = self.end_frame - self.start_frame + 1
        # load detailed information about the scenario
        self.osm, self.centers, self.bboxes, self.frames, self.where, self.surround, self.frame_dict = \
            self.data[self.num]["osm"], self.data[self.num]["centers"], self.data[self.num]["bboxes"], \
            self.data[self.num]["frames"], self.data[self.num]["where"], self.data[self.num]["surround"], self.data[self.num]["frame_dict"]
        self.start_lanelet_centers = self.osm.get_relation(self.start_lanelet_id).mid
        self.end_lanelet_centers = self.osm.get_relation(self.end_lanelet_id).mid
        # use the right_alongside or left_alongside data depending on where the target lane is
        if right_or_left == 1:
            self.alongside = self.data[self.num]["right_alongside"]
        else:
            self.alongside = self.data[self.num]["left_alongside"]
        # my length and width
        self.my_l, self.my_w = find_l_w(self.bboxes[self.veh_id][self.start_frame])
        # place ego at start position ox, oy
        ox, oy = self.centers[self.veh_id][self.start_frame][0], self.centers[self.veh_id][self.start_frame][1]
        # find a vector for displacement w.r.t. closest lanelet point --- maintain this small displacement throughout episode
        closest_node, self.ipline = find_closest_node(self.start_lanelet_centers, ox, oy)
        self.max_lon = 0
        for node_idx in range(len(self.start_lanelet_centers)-1):
            self.max_lon += euc_dist(
                self.start_lanelet_centers[node_idx].x,
                self.start_lanelet_centers[node_idx].y,
                self.start_lanelet_centers[node_idx+1].x,
                self.start_lanelet_centers[node_idx+1].y,
            )
        self.dr = (ox - closest_node.x, oy - closest_node.y)
        # initialize longitudinal, lateral movement
        self.lon_dist, self.lat_dist = 0, 0
        # time step of episode
        self.i = 0
        # current bounding box of ego
        self.curr_bbox = copy.deepcopy(self.bboxes[self.veh_id][self.start_frame + self.i])
        # populate initial x/y data
        new_x, new_y, dc_lat, dc_lon, self.reference_vlon = find_point(self, self.start_frame + self.i)
        self.curr_x, self.curr_y = new_x, new_y
        # populate initial state
        target_lead, target_rear, overlap = find_target_lead_rear(self, new_x, new_y, self.start_frame + self.i)
        ipline_target = LineString([(node.x, node.y) for node in self.end_lanelet_centers])
        rightside = ipline_target.buffer(100, single_sided=True)
        rightside_coords = list(rightside.exterior.coords)
        if rightside_coords[-1] == rightside_coords[0]:
            rightside_coords = rightside_coords[:-1]
        side = -1 if contains(rightside_coords, (new_x, new_y)) else 1
        new_point2 = ipline_target.interpolate(self.lon_dist)
        dist_target = Point(new_x, new_y).distance(new_point2)
        self.curr_state = np.array([side * dist_target / 5., target_lead / 100., target_rear / 100., float(overlap)])
        # self.curr_state = np.array([side * dist_target/5.])
        # other initialization stuff
        self.initial_side = side
        self.initial_dist_target = side * dist_target
        self.terminated = False
        self.info = {}
        self.all_rewards = []
        # progress indicator for rendering
        if self.debug:
            self.pbar = tqdm.tqdm(range(self.end_frame-self.start_frame-2))
        return self.curr_state
    def find_state(self, lat, lon):
        # what is the state if lat/lon are set to these values?
        new_point = self.ipline.interpolate(lon)
        segpt1, segpt2 = find_segment(self.ipline, new_point.x, new_point.y)
        lon_vec_length = ((segpt1[0] - segpt2[0]) ** 2 + (segpt1[1] - segpt2[1]) ** 2) ** 0.5
        dx = (segpt2[0] - segpt1[0]) / lon_vec_length
        dy = (segpt2[1] - segpt1[1]) / lon_vec_length
        new_x = self.dr[0] + new_point.x + dy * lat
        new_y = self.dr[1] + new_point.y - dx * lat
        target_lead, target_rear, overlap = find_target_lead_rear(self, new_x, new_y, self.start_frame + self.i)
        ipline_target = LineString([(node.x, node.y) for node in self.end_lanelet_centers])
        rightside = ipline_target.buffer(100, single_sided=True)
        rightside_coords = list(rightside.exterior.coords)
        if rightside_coords[-1] == rightside_coords[0]:
            rightside_coords = rightside_coords[:-1]
        side = -1 if contains(rightside_coords, (new_x, new_y)) else 1
        new_point2 = ipline_target.interpolate(self.lon_dist)
        dist_target = Point(new_x, new_y).distance(new_point2)
        return np.array([side * dist_target / 5., target_lead / 100., target_rear / 100., float(overlap)]), new_x, new_y
        # return np.array([side * dist_target/5.]), new_x, new_y
    def step(self, action=None):
        # time step += 1
        self.i += 1
        # check for termination
        self.terminated = self.terminated | (self.start_frame + self.i + 2 > self.end_frame)
        if self.debug:
            self.pbar.update(1)
            if self.terminated:
                self.pbar.close()
        # find current x/y and lateral/longitudinal movement given the action
        new_x, new_y, dc_lat, dc_lon, _ = find_point(self, self.start_frame + self.i, action=action, only_lat=True)
        # adjust ego bounding box
        self.curr_bbox[:, 0] += new_x - self.curr_x
        self.curr_bbox[:, 1] += new_y - self.curr_y
        self.curr_x, self.curr_y = new_x, new_y
        # do the movement
        self.lat_dist += dc_lat
        self.lon_dist += dc_lon
        # compute state
        target_lead, target_rear, overlap = find_target_lead_rear(self, new_x, new_y, self.start_frame + self.i)
        ipline_target = LineString([(node.x, node.y) for node in self.end_lanelet_centers])
        rightside = ipline_target.buffer(100, single_sided=True)
        rightside_coords = list(rightside.exterior.coords)
        if rightside_coords[-1] == rightside_coords[0]:
            rightside_coords = rightside_coords[:-1]
        side = -1 if contains(rightside_coords, (new_x, new_y)) else 1
        new_point2 = ipline_target.interpolate(self.lon_dist)
        dist_target = Point(new_x, new_y).distance(new_point2)
        self.curr_state = np.array([side * dist_target / 5., target_lead / 100., target_rear / 100., float(overlap)])
        # self.curr_state = np.array([side * dist_target/5.])
        # are we outside bounds?
        my_id = localize(self.osm, new_x, new_y)
        outside = False
        if my_id not in [self.start_lanelet_id, self.end_lanelet_id]:
            self.terminated = True
            outside = True
        # compute reward
        reward = np.clip(1 - side * dist_target / self.initial_dist_target, 0, 1)
        # reward = np.exp(- 4 * abs(dist_target) / self.initial_dist_target)
        reward /= self.max_length
        self.all_rewards += [reward]
        if outside:
            reward -= sum(self.all_rewards)
        # return next_state, reward, done, info
        return {
            "next_state": self.curr_state,
            "reward": reward,
            "done": self.terminated,
            "info": self.info,
        }
    def render(self, **kwargs):
        # construct plotting stuff
        if not hasattr(self, "display_bboxes"):
            self.fig, self.ax = plt.subplots(1, 1)
            self.ax.axis("equal")
            self.fig.tight_layout()
            self.ax.axis("off")
            self.osm.plot({self.start_lanelet_id: "gray", self.end_lanelet_id: "gray"}, show_all=False, ax=self.ax)
            self.display_bboxes = {}
            self.scatter = None
            self.plot = None
        # if there are constraints, show them
        if len(self.constraints) > 0 and self.V == None:
            fwd = []
            x, y = [], []
            for lon_delta in np.arange(-30, 30, 1):
                for lat_delta in np.arange(-5, 5, 1):
                    if not (0 <= self.lon_dist + lon_delta <= self.max_lon):
                        continue
                    new_state, new_x, new_y = self.find_state(self.lat_dist + lat_delta, self.lon_dist + lon_delta)
                    x += [new_x]
                    y += [new_y]
                    tmp = []
                    for v in [-1]: # -0.5, -1., -1.5, -2.]: # act_dim=4
                        possible_action = v * self.initial_side
                        tmp += [[*new_state, possible_action]]
                    fwd += [tmp]
            fwd = torch.tensor(fwd, device=basic.device, dtype=torch.float) # lonlat x act_dim x constraint_dim
            new_state2, _, _ = self.find_state(self.lat_dist, self.lon_dist)
            fwd2 = []
            for possible_action2 in np.arange(-3, 3, 0.2):
                fwd2 += [[*new_state2, possible_action2]]
            fwd2 = torch.tensor(fwd2, device=basic.device, dtype=torch.float)
            vals = []
            cvals = []
            for constraint_fn in self.constraints:                            
                vals += [constraint_fn(fwd).detach().cpu().numpy()]
                if self.overlap_debug:
                    p, q = (
                        torch.sum(torch.multiply(torch.tensor(vals[-1]), (fwd[:, :, -2] == 1.).float().unsqueeze(-1))), 
                        torch.sum((fwd[:, :, -2] == 1.).float().unsqueeze(-1))
                    )
                    if p != 0. and q != 0.:
                        self.overlap_constraint_values += [(p, q)]
                    p2, q2 = (
                        torch.sum(torch.multiply(torch.tensor(vals[-1]), (fwd[:, :, -2] == 0.).float().unsqueeze(-1))),
                        torch.sum((fwd[:, :, -2] == 0.).float().unsqueeze(-1))
                    )
                    if p2 != 0. and q2 != 0.:
                        self.non_overlap_constraint_values += [(p2, q2)]
                cvals += [constraint_fn(fwd2).detach().cpu().view(-1).numpy()]
            vals = np.array(vals).squeeze(-1) # seeds x lonlat x act_dim
            vals = np.mean(vals, axis=-1) # seeds x lonlat
            vals_mean = np.mean(vals, axis=0)
            # vals_std = np.std(vals, axis=0) # small -> go to 0, large --> go to 1
            c = [item for item in vals_mean]
            # s = [2.5*(np.clip(1.-item, 0, 1)) for item in vals_std]
            if self.scatter is None:
                self.scatter = self.ax.scatter(x, y, s=2.5, c=c, cmap='rainbow', alpha=0.8) # , vmin=0, vmax=1)
            else:
                self.scatter.set_offsets(np.array([x, y]).T)
                # self.scatter.set_sizes(s)
                self.scatter.set_array(c)
            cvals = np.mean(cvals, axis=0)
            if self.plot is None:
                loc = 3
                # temporary for plotting (TODO: fix)
                if int(self.given_num) in [187, 309, 317, 392]:
                    loc = 2
                self.ax2 = mpl_il.inset_axes(self.ax, width="30%", height="30%", loc=loc)
                self.plot, = self.ax2.plot(np.arange(-3, 3, 0.2), cvals)
                self.ax2.set_ylim(min(cvals), max(cvals))
            else:
                self.plot.set_ydata(cvals)
                self.ax2.set_ylim(min(cvals), max(cvals)), lead_value, rear_value, overlap
        # show items in V
        if self.V != None:
            if not hasattr(self, 'plotted_v'):
                self.plotted_v = True
                for k, v in self.V.items():
                    (_, cost, X, Y, B) = v
                    nX, nY = [], []
                    for (x, y) in zip(X, Y):
                        my_id = localize(self.osm, x, y)
                        if my_id not in [self.start_lanelet_id, self.end_lanelet_id]:
                            break
                        else:
                            nX += [x]
                            nY += [y]
                    if len(nX) > 0 and len(nY) > 0:
                        print(k, len(nX), len(nY), cost)
                    if cost >= 5:
                        color = 'red'
                        opac = 1./(1+np.exp(-cost.detach().cpu().numpy()+5))
                    else:
                        color = 'green'
                        opac = 0.5+0.5*(5-cost.detach().cpu().numpy())/5.
                    self.ax.plot(nX, nY, color=color, alpha=float(opac)*0.5)
        # create or update bounding boxes of all vehicles
        unused_keys_so_far = set(self.display_bboxes.keys())
        for other_veh_id, other_idx in self.frame_dict[self.frames[self.veh_id][self.start_frame + self.i]]:
            if other_veh_id == self.veh_id:
                if self.V == None:
                    if self.veh_id not in self.display_bboxes.keys():
                        self.display_bboxes[self.veh_id] = Polygon2(copy.deepcopy(self.curr_bbox), facecolor="black", zorder=1000)    
                        self.ax.add_patch(self.display_bboxes[self.veh_id])
                    else:
                        self.display_bboxes[self.veh_id].set_xy(copy.deepcopy(self.curr_bbox))
                    if self.veh_id in unused_keys_so_far:
                        unused_keys_so_far.remove(self.veh_id)
                else:
                    _, new_x, new_y = self.find_state(self.lat_dist + 10, self.lon_dist)
                    _, new_x2, new_y2 = self.find_state(self.lat_dist - 10, self.lon_dist)
                    if self.veh_id not in self.display_bboxes.keys():
                        self.display_bboxes[self.veh_id] = self.ax.plot([new_x, new_x2], [new_y, new_y2], linewidth=3, color='black')[0]
                    else:
                        self.display_bboxes[self.veh_id].set_xdata([new_x, new_x2])
                        self.display_bboxes[self.veh_id].set_ydata([new_y, new_y2])
            else:
                veh_x, veh_y = self.centers[other_veh_id][other_idx][0], self.centers[other_veh_id][other_idx][1],
                visible = self.where[other_veh_id][other_idx] == self.start_lanelet_id \
                    or self.where[other_veh_id][other_idx] == self.end_lanelet_id
                if not visible:
                    continue
                if other_veh_id not in self.display_bboxes.keys():
                    self.display_bboxes[other_veh_id] = Polygon2(copy.deepcopy(self.bboxes[other_veh_id][other_idx]), facecolor='darkgreen',zorder=1001)
                    self.ax.add_patch(self.display_bboxes[other_veh_id])
                else:
                    self.display_bboxes[other_veh_id].set_xy(copy.deepcopy(self.bboxes[other_veh_id][other_idx]))
                if other_veh_id in unused_keys_so_far:
                    unused_keys_so_far.remove(other_veh_id)
        # show state information in plot
        self.fig.suptitle(",".join([str(round(decimal.Decimal(item), 2)) for item in self.state]))
        # remove unnecessary bounding boxes
        for key in unused_keys_so_far:
            self.display_bboxes[key].remove()
            del self.display_bboxes[key]
        # mode=rgb_array
        if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
            self.fig.canvas.draw()
            img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return img
