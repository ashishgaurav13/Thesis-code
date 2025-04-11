import random, torch
import numpy as np
import tools
import gym
from gym.utils.step_api_compatibility import step_api_compatibility
from gym import ActionWrapper
from gym.spaces import Box
from typing import Any, Callable
from gym import RewardWrapper
import time
from collections import deque
from typing import Optional
import argparse
import pathlib, os

parser = argparse.ArgumentParser()
parser.add_argument("-env", type=str, default="")
parser.add_argument("-save_dir", type=str, default="save")
parser.add_argument("-noise", type=float, default=0)
parser.add_argument("-reduced", action="store_true", default=False)
parser.add_argument("-seed", type=int, default=1)
parser.add_argument("-baseline", type=str, default="")
args = parser.parse_args()


env_name = args.env
# env_name = 'gridworld'
# env_name = 'cartpole'
# env_name = 'cartpoletest2'
# env_name = 'highd'
# env_name = 'ant'
# env_name = 'hc'
save_dir = args.save_dir
noise = args.noise
seed = args.seed


if not os.path.exists(save_dir):
    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)


device = torch.device("cpu")






def zero_constraint_function(sa):
    return 0




def seed_fn(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True





## mujoco stuff





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





if env_name == 'gridworld':

    n_features = 3
    N = 10 # icl iterations

    def create_env():
        r = np.zeros((7, 7))
        r[6, 0] = 1.0
        t = [(6, 0)]
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
        env.seed(seed)
        return env
    


    def true_constraint_function(sa):
        s, a = sa
        x, y = s[0], s[1]
        u = [(ui, uj) for ui in [3] for uj in [0, 1, 2, 3]]
        if (x, y) in u:
            return 1
        else:
            return 0



    sa_func = lambda s, a: (*s, a)
    skip = []


    continuous_f = {
        0: False,
        1: False,
        2: False,
    }


    seed_fn(seed)
    gridworld_dim = 7
    n_actions = 8
    constraint_fn_input_dim = 3
    hidden_dim = 64
    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n
    discount_factor = 1.0
    episodes_per_epoch = 20
    ppo_subepochs = 25
    replay_buffer_size = 10000
    learning_rate = 5e-4
    learning_rate_feasibility = 2.5e-5
    minibatch_size = 64
    clip_param = 0.1
    entropy_coef = 0.01
    beta = 0.99
    alpha = 15
    n_iters = 10
    flow_iters = 20
    ppo_iters = 500
    policy_add_to_mix_every = 250
    ca_iters = 20
    continuous_actions = False
    n_trajs = 50 # 1000 # 50
    use_ppo_lag = False
    delta = None




elif env_name == 'cartpole':
    
    n_features = 5
    N = 10 # icl iterations 

    def create_env():
        env = tools.environments.GymEnvironment(
            "CustomCartPole", 
            start_pos=[[-2, 2]],
        )
        env = tools.environments.TimeLimit(env, 200)
        env = tools.environments.FollowGymAPI(env)
        env.seed(seed)
        return env


   
    def true_constraint_function(sa):
        s, a = sa
        if (s[0] < -1 or s[0] > 1):
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()



    sa_func = lambda s, a: (*s, a)
    skip = []


    continuous_f = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: False,
    }



    
    seed_fn(seed)
    max_steps_per_epoch = 4000
    ppo_subepochs = 25
    time_limit = 200
    use_gae = False
    use_early_stopping = True
    learning_rate = 5e-4 # only for test time, ppo pen gae
    learning_rate_feasibility = 2.5e-5 # learning_rate * 100 # only for test time, ppo pen gae
    discount_factor = 0.99
    ppo_iters = 300
    policy_add_to_mix_every = 150
    hidden_dim = 64
    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n
    replay_buffer_size = 10000
    episodes_per_epoch = 50
    minibatch_size = 64
    clip_param = 0.1
    entropy_coef = 0.01
    beta = 20
    alpha = 15
    n_iters = 10
    flow_iters = 20
    ca_iters = 20
    n_trajs = 50 # 400
    gae_lambda = 0.97
    delta = None
    continuous_actions = False
    window_size = 100
    use_ppo_lag = False




elif env_name == 'cartpoletest':
    

    n_features = 6
    N = 10 # icl iterations


    def create_env():
        env = tools.environments.GymEnvironment(
            "CustomCartPoleTest", 
            start_pos=[[-0.2, 0.2]], #-2,2
            noise=noise
        )
        env = tools.environments.TimeLimit(env, 200)
        env = tools.environments.FollowGymAPI(env)
        env.seed(seed)
        return env


   
    def true_constraint_function(sa):
        s, a = sa
        if s[0] <= -0.2:
        #if (s[0] < -1 or s[0] > 1):
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()



    sa_func = lambda s, a: (*s, a)
    skip = []


    continuous_f = {
        0: True,
        1: True,
        2: True,
        3: True,
        4: True,
        5: False,
    }



    
    seed_fn(seed)
    max_steps_per_epoch = 4000
    ppo_subepochs = 25
    time_limit = 200
    use_gae = False
    use_early_stopping = True
    learning_rate = 5e-4 # only for test time, ppo pen gae
    learning_rate_feasibility = 2.5e-5 # learning_rate * 100 # only for test time, ppo pen gae
    discount_factor = 0.99
    ppo_iters = 300
    policy_add_to_mix_every = 150
    hidden_dim = 64
    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.n
    replay_buffer_size = 10000
    episodes_per_epoch = 50
    minibatch_size = 64
    clip_param = 0.1
    entropy_coef = 0.01
    beta = 20
    alpha = 15
    n_iters = 10
    flow_iters = 20
    ca_iters = 20
    n_trajs = 50
    gae_lambda = 0.97
    delta = None
    continuous_actions = False
    window_size = 100
    use_ppo_lag = False



elif env_name == 'highd':


    n_features = 16
    N = 3 # icl iterations

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
            observation = [observation[i]/state_scaling[i] for i in range(len(observation))]
            if type(step_data) == dict:
                return {
                    "next_state": np.array(observation), 
                    "reward": reward, 
                    "done": done, 
                    "info": info
                }
            else:
                return np.array(observation), reward, done, info
    
        def seed(self, s=None):
            return self.env.seed(s=s)
    
        @property
        def state(self):
            return [self.env.state[i]/state_scaling[i] for i in range(len(state_scaling))]
    
        def reset(self, **kwargs):
            s = self.env.reset(**kwargs)
            assert(s is not tuple)
            s = [s[i]/state_scaling[i] for i in range(len(s))]
            return np.array(s)
        
        def render(self, **kwargs):
            return self.env.render(**kwargs)


    
    def create_env():
        env = tools.environments.HighDSampleEnvironmentWrapper(discrete=False)
        env = HighDStateScaling(env)
        env = tools.environments.TimeLimit(env, 1000)
        env = tools.environments.FollowGymAPI(env)    
        # env = ClipAction(env, 0.0, 2.0)
        return env


    def true_constraint_function(sa):
        s, a = sa
        return torch.tensor(0).float()


    sa_func = lambda s, a: (*s, *a)
    if args.reduced:
        skip = [4,5,6,7,8,9,10,11,12,14,15]
    else:
        skip = [] # [4,5,6,7,8,9,10,11,12,14,15]
    state_scaling = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


    continuous_f = {idx: True for idx in range(16)}
    continuous_f[4] = False
    continuous_f[5] = False
    continuous_f[6] = False
    continuous_f[7] = False
    continuous_f[8] = False
    continuous_f[9] = False
    continuous_f[10] = False
    continuous_f[11] = False
    continuous_f[15] = False


    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.shape[0]    
    episodes_per_epoch = 20
    seed_fn(seed)
    max_steps_per_epoch = 4000
    ppo_subepochs = 25
    time_limit = 1000
    use_gae = True
    use_early_stopping = True
    learning_rate = 5e-4 # only for test time, ppo pen gae
    learning_rate_feasibility = learning_rate * 100 # only for test time, ppo pen gae
    discount_factor = 1.0
    ppo_iters = 50
    policy_add_to_mix_every = 50
    hidden_dim = 64
    minibatch_size = 64
    clip_param = 0.3
    entropy_coef = 0.01
    beta = 1
    alpha = 15
    n_iters = 10
    flow_iters = 20
    ca_iters = 20
    n_trajs = 118
    gae_lambda = 0.97
    delta = None
    continuous_actions = True
    window_size = 50
    use_ppo_lag = True







elif env_name == 'ant':


    n_features = 121
    N = 10 # icl iterations
    
    def create_env():
        env = tools.environments.GymEnvironment('AntWall-v0')
        env = tools.environments.FollowGymAPI(env)    
        env = RecordEpisodeStatistics(env)
        env = tools.environments.TimeLimit(env, 500)
        env = ClipAction(env)
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env.seed(seed)
        return env


    def true_constraint_function(sa):
        global env
        s, a = sa
        s = env.unnormalize(s)
        if s[0] <= -1:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()


    sa_func = lambda s, a: (*s, *a)
    skip = []

    continuous_f = {idx: True for idx in range(121)}

    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.shape[0]    
    hidden_dim = 64
    episodes_per_epoch = 100
    seed_fn(seed)
    continuous_actions = True
    n_iters = 10
    flow_iters = 20
    ca_iters = 10
    discount_factor = 0.99
    max_steps_per_epoch = 4000
    ppo_subepochs = 25
    time_limit = 500
    use_gae = True
    use_early_stopping = True
    minibatch_size = 64
    clip_param = 0.2
    entropy_coef = 0
    gae_lambda = 0.97
    beta = 15
    ppo_iters = 100
    policy_add_to_mix_every = 100
    learning_rate = 5e-4 # only for test time, ppo pen gae
    learning_rate_feasibility = learning_rate * 100 # only for test time, ppo pen gae
    alpha = 1
    n_trajs = 50 # 75
    delta = None    
    window_size = 50
    use_ppo_lag = True



elif env_name == 'hc':

    n_features = 24
    N = 10 # icl iterations

    
    def create_env():
        env = tools.environments.GymEnvironment('HCWithPos-v0')
        env = tools.environments.FollowGymAPI(env)    
        env = RecordEpisodeStatistics(env)
        env = tools.environments.TimeLimit(env, 500)
        env = ClipAction(env)
        env = NormalizeObservation(env)
        env = TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env.seed(seed)
        return env


    def true_constraint_function(sa):
        global env
        s, a = sa
        s = env.unnormalize(s)
        if s[0] <= -1:
            return torch.tensor(1).float()
        else:
            return torch.tensor(0).float()


    sa_func = lambda s, a: (*s, *a)
    skip = []

    continuous_f = {idx: True for idx in range(24)}

    env = create_env()
    obs_n = env.observation_space.shape[0]
    act_n = env.action_space.shape[0]    
    hidden_dim = 64
    episodes_per_epoch = 100
    seed_fn(seed)
    continuous_actions = True
    n_iters = 10
    flow_iters = 20
    ca_iters = 5
    discount_factor = 0.99
    max_steps_per_epoch = 8000
    ppo_subepochs = 25
    time_limit = 500
    use_gae = True
    use_early_stopping = True
    minibatch_size = 64
    clip_param = 0.2
    entropy_coef = 0
    gae_lambda = 0.97
    beta = 15
    ppo_iters = 100
    policy_add_to_mix_every = 20
    learning_rate = 5e-4 # only for test time, ppo pen gae
    learning_rate_feasibility = learning_rate * 100 # only for test time, ppo pen gae
    alpha = 1
    n_trajs = 50 # 75
    delta = None    
    window_size = 50
    use_ppo_lag = True






# ppo_iters = 4
# policy_add_to_mix_every = 2
# ca_iters = 1
# flow_iters = 1
# n_trajs = 5
# N = 1