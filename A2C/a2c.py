from typing import Any, Dict, Optional, Type, Union, List, Tuple
import time
import gym
import datetime
import torch as th
import numpy as np
from gym import spaces
from torch.nn import functional as F
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt

# from tools.on_policy_algorigthms import OnPolicyAlgorithm
from A2C.policy import ActorCriticPolicy
from preprocessor.preprocessors import series_decomposition
from stable_baselines3.common import utils
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.utils import explained_variance, set_random_seed, get_device, update_learning_rate, get_schedule_fn, obs_as_tensor, safe_mean
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.env_util import is_wrapped
from stable_baselines3.common.monitor import Monitor
from tools.buffers import RolloutBuffer



class A2C():
    """
    Advantage Actor Critic (A2C)

    Paper: https://arxiv.org/abs/1602.01783
    Code: This implementation borrows code from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (https://github.com/hill-a/stable-baselines)

    Introduction to A2C: https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        # policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 7e-4,
        agent_num: int = 4,
        max_level: int=3,
        n_steps: int = 5,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        rms_prop_eps: float = 1e-5,
        use_rms_prop: bool = True,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        normalize_advantage: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        self.normalize_advantage = normalize_advantage
        # self.logger = None
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        # self.n_envs = env.num_envs
        self.num_timesteps = 0
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.gamma = gamma
        self.seed = seed
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.verbose = verbose
        self.use_sde = use_sde
        self.device = get_device(device)
        self.sde_sample_freq = sde_sample_freq
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.action_noise = None
        self.ep_info_buffer = None
        self.ep_success_buffer = None
        # self._last_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]
        self._last_state = None
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        self.last_action = None
        # When using VecNormalize:
        self._last_original_obs = None  # type: Optional[Union[np.ndarray, Dict[str, np.ndarray]]]，set 1 in _setup_learn()
        self._episode_num = 0
        self._vec_normalize_env = None
        self._custom_logger = False
        self.tensorboard_log = tensorboard_log
        self._n_updates = 0
        self.agent_num = agent_num
        self.max_level = max_level
        self.Reward_total = [] 
        self.reward_episode = []

        if env is not None:
            env = self._wrap_env(env, self.verbose, monitor_wrapper)

        self.n_envs = env.num_envs
        self.env = env
        # Update optimizer inside the policy if we want to use RMSProp
        # (original implementation) rather than Adam
        if use_rms_prop and "optimizer_class" not in self.policy_kwargs:
            self.policy_kwargs["optimizer_class"] = th.optim.RMSprop
            self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)

        if _init_setup_model:
            self._setup_model()

    def collect_rollouts(self, env: VecEnv, rollout_buffer: RolloutBuffer, n_rollout_steps: int,) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_state is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
            # self.policy.reset_noise(1)
        # last_action = np.zeros((self.agent_num, self.action_space.shape[0]))  #the agent input considers the last step action
        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            if self._last_episode_starts == 1:
                self.last_action = th.zeros((self.agent_num, self.action_space.shape[0]))  #the agent input considers the last step action
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                actions = th.zeros([self.agent_num, self.action_space.shape[0]])
                action_mean = th.zeros([self.agent_num, self.action_space.shape[0]])
                log_probs = th.zeros([self.agent_num, 1])
                agent_oh = th.eye(self.agent_num)
                multi_obs = series_decomposition(self._last_state[0], self.max_level)   #multi_obs.shape=[4, 30, 252]
                obs_tensor = obs_as_tensor(multi_obs, self.device)
                state_tensor = obs_as_tensor(self._last_state[0], self.device)   # self._last_state.shape=[1，252，30]  
                for agent_id in range(self.agent_num):
                    action, log_prob, mu = self.policy.forward(obs_tensor[agent_id].unsqueeze(0), \
                                                        self.last_action[agent_id].unsqueeze(0),  \
                                                        agent_oh[agent_id].unsqueeze(0))
                    log_probs[agent_id] = log_prob
                    actions[agent_id] = action
                    action_mean[agent_id] = mu
                    self.last_action[agent_id] = action     #update the last actions array
            
                '''
                  Critic receive the state, obs, and actions from all agents as input
                '''
                critic_input_latent, critic_cf_input_latent = self.policy.get_critic_input(actions.to(self.device), obs_tensor, state_tensor.t(), action_mean)
                values = self.policy.value_net(critic_input_latent)  # values.shape=[agent_num, 1]
                values_baseline = self.policy.value_net(critic_cf_input_latent)


            '''
              the actions contains actions of different agents with shape [agent_num, stock_num], but the env only receive [1, stock_num]
              so we need to further process the actions from multi-agents, the most easier way is mean the actions from multi-agents.
            '''
            actions = actions.cpu().numpy()
            joint_actions = np.mean(actions, axis=0)

            # Rescale and perform action
            clipped_actions = joint_actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)

            new_state, rewards, dones, infos = env.step(clipped_actions.reshape(1,-1))

            if dones:
                self.Reward_total.append(sum(self.reward_episode))
                self.reward_episode = []
            else:
                self.reward_episode.append(rewards)

            self.num_timesteps += env.num_envs

            n_steps += 1

            # multi_obs.shape = (agent_num, stock_num, lookback)
            # _last_state.shape = (1, lookback, stock_num)
            rollout_buffer.add(self._last_state, multi_obs, actions, self.last_action, rewards, \
                                self._last_episode_starts, values, values_baseline, log_probs)
            self._last_state = new_state
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            last_timestep_action = th.zeros([self.agent_num, self.action_space.shape[0]])
            last_timestep_action_mean = th.zeros([self.agent_num, self.action_space.shape[0]])
            multi_obs = series_decomposition(self._last_state[0], self.max_level)
            obs_tensor = obs_as_tensor(multi_obs, self.device)
            state_tensor = obs_as_tensor(self._last_state[0], self.device)
            critic_last_input_latent, critic_last_cf_input_latent = self.policy.get_critic_input(last_timestep_action.to(self.device), \
                                                                                        obs_tensor, state_tensor.t(), last_timestep_action_mean)
            values = self.policy.value_net(critic_last_input_latent)  # values.shape=[agent_num, 1]
            
            # obs_tensor = obs_as_tensor(new_state, self.device)
            # _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        return True

    def train(self) -> None:
        """
        Update policy using the currently gathered
        rollout buffer (one gradient step over whole data).
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)

        # This will only loop once (get all data in one go)
        for rollout_data in self.rollout_buffer.get(batch_size=None):

            actions = rollout_data.actions
            last_actions = rollout_data.last_actions
            observations = rollout_data.observations
            states = rollout_data.states
            # if isinstance(self.action_space, spaces.Discrete):
                # Convert discrete action from float to long
                # actions = actions.long().flatten()
            # values, log_probs, entropys = [], [], []
            agent_oh = th.eye(self.agent_num)
            values = th.zeros([actions.shape[0], actions.shape[1], 1])
            log_probs = th.zeros([actions.shape[0], actions.shape[1]])
            entropys = th.zeros([actions.shape[0], actions.shape[1]])
            for transistion_id in range(actions.shape[0]):
                # TODO: avoid second computation of everything because of the gradient
                value, value_baseline, log_prob, entropy = self.policy.evaluate_actions(states[transistion_id], \
                                            observations[transistion_id], actions[transistion_id], \
                                            last_actions[transistion_id], agent_oh)
                # value = value.flatten()
                values[transistion_id] = value
                log_probs[transistion_id] = log_prob
                entropys[transistion_id] = entropy
            # Z-Score Normalize advantage (not present in the original implementation)
            # advantages = rollout_data.advantages
            advantages = rollout_data.co_adv
            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Policy gradient loss
            policy_loss = -(advantages * log_prob).mean()
            # Value loss using the TD(gae_lambda) target
            # print(rollout_data.returns.shape,values.shape)
            # exit()
            value_loss = F.mse_loss(rollout_data.returns, values.cuda())

            # Entropy loss favor exploration
            if entropy is None:
                # Approximate entropy when no analytical form
                entropy_loss = -th.mean(-log_prob)
            else:
                entropy_loss = -th.mean(entropy)

            loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

            # Optimization step
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip grad norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        # explained_var = explained_variance(self.rollout_buffer.values.sum(2).flatten(), self.rollout_buffer.returns.flatten())


        self._n_updates += 1
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/entropy_loss", entropy_loss.item())
        self.logger.record("train/policy_loss", policy_loss.item())
        self.logger.record("train/value_loss", value_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

    def learn(self, total_timesteps: int, log_interval: int = 100, eval_env: Optional[GymEnv] = None, eval_freq: int = -1,
        n_eval_episodes: int = 5, tb_log_name: str = "A2C", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True) -> "A2C":
        iteration = 0
        total_timesteps = self._setup_learn(
            total_timesteps, eval_env, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )
        portfolio_return = []
        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(self.env, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                # if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    # self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    # self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

            # evaluation
        #     df_account_value, df_actions = self.DRL_prediction(
        #         environment = eval_env
        #     )

        #     portfolio_return.append(df_account_value['daily_return'].sum())
        # pd_portfolio_return = pd.DataFrame(columns=['sum'],data=portfolio_return)
        # now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
        # pd_portfolio_return.to_csv("./results/iter_account_value_" + now + str(self.agent_num) + ".csv")
        pd_episode_reward = pd.DataFrame(columns=['reward'],data=self.Reward_total)
        pd_episode_reward.to_csv("./results/episode_reward.csv")
        plt.plot(range(len(self.Reward_total)), self.Reward_total, "r")
        plt.savefig("results/episode_reward.png")
        plt.close()

        return self

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
        buffer_cls = RolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            agent_num = self.agent_num,
        )
        self.policy = ActorCriticPolicy(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.agent_num,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def set_random_seed(self, seed: Optional[int] = None) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device.type == th.device("cuda").type)
        self.action_space.seed(seed)
        if self.env is not None:
            self.env.seed(seed)
        # if self.eval_env is not None:
        #     self.eval_env.seed(seed)

    def _wrap_env(self, env: GymEnv, verbose: int = 0, monitor_wrapper: bool = True) -> VecEnv:
        """ "
        Wrap environment with the appropriate wrappers if needed.
        For instance, to have a vectorized environment
        or to re-order the image channels.

        :param env:
        :param verbose:
        :param monitor_wrapper: Whether to wrap the env in a ``Monitor`` when possible.
        :return: The wrapped environment.
        """
        if not isinstance(env, VecEnv):
            if not is_wrapped(env, Monitor) and monitor_wrapper:
                if verbose >= 1:
                    print("Wrapping the env with a `Monitor` wrapper")
                env = Monitor(env)
            if verbose >= 1:
                print("Wrapping the env in a DummyVecEnv.")
            env = DummyVecEnv([lambda: env])
        return env

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = get_schedule_fn(self.learning_rate)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) ->int:
        #-> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=100)
            self.ep_success_buffer = deque(maxlen=100)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_state is None:
            self._last_state = self.env.reset()  # pytype: disable=annotation-type-mismatch
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            # self._last_episode_starts = np.ones((1,), dtype=bool)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                pass
                # self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # if eval_env is not None and self.seed is not None:
        #     eval_env.seed(self.seed)

        # eval_env = self._get_eval_env(eval_env)

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        # callback = self._init_callback(callback, eval_env, eval_freq, n_eval_episodes, log_path)

        return total_timesteps

    def _update_learning_rate(self, optimizers: Union[List[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        # Log the current learning rate
        self.logger.record("train/learning_rate", self.lr_schedule(self._current_progress_remaining))

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, self.lr_schedule(self._current_progress_remaining))

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    @property
    def logger(self) -> Logger:
        """Getter for the logger object."""
        return self._logger

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Get the model's action(s) from an observation

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param mask: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        return self.policy.predict(observation, state, mask, deterministic)

    def DRL_prediction(self, environment):
        test_env, test_state = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        episode_start = 1
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            test_multi_obs = series_decomposition(test_state[0], self.max_level) 
            if episode_start == 1:
                last_action = th.zeros([self.agent_num, test_state.shape[2]])
            action, agent_actions = self.predict(test_multi_obs, last_action, th.eye(self.agent_num))
            #account_memory = test_env.env_method(method_name="save_asset_memory")
            #actions_memory = test_env.env_method(method_name="save_action_memory")
            test_state, rewards, dones, info = test_env.step(action)
            episode_start = dones
            last_action = th.from_numpy(agent_actions)
            if i == (len(environment.df.index.unique()) - 2):
              account_memory = test_env.env_method(method_name="save_asset_memory")
              actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]