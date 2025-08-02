import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from src.TD3.actor_critic_agent import ActorCriticAgent, ReplayBuffer
from src.evaluate.performance_metrics import PerformanceMetrics
from src.util.plotter import record_gif
from src.custom_logger import CustomLogger

##############################
# Logger
##############################
logger = CustomLogger.get_project_logger()

##############################


class TD3Trainer:

    def __init__(self,
                 env: gym,
                 agent: ActorCriticAgent,
                 policy_update_start: int = 1e3,  # as per academic paper!
                 replay_buffer_size: int = 1e6,  # as per academic paper!
                 replay_batch_size: int = 100,  # as per academic paper!
                 n_trials: int = 10,  # as per academic paper!
                 actor_update_delay: int = 2,  # as per academic paper!
                 evaluate_interval: int = 5000,  # as per academic paper!
                 show_policy_interval: int = 10000,
                 epochs: int = 1e6,  # as per academic paper!
                 ):
        # TODO consider base trainer class for inheritence
        self._env = env
        self.agent = agent
        self._n_trials = n_trials
        self._evaluate_interval = evaluate_interval
        self._show_policy_interval = show_policy_interval
        self.metrics = PerformanceMetrics()
        self._replay_buf_size = replay_buffer_size
        self._update_start = policy_update_start
        self._batch_size = replay_batch_size
        self._actor_update_delay = actor_update_delay
        self._epochs = epochs

    def train(self):
        """
        Run the training loop
        """

        # initialise replay buffer - TODO doesn't need to be a classmethod - maintains all prior experience
        replay_buffer = ReplayBuffer.init(self._replay_buf_size)

        for trial in range(self._n_trials):
            # start a new episode
            logger.info("Training episode: %d" % trial)
            done = False
            obs, _ = self._env.reset()
            exploration_noise = 0

            # count all time step per episode
            time_steps = 0
            eval_means = []
            eval_sds = []
            while time_steps < self._epochs:
                if time_steps < self._update_start:
                    logger.info("random action sampling, step: %d" %
                                time_steps)
                    # randomly sample the action space
                    action = self._env.action_space.sample()
                else:
                    logger.info("Model action training, step: %d" % time_steps)
                    # get action with some clipped exploration noise
                    action = self.agent.get_action(obs)
                    exploration_noise = torch.clamp(torch.randn_like(
                        torch.tensor(action), dtype=torch.float32) * 0.1,
                        -0.5, 0.5).numpy()
                    action = np.add(action, exploration_noise)

                # Take action and update replay buffer
                s_next, reward, terminated, truncated, _ = self._env.step(
                    action)
                done = terminated or truncated
                replay_buffer.add(obs, action, reward, s_next, int(done))

                # Model learning
                if time_steps >= self._update_start and replay_buffer.min_buf_size:
                    # sample a mini-batch of replays
                    states, actions, rewards, next_states, dones = zip(
                        *replay_buffer.sample(self._batch_size))

                    # direct conversion to tensors during stacking should be more efficient
                    states_tensor = torch.stack(
                            [torch.tensor(state) for state in states], dim=0)
                    actions_tensor = torch.stack(
                            [torch.tensor(action) for action in actions], dim=0)
                    rewards_tensor = torch.stack(
                            [torch.tensor(reward) for reward in rewards], dim=0)
                        next_states_tensor = torch.stack([torch.tensor(state_n)
                                                      for state_n in next_states],
                                                     dim=0)
                    done_tensor = torch.stack(
                        [torch.tensor(done) for done in dones], dim=0
                    )

                    target_actions = self.agent.get_target_action(
                        next_states_tensor)

                    clipped_noise = torch.clamp(
                        torch.rand_like(
                            target_actions, dtype=torch.float32
                            # see Fujimoto et al. Evaluation.
                        ) * 0.2, -0.5, 0.5
                    )

                    target_action_eps = target_actions + clipped_noise

                    # compute Q_target = r + gamma * min(Q1', Q2')
                    q1_targets = self.agent.get_target_q1(
                        target_action_eps, next_states_tensor)
                    q2_targets = self.agent.get_target_q2(
                        target_action_eps, next_states_tensor)

                    # done tensor ensures that terminal states (1) have no future value when updating targets
                    q_targets = rewards_tensor + self.agent._gamma * \
                        (1 - done_tensor) * torch.min(q1_targets, q2_targets,
                                                      dim=1).values  # returns a NamedTuple

                    self.agent.update_critics(
                        actions_tensor, states_tensor, q_targets)

                    if trial % self._actor_update_delay == 0:
                        # update actor
                        self.agent.update_actor(states_tensor)
                        # update target networks
                        self.agent.update_target_networks()

                if time_steps % self._evaluate_interval == 0:
                    mean, sd = self.evaluate()
                    logger.info(
                        "Evaluating model - Time step: %d - Mean returns: %3.2f" % (time_steps, mean))
                    eval_means.append(mean)
                    eval_sds.append(sd)

                if done:
                    # reset the environment so learning can continue through this epoch/trial
                    # training for 1e6 time steps will require multiple training episodes
                    s_next, _ = self._env.reset()

                obs = s_next
                time_steps += 1

            # at the end of each trial, update the learning rate data
            self.metrics.update_td3_average(eval_means, eval_sds)

    def evaluate(self, eval_episodes=10):
        # TODO this can potentially be part of the base class implementation
        # set model to eval mode
        self.agent.actor.eval()

        # make a new gym env with the same spec as the one used for training
        eval_env = gym.make(self._env.spec.id, render_mode="rgb_array")
        eval_frames = []
        episode_returns = []

        for ep in range(eval_episodes):
            # TODO think about seeding!
            obs, _ = eval_env.reset()
            eval_time_steps = 0
            episode_reward = 0
            done = False

            while not done:
                # get an action w/ no exploration noise
                action = self.agent.get_action(obs)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                eval_time_steps += 1
                # get the cumulative or average return over all the time steps
                episode_reward += reward
                if ep == eval_episodes-1:
                    eval_frames.append(eval_env.render())

            episode_returns.append(episode_reward)

        # back to training mode
        self.agent.actor.train()

        eval_env.close()

        # record data
        record_gif(eval_frames,
                   filename="TD3",
                   epochs=10)

        return np.mean(episode_returns), np.std(episode_returns)
