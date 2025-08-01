import torch
import torch.nn as nn
import gymnasium as gym
from actor_critic_agent import ActorCriticAgent, ReplayBuffer
from src.evaluate.performance_metrics import PerformanceMetrics


class TD3Trainer:

    def __init__(self,
                 env: gym,
                 agent: ActorCriticAgent,
                 n_episodes: int,
                 policy_update_start: int,
                 replay_buffer_size: int,
                 replay_batch_size: int,
                 actor_update_delay: int = 2,
                 evaluate_interval: int = 100,  # TODO update as per paper!
                 show_policy_interval: int = 10000,
                 ):
        # TODO consider base trainer class for inheritence
        self._env = env
        self.agent = agent
        self._n_episodes = n_episodes
        self._evaluate_interval = evaluate_interval
        self._show_policy_interval = show_policy_interval
        self.metrics = PerformanceMetrics()
        self._replay_buf_size = replay_buffer_size
        self._update_start = policy_update_start
        self._batch_size = replay_batch_size
        self._actor_update_delay = actor_update_delay

    def train(self):
        """
        Run the training loop
        """

        # initialise replay buffer - doesn't need to be a classmethod - maintains all prior experience
        replay_buffer = ReplayBuffer.init(self._replay_buf_size)

        # count the time steps across all epochs
        time_steps = 0
        for episode_n in range(self._n_episodes):
            # start a new episode
            done = False
            obs, _ = self._env.reset()
            exploration_noise = 0

            while not done:
                if time_steps < 1000:
                    # randomly sample the action space
                    action = self._env.action_space.sample()
                else:
                    # get action with some clipped exploration noise
                    action = self.agent.get_action(obs)
                    exploration_noise = torch.clamp(torch.randn_like(
                        torch.tensor(action), dtype=torch.float32) * 0.1,
                        -0.5, 0.5)
                    action += exploration_noise

                    if replay_buffer.min_buf_size:
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

                        if episode_n % self._actor_update_delay == 0:
                            # update actor
                            self.agent.update_actor(states_tensor)
                            # update target networks
                            self.agent.update_target_networks()

                s_next, reward, terminated, truncated, _, _ = self._env.step(
                    action)
                done = terminated or truncated
                time_steps += 1

                replay_buffer.add(obs, action, reward, s_next, int(done))
                obs = s_next
