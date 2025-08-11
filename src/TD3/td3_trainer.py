import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from torchinfo import summary
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
                 time_steps: int = 1e6,  # as per academic paper!
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
        self._time_steps = time_steps
        self.diag = EnvDiagnostics()

    def train(self):
        """
        Run the training loop
        """

        # initialise replay buffer - TODO doesn't need to be a classmethod - maintains all prior experience
        replay_buffer = ReplayBuffer.init(self._replay_buf_size)
        self.diag.diagnostics(self._env, self.agent)

        for trial in range(self._n_trials):
            np.random.seed(trial)
            torch.manual_seed(trial)
            # start a new episode
            logger.info("Training trial: %d" % trial)
            done = False
            obs, _ = self._env.reset(seed=trial)
            exploration_noise = 0

            # count all time step per episode - start from one to avoid 0 mod interval.
            time_steps = 1
            eval_means = []
            eval_sds = []
            initial_actor_weight = None
            while time_steps <= self._time_steps:
                if not initial_actor_weight:
                    initial_actor_weight = list(self.agent.actor.parameters())[
                        0][0, 0].item()
                    logger.info("initial actor weight: %f" %
                                initial_actor_weight)
                if time_steps <= self._update_start:
                    # logger.info("random action sampling, step: %d" %
                    #             time_steps)
                    # randomly sample the action space
                    action = self._env.action_space.sample()
                else:
                    # logger.info("Model action training, step: %d" % time_steps)
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
                        [torch.tensor(state, dtype=torch.float32) for state in states], dim=0)
                    actions_tensor = torch.stack(
                        [torch.tensor(action, dtype=torch.float32) for action in actions], dim=0)
                    rewards_tensor = torch.stack(
                        [torch.tensor(reward, dtype=torch.float32) for reward in rewards], dim=0)
                    next_states_tensor = torch.stack([torch.tensor(state_n, dtype=torch.float32)
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

                    # check input and q1 sizes.
                    # logger.info(
                    #     f"states tensor shape: {states_tensor.shape}")
                    # logger.info(
                    #     f"actions tensor shape: {actions_tensor.shape}")
                    # logger.info(
                    #     f"rewards tensor shape: {rewards_tensor.shape}")
                    # logger.info(
                    #     f"dones tensor shape: {done_tensor.shape}")

                    # compute Q_target = r + gamma * min(Q1', Q2')
                    q1_targets = self.agent.get_target_q1(
                        next_states_tensor, target_action_eps)
                    q2_targets = self.agent.get_target_q2(
                        next_states_tensor, target_action_eps)

                    # logger.info(
                    #     f"q1 targets shape: {q1_targets.shape}")
                    # logger.info(
                    #     f"q2 targets shape: {q2_targets.shape}")

                    # done tensor ensures that terminal states (1) have no future value when updating targets
                    q_targets = rewards_tensor.unsqueeze(1) + self.agent._gamma * \
                        (1 - done_tensor.unsqueeze(1)) * \
                        torch.min(q1_targets, q2_targets)

                    q1_pred, q2_pred = self.agent.update_critics(
                        actions_tensor, states_tensor, q_targets)

                    # logger.info(
                    #     f"q1 pred shape: {q1_pred.shape}; q2 pred shape: {q2_pred.shape}"
                    # )

                    if time_steps % self._actor_update_delay == 0:
                        # update actor
                        total_grad_norm = self.agent.update_actor(
                            states_tensor)

                        if time_steps % 5000 == 0:
                            logger.info(
                                f"q targets shape: {q_targets.shape}"
                            )
                            logger.info(
                                f"total grad norm after 5000 time steps: {total_grad_norm}")
                        # update target networks
                        self.agent.update_target_networks()

                if time_steps % self._evaluate_interval == 0:
                    mean, sd = self.evaluate(current_time_step=time_steps,
                                             current_trial=trial)
                    logger.info(
                        "Evaluating model - Time step: %d - Mean returns: %3.2f" % (time_steps, mean))
                    eval_means.append(mean)
                    eval_sds.append(sd)

                    current_actor_weight = list(self.agent.actor.parameters())[
                        0][0, 0].item()
                    logger.info(
                        f"Current actor weight: {current_actor_weight}")
                    logger.info(
                        f"Weight changed: {abs(current_actor_weight - initial_actor_weight) > 1e-6}")

                if done:
                    # reset the environment so learning can continue through this epoch/trial
                    # training for 1e6 time steps will require multiple training episodes
                    s_next, _ = self._env.reset(seed=trial)

                obs = s_next
                time_steps += 1

            # at the end of each trial, update the learning rate data
            self.metrics.update_td3_average(eval_means, eval_sds)

        model_summary = summary(
            self.agent.actor, input_size=(self.agent._obs_dim,),
            device='cpu', verbose=0)

        logger.info(
            f"=== TD3 Model Summary ===\n"
            f"{str(model_summary)}\n"
            f"--- Epochs / timesteps ---\n"
            f"    {self._time_steps}\n"
            f"--- Trials ---\n"
            f"    {self._n_trials}\n"
            f"=== Agent input dimensions ===\n"
            f"   (observation space): {self.agent._obs_dim}\n"
        )

        # TODO dataclass
        checkpoint = {
            "epoch": self._time_steps,
            "trials": self._n_trials,
            "actor_state_dict": self.agent.actor.state_dict(),
            "actor_optimiser_state_dict": self.agent.actor_optimizer.state_dict(),
            "critic_1_state": self.agent.critic1.state_dict(),
            "critic_1_optimiser": self.agent.critic1_optimizer.state_dict(),
            "critic_2_state": self.agent.critic2.state_dict(),
            "critic_2_optimiser": self.agent.critic2_optimizer.state_dict(),
            "target_actor_state": self.agent.target_action.state_dict(),
            "target_critic_1": self.agent.targetQ1.state_dict(),
            "target_critic_2": self.agent.targetQ2.state_dict(),
            # mean values
            "returns": self.metrics.get_td3_learning()[0]
        }

        self.agent.save_model(checkpoint)

    def evaluate(self, current_time_step: int, current_trial: int, eval_episodes=10):
        # TODO this can potentially be part of the base class implementation
        # set model to eval mode
        self.agent.actor.eval()

        # make a new gym env with the same spec as the one used for training
        eval_env = gym.make(self._env.spec.id, render_mode="rgb_array")
        eval_frames = []
        episode_returns = []

        with torch.no_grad():
            for ep in range(eval_episodes):
                obs, _ = eval_env.reset(seed=current_trial)
                eval_time_steps = 0
                episode_reward = 0
                done = False

                while not done:
                    # get an action w/ no exploration noise
                    action = self.agent.get_action(obs)
                    obs, reward, terminated, truncated, _ = eval_env.step(
                        action)
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

        if current_time_step > 0 and \
                current_time_step % self._time_steps == 0:
            # record data
            record_gif(eval_frames,
                       filename=f"TD3-Trial-{current_trial}",
                       epochs=current_time_step)

        return np.mean(episode_returns), np.std(episode_returns)


class EnvDiagnostics:
    def __init__(self):
        self._env = None

    @classmethod
    def diagnostics(cls, env: gym.Env, agent: ActorCriticAgent):
        cls._env = env
        # Diagnostic test
        logger.info("=== TD3 Diagnostic ===")
        logger.info(f"Environment: {env.spec.id}")
        logger.info(f"Action space: {env.action_space}")
        logger.info(f"Observation space: {env.observation_space}")

        # Test actor output
        obs, _ = env.reset()
        action = agent.get_action(obs)
        logger.info(f"Actor output: {action}")
        logger.info(
            f"Actor output range: [{action.min():.3f}, {action.max():.3f}]")

        # Test random baseline (quick version)
        episode_return = 0
        obs, _ = env.reset()
        for _ in range(1000):  # Fixed length episode
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            if terminated or truncated:
                break
        logger.info(f"Sample random episode return: {episode_return:.2f}")
