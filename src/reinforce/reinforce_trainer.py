import numpy as np
import gymnasium as gym
import torch
from torchinfo import summary
import glfw
from src.reinforce.reinforce_agent import ReinforceAgent
from src.util.plotter import record_gif
from src.custom_logger import CustomLogger
from src.evaluate.performance_metrics import PerformanceMetrics

########################################
# logger
########################################
logger = CustomLogger.get_project_logger()
########################################


class ReinforceTrainer:
    """Train a REINFORCE agent on a given gym environment.

    TODOs
    per-timestep evaluation for better A/B comparison
    """

    def __init__(self,
                 env: gym,
                 agent: ReinforceAgent,
                 n_trials: int = 10,
                 n_timesteps: int = 1e6,
                 evaluate_interval: int = 5000,
                 show_policy_interval: int = 10000
                 ):
        """
        Args:
            env (gym.Env): A gym environment
            agent (ReinforceAgent): The REINFORCE agent
            evaluate_interval (int): Number of episodes between two evaluations
            show_policy_interval (int): Number of episodes between policy displays
        """
        self.env = env
        self.agent = agent
        self.evaluate_interval = evaluate_interval
        self.show_policy_interval = show_policy_interval
        self._n_trials = n_trials
        self._n_timesteps = int(n_timesteps)
        self.metrics = PerformanceMetrics()

    def train(self):
        """Run the training loop.
        """
        logger.info(f"Evaluation interval: {self.evaluate_interval}")
        logger.info(f"Number of trials: {self._n_trials}")
        logger.info(f"Number of timesteps: {self._n_timesteps}")

        for trial in range(self._n_trials):
            torch.manual_seed(trial)
            current_time_step = 0
            next_eval_interval = 1
            episode_returns = []
            eval_interval = self.evaluate_interval

            # per-trial evaluation metrics
            trial_evaluation_metrics = []

            logger.info("Training REINFORCE model - trial %d" % trial)

            while current_time_step < self._n_timesteps:
                episode_done = False
                # start a new episode
                obs, _ = self.env.reset(seed=trial)

                rewards = []
                log_probs = []
                entropies = []
                current_episode_return = 0

                while not episode_done:
                    # get the agent's action from the current observation
                    agent_action, log_prob, entropy = self.agent.get_action(
                        obs)

                    # perform action in the env, store the reward and the next obs
                    obs, reward, terminated, truncated, _ = self.env.step(
                        agent_action)

                    episode_done = terminated or truncated

                    rewards.append(reward)
                    log_probs.append(log_prob)
                    entropies.append(entropy)

                    current_episode_return += reward

                    current_time_step += 1

                self.agent.update(log_probs, rewards)
                episode_returns.append(current_episode_return)

                # maybe evaluate
                if current_time_step >= eval_interval:

                    # log policy distributions
                    means, std_devs = self.agent.get_action_metadata()
                    if means is not None:
                        logger.info(
                            f"mean range: {means.min():.3f} <-> {means.max():.3f}")
                        logger.info(
                            f"std dev range: {std_devs.min():.3f} <-> {std_devs.max():.3f}")

                    mean, sd = self.evaluate(current_time_step,
                                             trial)
                    logger.info(
                        f"Evaluating model - time step: {current_time_step}")
                    trial_evaluation_metrics.append(
                        (current_time_step, mean, sd))

                    next_eval_interval += 1

                    eval_interval = self.evaluate_interval * next_eval_interval

            checkpoint = {
                "episodes": len(episode_returns),
                "model_state_dict": self.agent.policy.state_dict(),
                "optimiser_state_dict": self.agent.optimizer.state_dict(),
                "returns": episode_returns
            }

            model_summary = summary(
                self.agent.policy, input_size=(self.agent.obs_dim,),
                device='cpu', verbose=0)

            logger.info(
                f"=== Model Summary ===\n"
                f"{str(model_summary)}\n"
                f"--- Episodes ---\n"
                # f"    {self.n_episodes}\n"
                f"    {len(episode_returns)}\n"
                f"=== Agent input dimensions ===\n"
                f"   (observation space): {self.agent.obs_dim}\n"
            )

            self.agent.save_model(checkpoint)

            # update per-trial evaluation metrics
            if trial_evaluation_metrics:
                steps, means, sds = zip(*trial_evaluation_metrics)
                self.metrics.update_reinforce_trial(
                    steps, means, sds
                )

        return episode_returns

    def evaluate(self, current_time_step: int, current_trial: int, eval_episodes=10):
        # TODO this can potentially be part of the base class implementation
        # set model to eval mode
        self.agent.policy.eval()

        # make a new gym env with the same spec as the one used for training
        eval_env = gym.make(self.env.spec.id, render_mode="rgb_array")
        eval_frames = []
        episode_returns = []

        with torch.no_grad():
            for ep in range(eval_episodes):
                obs, _ = eval_env.reset(seed=current_trial)
                episode_reward = 0
                done = False

                while not done:
                    # get an action w/ no exploration noise
                    action, _, _ = self.agent.get_action(obs)
                    obs, reward, terminated, truncated, _ = eval_env.step(
                        action)
                    done = terminated or truncated
                    # get the cumulative or average return over all the time steps
                    episode_reward += reward
                    if ep == eval_episodes-1:
                        eval_frames.append(eval_env.render())

                episode_returns.append(episode_reward)

        # back to training mode
        self.agent.policy.train()

        eval_env.close()

        if current_time_step > 0 and \
                current_time_step % self._n_timesteps == 0:
            # record data
            record_gif(eval_frames,
                       filename=f"REINFORCE-Trial-{current_trial}",
                       epochs=current_time_step)

        return np.mean(episode_returns), np.std(episode_returns)

    def show_policy(self, epoch: int):
        """
        Run a single episode in the environment and render a recording
        to view the agent's current policy.
        """
        logger.info("Recording episode")
        vis_env = gym.make(self.env.spec.id, render_mode='rgb_array')
        obs, _ = vis_env.reset()
        initial_frame = vis_env.render()
        done = False

        record_data = [initial_frame]

        while not done:
            action, _, _ = self.agent.get_action(obs)
            obs, _, terminated, truncated, _ = vis_env.step(action)
            record_data.append(vis_env.render())

            done = terminated or truncated

        vis_env.close()

        # TODO add epoch to the filename
        record_gif(record_data, epochs=epoch)
