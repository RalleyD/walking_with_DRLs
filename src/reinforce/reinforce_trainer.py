import numpy as np
import gymnasium as gym
from torchsummary import summary
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
    """

    def __init__(self,
                 env: gym,
                 agent: ReinforceAgent,
                 n_episodes: int,
                 evaluate_interval: int = 100,
                 show_policy_interval: int = 10000
                 ):
        """
        Args:
            env (gym.Env): A gym environment
            agent (ReinforceAgent): The REINFORCE agent
            n_episodes (int): Number of episodes to run the environment
            evaluate_interval (int): Number of episodes between two evaluations
            show_policy_interval (int): Number of episodes between policy displays
        """
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.evaluate_interval = evaluate_interval
        self.show_policy_interval = show_policy_interval
        self.metrics = PerformanceMetrics()

    def train(self):
        """Run the training loop.
        """
        # episode_lengths = []
        episode_returns = []

        for episode_n in range(self.n_episodes):
            # start a new episode
            done = False
            obs, _ = self.env.reset()

            rewards = []
            log_probs = []
            entropies = []
            current_episode_return = 0

            while not done:
                # get the agent's action from the current observation
                agent_action, log_prob, entropy = self.agent.get_action(obs)

                # perform action in the env, store the reward and the next obs
                obs, reward, terminated, truncated, info = self.env.step(
                    agent_action)

                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(log_prob)
                entropies.append(entropy)

                current_episode_return += reward

            self.agent.update(log_probs, rewards)
            # episode_lengths.append(len(rewards))
            episode_returns.append(current_episode_return)

            # print("\n=== Training stats: ===")
            # print("\tAverage episode length: ", np.mean(episode_lengths))
            self.metrics.update(episode_return=current_episode_return,
                                policy_entropy=np.mean(entropies))

            if episode_n % self.evaluate_interval-1 == 0:
                logger.info(
                    f"\n=== Episode {episode_n} ===\n"
                    f"     Mean reward from last {self.evaluate_interval}"
                    f"returns: {np.mean(episode_returns[-self.evaluate_interval:])}\n"
                )

            if episode_n % self.show_policy_interval-1 == 0:
                self.show_policy()

        checkpoint = {
            "epoch": self.n_episodes,
            "model_state_dict": self.agent.policy.state_dict(),
            "optimiser_state_dict": self.agent.optimizer.state_dict(),
            "returns": episode_returns
        }

        model_summary = summary(
            self.agent.policy, input_size=(self.agent.obs_dim,))

        logger.info(
            f"=== Model Summary ===\n"
            f"{str(model_summary)}\n"
            f"--- Epochs ---\n"
            f"    {self.n_episodes}\n"
            f"=== Agent input dimensions ===\n"
            f"   (observation space): {self.agent.obs_dim}\n"
        )

        self.agent.save_model(checkpoint)

        return episode_returns

    def show_policy(self):
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
        record_gif(record_data, epochs=self.n_episodes)
