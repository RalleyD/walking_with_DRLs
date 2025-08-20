import numpy as np
import torch
from torchinfo import summary
from src.reinforce.enhanced_policy_network import EnhancedPolicyNetwork
from typing import Optional
from pathlib import Path
from src.util.plotter import PRJ_ROOT

from src.custom_logger import CustomLogger

##############################
# Logger
##############################
logger = CustomLogger.get_project_logger()

##############################


class ReinforceAgent:
    """An agent that learns a policy via the REINFORCE algorithm"""

    def __init__(
        self,
        policy: EnhancedPolicyNetwork,
        obs_dim: int,
        action_dim: int,
        hidden_size1: int,
        hidden_size2: int,
        learning_rate: float,
        gamma: float,
        max_gradient_norm: float,
        device: str = 'cpu'
    ):
        """
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_size1: Size of the first hidden layer
            hidden_size2: Size of the second hidden layer            
            learning_rate: The learning rate
            gamma: The discount factor
            max_gradient_norm: gradient clipping factor
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self._gradient_norm = max_gradient_norm

        self._device = device

        self.policy = policy.to(self._device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate)

        self._agent_meta_means = None
        self._agent_meta_stds = None

        with torch.no_grad():
            # initial mean range and std values
            dummy_input = torch.randn(1, 17).to(self._device)
            means, stds = self.policy(dummy_input)
            logger.info(
                f"Initial mean range {means.min():.3f} <-> {means.max():.3f}")
            logger.info(
                f"Initial S.D range {stds.min():.3f} <-> {stds.max():.3f}")

    def get_action(self, obs: np.array) -> tuple:
        """Returns an action, conditioned on the policy and observation.
        Args:
            obs: Observation from the environment
        Returns:
            action: An action to be performed
            log_prob: The logarithm of the action probability
            entropy: randomness indicator
        """
        # unsqueeze at axis 0:
        # consider the input observation a single mini batch
        # the input to the policy network expects the batch
        # and the second dimension is the vector itself.
        # i.e, the in_features
        obs_torch = torch.from_numpy(obs).float().unsqueeze(0).to(self._device)

        # forward pass through the network
        means, std_devs = self.policy(obs_torch)
        # means = means.cpu()
        # std_devs = std_devs.cpu()

        # set current metadata
        self._set_agent_metadata(means, std_devs)

        # get a normal distribution of the forward pass that
        # can be sampled
        norm_dist = torch.distributions.Normal(
            loc=means,
            scale=std_devs
        )

        # sample the actions from the predicted distributions
        # this is policy(a | s)
        action = norm_dist.sample()

        # get the log probability of this action i.e
        # how likely is the policy to chose all the joint
        # angles together
        # sum over the action dimensions
        # n.b mean will cause artificial inflation of the
        # probabilities, which will affect gradient magnitude
        prob = norm_dist.log_prob(action).sum()

        # get action entropy
        # high entropy - indicator of randomness
        # low entropy - indicator of determinism
        entropy = norm_dist.entropy().detach().mean()

        # TODO detatch so that the gradients aren't maintianed during env interaction?
        return action.squeeze(0).cpu().numpy(), prob.cpu(), entropy.cpu()

    def _set_agent_metadata(self, means, stds):
        self._agent_meta_means = means
        self._agent_meta_stds = stds

    def get_action_metadata(self):
        return self._agent_meta_means, self._agent_meta_stds

    def get_model_summary(self) -> str:
        """Get a summary of the model architecture
        TODO refactor duplication"""

        model_summary = summary(
            self.policy, input_size=(self.obs_dim,),
            device=self._device, verbose=0)

        return str(model_summary)

    def update(self, log_probs, rewards):
        """Update the policy network's weights.
        Args:
            log_probs: Logarithms of the action probabilities 
            rewards: The rewards received for taking that actions
        """
        action_rewards = self.compute_returns(rewards)

        # scale the rewards due to the high variance
        action_rewards = torch.tensor(action_rewards, dtype=torch.float32)
        # action_rewards = np.array(action_rewards, dtype=np.float32)
        if len(action_rewards) > 1:
            action_rewards = action_rewards - \
                action_rewards.mean() / (action_rewards.std() +
                                         # add a very small value in the unlikely event all returns are equal (S.D is 0)
                                         1e-6)

        loss = torch.tensor(0.0)
        # take the negative because we need gradient ascent
        for gt, log_prob in zip(action_rewards, log_probs):
            loss += -log_prob * gt

        # to balance the different episode lengths, averange the loss.
        loss = loss.mean()

        # determine the gradients
        # reset the gradient to prevent accumulation
        self.optimizer.zero_grad()
        loss.backward()
        # clip the gradients (in place _)
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), self._gradient_norm)

        # update the policy's parameters
        self.optimizer.step()

    def compute_returns(self, rewards):
        """Compute the returns Gt for all the episode steps."""
        returns = []
        current_return = 0

        for reward in reversed(rewards):
            current_return = reward + self.gamma * current_return
            returns.insert(0, current_return)
        return returns

    def save_model(self, state_data: dict, out_path: Optional[Path] = None) -> None:
        """Save model, state and experiment details"""
        from datetime import datetime
        filename = "reinforce-" + datetime.now().strftime("%Y-%m-%d_%H_%M_%S") + ".pth"

        if out_path is None:
            out_path = PRJ_ROOT / "models"

        out_path.mkdir(parents=True, exist_ok=True)

        print(
            f"\nSaving model state: {', '.join(state_data.keys())} to {out_path / filename}.")

        torch.save(state_data, out_path / filename)
