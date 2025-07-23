import numpy as np
import torch
from src.reinforce.policy_network import PolicyNetwork
from typing import Optional
from pathlib import Path
from src.util.plotter import PRJ_ROOT


class ReinforceAgent:
    """An agent that learns a policy via the REINFORCE algorithm"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size1: int,
        hidden_size2: int,
        learning_rate: float,
        gamma: float
    ):
        """
        Args:
            obs_dim: Dimension of the observation space
            action_dim: Dimension of the action space
            hidden_size1: Size of the first hidden layer
            hidden_size2: Size of the second hidden layer            
            learning_rate: The learning rate
            gamma: The discount factor
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy = PolicyNetwork(
            obs_dim, action_dim, hidden_size1, hidden_size2)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=learning_rate)

    def get_action(self, obs: np.array) -> np.array:
        """Returns an action, conditioned on the policy and observation.
        Args:
            obs: Observation from the environment
        Returns:
            action: An action to be performed
            log_prob: The logarithm of the action probability
        """
        # unsqueeze at axis 0:
        # consider the input observation a single mini batch
        # the input to the policy network expects the batch
        # and the second dimension is the vector itself.
        # i.e, the in_features
        obs_torch = torch.as_tensor(obs).float().unsqueeze(0)
        means, std_devs = self.policy(obs_torch)

        # get a normal distribution of the forward pass that
        # can be sampled
        norm_dist = torch.distributions.Normal(
            loc=means,
            scale=std_devs
        )

        # sample the actions from the predicted distributions
        # this is policy(a | s)
        action = norm_dist.sample()
        # get the log probability of this action
        prob = norm_dist.log_prob(action).mean()

        return action.squeeze(0).numpy(), prob

    def update(self, log_probs, rewards):
        """Update the policy network's weights.
        Args:
            log_probs: Logarithms of the action probabilities 
            rewards: The rewards received for taking that actions
        """
        action_rewards = self.compute_returns(rewards)
        loss = torch.tensor(0.0)
        # take the negative because we need gradient ascent
        for gt, log_prob in zip(action_rewards, log_probs):
            loss += -log_prob * gt
        # determine the gradients
        # reset the gradient to prevent accumulation
        self.optimizer.zero_grad()
        loss.backward()
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

        print(
            f"\nSaving model state: {', '.join(state_data.keys())} to {out_path / filename}.")

        torch.save(state_data, out_path / filename)
