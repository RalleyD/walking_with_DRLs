import torch
import torch.nn as nn
import numpy as np


class ActorPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=1.0, device='cpu'):
        super().__init__()

        self._max_action = torch.Tensor([max_action]).to(device)

        # define sequential model, without RELU on the output
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh gives a desired joint output range [-1, 1]
        return self._max_action * torch.tanh(self.policy_net(x))
