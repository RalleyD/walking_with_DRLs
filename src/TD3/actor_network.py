import torch
import torch.nn as nn
import numpy as np


class ActorPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=1.0):
        super().__init__()

        self._max_action = torch.Tensor([max_action])

        # define sequential model, without RELU on the output
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh gives a desired joint output range [-1, 1]
        return self._max_action * torch.tanh(self.policy_net(x))
