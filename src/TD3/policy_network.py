import torch
import torch.nn as nn


class ActorPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, max_action=1.0):
        super().__init__(obs_dim, action_dim, max_action)

        self._max_action = max_action

        # define sequential model, without RELU on the output
        self.policy_net = nn.Sequential(
            nn.Lienar(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        # tanh gives a desired joint output range [-1, 1]
        return self._max_action * torch.tanh(self.policy_net(x))
