import torch
import torch.nn as nn


class CriticPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # define sequential model, without RELU on the output
        self.policy_net = nn.Sequential(
            nn.Lienar(obs_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # the output is a Q value (prediction of value)
            nn.Linear(128, 1)
        )

        def forward(self, a, s):
            # to meet the input dimensions, stack inputs horizontally
            return self.policy_net(
                torch.cat([s, a], dim=1)
            )
