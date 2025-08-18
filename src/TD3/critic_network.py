import torch
import torch.nn as nn


class CriticPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, device='cpu'):
        super().__init__()

        # define sequential model, without RELU on the output
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            # the output is a Q value (prediction of value)
            nn.Linear(300, 1)
        ).to(device)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # to meet the input dimensions, stack inputs horizontally
        x = self.policy_net(
            torch.cat([s, a], dim=1)
        )
        return x
