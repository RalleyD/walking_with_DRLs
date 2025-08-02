import torch
import torch.nn as nn
import numpy as np
from src.TD3.critic_network import CriticPolicy
from TD3.actor_network import ActorPolicy


class ReplayBuffer:
    def __init__(self):
        self._buf_size = None
        self._buf_ptr = None
        self._replay_buf = []
        self.min_buf_size = False
        self._min_batch_samples = 1000
        pass

    @classmethod
    def init(cls, replay_buf_size: int):
        instance = cls()
        instance._buf_ptr = 0
        instance._buf_size = replay_buf_size
        return instance

    # state_current, action_current, reward, state_next, done_state: int):
    def add(self, *args):
        """
        """
        if len(self._replay_buf) == self._buf_size:
            # circular update
            self._replay_buf[self._buf_ptr] = args
            self._buf_ptr = (self._buf_ptr + 1) % self._buf_size
        else:
            self._replay_buf.append(args)
            self.min_buf_size = len(
                self._replay_buf) >= self._min_batch_samples

    def sample(self, batch_size: int) -> list:
        """
        """
        indices = np.random.randint(
            0, len(self._replay_buf)-1, size=batch_size)

        # return a mini batc of replay samples
        return [self._replay_buf[i] for i in indices]


class ActorCriticAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        learning_rate: float,
        gamma: float,
        soft_target_update_tau=0.005,
    ):
        # TODO consider a base Agent class for inheritence
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._gamma = gamma
        self._learning_rate = learning_rate
        self._tau = soft_target_update_tau

        # Actor
        self.actor = ActorPolicy(
            self._obs_dim, self._action_dim, max_action=1
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=learning_rate)

        # Two critics
        self.critic1 = CriticPolicy(
            self._obs_dim, self._action_dim
        )
        self.critic1_optimizer = torch.optim.AdamW(
            self.critic1.parameters(), lr=learning_rate)

        self.critic2 = CriticPolicy(
            self._obs_dim, self._action_dim
        )
        self.critic2_optimizer = torch.optim.AdamW(
            self.critic2.parameters(), lr=learning_rate)

        # Target policies
        self.target_action = ActorPolicy(
            self._obs_dim, self._action_dim, max_action=1
        )
        self.target_actor_optimizer = torch.optim.AdamW(
            self.target_action.parameters(), lr=learning_rate)

        self.targetQ1 = CriticPolicy(
            self._obs_dim, self._action_dim
        )
        self.tq1_optimizer = torch.optim.AdamW(
            self.targetQ1.parameters(), lr=learning_rate)

        self.targetQ2 = CriticPolicy(
            self._obs_dim, self._action_dim
        )
        self.tq2_optimizer = torch.optim.AdamW(
            self.targetQ2.parameters(), lr=learning_rate)

    def get_action(self, obs: np.array) -> np.ndarray:
        obs_torch = torch.as_tensor(obs).float().unsqueeze(0)
        x = self.actor(obs_torch)
        # detatch here as we don't want to maintain gradients during environment interaction
        return x.squeeze(0).detach().cpu().numpy()

    def get_target_action(self, obs: torch.Tensor) -> torch.Tensor:
        # obs_torch = torch.as_tensor(obs).float().unsqueeze(0)
        return self.target_action(obs).detach().cpu()  # .squeeze(0).numpy()

    def get_q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic1(state, action)  # .squeeze(0).numpy()

    def get_q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.critic2(state, action)  # .squeeze(0).numpy()

    def get_target_q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # detach at the point the data enters the policy - breaks the tensor from the current computation graph.
        return self.targetQ1(state.detach(), action.detach())

    def get_target_q2(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.targetQ2(state.detach(), action.detach())

    def update_actor(self, states: torch.Tensor):
        # tau == soft update to the targets
        actor_loss = -self.critic1(states,
                                   self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_critics(self, actions: torch.Tensor, states: torch.Tensor, q_targets: torch.Tensor):
        # MSE loss functions (mean!)
        # using the same loss function will accumulate
        q1_loss_fn = nn.MSELoss(reduction="mean")
        q2_loss_fn = nn.MSELoss(reduction="mean")
        q_loss = q1_loss_fn(
            self.agent.get_q1(actions, states), q_targets) + \
            q2_loss_fn(self.agent.get_q2(
                actions, states), q_targets)

        # TODO move to critic agent update method
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        q_loss.backward()

        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

    def update_target_networks(self):
        # no gradient tracking or accumulation required for this process
        # explicitly not part of any backpropagation.
        with torch.no_grad():
            self._soft_update(self.actor, self.target_action)
            self._soft_update(self.critic1, self.targetQ1)
            self._soft_updates(self.critic2, self.targetQ2)

    def _soft_update(self, source: torch.Module, target: torch.Module):
        for param, target_param in zip(source.parameters(),
                                       target.parameters()):
            target_param.copy_(
                src=self._tau * param.data +
                (1 - self._tau) * target_param.data
            )
