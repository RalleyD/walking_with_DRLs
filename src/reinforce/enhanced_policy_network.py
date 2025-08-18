import torch
import torch.nn as nn


class EnhancedPolicyNetwork(nn.Module):
    """A neural network that estimates the mean and standard deviation of a normal distribution
    from which the agent's action is sampled."""

    def __init__(self, obs_dim, action_dim, hidden_size1=256, hidden_size2=256, hidden_size3=128) -> None:
        """
        Args:
            obs_dim (int): Dimension of the observation space
            action_dim (int): Dimension of the action space
            hidden_size1 (int): Size of the first hidden layer
            hidden_size2 (int): Size of the second hidden layer
        """
        super().__init__()

        self._final_hidden = hidden_size3

        # Shared layers with increased complexity based on research
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, self._final_hidden),
            nn.ReLU()
        )

        # Mean output layer
        self.mean_net = nn.Sequential(
            nn.Linear(self._final_hidden, action_dim)
        )

        # xavier initialisation (weights)
        # this scales the weights so that the variance of the outputs matches the inputs
        # (Glorot & Bengio, 2010).
        nn.init.xavier_uniform_(self.mean_net[-1].weight)

        # Log of standard deviation output layer
        self.log_std_net = nn.Sequential(
            nn.Linear(self._final_hidden, action_dim)
        )

        # xavier log std initialisation
        nn.init.xavier_uniform_(self.log_std_net[-1].weight)

        # log std final layer initial bias to 0
        self.log_std_net[-1].bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor):
        """Given an observation, this function returns the means and standard deviations of
        the normal distributions from which the action components are sampled.

        Args:
            x (torch.Tensor): Observation from the environment
        Returns:
            means: Predicted means of the normal distributions
            stddevs: Predicted standard deviations of the normal distributions
        """
        shared_features = self.shared_net(x)
        means = torch.tanh(self.mean_net(shared_features))

        # if the log_std_net produces very negative values
        # then sigma, exp(log_std) becomes very small.
        # as a result the numerical stability drops and the log probs will lead to
        # unstable gradients
        # automatically clamp exp values prior to performing exp
        # thereby preventing extreme log values before the exp calculation
        stddevs = torch.clamp(
            self.log_std_net(shared_features),
            min=1e-6,  # Don't go negative
            max=10,   # experiment to find a good value. Relates to exploding gradients. Through experimentation you've seen gradients over 5 cause swinging returns
        )
        stddevs = torch.exp(stddevs)

        return means, stddevs
