import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from sac.config import Config


class SACActor(nn.Module):
    """Diagonal Gaussian actor network for SAC algorithm. Consists of 2 linear layers and two output layers
    returning a mean and log std of the resulting policy distribution."""

    def __init__(
        self,
        config: Config,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        epsilon: float = 1e-6,
    ) -> None:
        super(SACActor, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, output_dim)
        self.log_std_linear = nn.Linear(hidden_dim, output_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.epsilon = epsilon  # regulariser for log
        self.args = config

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def get_action(self, state: torch.Tensor, training: bool) -> torch.Tensor:
        """Returns an action sampled from the policy distribution via the reparametrisation trick if training,
        and returns mean action if not. Resulting actions are scaled by tanh."""
        mean, log_std = self.forward(state)
        if training:
            std = log_std.exp()
            zeta = torch.randn(self.args.action_dim).to(self.args.device)
            action = torch.tanh(mean + std * zeta)
        else:
            action = torch.tanh(mean)
        return action.detach()

    def evaluate(self, state_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a batch of actions sampled from the policy distributions for a given batch of states, as well as
        the log probability of those actions being selected"""
        mean, log_std = self.forward(state_batch)
        std = log_std.exp()

        zeta = torch.randn((self.args.batch_size, self.args.action_dim)).to(self.args.device)
        actions = torch.tanh(mean + std * zeta)
        log_prob = Normal(mean, std).log_prob(mean + std * zeta) - torch.log(
            1 - actions**2 + self.epsilon
        )

        return actions, log_prob.sum(-1, keepdim=True)


class SACCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(SACCritic, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
