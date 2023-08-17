import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

from network import SACActor, SACCritic
from config import Config


class SACLearner:
    """Learner class for the SAC algorithm"""

    def __init__(self, config: Config) -> None:

        self.actor: SACActor = SACActor(
            config, config.state_dim, config.actor_hidden_dim, config.action_dim
        ).to(config.device)
        self.q_net_1 = SACCritic(
            config.state_dim + config.action_dim, config.critic_hidden_dim, 1
        ).to(config.device)
        self.q_net_2 = SACCritic(
            config.state_dim + config.action_dim, config.critic_hidden_dim, 1
        ).to(config.device)

        self.q_net_1_target = SACCritic(
            config.state_dim + config.action_dim, config.critic_hidden_dim, 1
        ).to(config.device)
        self.q_net_2_target = SACCritic(
            config.state_dim + config.action_dim, config.critic_hidden_dim, 1
        ).to(config.device)
        self.hard_update(self.q_net_1_target, self.q_net_1)
        self.hard_update(self.q_net_2_target, self.q_net_2)

        self.actor_optimiser = optim.Adam(params=self.actor.parameters(), lr=config.actor_lr)
        self.q_net_1_optimiser = optim.Adam(params=self.q_net_1.parameters(), lr=config.critic_lr)
        self.q_net_2_optimiser = optim.Adam(params=self.q_net_2.parameters(), lr=config.critic_lr)

        self.log_temp = torch.tensor(config.init_temp).log().to(config.device)
        self.log_temp.requires_grad = True
        self.log_temp_optimiser = optim.Adam([self.log_temp], lr=config.temp_lr)
        self.entropy_target = -config.action_dim  # entropy target is -dim(A)

        self.config = config

    @property
    def temp(self) -> torch.Tensor:
        return self.log_temp.exp()

    def step(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        not_done_mask: torch.Tensor,
    ) -> None:
        """Performs a step of optimisation on the agent
        batches are of dimension (batch_size, variable_dim)"""

        # perform training steps on critic, actor and temperature
        self.train_critic(state_batch, action_batch, reward_batch, next_state_batch, not_done_mask)
        log_prob = self.train_actor(state_batch)
        self.train_temp(log_prob)

        # soft-update target networks
        if not self.t % self.config.update_targets:
            self.soft_update(self.q_net_1_target, self.q_net_1, self.config.tau)
            self.soft_update(self.q_net_2_target, self.q_net_2, self.config.tau)

    def train_critic(
        self,
        state_batch: torch.Tensor,
        action_batch: torch.Tensor,
        reward_batch: torch.Tensor,
        next_state_batch: torch.Tensor,
        not_done_mask: torch.Tensor,
    ) -> None:
        """Performs a step of optimisation on the two Q-networks"""

        # sample next actions
        next_action_batch, next_log_prob = self.actor.evaluate(next_state_batch)

        # calculate target values
        target_inputs = torch.cat((next_state_batch, next_action_batch), dim=-1)
        q1_targets = self.q_net_1_target(target_inputs)
        q2_targets = self.q_net_2_target(target_inputs)
        v_targets = torch.min(q1_targets, q2_targets) - self.temp.detach() * next_log_prob
        q_targets = (reward_batch + (not_done_mask * self.config.gamma * v_targets)).detach()

        # calculate Q-values for current state
        inputs = torch.cat((state_batch, action_batch), dim=-1)
        q1_values = self.q_net_1(inputs)
        q2_values = self.q_net_2(inputs)

        # calculate critic loss
        q1_loss = F.mse_loss(q1_values, q_targets)
        q2_loss = F.mse_loss(q2_values, q_targets)

        # perform optimisation step
        self.q_net_1_optimiser.zero_grad()
        self.q_net_2_optimiser.zero_grad()
        q1_loss.backward()
        q2_loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.q_net_1.parameters(), self.config.grad_clip)
            torch.nn.utils.clip_grad_norm_(self.q_net_2.parameters(), self.config.grad_clip)
        self.q_net_1_optimiser.step()
        self.q_net_2_optimiser.step()

    def train_actor(self, state_batch: torch.Tensor) -> torch.Tensor:
        """Performs a step of optimisation on the actor network"""

        # sample actions and calculate log probability
        actions, log_prob = self.actor.evaluate(state_batch)

        # calculate Q-values
        inputs = torch.cat((state_batch, actions), dim=-1)
        q1_values = self.q_net_1(inputs)
        q2_values = self.q_net_2(inputs)
        q_values = torch.min(q1_values, q2_values)

        # calculate loss
        loss = torch.mean(self.temp.detach() * log_prob - q_values)

        # perform optimisation step
        self.actor.zero_grad()
        loss.backward()
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
        self.actor_optimiser.step()

        return log_prob

    def train_temp(self, log_prob: torch.Tensor) -> None:
        """Performs a step of optimisation on the learnable temperature parameter"""

        # calculate loss
        loss = torch.mean(self.temp * (-log_prob - self.entropy_target).detach())

        # perform optimisation step
        self.log_temp_optimiser.zero_grad()
        loss.backward()
        self.log_temp_optimiser.step()

    @staticmethod
    def hard_update(target: nn.Module, network: nn.Module) -> None:
        """Hard updates the target network"""
        target.load_state_dict(network.state_dict())

    @staticmethod
    def soft_update(target: nn.Module, network: nn.Module, tau: float) -> None:
        """Soft updates the target network"""
        for target_param, param in zip(target.parameters(), network.parameters()):
            target_param.data.copy_(param.data * tau + target_param.data * (1.0 - tau))