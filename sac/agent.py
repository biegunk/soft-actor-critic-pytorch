import json
from pathlib import Path

import numpy as np
import torch
from sac.replay_buffer import ReplayBuffer, Transition

from sac.config import Config
from sac.learner import SACLearner


class SACAgent:
    def __init__(
        self,
        config: Config,
        learner: SACLearner,
        replay_buffer: ReplayBuffer,
    ) -> None:
        self.config = config

        self.learner = learner
        self.replay_buffer = replay_buffer

        self.is_training: bool = True
        self.t: int = 0  # keeps track of number of steps taken

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """Returns action for a given observation."""

        # takes random action before we start learning (only during training episodes)
        if self.is_training and self.t < self.config.burn_in:
            action: torch.Tensor = torch.from_numpy(
                np.random.uniform(
                    -self.config.max_action, self.config.max_action, size=self.config.action_dim
                )
            )

        # otherwise takes action according to agent's policy
        else:
            state = state.to(self.config.device)
            action = self.learner.actor.get_action(state, self.is_training).cpu()

        # clip action to ensure it is within bounds of environment
        action = torch.clip(action, self.config.min_action, self.config.max_action)

        return action.cpu()

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Performs one step of training on the agent
        All batches are of dimension (batch_size, variable_dim)"""

        if not self.is_training:
            return

        self.t += 1

        # push current transition to replay buffer
        self.save_transition(state, action, reward, next_state, done)

        # prevents training until we have taken burn_in steps on the environment or if not enough experience in buffer
        if self.t < self.config.burn_in or len(self.replay_buffer) < self.config.batch_size:
            return

        # train learner on batch
        batch = self.get_batch()
        self.learner.step(*batch)

    def get_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a batch of experience sampled uniformly from replay buffer
        All batches are of dimension (batch_size x variable_dim)"""

        # sample transitions
        transitions = self.replay_buffer.sample(self.config.batch_size)

        # convert transitions to batch
        batch = Transition(*zip(*transitions))

        # get data from batch
        state_batch = torch.stack(batch.state).to(self.config.device)  # batch_size x state_dim
        action_batch = torch.stack(batch.action).to(self.config.device)  # batch_size x 1
        reward_batch = torch.stack(batch.reward).to(self.config.device)  # batch_size x 1
        next_state_batch = torch.stack(batch.next_state).to(
            self.config.device
        )  # batch_size x state_dim
        not_done_mask = 1 - torch.stack(batch.done).to(self.config.device)  # batch_size x 1
        return state_batch, action_batch, reward_batch, next_state_batch, not_done_mask

    def save_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """Pushes transition to replay buffer"""

        self.replay_buffer.push(
            state.float(),
            action.float(),
            torch.tensor([reward]).float(),
            next_state.float(),
            torch.tensor([done]).float(),
        )

    def save_policy(self, path: Path) -> None:
        """Saves weights of actor network"""
        torch.save(self.learner.actor, (path / "actor_weights.pt").as_posix())

    def load_policy(self, path: Path) -> None:
        """Loads weights of actor network"""
        self.learner.actor = torch.load(
            (path / "actor_weights.pt").as_posix(), map_location=self.config.device
        )

    def save_config_dict(self, path: Path) -> None:
        with (path / "config.json").open("w+") as f:
            json.dump(self.config, f)
