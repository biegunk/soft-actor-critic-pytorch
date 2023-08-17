import random
from collections import namedtuple

import torch

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer(object):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: list = []
        self.position = 0

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) -> None:
        """Stores transition in experience buffer"""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, reward, next_state, done)

        # overwrite old experience if capacity is reached
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list:
        """Returns a random sample of experience"""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
