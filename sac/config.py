import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from utils.helpers import is_gpu


@dataclass
class Config:
    """Dataclass to store all the hyperparameters shared between parts of the agent"""

    # action dimension
    action_dim: int

    # dimension of observed state
    state_dim: int

    # number of hidden nodes in the actor network
    actor_hidden_dim: int = 256

    # number of hidden nodes in the critic network
    critic_hidden_dim: int = 256

    # number of transitions in training batch
    batch_size: int = 32

    # action bounds output by the agent (set to -1., 1. if using normalised actions)
    min_action: float = -1.0
    max_action: float = 1.0

    # size of gradient norm clipping (if 0 then doesn't clip)
    grad_clip: float = 0

    # critic network learning rate
    critic_lr: float = 0.001

    # actor network learning rate
    actor_lr: float = 0.001

    # discount factor for RL learner
    gamma: float = 0.99

    # soft update parameter
    tau: float = 0.001

    # number of steps between target network updates
    update_targets: int = 1

    # minimum number of steps on environment before training
    burn_in: int = 100

    # initial temperature parameter
    init_temp: float = 1.0

    # temperature learning rate
    temp_lr: float = 0.001

    # device to run model on: CUDA if available else CPU
    device: str = is_gpu()


def write_out_config(config: Config, out_dir: Path) -> None:
    with (out_dir / "config.json").open("+w") as f:
        json.dump(asdict(config), f, indent=4)
