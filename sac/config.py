import json
from dataclasses import asdict, dataclass
from pathlib import Path

from util.utils import is_gpu


@dataclass
class Config:
    """Dataclass to store all the hyperparameters shared between parts of the agent"""

    # action dimension
    action_dim: int

    # observation dimension
    obs_dim: int

    # number of hidden nodes in the actor network
    actor_hidden_dim: int = 256

    # number of hidden nodes in the critic network
    critic_hidden_dim: int = 256

    # number of transitions in training batch
    batch_size: int = 256

    # action bounds output by the agent (set to -1., 1. if using normalised actions)
    min_action: float = -1.0
    max_action: float = 1.0

    # size of gradient norm clipping (if 0 then doesn't clip)
    grad_clip: float = 0

    # critic network learning rate
    critic_lr: float = 3e-4

    # actor network learning rate
    actor_lr: float = 3e-4

    # discount factor for RL learner
    gamma: float = 0.99

    # soft update parameter
    tau: float = 0.005

    # number of steps between target network updates
    update_targets: int = 1

    # minimum number of steps on environment before training
    burn_in: int = 1024

    # initial temperature parameter
    init_temp: float = 1.0

    # temperature learning rate
    temp_lr: float = 3e-4

    # size of replay buffer
    buffer_size: int = int(1e6)

    # device to run model on: CUDA if available else CPU
    device: str = "cpu"


def write_out_config(config: Config, out_dir: Path) -> None:
    with (out_dir / "config.json").open("+w") as f:
        json.dump(asdict(config), f, indent=4)
