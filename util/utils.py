import argparse
from enum import Enum
import json
import os
import random
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn



class EnumWithChoices(Enum):
    @classmethod
    def choices(cls) -> list[str]:
        return [x.value for x in cls]


def set_seeds(seed: int = 42) -> None:
    """
    Fix random seeds for reproducible results
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def is_gpu() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


def write_out_args(args: argparse.Namespace, out_dir: Path) -> None:
    with (out_dir / "args.json").open("+w") as f:
        json.dump(vars(args), f, indent=4)
