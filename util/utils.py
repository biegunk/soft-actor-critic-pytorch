# pyright: reportGeneralTypeIssues=false

import argparse
import json
import os
import random
from enum import Enum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch


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


def is_gpu(device: Optional[str] = None) -> str:
    if torch.cuda.is_available() and device in {"cuda", "gpu", None}:
        print("Running on CUDA")
        return "cuda"
    elif torch.backends.mps.is_available() and device in {"mps", "gpu", None}:
        print("Running on MPS")
        return "mps"
    else:
        print("Running on CPU")
        return "cpu"


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w), "valid") / w


def write_out_args(args: argparse.Namespace, out_dir: Path) -> None:
    with (out_dir / "args.json").open("+w") as f:
        json.dump(vars(args), f, indent=4)


def convert_arrays_to_video(
    frames: list[np.ndarray], out_dir: Path, fps: int = 60, suffix: str = ""
) -> None:
    first_array: np.ndarray = frames[0]
    height, width, _ = first_array.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    video_writer = cv2.VideoWriter(
        (out_dir / f"video{suffix}.mp4").as_posix(), fourcc, fps, (width, height)
    )
    print(out_dir.as_posix())
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()

    print("Video conversion completed.")
