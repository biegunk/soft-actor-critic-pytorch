import argparse
import time
from pathlib import Path

import dm_control.suite as suite
from torch.utils.tensorboard.writer import SummaryWriter

from run.runner import test
from sac.agent import SACAgent
from sac.config import Config, write_out_config
from sac.learner import SACLearner
from sac.replay_buffer import ReplayBuffer
from util.utils import convert_arrays_to_video, is_gpu, set_seeds, write_out_args
from util.wrappers import DMCWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate SAC agent on specified environment")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name of dm-control-suite environment",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Task name of dm-control-suite environment"
    )
    parser.add_argument("--n-test", type=int, default=1, help="Number of test episodes")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out",
        help="Path to directory to save configs and videos",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Which device to run on, defaults to GPU if available else CPU",
        choices=["cpu", "gpu", "cuda", "mps"],
    )
    parser.add_argument(
        "--weight-path", type=str, required=False, help="Path to saved policy weights"
    )
    parser.add_argument(
        "--render", action="store_true", help="Whether to save video of test episodes"
    )
    parser.add_argument("--cam-ids", type=int, help="Camera IDs to render", default=0, nargs="+")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / args.domain / args.task / f"{time.time()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_out_args(args=args, out_dir=out_dir)

    set_seeds(args.seed)
    env = DMCWrapper(suite.load(domain_name=args.domain, task_name=args.task))
    config = Config(action_dim=env.action_dim, obs_dim=env.obs_dim, device=is_gpu(args.device))
    write_out_config(config=config, out_dir=out_dir)

    learner = SACLearner(config)
    replay_buffer = ReplayBuffer(config.buffer_size)
    agent = SACAgent(config, learner, replay_buffer)
    ep_rewards, frames = test(
        agent=agent,
        env=env,
        num_test_eps=args.n_test,
        render=args.render,
        camera_ids=args.cam_ids,
    )
    print(f"Mean Test Return: {ep_rewards}")
    if args.render:
        convert_arrays_to_video(frames, out_dir=out_dir)


if __name__ == "__main__":
    args = parse_args()
    run(args)
