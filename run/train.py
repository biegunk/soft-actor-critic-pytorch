import argparse
import time
from pathlib import Path

import dm_control.suite as suite
from torch.utils.tensorboard.writer import SummaryWriter

from run.runner import plot_test_curve, plot_train_curve, train
from sac.agent import SACAgent
from sac.config import Config, write_out_config
from sac.learner import SACLearner
from sac.replay_buffer import ReplayBuffer
from util.utils import is_gpu, set_seeds, write_out_args
from util.wrappers import DMCWrapper
from dm_control.suite import BENCHMARKING


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train SAC agent on specified environment")
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name of dm-control-suite environment",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task name of dm-control-suite environment",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=500,
        help="Number of train episodes",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=1,
        help="Number of test episodes per evaluation",
    )
    parser.add_argument(
        "--test-every",
        type=int,
        default=10,
        help="Number of train episodes between evaluations",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out",
    )
    parser.add_argument(
        "--device",
        type=str,
        required=False,
        help="Which device to run on, defaults to GPU if available else CPU",
        choices=["cpu", "gpu", "cuda", "mps"],
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir) / args.domain / args.task / f"{time.time()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_out_args(args=args, out_dir=out_dir)

    set_seeds()
    env = DMCWrapper(suite.load(domain_name=args.domain, task_name=args.task))
    config = Config(action_dim=env.action_dim, obs_dim=env.obs_dim, device=is_gpu(args.device))
    write_out_config(config=config, out_dir=out_dir)

    learner = SACLearner(config)
    replay_buffer = ReplayBuffer(config.buffer_size)
    agent = SACAgent(config, learner, replay_buffer)
    tb_writer = SummaryWriter((out_dir / "logs").as_posix())
    ep_rewards, test_ep_rewards = train(
        agent=agent,
        env=env,
        out_dir=out_dir,
        num_train_eps=args.n_train,
        test_every=args.test_every,
        num_test_eps=args.n_test,
        save_policy=True,
        tb_writer=tb_writer,
    )
    plot_train_curve(path=out_dir, train_rewards=ep_rewards)
    plot_test_curve(path=out_dir, test_rewards=test_ep_rewards, test_every=args.test_every)


if __name__ == "__main__":
    args = parse_args()
    run(args)
