import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from sac.agent import SACAgent
from util.wrappers import DMCWrapper
from util.utils import moving_average
from torch.utils.tensorboard.writer import SummaryWriter


def train(
    agent: SACAgent,
    env: DMCWrapper,
    out_dir: Path,
    num_train_eps: int = 200,
    test_every: int = 10,
    num_test_eps: int = 10,
    save_policy: bool = True,
    tb_writer: Optional[SummaryWriter] = None,
) -> tuple[list[float], list[float]]:
    """Performs [num_train_eps] episodes of training of the agent on the environment
    Performs [num_test_eps] test episodes every [test_every] episodes"""

    ep_rewards = []
    test_ep_rewards = []

    # ensure agent is training
    agent.is_training = True

    best_test_reward = -np.inf

    for episode in tqdm(range(1, num_train_eps + 1)):
        # run test episode every self.test_every episodes (including before and after training)
        if not (episode - 1) % test_every:
            test_start_time = time.time()
            test_reward, _ = test(
                agent=agent,
                env=env,
                num_test_eps=num_test_eps,
            )
            test_end_time = time.time()
            test_ep_rewards.append(test_reward)
            if tb_writer:
                tb_writer.add_scalar("reward/test", test_reward, episode)
            tqdm.write(
                f"Ep: {episode - 1} || # Test Eps: {num_test_eps} || Mean Test Reward: {test_reward:.2f} || "
                f"Total Test Time: {test_end_time - test_start_time:.2f}s"
            )

            # save policy if beats best score so far
            if test_reward > best_test_reward and save_policy:
                agent.save_policy(out_dir / "best")
                best_test_reward = test_reward
            agent.save_policy(out_dir / "last")

            # ensure agent is training
            agent.is_training = True

        # run episode and store reward
        ep_reward, _ = single_episode(env=env, agent=agent)
        ep_rewards.append(ep_reward)
        if tb_writer:
            tb_writer.add_scalar("reward/train", ep_reward, episode)

    # run test episode after training
    test_start_time = time.time()
    test_reward, _ = test(
        agent=agent,
        env=env,
        num_test_eps=num_test_eps,
    )
    test_end_time = time.time()
    test_ep_rewards.append(test_reward)
    print(
        f"Ep: {num_train_eps} || # Test Eps: {num_test_eps} || Mean Test Reward: {test_reward:.2f} || "
        f"Total Test Time: {test_end_time - test_start_time:.2f}s"
    )
    return ep_rewards, test_ep_rewards


def test(
    agent: SACAgent,
    env: DMCWrapper,
    num_test_eps: int = 10,
    render: bool = False,
) -> tuple[float, list[np.ndarray]]:
    """Performs [num_test_eps] test episodes of the agent on the environment
    and saves trajectories to disc (optional)"""

    # ensure agent is not training
    agent.is_training = False

    ep_rewards = []
    frames: list[np.ndarray] = []
    for episode in range(1, num_test_eps + 1):
        # run episode and store rewards and trajectories
        ep_reward, frames = single_episode(env=env, agent=agent, render=render)
        ep_rewards.append(ep_reward)

    mean_rewards = np.mean(ep_rewards)
    return float(mean_rewards), frames


def single_episode(
    env: DMCWrapper,
    agent: SACAgent,
    render: bool = False,
) -> tuple[float, list[np.ndarray]]:
    """Runs one episode of the environment"""

    obs, reward, done = env.reset()
    frames: list[np.ndarray] = []
    if render:
        frame = np.hstack([env.render(camera_id=0), env.render(camera_id=1)])
        frames.append(frame)
    ep_reward = 0.0
    done = False

    while not done:
        action = agent.get_action(torch.from_numpy(obs).float()).numpy()
        next_obs, reward, done = env.step(action)

        # ignore done signal if it comes from hitting time limit
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        if env.step_count >= env.step_limit:
            terminal = False
        else:
            terminal = done

        if render:
            frame = np.hstack([env.render(camera_id=0), env.render(camera_id=1)])
            frames.append(frame)

        agent.step(
            torch.from_numpy(obs),
            torch.from_numpy(action),
            reward,
            torch.from_numpy(next_obs),
            terminal,
        )

        obs = next_obs
        ep_reward += reward
    return ep_reward, frames


def plot_train_curve(path: Path, train_rewards: list[float]) -> None:
    """Plots the smoothed episodic returns during training"""
    smoothed_rewards = moving_average(np.array(train_rewards), 20)
    plt.figure()
    plt.plot(range(1, len(smoothed_rewards) + 1), smoothed_rewards)
    plt.xlabel("# Train Eps")
    plt.ylabel("Smoothed Train Return")
    plt.tight_layout()
    plt.savefig((path / "train_curve.png").as_posix())


def plot_test_curve(path: Path, test_rewards: list[float], test_every: int) -> None:
    """Plots the evolution of mean test reward throughout training"""
    plt.figure()
    plt.plot(np.arange(len(test_rewards)) * test_every, test_rewards)
    plt.xlabel("# Train Eps")
    plt.ylabel("Mean Test Return")
    plt.tight_layout()
    plt.savefig((path / "test_curve.png").as_posix())
