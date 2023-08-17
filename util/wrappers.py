from collections import OrderedDict

import numpy as np
from dm_control.rl.control import Environment


def flatten_obs(obs: OrderedDict[str, np.ndarray]) -> np.ndarray:
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCWrapper:
    """
    Wrapper for the DM control suite environments
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self.min_action = self._env.action_spec().minimum
        self.max_action = self._env.action_spec().maximum
        self.action_dim = self._env.action_spec().shape[0]
        self.obs_dim = sum([v.shape[0] for v in self._env.observation_spec().values()])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        timestep = self._env.step(action)
        return (
            flatten_obs(timestep.observation),
            timestep.reward,
            timestep.last(),
        )

    def reset(self) -> tuple[np.ndarray, float, bool]:
        timestep = self._env.reset()
        return (
            flatten_obs(timestep.observation),
            timestep.reward,
            timestep.last(),
        )
    
    @property
    def step_count(self) -> int:
        return self._env._step_count

    @property
    def step_limit(self) -> int:
        return self._env._step_limit