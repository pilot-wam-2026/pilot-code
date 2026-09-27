"""Explicit constructor/reset seeds, including vector-environment autoresets."""
import hashlib
import os
import random

import gymnasium as gym
import numpy as np


def seed_scene(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return seed


def require_hash_seed():
    value = os.environ.get("PYTHONHASHSEED")
    if value is None or not value.isdigit() or not 0 <= int(value) < 2**32:
        raise RuntimeError("Launch the simulator with an explicit numeric PYTHONHASHSEED (for example 0)")


def request_seed(base_seed, episode_ids, step_indices):
    payload = f"{int(base_seed)}:{list(episode_ids)}:{list(step_indices)}"
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") % (2**63)


class SeededEpisodes(gym.Wrapper):
    def __init__(self, env, base_seed, stride=1):
        super().__init__(env)
        self.base_seed = int(base_seed)
        self.stride = int(stride)
        self.episode_index = 0

    def reset(self, *, seed=None, options=None):
        if seed is None:
            seed = self.base_seed + self.episode_index * self.stride
        self.episode_index += 1
        seed_scene(seed)
        observation, info = self.env.reset(seed=int(seed), options=options)
        info = dict(info)
        info["wm4a_episode_seed"] = int(seed)
        return observation, info
