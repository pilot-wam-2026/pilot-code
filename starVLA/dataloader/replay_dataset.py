"""Isolate sample augmentation RNG from model RNG and loader prefetch timing."""
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info


class ReplayDataset(Dataset):
    def __init__(self, dataset, seed=42):
        self.dataset = dataset
        self.seed = int(seed)
        self.epoch = int(getattr(dataset, "epoch", 0))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        payload = f"{self.seed}:{self.epoch}:{int(index)}".encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)
        py, np_state = random.getstate(), np.random.get_state()
        with torch.random.fork_rng(devices=[]):
            try:
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.random.default_generator.manual_seed(seed)
                return self.dataset[index]
            finally:
                random.setstate(py)
                np.random.set_state(np_state)

    def __getattr__(self, name):
        if name == "dataset" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def state_dict(self):
        state = {"epoch": self.epoch, "seed": self.seed}
        if hasattr(self.dataset, "state_dict"):
            state["dataset"] = self.dataset.state_dict()
        if get_worker_info() is not None:
            state["rng"] = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        return state

    def load_state_dict(self, state):
        if state["seed"] != self.seed:
            raise ValueError("Dataset seed changed across resume")
        self.set_epoch(state["epoch"])
        if "dataset" in state:
            self.dataset.load_state_dict(state["dataset"])
        if "rng" in state:
            py, np_state, cpu = state["rng"]
            random.setstate(py)
            np.random.set_state(np_state)
            torch.set_rng_state(cpu)
