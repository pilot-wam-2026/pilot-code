"""Keep visualization and explicitly seeded requests out of the policy RNG stream."""
from functools import wraps
import random
from contextlib import contextmanager

import numpy as np
import torch


def module_cuda_devices(module):
    return sorted({p.device.index for p in module.parameters() if p.is_cuda})


def isolated_visualization(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with torch.random.fork_rng(devices=module_cuda_devices(self)):
            try:
                return function(self, *args, **kwargs)
            finally:
                self.backbone._intermediate_features.clear()
    return wrapped


def seeded_inference(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        seed = kwargs.pop("inference_seed", None)
        if seed is None:
            return function(self, *args, **kwargs)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError("inference_seed must be an integer in [0, 2**63)")
        devices = module_cuda_devices(self)
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            return function(self, *args, **kwargs)
    return wrapped


@contextmanager
def evaluation_context(model):
    modes = {m: m.training for m in model.modules()}
    py_state, np_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=module_cuda_devices(model)):
        try:
            model.eval()
            with torch.inference_mode():
                yield
        finally:
            for module, training in modes.items():
                module.training = training
            random.setstate(py_state)
            np.random.set_state(np_state)
