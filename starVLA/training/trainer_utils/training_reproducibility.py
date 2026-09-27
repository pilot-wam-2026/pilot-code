"""Seed model construction, with an explicit strict GPU determinism option."""
import os

import torch
from accelerate.utils import set_seed


def seed_model_initialization(config, rank=0):
    strict = config.trainer.get("deterministic_algorithms", False)
    if not isinstance(strict, bool):
        raise ValueError("trainer.deterministic_algorithms must be a boolean")
    if strict:
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
            raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before launching strict training")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    set_seed(int(getattr(config, "seed", 3047)) + int(rank), deterministic=strict)
