"""Read evaluation metadata without importing the training/model registry."""

import json
from pathlib import Path


def read_mode_config(pretrained_checkpoint):
    from omegaconf import OmegaConf

    checkpoint = Path(pretrained_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.suffix != ".pt":
        raise ValueError("Expected a .pt checkpoint.")
    run = checkpoint.parents[1]
    config_path = run / "config.full.yaml"
    if not config_path.exists():
        config_path = run / "config.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    with (run / "dataset_statistics.json").open() as stream:
        statistics = json.load(stream)
    return config, statistics
