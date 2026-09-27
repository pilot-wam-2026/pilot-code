"""Reject changed training semantics or dependency versions on full-state resume."""
import importlib.metadata
import json
from pathlib import Path

from omegaconf import OmegaConf

PATH_ONLY_TRAINER_KEYS = {
    "pretrained_checkpoint", "resume_training_state", "is_resume", "reload_modules",
    "resume_data_policy", "save_training_state", "save_interval", "logging_frequency",
}


def plain(config):
    config = getattr(config, "_cfg", config)
    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def semantics(config):
    config = plain(config)
    return {
        "seed": config.get("seed"),
        "framework": config.get("framework"),
        "datasets": config.get("datasets"),
        "trainer": {
            key: value for key, value in config.get("trainer", {}).items()
            if key not in PATH_ONLY_TRAINER_KEYS
        },
    }


def differences(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        keys = sorted(left.keys() | right.keys())
        paths = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(differences(left[key], right[key], path))
        return paths
    return [] if left == right else [prefix]


def validate_resume(config, directory, manifest=None):
    directory = Path(directory)
    manifest = manifest or json.loads((directory / "training_manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("Refusing an incomplete training-state checkpoint")
    saved_path = directory / "config.full.yaml"
    if not saved_path.is_file():
        raise ValueError("Full-state resume requires its saved config.full.yaml")
    saved = plain(OmegaConf.load(saved_path))
    current = plain(config)
    changed = differences(semantics(saved), semantics(current))
    if changed:
        raise ValueError(
            "Full-state resume changed training semantics: " + ", ".join(changed)
            + ". Restore the saved configuration; use explicit weight-only initialization "
            "for a new experiment instead of claiming exact resume."
        )
    packages = manifest.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Full-state resume is missing its dependency-version manifest")
    changed_packages = []
    for name, expected in packages.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        if actual != expected:
            changed_packages.append(f"{name}: saved={expected!r}, current={actual!r}")
    if changed_packages:
        raise ValueError("Full-state resume dependency drift: " + "; ".join(changed_packages))
