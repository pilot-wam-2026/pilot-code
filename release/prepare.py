"""Reconstruct backbone initialization files from the complete policy checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def select(state, prefix):
    subset = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not subset:
        raise ValueError(f"No checkpoint parameters under {prefix!r}")
    return subset


def validate_checkpoint(checkpoint, expected_sha256=None, skip_checksum=False):
    if not expected_sha256 and not skip_checksum:
        raise ValueError("Supply --checkpoint-sha256, or explicitly use --skip-checksum for a trusted file.")
    if expected_sha256 and (
        len(expected_sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in expected_sha256)
    ):
        raise ValueError("Expected checkpoint SHA-256 must contain 64 hexadecimal characters.")
    actual_hash = digest(checkpoint)
    if not skip_checksum and actual_hash != expected_sha256.lower():
        raise ValueError("Checkpoint SHA-256 mismatch.")
    return actual_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="Locally supplied training configuration.")
    parser.add_argument("--dataset-statistics", type=Path, required=True, help="Locally supplied dataset statistics JSON.")
    parser.add_argument("--checkpoint-sha256", help="Expected SHA-256 from a trusted checkpoint source.")
    parser.add_argument("--cache", type=Path, default=ROOT / ".runtime")
    parser.add_argument(
        "--skip-checksum", action="store_true",
        help="Skip comparison to a supplied checksum for trusted files only; still hash the cache input.",
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    statistics_path = args.dataset_statistics.resolve(strict=True)
    cache = args.cache.resolve()
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    expected_hash = args.checkpoint_sha256
    actual_hash = validate_checkpoint(checkpoint, expected_hash, args.skip_checksum)
    cache.mkdir(parents=True, exist_ok=True)
    stamp_path = cache / "materialization.json"
    if stamp_path.exists():
        stamp = json.loads(stamp_path.read_text())
        if stamp["checkpoint_bytes"] != checkpoint.stat().st_size:
            raise ValueError("The runtime cache belongs to another checkpoint.")
        if stamp.get("verified_sha256") != actual_hash:
            raise ValueError("The runtime cache belongs to another checkpoint hash.")
        for relative, size in stamp["files"].items():
            if (cache / relative).stat().st_size != size:
                raise ValueError(f"Incomplete runtime cache: {relative}")
        print(f"Using verified existing cache: {cache}", flush=True)
    else:
        import torch

        torch.set_num_threads(1)
        state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
        staging = Path(tempfile.mkdtemp(prefix="materialize-", dir=cache))
        files = {}
        try:
            for label in manifest["metadata_labels"]:
                shutil.copytree(ROOT / "model_metadata" / label, staging / "backbones" / label)
            for component in manifest["components"]:
                output = staging / component["output"]
                output.parent.mkdir(parents=True, exist_ok=True)
                if "prefix" in component:
                    subset = select(state, component["prefix"])
                else:
                    subset = {name: select(state, prefix) for name, prefix in component["prefixes"].items()}
                print(f"Writing {component['output']}", flush=True)
                torch.save(subset, output)
                files[component["output"]] = output.stat().st_size
                del subset
            if (cache / "backbones").exists():
                raise FileExistsError("Refusing to overwrite an unrecognized backbone cache.")
            (staging / "backbones").rename(cache / "backbones")
            stamp = {
                "expected_sha256": expected_hash,
                "verified_sha256": actual_hash,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "files": files,
                "source_checkpoint": str(checkpoint),
                "parameter_conversion": "none; original keys and tensor dtypes retained",
            }
            stamp_path.write_text(json.dumps(stamp, indent=2) + "\n")
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        del state

    from omegaconf import OmegaConf

    # Model factories select classes using canonical backbone path names.
    for alias, label in manifest.get("backbone_aliases", {}).items():
        target = cache / "backbones" / alias
        if not target.exists():
            target.symlink_to(label, target_is_directory=True)
        if target.resolve() != (cache / "backbones" / label).resolve():
            raise ValueError(f"Unexpected backbone alias: {target}")
    config = OmegaConf.load(config_path)
    for key, relative in manifest["config_paths"].items():
        resolved = ROOT / relative if relative.startswith("starVLA/") else cache / relative
        OmegaConf.update(config, key, str(resolved), merge=False)
    run = cache / "run"
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = run / "checkpoints/model.pt"
    if target.is_symlink():
        if target.resolve() != checkpoint:
            raise FileExistsError(f"Runtime checkpoint link already points elsewhere: {target}")
    elif target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    else:
        target.symlink_to(checkpoint)
    OmegaConf.save(config, run / "config.yaml")
    shutil.copy2(statistics_path, run / "dataset_statistics.json")
    print(f"Prepared checkpoint: {target}", flush=True)
    print("The supplied checkpoint and training config were not modified.", flush=True)


if __name__ == "__main__":
    main()
