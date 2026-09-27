"""Verify the exact released checkpoint and, optionally, packaged resources."""
import argparse
import hashlib
import json
from pathlib import Path
from release.unpack_assets import resource_path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    checkpoint = args.checkpoint or ROOT / manifest["checkpoint_file"]
    if checkpoint.stat().st_size != manifest["checkpoint_bytes"]:
        raise ValueError("Incomplete or wrong checkpoint: file size mismatch.")
    if digest(checkpoint) != manifest["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 mismatch.")
    print("Original 340000 checkpoint verified.")
    if args.assets:
        inventory = json.loads((ROOT / "environment/resource_manifest.json").read_text())
        for relative, expected in inventory["files"].items():
            path = resource_path(ROOT, relative)
            if path.is_symlink() or not path.is_file() or digest(path) != expected:
                raise ValueError(f"Missing, changed, or external resource: {relative}")
        print(f"Verified {len(inventory['files'])} environment resource files.")


if __name__ == "__main__":
    main()
