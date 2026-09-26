"""Check a sealed evaluation source snapshot, optionally including large assets."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def verify_sources(root=ROOT, include_assets=False):
    manifests = [root / "provenance/release_sources.sha256.json"]
    if include_assets:
        manifests.append(root / "provenance/simulation_assets.sha256.json")
    count = 0
    for manifest in manifests:
        entries = json.loads(manifest.read_text())
        for relative, expected in entries.items():
            path = root / relative
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError(f"Released file is missing or changed: {relative}")
            count += 1
    return {"verified_files": count, "assets_included": include_assets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_sources(include_assets=args.assets)))


if __name__ == "__main__":
    main()
