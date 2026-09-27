"""Safely restore the simulator archive, retaining existing identical files."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def safe_relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe archive path: {name}")
    if path.as_posix() != name:
        raise ValueError(f"Non-canonical archive path: {name}")
    if path.parts[0] not in ("third_party", "runtime_assets"):
        raise ValueError(f"Unexpected archive root: {name}")
    return path


def resource_path(root, name):
    relative = safe_relative(name)
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Symlink in resource path: {name}")
    return current


def unpack(root=ROOT):
    inventory = json.loads((root / "environment/resource_manifest.json").read_text())
    archive = root / "archives/simulation_assets.tar.gz"
    expected = inventory["files"]
    if archive.exists():
        if sha256(archive) != inventory["archive_sha256"]:
            raise ValueError("Simulator archive SHA-256 mismatch.")
        seen = set()
        with tarfile.open(archive, "r:gz") as source:
            for member in source:
                relative = safe_relative(member.name)
                destination = resource_path(root, member.name)
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f"Unexpected or duplicate archive entry: {member.name}")
                seen.add(member.name)
                # Resolve before creating anything; pre-existing symlinks must not redirect writes.
                if root.resolve() not in destination.resolve().parents or destination.is_symlink():
                    raise ValueError(f"Unsafe extraction target: {member.name}")
                if destination.exists():
                    if sha256(destination) != expected[member.name]:
                        raise ValueError(f"Refusing to overwrite modified resource: {member.name}")
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                data = source.extractfile(member)
                temp = destination.with_name(destination.name + ".extracting")
                value = hashlib.sha256()
                created = False
                try:
                    with temp.open("xb") as output:
                        created = True
                        for chunk in iter(lambda: data.read(8 * 1024 * 1024), b""):
                            value.update(chunk)
                            output.write(chunk)
                    if value.hexdigest() != expected[member.name]:
                        raise ValueError(f"Resource hash mismatch: {member.name}")
                    temp.chmod(member.mode & 0o777)
                    # Atomic no-clobber installation if another process created the target.
                    os.link(temp, destination)
                finally:
                    if created and temp.exists():
                        temp.unlink()
        if seen != set(expected):
            raise ValueError("Simulator archive is incomplete.")
    for name, expected_hash in expected.items():
        path = resource_path(root, name)
        if root.resolve() not in path.resolve().parents or path.is_symlink() or sha256(path) != expected_hash:
            raise ValueError(f"Resource verification failed: {name}")
    print(f"Verified {len(expected)} simulator resource files.")


if __name__ == "__main__":
    unpack()
