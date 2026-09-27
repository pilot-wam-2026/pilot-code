import ast
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from release.evaluate import PROTOCOL, allocated_gpus, check_headroom, stop_owned, validate_result
from release.unpack_assets import resource_path, safe_relative, unpack
from release.prepare import verify_cache

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_hub_does_not_ignore_release_payload(self):
        rules = (ROOT / "release/huggingface.gitignore").read_text().splitlines()
        for required in ("checkpoints/", "archives/", "environment/resource_manifest.json"):
            self.assertNotIn(required, rules)
        self.assertIn(".runtime/", rules)
        self.assertIn("*.partial", rules)

    def test_selected_checkpoint(self):
        manifest = json.loads((ROOT / "release_manifest.json").read_text())
        self.assertEqual(manifest["checkpoint_step"], 340000)
        self.assertEqual(manifest["checkpoint_bytes"], 23913608021)
        self.assertEqual(manifest["checkpoint_keys"], 2596)
        self.assertEqual(manifest["latent_normalization"], "legacy")
        self.assertEqual(manifest["future_generation_frames"], 5)
        self.assertEqual(len(manifest["tasks"]), 24)
        self.assertEqual(len(set(manifest["tasks"])), 24)

    def test_published_counts(self):
        result = json.loads((ROOT / "results/robocasa_340000.json").read_text())
        self.assertEqual(sum(x["successes"] for x in result["tasks"]), 717)
        self.assertEqual(sum(x["episodes"] for x in result["tasks"]), 1200)
        self.assertEqual(result["success_rate"], 717 / 1200)
        self.assertFalse(result["protocol"]["early_success_termination"])

    def test_allocation_restrictions(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,3"}, clear=True):
            self.assertEqual(allocated_gpus("2,3", 3), [2, 3])
            with self.assertRaises(ValueError):
                allocated_gpus("0", 3)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "10"}, clear=True):
            with self.assertRaises(ValueError):
                allocated_gpus("0", 0)
        with patch.dict(os.environ, {}, clear=True):
            for value in ("0,0", "0,1,2", "-1"):
                with self.assertRaises(ValueError):
                    allocated_gpus(value, 0)

    def test_headroom(self):
        snapshot = {"cards": {0: {"free_mib": 60 * 1024}, 1: {"free_mib": 70 * 1024}}}
        check_headroom(snapshot, [0, 1], 1, 40)
        snapshot["cards"][1]["free_mib"] = 55 * 1024
        with self.assertRaises(RuntimeError):
            check_headroom(snapshot, [0, 1], 1, 40)
        for cap in (-1, 0):
            with self.assertRaises(ValueError):
                check_headroom(snapshot, [0], 0, cap)

    def test_result_protocol_validation(self):
        result = {
            "protocol": PROTOCOL, "task": "test-task", "episodes": 2,
            "seed": 9000, "mode": "legacy", "checkpoint": "/test/model.pt",
            "inference_calls": 120, "successes": 1, "success_rate": 0.5,
            "metadata": {"latent_normalization": "legacy", "action_steps": 20},
        }
        validate_result(result, "test-task", 2, 9000, Path("/test/model.pt"))
        for key, value in (
            ("task", "wrong-task"), ("seed", 0), ("mode", "canonical"),
            ("episodes", 1), ("inference_calls", 119), ("checkpoint", "/other/model.pt"),
            ("successes", True), ("successes", 3), ("success_rate", 1),
            ("metadata", {"latent_normalization": "legacy", "action_steps": 5}),
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_result(dict(result, **{key: value}), "test-task", 2, 9000, Path("/test/model.pt"))

    def test_owned_cleanup_does_not_signal_completed_child(self):
        from unittest.mock import Mock
        child = Mock()
        child.poll.return_value = 0
        stop_owned(child)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()

    def test_materialized_cache_detects_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = "backbones/model.bin"
            path = root / name
            path.parent.mkdir()
            path.write_bytes(b"abcd")
            manifest = {"components": [{"output": name}], "metadata_labels": [],
                        "checkpoint_bytes": 123, "checkpoint_sha256": "pinned"}
            stamp = {"schema_version": 2, "checkpoint_bytes": 123, "expected_sha256": "pinned",
                     "files": {name: {"bytes": 4, "sha256": hashlib.sha256(b"abcd").hexdigest()}}}
            verify_cache(root, stamp, manifest)
            path.write_bytes(b"abce")
            with self.assertRaises(ValueError):
                verify_cache(root, stamp, manifest)
            path.write_bytes(b"abcd")
            stamp["files"] = {}
            with self.assertRaises(ValueError):
                verify_cache(root, stamp, manifest)

    def test_safe_archive_paths(self):
        for name in ("/etc/passwd", "../escape", "third_party/../../escape", "checkpoints/model.pt"):
            with self.assertRaises(ValueError):
                safe_relative(name)
        self.assertEqual(str(safe_relative("runtime_assets/GR1T2/robot.urdf")),
                         "runtime_assets/GR1T2/robot.urdf")

    def test_resource_parent_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other").mkdir()
            (root / "third_party").symlink_to(root / "other", target_is_directory=True)
            with self.assertRaises(ValueError):
                resource_path(root, "third_party/model.xml")

    def test_unpack_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "archives").mkdir()
            (root / "environment").mkdir()
            archive = root / "archives/simulation_assets.tar.gz"
            data = b"<robot/>"
            name = "runtime_assets/test.urdf"
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                entry.mode = 0o755
                bundle.addfile(entry, io.BytesIO(data))
            inventory = {"files": {name: hashlib.sha256(data).hexdigest()},
                         "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
            (root / "environment/resource_manifest.json").write_text(json.dumps(inventory))
            unpack(root)
            self.assertEqual((root / name).read_bytes(), data)
            self.assertEqual((root / name).stat().st_mode & 0o777, 0o755)
            unpack(root)
            (root / name).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                unpack(root)

    def test_source_syntax(self):
        for relative in ("release", "starVLA", "examples", "deployment"):
            for path in (ROOT / relative).rglob("*.py"):
                ast.parse(path.read_text(), filename=str(path))

    def test_no_machine_binding(self):
        for path in (ROOT / "release").glob("*.py"):
            source = path.read_text()
            self.assertNotIn("nb-5pl2apckdj-0", source)
            self.assertNotIn("/mnt/workspace/", source)


if __name__ == "__main__":
    unittest.main()
