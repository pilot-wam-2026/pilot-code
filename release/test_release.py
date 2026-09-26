"""CPU-only tests for release metadata and relocation helpers."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from release.prepare import select, validate_checkpoint
from release.verify import verify_sources

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_manifest_protocol(self):
        manifest = json.loads((ROOT / "release_manifest.json").read_text())
        self.assertEqual(len(manifest["tasks"]), 24)
        self.assertEqual(len(set(manifest["tasks"])), 24)
        self.assertEqual(manifest["evaluation_protocol"]["episodes_per_task"], 50)
        self.assertTrue(manifest["source_only"])
        self.assertFalse(any(key.startswith(("historical_", "checkpoint_")) for key in manifest))
        self.assertNotIn("source_host", manifest)

    def test_checkpoint_requires_explicit_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint")
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint)

    def test_checkpoint_matches_external_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"test checkpoint")
            expected = hashlib.sha256(b"test checkpoint").hexdigest()
            self.assertEqual(validate_checkpoint(checkpoint, expected.upper()), expected)
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, "0" * 64)
            with self.assertRaises(ValueError):
                validate_checkpoint(checkpoint, "invalid")

    def test_trusted_checkpoint_still_has_cache_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(b"model A")
            first = validate_checkpoint(checkpoint, skip_checksum=True)
            checkpoint.write_bytes(b"model B")
            second = validate_checkpoint(checkpoint, skip_checksum=True)
            self.assertNotEqual(first, second)

    def test_parameter_prefixes(self):
        self.assertEqual(select({"base.a": 1, "base.b": 2, "other": 3}, "base."), {"a": 1, "b": 2})
        with self.assertRaises(ValueError):
            select({"a": 1}, "missing.")

    def test_source_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "provenance").mkdir()
            (root / "model.py").write_text("original")
            expected = {"model.py": hashlib.sha256(b"original").hexdigest()}
            (root / "provenance/release_sources.sha256.json").write_text(json.dumps(expected))
            self.assertEqual(verify_sources(root)["verified_files"], 1)
            (root / "model.py").write_text("edited")
            with self.assertRaises(RuntimeError):
                verify_sources(root)

    def test_no_machine_path_in_retarget(self):
        for name in ["gr1_pos_transform.py", "gr1_pos_transform_new.py"]:
            text = (ROOT / "examples/Robocasa_tabletop/eval_files" / name).read_text()
            self.assertNotIn('Path("/mnt/workspace/', text)


if __name__ == "__main__":
    unittest.main()
