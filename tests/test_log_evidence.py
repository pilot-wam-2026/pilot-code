import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from release.audit_logs import audit_records, read_bundle

ROOT = Path(__file__).resolve().parents[1]


class LogEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "results/evaluation_340000_manifest.json").read_text())
        cls.reference = json.loads((ROOT / "results/robocasa_340000.json").read_text())
        cls.path = ROOT / "results/evaluation_340000_logs.tar.gz"
        cls.payload = read_bundle(cls.path, cls.manifest)

    def test_complete_evidence(self):
        result = audit_records(self.payload, self.reference)
        self.assertEqual((result["successes"], result["episodes"]), (717, 1200))
        self.assertEqual(result["training_metric_rows"], 680)

    def test_corrupt_archive(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            read_bundle(self.path, manifest)

    def test_wrong_seed_rejected(self):
        payload = dict(self.payload)
        name = next(key for key in payload if key.endswith("/simulation.log"))
        payload[name] = payload[name].replace(b'"scene_seed": 9000', b'"scene_seed": 9001', 1)
        with self.assertRaisesRegex(ValueError, "seed mismatch"):
            audit_records(payload, self.reference)

    def test_changed_success_or_source_rejected(self):
        reference = copy.deepcopy(self.reference)
        reference["tasks"][0]["successes"] += 1
        with self.assertRaisesRegex(ValueError, "Success count"):
            audit_records(self.payload, reference)
        payload = dict(self.payload)
        audit = json.loads(payload["benchmark_final_audit.json"])
        audit["tasks"][0]["source"] = "missing/result.json"
        payload["benchmark_final_audit.json"] = json.dumps(audit).encode()
        with self.assertRaisesRegex(ValueError, "Source member path"):
            audit_records(payload, self.reference)

    def test_missing_episode_rejected(self):
        payload = dict(self.payload)
        name = next(key for key in payload if key.endswith("/simulation.log"))
        lines = payload[name].decode().splitlines(keepends=True)
        index = next(i for i, line in enumerate(lines) if line.startswith("EPISODE_RESULT "))
        del lines[index]
        payload[name] = "".join(lines).encode()
        with self.assertRaisesRegex(ValueError, "Missing episode"):
            audit_records(payload, self.reference)

    def test_unsafe_member_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                member = tarfile.TarInfo("../outside")
                member.size = 1
                archive.addfile(member, io.BytesIO(b"x"))
            manifest = {"archive_bytes": path.stat().st_size,
                        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "files": {}}
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                read_bundle(path, manifest)


if __name__ == "__main__":
    unittest.main()
