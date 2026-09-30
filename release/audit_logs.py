"""Verify the published log bundle and recompute its recorded success counts."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "wm4a_contract_v2_any_physical_step"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_bundle(path, manifest):
    require(path.stat().st_size == manifest["archive_bytes"], "Archive size mismatch.")
    require(digest(path.read_bytes()) == manifest["archive_sha256"], "Archive SHA-256 mismatch.")
    payload = {}
    total = 0
    # Inspect in memory: no archive member can write a path or follow a symlink.
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = member.name
            parts = PurePosixPath(name)
            require(member.isfile() and not parts.is_absolute()
                    and ".." not in parts.parts and "\\" not in name,
                    "Unsafe archive member.")
            require(name not in payload and name in manifest["files"],
                    "Duplicate or unexpected archive member.")
            total += member.size
            require(0 <= member.size <= 32 * 1024**2 and total <= 64 * 1024**2,
                    "Archive exceeds the evidence size limit.")
            expected = manifest["files"][name]
            require(member.size == expected["bytes"], f"Member size mismatch: {name}")
            data = archive.extractfile(member).read()
            require(digest(data) == expected["sha256"], f"Member SHA-256 mismatch: {name}")
            payload[name] = data
    require(set(payload) == set(manifest["files"]), "Missing archive member.")
    return payload


def audit_records(payload, reference):
    audit = json.loads(payload["benchmark_final_audit.json"])
    tasks = audit["tasks"]
    expected = {task["env"]: task for task in reference["tasks"]}
    require(len(tasks) == len(expected) == 24, "Expected exactly 24 task records.")
    require({task["task"] for task in tasks} == set(expected), "Task set mismatch.")
    require(audit["complete"] is True and audit["pending"] == []
            and audit["protocol"] == PROTOCOL and audit["mode"] == "legacy",
            "Incomplete or incompatible source audit.")
    successes = episodes = requests = endpoint = final = 0
    categories = Counter()
    for task in tasks:
        name = task["task"]
        prefix = "tasks/" + name.replace("/", "_")
        require(task["source"] == prefix + "/result.json",
                f"Source member path mismatch: {name}")
        result = json.loads(payload[prefix + "/result.json"])
        lines = payload[prefix + "/simulation.log"].decode().splitlines()
        records = [json.loads(line.split("EPISODE_RESULT ", 1)[1])
                   for line in lines if line.startswith("EPISODE_RESULT ")]
        diagnostics = task["diagnostics"]
        require(len(records) == len(diagnostics) == 50, f"Missing episode: {name}")
        require([row["episode_id"] for row in records] == list(range(50)),
                f"Repeated or missing log episode IDs: {name}")
        require([row["episode"] for row in diagnostics] == list(range(50)),
                f"Repeated or missing diagnostic episode IDs: {name}")
        count = sum(row["success"] is True for row in records)
        require(count == task["successes"] == result["successes"]
                == expected[name]["successes"], f"Success count mismatch: {name}")
        for key, value in {
            "task": name, "protocol": PROTOCOL, "mode": "legacy", "seed": 9000,
            "episodes": 50, "inference_calls": 3000, "success_rate": count / 50,
        }.items():
            require(result.get(key) == value, f"Result contract mismatch: {name}/{key}")
        require(task["episodes"] == 50 and task["success_rate"] == count / 50,
                f"Audit task count/rate mismatch: {name}")
        metadata = result["metadata"]
        require(metadata["latent_normalization"] == "legacy"
                and metadata["num_frames"] == 5 and metadata["action_steps"] == 20,
                f"Policy metadata mismatch: {name}")
        task_categories = Counter()
        task_endpoint = 0
        for i, (row, diagnostic) in enumerate(zip(records, diagnostics)):
            require(row["scene_seed"] == diagnostic["seed"] == 9000 + i,
                    f"Episode seed mismatch: {name}/{i}")
            require(type(row["success"]) is bool
                    and row["success"] is diagnostic["success"],
                    f"Episode success mismatch: {name}/{i}")
            require(row["ik_cache_cleared"] is True, f"IK reset missing: {name}/{i}")
            require(diagnostic["normalized_model_actions"]["requests"] == 60,
                    f"Action request count mismatch: {name}/{i}")
            require(diagnostic["success"] is (diagnostic["first_success"] is not None),
                    f"First-success mismatch: {name}/{i}")
            task_endpoint += diagnostic["endpoint_sampled_success_same_trajectory"] is True
            final += all(c["final"] for c in diagnostic["components"].values())
            task_categories[diagnostic["diagnostic_category"]] += 1
        require(task_endpoint == task["endpoint_sampled_successes_same_trajectories"],
                f"Block-end score mismatch: {name}")
        require(dict(task_categories) == task["failure_categories"],
                f"Failure-category mismatch: {name}")
        categories.update(task_categories)
        successes += count
        episodes += 50
        requests += result["inference_calls"]
        endpoint += task_endpoint
    require(successes == audit["successes_in_complete_tasks"] == reference["successes"] == 717,
            "Aggregate success mismatch.")
    require(episodes == audit["audited_episodes_in_complete_tasks"]
            == audit["expected_episodes"] == reference["episodes"] == 1200,
            "Aggregate episode mismatch.")
    require(successes / episodes == audit["full_benchmark_success_rate"]
            == reference["success_rate"] == 0.5975, "Aggregate rate mismatch.")
    require(endpoint == audit["endpoint_sampled_successes_same_trajectories"] == 712
            and final == 637, "Same-trajectory rescore mismatch.")
    require(dict(categories) == audit["failure_categories"], "Aggregate category mismatch.")
    provenance = json.loads(payload["source_provenance.json"])
    require(provenance["tasks"] == 24 and provenance["episodes"] == episodes
            and provenance["request_seeds_and_float32_hashes_verified"] == requests == 72000,
            "Source provenance count mismatch.")
    rows = list(csv.DictReader(io.StringIO(payload["training_metrics_340000.csv"].decode())))
    require([int(row["step"]) for row in rows] == list(range(500, 340001, 500)),
            "Training metric steps mismatch.")
    require(all(math.isfinite(float(value)) for row in rows for value in row.values()),
            "Non-finite training metric.")
    return dict(tasks=24, episodes=episodes, successes=successes,
                success_rate=successes / episodes, block_end_successes=endpoint,
                final_step_successes=final, recorded_action_requests=requests,
                training_metric_rows=len(rows),
                scope="Recorded evidence consistency only; no simulation or action regeneration.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "results/evaluation_340000_logs.tar.gz")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/evaluation_340000_manifest.json")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    reference = json.loads((ROOT / "results/robocasa_340000.json").read_text())
    require(manifest["schema_version"] == 1 and manifest["checkpoint_step"] == 340000
            and manifest["checkpoint_sha256"] == reference["checkpoint_sha256"],
            "Evidence belongs to a different release.")
    payload = read_bundle(args.archive, manifest)
    require(payload["training_metrics_340000.csv"]
            == (ROOT / "results/training_metrics_340000.csv").read_bytes(),
            "Standalone training CSV differs from the archived copy.")
    print(json.dumps(audit_records(payload, reference), indent=2))


if __name__ == "__main__":
    main()
