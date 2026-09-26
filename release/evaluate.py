"""Own only this run's processes; never kill processes occupying another port."""

import argparse
import concurrent.futures
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def runtime_environment(root):
    env = os.environ.copy()
    paths = [
        root, root / "third_party/robocasa", root / "third_party/robosuite",
        root / "third_party/pykdl_utils/src", root / "third_party/hrl_geom/src",
    ]
    env.update({
        "PYTHONPATH": os.pathsep.join(map(str, paths)),
        "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
        "PYTHONFAULTHANDLER": "1",
        "MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl",
        "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
    })
    if env.get("EGL_LIBRARY_PATH"):
        env["LD_LIBRARY_PATH"] = env["EGL_LIBRARY_PATH"] + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def gpu_snapshot():
    fields = "index,uuid,name,memory.total,memory.free,utilization.gpu"
    output = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], text=True,
    )
    return [
        dict(zip(["index", "uuid", "name", "total_mib", "free_mib", "utilization"], map(str.strip, row)))
        for row in csv.reader(output.splitlines())
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical IDs or UUIDs, comma-separated; no automatic reservations.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".runtime/run/checkpoints/model.pt")
    parser.add_argument("--policy-python", default=os.environ.get("POLICY_PYTHON", sys.executable))
    parser.add_argument("--sim-python", default=os.environ.get("SIM_PYTHON", sys.executable))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--task", action="append")
    parser.add_argument("--base-port", type=int, default=23171)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, help="Optional deterministic seed for this evaluation.")
    parser.add_argument("--no-video", action="store_true", help="Skip all video encoding; keep future-image inference unchanged.")
    parser.add_argument("--load-timeout", type=int, default=900)
    parser.add_argument("--task-timeout", type=int, default=21600, help="Per-task wall-clock limit in seconds.")
    args = parser.parse_args()
    from release.verify import verify_sources
    source_verification = verify_sources()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.task_timeout < 1:
        parser.error("--task-timeout must be positive")
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus or len(set(gpus)) != len(gpus):
        parser.error("--gpus must contain distinct GPU identifiers")
    allowed = os.environ.get("CUDA_VISIBLE_DEVICES")
    if allowed and not set(gpus) <= set(allowed.split(",")):
        raise RuntimeError("Requested GPUs are outside the inherited CUDA_VISIBLE_DEVICES allocation.")
    if any(name.startswith(("SLURM_", "PBS_", "LSB_")) for name in os.environ) and not allowed:
        raise RuntimeError("Scheduler detected without a visible GPU allocation; obtain an explicit allocation first.")
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    tasks = args.task or manifest["tasks"]
    if any(task not in manifest["tasks"] for task in tasks) or len(set(tasks)) != len(tasks):
        raise ValueError("Unknown or duplicate evaluation task.")
    checkpoint = args.checkpoint.absolute()
    if not checkpoint.is_file():
        raise FileNotFoundError("Prepare the checkpoint first with python -m release.prepare.")
    for executable in [args.policy_python, args.sim_python]:
        if not Path(executable).is_file():
            raise FileNotFoundError(f"Use an absolute Python interpreter path: {executable}")
    snapshot = gpu_snapshot()
    selected = []
    for gpu in gpus:
        row = next((r for r in snapshot if gpu in (r["index"], r["uuid"])), None)
        if row is None:
            raise ValueError(f"GPU {gpu} is not visible in this allocation.")
        required = manifest["max_policy_vram_gib"] + 8
        if int(row["free_mib"]) < required * 1024:
            raise RuntimeError(f"GPU {gpu} has {row['free_mib']} MiB free; need at least {required} GiB including safety headroom.")
        selected.append(row)
    for offset in range(len(gpus)):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.base_port + offset))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    output = args.output.absolute() if args.output else ROOT / ".runtime/evaluations" / run_id
    output.mkdir(parents=True, exist_ok=False)
    processes = []
    lock = threading.Lock()
    token = uuid.uuid4().hex
    stopping = threading.Event()
    evidence = {
        "run_id": run_id, "host": socket.gethostname(), "selected_gpus": selected,
        "all_gpu_snapshot": snapshot, "checkpoint": str(checkpoint), "tasks": tasks,
        "episodes_per_task": args.episodes, "seed": args.seed,
        "policy_python": args.policy_python, "sim_python": args.sim_python,
        "original_command": sys.argv, "processes": [],
        "source_verification": source_verification,
    }
    evidence["existing_compute_processes"] = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"], text=True,
    )
    (output / "run.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"RUN_OUTPUT={output}", flush=True)

    def start(command, logfile, env):
        stream = open(logfile, "w")
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stream.close()
        with lock:
            processes.append(process)
            evidence["processes"].append({"pid": process.pid, "command": command, "log": str(logfile)})
            (output / "run.json").write_text(json.dumps(evidence, indent=2) + "\n")
        return process

    def stop_owned(process):
        if process.poll() is not None:
            return
        environ = Path(f"/proc/{process.pid}/environ")
        if not environ.exists() or f"PILOT_RELEASE_TASK_ID={token}".encode() not in environ.read_bytes().split(b"\0"):
            raise RuntimeError(f"Cannot verify ownership of PID {process.pid}; refusing to signal it.")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            if process.poll() is None and environ.exists() and f"PILOT_RELEASE_TASK_ID={token}".encode() in environ.read_bytes().split(b"\0"):
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)

    def worker(index, gpu):
        port = args.base_port + index
        env = runtime_environment(ROOT)
        env.update({"CUDA_VISIBLE_DEVICES": gpu, "PILOT_RELEASE_TASK_ID": token, "MUJOCO_EGL_DEVICE_ID": selected[index]["index"]})
        command = [
            args.policy_python, "-m", "release.serve", "--checkpoint", str(checkpoint),
            "--port", str(port), "--evidence", str(output / f"load_gpu{gpu}.json"),
            "--max-vram-gib", str(manifest["max_policy_vram_gib"]),
        ]
        if args.seed is not None:
            command += ["--seed", str(args.seed)]
        server = start(command, output / f"server_gpu{gpu}.log", env)
        try:
            deadline = time.monotonic() + args.load_timeout
            while True:
                if stopping.is_set():
                    return
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited with {server.returncode}; inspect server_gpu{gpu}.log")
                try:
                    from websockets.sync.client import connect
                    with connect(f"ws://127.0.0.1:{port}", open_timeout=1, close_timeout=1, proxy=None) as client:
                        client.recv(timeout=2)
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Policy load timed out on GPU {gpu}")
                    time.sleep(2)
            for task in tasks[index::len(gpus)]:
                if stopping.is_set():
                    return
                task_name = task.replace("/", "_")
                task_output = output / task_name
                command = [
                    args.sim_python, "-m", "release.simulate", "--checkpoint", str(checkpoint),
                    "--task", task, "--port", str(port), "--episodes", str(args.episodes),
                    "--inference-steps", str(manifest["inference_steps"]), "--output", str(task_output),
                ]
                if args.seed is not None:
                    command += ["--seed", str(args.seed)]
                if args.no_video:
                    command += ["--no-video"]
                simulation = start(command, output / f"eval_{task_name}.log", env)
                try:
                    code = simulation.wait(timeout=args.task_timeout)
                except subprocess.TimeoutExpired:
                    stop_owned(simulation)
                    raise TimeoutError(f"Evaluation exceeded {args.task_timeout}s for {task}")
                if code:
                    raise RuntimeError(f"Evaluation failed for {task} with exit code {code}")
                print(f"COMPLETED gpu={gpu} task={task}", flush=True)
        finally:
            stop_owned(server)

    def interrupted(signum, frame):
        stopping.set()
        with lock:
            owned = list(processes)
        for process in reversed(owned):
            stop_owned(process)
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    failure = None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
            futures = [executor.submit(worker, i, gpu) for i, gpu in enumerate(gpus)]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    stopping.set()
                    failure = repr(exc)
                    with lock:
                        owned = list(processes)
                    for process in reversed(owned):
                        stop_owned(process)
                    raise
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        stopping.set()
        for process in reversed(processes):
            stop_owned(process)
        results = []
        for task in tasks:
            file = output / task.replace("/", "_") / "result.json"
            if file.exists():
                results.append(json.loads(file.read_text()))
        with open(output / "summary.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["env_name", "success_rate", "episodes", "successes"])
            writer.writeheader()
            writer.writerows({key: row[key] for key in writer.fieldnames} for row in results)
        summary = {
            "requested_tasks": len(tasks), "completed_tasks": len(results),
            "episodes": sum(r["episodes"] for r in results),
            "mean_success_rate": sum(r["success_rate"] for r in results) / len(results) if results else None,
            "full_evaluation_protocol": (
                len(results) == len(manifest["tasks"])
                and args.episodes == manifest["evaluation_protocol"]["episodes_per_task"]
            ),
            "failure": failure,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
