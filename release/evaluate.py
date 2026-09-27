"""Run the fixed PILOT RoboCasa protocol using one or two allocated GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "wm4a_contract_v2_any_physical_step"


def gpu_snapshot():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    cards = {}
    for line in raw.strip().splitlines():
        index, free, total, utilization = map(int, line.split(","))
        cards[index] = dict(free_mib=free, total_mib=total, utilization=utilization)
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
        "--format=csv,noheader",
    ], text=True)
    return {"time": time.time(), "cards": cards, "processes": processes}


def allocated_gpus(text, render):
    result = [int(x) for x in text.split(",")]
    if not 1 <= len(result) <= 2 or len(set(result)) != len(result) or min(result + [render]) < 0:
        raise ValueError("Specify one or two distinct physical GPU IDs.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        allowed = visible.split(",")
        if any(str(x) not in allowed for x in result + [render]):
            raise ValueError("Requested GPU is outside the inherited CUDA_VISIBLE_DEVICES allocation.")
    elif any(os.environ.get(x) for x in ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID")):
        raise ValueError("A scheduler job must expose its allocated CUDA_VISIBLE_DEVICES.")
    return result


def check_headroom(snapshot, gpus, render, cap):
    if not 0 < cap < 1024:
        raise ValueError("The policy memory cap must be positive and expressed in GiB.")
    for gpu in set(gpus + [render]):
        if gpu not in snapshot["cards"]:
            raise ValueError(f"GPU {gpu} does not exist.")
        # This is admission control, not a memory reservation.
        required = (cap * 1024 if gpu in gpus else 0) + (16 * 1024 if gpu == render else 8 * 1024)
        if snapshot["cards"][gpu]["free_mib"] < required:
            raise RuntimeError(f"Insufficient current headroom on GPU {gpu}: need {required} MiB.")


def stop_owned(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def validate_result(result, task, episodes, seed, checkpoint):
    expected = {
        "protocol": PROTOCOL, "task": task, "episodes": episodes,
        "seed": seed, "mode": "legacy", "checkpoint": str(checkpoint),
        "inference_calls": episodes * 60,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(f"Incomplete or incompatible task result: {key}")
    successes = result.get("successes")
    if type(successes) is not int or not 0 <= successes <= episodes:
        raise ValueError("Invalid success count.")
    if result.get("success_rate") != successes / episodes:
        raise ValueError("Success count and rate disagree.")
    metadata = result.get("metadata", {})
    if metadata.get("latent_normalization") != "legacy" or metadata.get("action_steps") != 20:
        raise ValueError("Result was not produced by the selected policy protocol.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Physical policy GPU IDs, e.g. 0,1.")
    parser.add_argument("--render-gpu", type=int, required=True, help="Physical EGL device index.")
    parser.add_argument("--policy-python", required=True)
    parser.add_argument("--sim-python", required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / ".runtime/run/checkpoints/model.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--workers-per-policy", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--port", type=int, default=19400)
    parser.add_argument("--task", action="append", help="Omit to run all 24 tasks.")
    parser.add_argument("--visualize-every", type=int, default=0)
    parser.add_argument("--max-vram-gib", type=int, default=40)
    args = parser.parse_args()
    if args.episodes < 1 or args.seed < 0:
        parser.error("Episodes must be positive and seed must be nonnegative.")
    gpus = allocated_gpus(args.gpus, args.render_gpu)
    if not 1024 <= args.port <= 65536 - len(gpus):
        parser.error("Choose an unprivileged port range entirely below 65536.")
    if args.max_vram_gib <= 0 or args.visualize_every < 0:
        parser.error("Memory cap must be positive; visualization interval must be nonnegative.")
    # Keep the checkpoint symlink location: its parent run holds config and statistics.
    args.checkpoint = Path(os.path.abspath(args.checkpoint.expanduser()))
    args.output = args.output.expanduser().resolve()
    for name in ("policy_python", "sim_python"):
        interpreter = Path(getattr(args, name)).expanduser()
        if not interpreter.is_absolute() or not interpreter.is_file():
            parser.error(f"--{name.replace('_', '-')} must name an absolute interpreter path.")
        setattr(args, name, str(interpreter))
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    tasks = args.task or manifest["tasks"]
    if len(set(tasks)) != len(tasks) or any(t not in manifest["tasks"] for t in tasks):
        parser.error("Unknown or repeated task.")
    if not args.checkpoint.is_file():
        parser.error("Run python -m release.prepare first.")
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = gpu_snapshot()
    (args.output / "gpu_admission.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    check_headroom(snapshot, gpus, args.render_gpu, args.max_vram_gib)
    for i in range(len(gpus)):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port + i))
    env = os.environ.copy()
    env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", PYTHONPATH=str(ROOT))
    children, mutex, cancelled = [], threading.Lock(), threading.Event()
    servers, handles, results = [], [], []

    def start(command, environment, logfile):
        with mutex:
            if cancelled.is_set():
                raise RuntimeError("Evaluation cancelled.")
            handle = logfile.open("x")
            handles.append(handle)
            child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                     stdout=handle, stderr=subprocess.STDOUT)
            children.append(child)
        return child

    def simulate(index, task):
        if cancelled.is_set():
            raise RuntimeError("Evaluation cancelled.")
        directory = args.output / task.replace("/", "_")
        directory.mkdir()
        simulation = env.copy()
        simulation.update(CUDA_VISIBLE_DEVICES=str(args.render_gpu),
            MUJOCO_EGL_DEVICE_ID=str(args.render_gpu), MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl", __GLX_VENDOR_LIBRARY_NAME="nvidia",
            OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            PYTHONPATH=os.pathsep.join(str(ROOT / x) for x in (
                "", "third_party/robocasa", "third_party/robosuite",
                "third_party/pykdl_utils/src", "third_party/hrl_geom/src")))
        if os.environ.get("EGL_LIBRARY_PATH"):
            simulation["LD_LIBRARY_PATH"] = os.environ["EGL_LIBRARY_PATH"] + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        child = start([
            args.sim_python, "-u", "-m", "release.simulate", "--checkpoint", str(args.checkpoint),
            "--port", str(args.port + index), "--seed", str(args.seed),
            "--task", task, "--episodes", str(args.episodes),
            "--render-gpu", str(args.render_gpu), "--output", str(directory / "rollouts"),
            "--visualize-every", str(args.visualize_every),
        ], simulation, directory / "simulation.log")
        while child.poll() is None:
            if cancelled.wait(1) or any(server.poll() is not None for server in servers):
                stop_owned(child)
                raise RuntimeError("Policy exited or evaluation was cancelled.")
        if child.returncode != 0:
            raise RuntimeError(f"Simulation failed; see {directory / 'simulation.log'}")
        result = json.loads((directory / "rollouts/result.json").read_text())
        validate_result(result, task, args.episodes, args.seed, args.checkpoint)
        print(f"{task}: {result['successes']}/{result['episodes']}", flush=True)
        return result

    def interrupt(signum, frame):
        cancelled.set()
        raise KeyboardInterrupt

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, interrupt)
    run = {"protocol": PROTOCOL, "tasks": tasks, "gpus": gpus, "render_gpu": args.render_gpu,
           "episodes_per_task": args.episodes, "seed": args.seed, "complete": False,
           "checkpoint": str(args.checkpoint), "host": socket.gethostname(),
           "supervisor_pid": os.getpid(), "started_at": time.time()}
    try:
        for index, gpu in enumerate(gpus):
            policy_env = dict(env, CUDA_VISIBLE_DEVICES=str(gpu))
            server = start([
                args.policy_python, "-u", "-m", "release.serve",
                "--checkpoint", str(args.checkpoint), "--port", str(args.port + index),
                "--max-vram-gib", str(args.max_vram_gib),
                "--output", str(args.output / f"load_gpu{gpu}.json"),
            ], policy_env, args.output / f"server_gpu{gpu}.log")
            servers.append(server)
            deadline = time.monotonic() + 900
            while True:
                log = (args.output / f"server_gpu{gpu}.log").read_text(errors="replace")
                if "server listening on" in log:
                    break
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"Policy startup failed on GPU {gpu}.")
                time.sleep(1)
        # Separate pools preserve the fixed round-robin policy/GPU task assignment.
        pools = [ThreadPoolExecutor(max_workers=args.workers_per_policy) for _ in gpus]
        try:
            futures = [pools[i % len(gpus)].submit(simulate, i % len(gpus), task)
                       for i, task in enumerate(tasks)]
            for future in as_completed(futures):
                results.append(future.result())
        finally:
            cancelled.set()
            for pool in pools:
                pool.shutdown(wait=True, cancel_futures=True)
        run.update(complete=True, successes=sum(r["successes"] for r in results),
                   episodes=sum(r["episodes"] for r in results), task_results=results)
        run["success_rate"] = run["successes"] / run["episodes"]
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            run["error"] = f"{type(error).__name__}: {error}"
        cancelled.set()
        for child in reversed(children):
            stop_owned(child)
        for handle in handles:
            handle.close()
        run["finished_at"] = time.time()
        (args.output / "summary.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
