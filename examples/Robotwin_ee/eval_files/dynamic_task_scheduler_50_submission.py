# Copyright 2025 starVLA community. All rights reserved.
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by StarVLA community in 2025.

"""
Dynamic task scheduler for parallel evaluation of all 50 robotwin tasks.
Maintains a task queue and assigns tasks to GPUs as they become available.
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# All 50 robotwin tasks from mixtures.py
EVAL_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


@dataclass
class EvalConfig:
    """Configuration for evaluation."""

    ckpt_path: str = "/path/to/policy_run/checkpoints/model.pt"
    server_port: int = 5693
    server_port_map: Optional[Dict[int, int]] = None
    task_config: str = "demo_clean"
    ckpt_setting: str = "nopre_550_step4"
    eval_seed: int = 0
    starvla_path: str = ""
    robotwin_path: str = ""
    python_path: str = "python"
    log_dir: str = ""
    eval_save_root: str = "/path/to/output/robotwin"
    eval_timestamp: str = ""
    test_num: int = 100
    num_traj_workers: int = 1
    slots_per_gpu: int = 1
    gpu_start_id: int = 1
    resource_log_interval_s: int = 150  # Check every 5 seconds for peak detection
    enable_resource_monitor: bool = True


@dataclass
class EvalJob:
    """One sharded evaluation job for a task."""

    task_name: str
    worker_id: int
    num_workers: int
    start_seed: int
    test_num: int

    @property
    def job_name(self) -> str:
        return f"{self.task_name}#w{self.worker_id}"


def strip_ansi_codes(text: str) -> str:
    """Remove ANSI color codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)


def get_default_paths() -> Tuple[str, str]:
    """Infer default paths from this file location and workspace layout."""
    current_file = Path(__file__).resolve()
    # .../JoyRA_v1_paral_pass_bak/examples/Robotwin_ee/eval_files/dynamic_task_scheduler_50.py
    # -> STARVLA root is parents[3]
    starvla_root = current_file.parents[3]
    sibling_robotwin = starvla_root.parent / "RoboTwin"
    if sibling_robotwin.exists():
        return str(starvla_root), str(sibling_robotwin)
    # Fallback to legacy in-tree layout
    return str(starvla_root), str(starvla_root / "RoboTwin")


def split_test_num(total_test_num: int, num_workers: int) -> List[int]:
    """Split test_num into near-equal shards."""
    base = total_test_num // num_workers
    rem = total_test_num % num_workers
    return [base + (1 if i < rem else 0) for i in range(num_workers)]


def format_seconds(seconds: float) -> str:
    """Format seconds to HH:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def render_progress_bar(completed: int, total: int, width: int = 30) -> str:
    """Render ASCII progress bar."""
    if total <= 0:
        total = 1
    ratio = completed / total
    filled = int(width * ratio)
    return f"[{'#' * filled}{'-' * (width - filled)}] {completed}/{total} ({ratio * 100:.1f}%)"


class SystemResourceMonitor:
    """Lightweight system monitor based on /proc (no external dependency)."""

    def __init__(self) -> None:
        self._prev_total: Optional[int] = None
        self._prev_idle: Optional[int] = None

    def _read_cpu_times(self) -> Optional[Tuple[int, int]]:
        try:
            with open("/proc/stat", "r", encoding="utf-8") as f:
                first = f.readline().strip().split()
            if len(first) < 5 or first[0] != "cpu":
                return None
            values = [int(x) for x in first[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
            total = sum(values)
            return total, idle
        except Exception:
            return None

    def _read_memory(self) -> Optional[Tuple[int, int]]:
        mem_total_kb = None
        mem_available_kb = None
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        mem_available_kb = int(line.split()[1])
                    if mem_total_kb is not None and mem_available_kb is not None:
                        break
        except Exception:
            return None

        if mem_total_kb is None or mem_available_kb is None:
            return None
        used_kb = max(0, mem_total_kb - mem_available_kb)
        return used_kb, mem_total_kb

    def sample(self) -> str:
        cpu_usage_str = "N/A"
        cpu_times = self._read_cpu_times()
        if cpu_times is not None:
            total, idle = cpu_times
            if self._prev_total is not None and self._prev_idle is not None:
                delta_total = total - self._prev_total
                delta_idle = idle - self._prev_idle
                if delta_total > 0:
                    cpu_usage = 100.0 * (1.0 - (delta_idle / delta_total))
                    cpu_usage_str = f"{cpu_usage:.1f}%"
            self._prev_total = total
            self._prev_idle = idle

        mem_str = "N/A"
        memory = self._read_memory()
        if memory is not None:
            used_kb, total_kb = memory
            used_gb = used_kb / (1024 * 1024)
            total_gb = total_kb / (1024 * 1024)
            pct = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
            mem_str = f"{used_gb:.1f}/{total_gb:.1f} GiB ({pct:.1f}%)"

        load_str = "N/A"
        runnable_str = "N/A"
        try:
            l1, l5, l15 = os.getloadavg()
            cpu_count = os.cpu_count() or 1
            load_str = f"{l1:.2f}/{l5:.2f}/{l15:.2f} (l1/core={l1 / cpu_count:.2f})"
            with open("/proc/loadavg", "r", encoding="utf-8") as f:
                fields = f.read().strip().split()
            if len(fields) >= 4 and "/" in fields[3]:
                runnable_str = fields[3]
        except Exception:
            pass

        return (
            f"[RESOURCE] CPU: {cpu_usage_str} | Load: {load_str} | "
            f"RAM: {mem_str} | Runnable/Total: {runnable_str}"
        )


def parse_final_result_from_log(log_file: str) -> Optional[Tuple[int, int]]:
    """Parse FINAL_RESULT line from log and return (success, total)."""
    if not os.path.exists(log_file):
        return None

    final_pattern = re.compile(r"FINAL_RESULT .*success=(\d+)\s+total=(\d+)")
    success = None
    total = None

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = final_pattern.search(line)
            if m:
                success = int(m.group(1))
                total = int(m.group(2))

    if success is None or total is None:
        return None
    return success, total


def build_jobs(tasks: List[str], config: EvalConfig) -> List[EvalJob]:
    """
    Create sharded jobs for all tasks with expanded seed ranges.
    
    To handle unstable seeds that get skipped, we allocate 2x seeds per worker.
    Example: 100 trajectories with 4 workers:
        Worker 0: seeds 0-49   (need 25 trajectories)
        Worker 1: seeds 50-99  (need 25 trajectories)
        Worker 2: seeds 100-149 (need 25 trajectories)
        Worker 3: seeds 150-199 (need 25 trajectories)
    
    This ensures no overlap even if up to 50% of seeds are unstable.
    """
    jobs: List[EvalJob] = []
    per_worker_counts = split_test_num(config.test_num, config.num_traj_workers)
    base_seed = 100000 * (1 + config.eval_seed)
    
    # Expanded seed range: allocate 4x seeds per worker to avoid overlap
    seed_range_per_trajectory = 4  # 4x buffer for unstable seeds

    for task_name in tasks:
        seed_offset = 0
        for worker_id, shard_test_num in enumerate(per_worker_counts):
            if shard_test_num <= 0:
                continue
            jobs.append(
                EvalJob(
                    task_name=task_name,
                    worker_id=worker_id,
                    num_workers=config.num_traj_workers,
                    start_seed=base_seed + seed_offset,
                    test_num=shard_test_num,
                )
            )
            # Allocate expanded seed range (2x the actual test_num)
            seed_offset += shard_test_num * seed_range_per_trajectory
    return jobs


def parse_server_ports(server_ports_str: str) -> Dict[int, int]:
    """
    Parse gpu->port mapping string.
    Format example: "0:5693,1:5694"
    """
    mapping: Dict[int, int] = {}
    if not server_ports_str:
        return mapping

    for item in server_ports_str.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(
                f"Invalid server port mapping '{pair}', expected format gpu_id:port"
            )
        gpu_str, port_str = pair.split(":", 1)
        gpu_id = int(gpu_str.strip())
        port = int(port_str.strip())
        mapping[gpu_id] = port
    return mapping


def run_dynamic_scheduler(
    tasks: List[str],
    num_gpus: int,
    config: EvalConfig,
) -> Tuple[Dict[str, Dict[str, int]], float]:
    """
    Run dynamic task scheduler.

    Args:
        tasks: List of task names to evaluate
        num_gpus: Number of GPUs to use (1 to num_gpus)
        config: Evaluation configuration

    Returns:
        Tuple of (task stats dict, total elapsed seconds)
    """
    import queue

    gpu_ids = list(range(config.gpu_start_id, config.gpu_start_id + num_gpus))
    jobs = build_jobs(tasks, config)
    server_port_map = config.server_port_map or {}

    # Create job queue
    task_queue = queue.Queue()
    for job in jobs:
        task_queue.put(job)

    # Track shard and task results
    results: Dict[str, Dict[str, int]] = {
        task: {
            "success": 0,
            "total": 0,
            "done_shards": 0,
            "failed_shards": 0,
        }
        for task in tasks
    }
    
    # Track task-level timing for summary
    task_start_times: Dict[str, float] = {}  # task_name -> first shard start time
    task_completed: Dict[str, bool] = {task: False for task in tasks}  # Track completion
    
    completed = 0
    total = len(jobs)
    start_time = time.time()
    last_progress_print = 0.0
    last_resource_print = 0.0

    # Track running processes per slot (gpu_id, slot_id)
    running_processes: Dict[Tuple[int, int], subprocess.Popen] = {}
    running_jobs: Dict[Tuple[int, int], EvalJob] = {}
    running_logs: Dict[Tuple[int, int], str] = {}
    running_start_ts: Dict[Tuple[int, int], float] = {}

    scheduler_log_file = os.path.join(config.log_dir, "scheduler_progress.log")
    with open(scheduler_log_file, "w", encoding="utf-8") as f:
        f.write("")

    def log_line(message: str = "") -> None:
        print(message)
        with open(scheduler_log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    log_line(f"\n{'='*60}")
    log_line("Dynamic Task Scheduler - 50 Tasks")
    log_line(f"{'='*60}")
    log_line(f"Total tasks: {len(tasks)}")
    log_line(f"Total shard jobs: {total}")
    log_line(f"GPU IDs: {gpu_ids}")
    if server_port_map:
        log_line(f"Server port map: {server_port_map}")
    else:
        log_line(f"Server port: {config.server_port} (shared)")
    log_line(f"Slots per GPU: {config.slots_per_gpu}")
    log_line(f"Trajectory workers per task: {config.num_traj_workers}")
    log_line(f"Each task test_num: {config.test_num}")
    log_line(
        f"Resource monitor: {config.enable_resource_monitor}, "
        f"interval={config.resource_log_interval_s}s"
    )
    log_line(f"Scheduler progress log: {scheduler_log_file}")
    log_line(f"{'='*60}\n")

    resource_monitor = SystemResourceMonitor()

    # Function to stream output and strip ANSI codes
    def stream_to_log(process, log_f):
        """Read process output, strip ANSI codes, and write to log file."""
        for line in process.stdout:
            clean_line = strip_ansi_codes(line)
            log_f.write(clean_line)
            log_f.flush()
        log_f.close()

    # Function to start a job on a specific GPU slot
    def start_job_on_slot(gpu_id: int, slot_id: int, job: EvalJob):
        log_file = os.path.join(
            config.log_dir,
            f"eval_gpu{gpu_id}_s{slot_id}_{job.task_name}_w{job.worker_id}.log",
        )
        eval_files_path = os.path.join(config.starvla_path, "examples/Robotwin_ee/eval_files")
        deploy_policy_path = os.path.join(eval_files_path, "deploy_policy.yml")

        env = os.environ.copy()
        env["PYTHONPATH"] = (
            f"{config.robotwin_path}:{config.starvla_path}:{eval_files_path}"
            f":{env.get('PYTHONPATH', '')}"
        )
        env["PYTHONWARNINGS"] = "ignore::UserWarning"
        # env["LD_LIBRARY_PATH"] = (
        #     f"/tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu:{env.get('LD_LIBRARY_PATH', '')}"
        # )
        env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"
        env["CUROBO_TORCH_COMPILE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env ["PYTHONUNBUFFERED"] = "1"
        
        server_port = server_port_map.get(gpu_id, config.server_port)
        cmd = [
            config.python_path,
            os.path.join(config.robotwin_path, "script/eval_policy.py"),
            "--config",
            deploy_policy_path,
            "--overrides",
            "--port",
            str(server_port),
            "--policy_ckpt_path",
            config.ckpt_path,
            "--task_name",
            job.task_name,
            "--task_config",
            config.task_config,
            "--ckpt_setting",
            config.ckpt_setting,
            "--seed",
            str(config.eval_seed),
            "--test_num",
            str(job.test_num),
            "--start_seed",
            str(job.start_seed),
            "--worker_id",
            str(job.worker_id),
            "--num_workers",
            str(job.num_workers),
            "--policy_name",
            "model2robotwin_interface_submission",
            "--eval_save_root",
            config.eval_save_root,
            "--eval_timestamp",
            config.eval_timestamp,
        ]

        log_line(
            f"[GPU {gpu_id}|SLOT {slot_id}] Starting: {job.job_name} "
            f"(seed={job.start_seed}, test_num={job.test_num}, port={server_port})"
        )

        log_f = open(log_file, "w")
        process = subprocess.Popen(
            cmd,
            env=env,
            cwd=config.robotwin_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # Start thread to stream output with ANSI stripping
        thread = threading.Thread(target=stream_to_log, args=(process, log_f))
        thread.daemon = True
        thread.start()
        return process, thread, log_file

    def pop_next_job() -> Optional[EvalJob]:
        try:
            job = task_queue.get_nowait()
            # Record task start time on first shard
            if job.task_name not in task_start_times:
                task_start_times[job.task_name] = time.time()
            return job
        except queue.Empty:
            return None

    # Initialize slots
    for gpu_id in gpu_ids:
        for slot_id in range(config.slots_per_gpu):
            job = pop_next_job()
            if job is None:
                break
            process, thread, log_file = start_job_on_slot(gpu_id, slot_id, job)
            slot_key = (gpu_id, slot_id)
            running_processes[slot_key] = process
            running_jobs[slot_key] = job
            running_logs[slot_key] = log_file
            running_start_ts[slot_key] = time.time()
            running_processes[slot_key].log_thread = thread

    # Main loop: monitor jobs and refill free slots
    while completed < total:
        for slot_key in list(running_processes.keys()):
            process = running_processes[slot_key]
            ret = process.poll()

            if ret is not None:  # Process finished
                gpu_id, slot_id = slot_key
                job = running_jobs[slot_key]
                task_name = job.task_name
                exit_code = ret

                # Wait for log thread to finish
                if hasattr(process, "log_thread"):
                    process.log_thread.join()

                elapsed_job = time.time() - running_start_ts[slot_key]
                status = "SUCCESS" if exit_code == 0 else f"FAILED (exit code: {exit_code})"
                log_line(
                    f"[GPU {gpu_id}|SLOT {slot_id}] {status}: {job.job_name} "
                    f"(elapsed={format_seconds(elapsed_job)})"
                )

                shard_result = parse_final_result_from_log(running_logs[slot_key])
                if shard_result is not None:
                    shard_success, shard_total = shard_result
                else:
                    shard_success = 0
                    shard_total = job.test_num

                results[task_name]["success"] += shard_success
                results[task_name]["total"] += shard_total
                results[task_name]["done_shards"] += 1
                if exit_code != 0:
                    results[task_name]["failed_shards"] += 1
                completed += 1

                # Check if task is fully completed (all shards done)
                task_total_shards = sum(1 for j in jobs if j.task_name == task_name)
                if results[task_name]["done_shards"] == task_total_shards and not task_completed[task_name]:
                    task_completed[task_name] = True
                    task_elapsed = time.time() - task_start_times[task_name]
                    task_success = results[task_name]["success"]
                    task_total = results[task_name]["total"]
                    task_rate = (task_success / task_total * 100) if task_total > 0 else 0.0
                    
                    log_line("")
                    log_line(f"{'='*60}")
                    log_line(f"✓ TASK COMPLETED: {task_name}")
                    log_line(f"{'='*60}")
                    log_line(f"  Total Time:    {format_seconds(task_elapsed)}")
                    log_line(f"  Success Rate:  {task_success}/{task_total} ({task_rate:.1f}%)")
                    log_line(f"  Shards:        {results[task_name]['done_shards']} completed, "
                             f"{results[task_name]['failed_shards']} failed")
                    log_line(f"{'='*60}")
                    log_line("")

                elapsed = time.time() - start_time
                eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                log_line(
                    f"  Progress: {render_progress_bar(completed, total)} | "
                    f"Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)}"
                )

                # Remove finished slot from tracking
                del running_processes[slot_key]
                del running_jobs[slot_key]
                del running_logs[slot_key]
                del running_start_ts[slot_key]

                # Assign next job to this free slot
                new_job = pop_next_job()
                if new_job is not None:
                    process, thread, log_file = start_job_on_slot(gpu_id, slot_id, new_job)
                    running_processes[slot_key] = process
                    running_jobs[slot_key] = new_job
                    running_logs[slot_key] = log_file
                    running_start_ts[slot_key] = time.time()
                    running_processes[slot_key].log_thread = thread

        now = time.time()
        if now - last_progress_print >= 300.0:
            elapsed = now - start_time
            eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
            log_line(
                f"[HEARTBEAT] {render_progress_bar(completed, total)} | "
                f"Elapsed: {format_seconds(elapsed)} | ETA: {format_seconds(eta)} | "
                f"Running slots: {len(running_processes)}"
            )
            last_progress_print = now

        if config.enable_resource_monitor and now - last_resource_print >= config.resource_log_interval_s:
            log_line(resource_monitor.sample())
            last_resource_print = now

        time.sleep(0.5)  # Poll interval

    return results, (time.time() - start_time)


def main():
    default_starvla_path, default_robotwin_path = get_default_paths()
    parser = argparse.ArgumentParser(
        description="Dynamic task scheduler for 50 robotwin tasks"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "list"],
        help="Mode: run (execute tasks), list (just list tasks)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=7,
        help="Number of GPUs used for evaluation",
    )
    parser.add_argument(
        "--gpu_start_id",
        type=int,
        default=1,
        help="First GPU ID for evaluation (set 0 to include GPU0)",
    )
    parser.add_argument(
        "--slots_per_gpu",
        type=int,
        default=1,
        help="Concurrent eval processes per GPU",
    )
    parser.add_argument(
        "--num_traj_workers",
        type=int,
        default=1,
        help="How many shards to split each task into",
    )
    parser.add_argument(
        "--test_num",
        type=int,
        default=100,
        help="Total trajectories per task",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/path/to/policy_run/checkpoints/model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--server_port",
        type=int,
        default=5693,
        help="Port for policy server",
    )
    parser.add_argument(
        "--server_ports",
        type=str,
        default="",
        help="Optional GPU-specific server ports, e.g. '0:5693,1:5694'",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="demo_clean",
        help="Task configuration",
    )
    parser.add_argument(
        "--ckpt_setting",
        type=str,
        default="nopre_550_step4",
        help="Checkpoint setting",
    )
    parser.add_argument(
        "--eval_seed",
        type=int,
        default=0,
        help="Base evaluation seed",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="",
        help="Directory for log files",
    )
    parser.add_argument(
        "--eval_save_root",
        type=str,
        default="/path/to/output/robotwin",
        help="Root directory for saving eval videos/images",
    )
    parser.add_argument(
        "--eval_timestamp",
        type=str,
        default="",
        help="Unified run timestamp (shared by log/video dirs)",
    )
    parser.add_argument(
        "--starvla_path",
        type=str,
        default=default_starvla_path,
        help="Path to StarVLA project root",
    )
    parser.add_argument(
        "--robotwin_path",
        type=str,
        default=default_robotwin_path,
        help="Path to RoboTwin project root",
    )
    parser.add_argument(
        "--python_path",
        type=str,
        default=sys.executable,
        help="Python executable used to launch eval workers",
    )
    parser.add_argument(
        "--resource_log_interval",
        type=int,
        default=5,
        help="System resource log interval in seconds (default: 5 for peak detection)",
    )
    parser.add_argument(
        "--disable_resource_monitor",
        action="store_true",
        help="Disable CPU/RAM resource logging",
    )

    args = parser.parse_args()

    if args.mode == "list":
        print("Tasks to evaluate (50 total):")
        for i, task in enumerate(EVAL_TASKS, 1):
            print(f"  {i:2d}. {task}")
        return

    # Setup log directory
    if not args.log_dir:
        args.log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "eval_logs_50tasks",
        )
    os.makedirs(args.log_dir, exist_ok=True)

    config = EvalConfig(
        ckpt_path=args.ckpt_path,
        server_port=args.server_port,
        server_port_map=parse_server_ports(args.server_ports),
        task_config=args.task_config,
        ckpt_setting=args.ckpt_setting,
        eval_seed=args.eval_seed,
        starvla_path=os.path.abspath(args.starvla_path),
        robotwin_path=os.path.abspath(args.robotwin_path),
        python_path=args.python_path,
        log_dir=args.log_dir,
        eval_save_root=args.eval_save_root,
        eval_timestamp=args.eval_timestamp,
        test_num=args.test_num,
        num_traj_workers=max(1, args.num_traj_workers),
        slots_per_gpu=max(1, args.slots_per_gpu),
        gpu_start_id=args.gpu_start_id,
        resource_log_interval_s=max(1, args.resource_log_interval),
        enable_resource_monitor=not args.disable_resource_monitor,
    )

    # Run scheduler
    print("######## WM4A-MODE(1/3): StarVLA/examples/Robotwin_ee/eval_files/dynamic_task_scheduler_50_submission.py")
    results, elapsed_total = run_dynamic_scheduler(EVAL_TASKS, args.num_gpus, config)

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")

    failed_tasks = []
    success_trajs = 0
    all_trajs = 0
    for task_name, stat in results.items():
        total = max(1, stat["total"])
        rate = stat["success"] / total * 100
        print(
            f"{task_name:<28} | "
            f"success {stat['success']:>3d}/{stat['total']:<3d} "
            f"({rate:>5.1f}%) | "
            f"shards {stat['done_shards']}/{config.num_traj_workers} | "
            f"failed_shards={stat['failed_shards']}"
        )
        if stat["failed_shards"] > 0:
            failed_tasks.append(task_name)
        success_trajs += stat['success']
        all_trajs += stat['total']

    success_count = len(results) - len(failed_tasks)
    fail_count = len(failed_tasks)
    print(f"\nTask Success: {success_count}/{len(results)}")
    print(f"Success Rate: {success_trajs}/{all_trajs}")
    print(f"Task Failed: {fail_count}/{len(results)}")
    print(f"Total elapsed: {format_seconds(elapsed_total)}")

    if fail_count > 0:
        print("\nFailed tasks:")
        for task in failed_tasks:
            print(f"  - {task}")

    print(f"{'='*60}")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
