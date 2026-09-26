#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import glob
import os
import re
from statistics import mean

RE_SUCCESS = re.compile(r"Success\s*rate:\s*([0-9]*\.?[0-9]+)")
RE_RESULTS_FOR = re.compile(r"Results\s+for\s+(\S+)")
RE_RUNNING = re.compile(r"Running\s+(\d+)\s+episodes", re.IGNORECASE)

def parse_one_log(path: str):
    env_name = None
    success = None
    episodes = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RE_RESULTS_FOR.search(line)
            if m:
                env_name = m.group(1)

            m = RE_RUNNING.search(line)
            if m:
                episodes = int(m.group(1))

            m = RE_SUCCESS.search(line)
            if m:
                success = float(m.group(1))  # 取最后一次出现

    if env_name is None:
        base = os.path.basename(path)
        env_name = base.replace("eval_env_", "").replace(".log", "")
        env_name = re.sub(r"_gpu\d+$", "", env_name)

    return {
        "env_name": env_name,
        "success_rate": success,
        "episodes": episodes,
        "log_file": os.path.basename(path),
        "path": path,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir", help="Directory containing locally generated evaluation logs")
    ap.add_argument("--pattern", default="eval_env_*.log")
    ap.add_argument("--out_csv", default="summary.csv")
    args = ap.parse_args()

    log_dir = args.log_dir.rstrip("/")
    files = sorted(glob.glob(os.path.join(log_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No log files matched in {log_dir} with pattern {args.pattern}")

    rows = [parse_one_log(p) for p in files]
    ok = [r for r in rows if r["success_rate"] is not None]
    missing = [r for r in rows if r["success_rate"] is None]

    rows_sorted = sorted(
        rows,
        key=lambda r: (float("inf") if r["success_rate"] is None else -r["success_rate"], r["env_name"]),
    )

    print(f"\nLog dir: {log_dir}")
    print(f"Found {len(rows)} logs")
    if ok:
        print(f"Parsed success rate: {len(ok)}/{len(rows)}  avg={mean([r['success_rate'] for r in ok]):.4f}")
    if missing:
        print(f"Missing success rate: {len(missing)} (maybe crashed / incomplete)")

    print("\n{:<4} {:<8} {:<8} {}".format("#", "success", "eps", "env"))
    print("-" * 100)
    for i, r in enumerate(rows_sorted, 1):
        s = "NA" if r["success_rate"] is None else f"{r['success_rate']:.4f}"
        e = "" if r["episodes"] is None else str(r["episodes"])
        print("{:<4} {:<8} {:<8} {}".format(i, s, e, r["env_name"]))

    out_csv_path = os.path.join(log_dir, args.out_csv)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["env_name", "success_rate", "episodes", "log_file", "path"])
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)

    print(f"\nWrote: {out_csv_path}\n")

if __name__ == "__main__":
    main()
