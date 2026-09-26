"""Run the original simulation loop with isolated imports and exact result records."""

import argparse
import json
import os
from pathlib import Path
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    import robocasa
    import robosuite
    from examples.Robocasa_tabletop.eval_files.model2robocasa_interface_ee_wm import PolicyWarper
    from examples.Robocasa_tabletop.eval_files.simulation_env import run_evaluation

    torch.set_num_threads(1)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    class RecordedPolicy(PolicyWarper):
        def step(self, observations):
            started = time.monotonic()
            action = super().step(observations)
            values = action["actions"] if isinstance(action, dict) else action
            if isinstance(values, dict):
                values = np.concatenate([np.asarray(value) for value in values.values()], axis=-1)
            else:
                values = np.asarray(values)
            if not np.isfinite(values).all():
                raise RuntimeError("Policy returned non-finite actions.")
            record = {
                "elapsed_seconds": time.monotonic() - started,
                "shape": list(values.shape), "finite": True,
                "min": float(values.min()), "max": float(values.max()),
            }
            with (args.output / "action_calls.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            return action
    print(json.dumps({
        "robocasa_source": robocasa.__file__, "robosuite_source": robosuite.__file__,
        "task": args.task, "episodes": args.episodes, "n_envs": 1,
        "n_action_steps": 12, "max_episode_steps": 720,
        "num_inference_steps": args.inference_steps, "shift": 5.0, "seed": args.seed,
    }), flush=True)
    model = RecordedPolicy(
        policy_ckpt_path=args.checkpoint, host="127.0.0.1", port=args.port,
        image_size=[224, 224], n_action_steps=12,
        num_inference_steps=args.inference_steps, shift=5.0,
        wm_future_image_dir=None if args.no_video else str(args.output / "wm_future_images"),
    )
    if not model.use_eepose:
        raise RuntimeError("GR1 retarget initialization failed; refusing silent joint-space fallback.")
    started = time.time()
    try:
        _, successes = run_evaluation(
            env_name=args.task, model=model, video_dir=None if args.no_video else str(args.output / "videos"),
            n_episodes=args.episodes, n_envs=1, n_action_steps=12, max_episode_steps=720,
        )
    finally:
        model.close_wm_videos()
    if len(successes) != args.episodes:
        raise RuntimeError("Simulation returned a different number of episodes than requested.")
    result = {
        "env_name": args.task, "episodes": len(successes),
        "successes": sum(bool(x) for x in successes),
        "success_rate": float(np.mean(successes)),
        "episode_successes": [bool(x) for x in successes],
        "elapsed_seconds": time.time() - started, "seed": args.seed,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
