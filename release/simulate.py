"""Seeded single-env evaluation; append results only after completed episodes."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--mode", choices=["legacy"], default="legacy")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--render-gpu", type=int, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--visualize-every", type=int, default=0)
args = parser.parse_args()
assert os.environ["CUDA_VISIBLE_DEVICES"] == str(args.render_gpu)
assert os.environ["MUJOCO_EGL_DEVICE_ID"] == str(args.render_gpu)
args.output.mkdir(exist_ok=False)
os.environ["WM4A_DIAGNOSTICS_DIR"] = str(args.output / "physical_steps")

import numpy as np
from examples.Robocasa_tabletop.eval_files.model2robocasa_interface_ee_wm import PolicyWarper
from examples.Robocasa_tabletop.eval_files.simulation_env import run_evaluation


class RecordedPolicy(PolicyWarper):
    def _update_vla_input(self, vla_input):
        super()._update_vla_input(vla_input)
        enabled = args.visualize_every > 0 and self.wm_env_step_indices[0] % args.visualize_every == 0
        vla_input["decode_future_images"] = enabled
        vla_input["num_inference_steps"] = 20
        vla_input["future_image_generation"].update(
            enabled=enabled, num_inference_steps=20, guidance_scale=7,
        )

    def finish_eval_episode(self, env_idx, episode_id, success):
        super().finish_eval_episode(env_idx, episode_id, success)
        assert not self.body_retargeter._ik_last_solution.get(env_idx)
        row = {"episode_id": int(episode_id), "scene_seed": args.seed + int(episode_id),
               "success": bool(success), "ik_cache_cleared": True, "time": time.time()}
        with (args.output / "episodes.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print("EPISODE_RESULT " + json.dumps(row), flush=True)


model = RecordedPolicy(
    policy_ckpt_path=args.checkpoint, host="127.0.0.1", port=args.port,
    image_size=[224, 224], n_action_steps=12, num_inference_steps=5, shift=5,
    policy_seed=args.seed, wm_future_image_dir=str(args.output / "future_images"),
)
assert model.use_eepose, "Invalid EEPose setup"
metadata = model.client.get_server_metadata()
assert metadata["latent_normalization"] == args.mode
assert metadata["checkpoint"] == args.checkpoint
original_predict = model.client.predict_action
calls = 0


def recorded_request(request):
    global calls
    response = original_predict(request)
    assert response.get("status") == "ok", response
    action = np.ascontiguousarray(response["data"]["normalized_actions"])
    assert np.isfinite(action).all()
    with (args.output / "action_requests.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "call": calls, "episode_ids": list(map(int, model.wm_episode_ids)),
            "chunk_indices": list(map(int, model.wm_env_step_indices)),
            "inference_seed": int(request["inference_seed"]), "sha256": hashlib.sha256(action.tobytes()).hexdigest(),
            "normalized_actions": action.tolist(),
        }) + "\n")
    if calls == 0:
        (args.output / "first_action.json").write_text(json.dumps({
            "sha256": hashlib.sha256(action.tobytes()).hexdigest(),
            "inference_seed": request["inference_seed"], "shape": list(action.shape),
        }, indent=2))
    calls += 1
    return response


model.client.predict_action = recorded_request
try:
    task, successes = run_evaluation(
        env_name=args.task, model=model, video_dir=str(args.output / "videos"),
        n_episodes=args.episodes, n_envs=1, n_action_steps=12, max_episode_steps=720,
        seed=args.seed,
    )
    assert len(successes) == args.episodes
    result = {
        "protocol": "wm4a_contract_v2_any_physical_step", "task": task, "checkpoint": args.checkpoint,
        "mode": args.mode, "seed": args.seed, "episodes": len(successes),
        "successes": sum(map(bool, successes)), "success_rate": float(np.mean(successes)),
        "inference_calls": calls, "metadata": metadata,
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print("TASK_RESULT " + json.dumps(result), flush=True)
finally:
    model.close_wm_videos()
    model.client.close()
