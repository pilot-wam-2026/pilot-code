import dataclasses
import json
import logging
import os
from pathlib import Path

import tyro

from examples.Robocasa_tabletop.eval_files.model2robocasa_interface_ee_wm import PolicyWarper
from examples.Robocasa_tabletop.eval_files.simulation_env import (
    Args,
    run_evaluation,
    start_debugpy_once,
)


def eval_gr1_unified(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")
    if os.getenv("DEBUG", False):
        start_debugpy_once()

    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
        shift=args.shift,
        wm_future_image_dir=str(Path(args.video_out_path) / "wm_future_images"),
    )
    try:
        run_evaluation(
            env_name=args.env_name,
            model=model,
            video_dir=args.video_out_path,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            n_action_steps=args.n_action_steps,
            max_episode_steps=args.max_episode_steps,
        )
    finally:
        model.close_wm_videos()


if __name__ == "__main__":
    tyro.cli(eval_gr1_unified)
