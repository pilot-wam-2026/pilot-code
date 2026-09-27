"""Read-only native predicates, executed actions and fresh rollout evidence."""
import hashlib
import inspect
import json
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np


NATIVE_SUCCESS_HASHES = {
    "9c2c8041b9714c4d58833408f0f9855b68c8ae8659e6338a1b3b781221187203",
    "4b9a6fe744822ec7617a1c88461df74745644a185f61a37a4e0940e5a46acf47",
    "a61d24efa80b5bf875b113a009baf4b82aebe63375f1bbf9948885d241d7c603",
    "e5c674fcd69264b4832bd060ada88369de72d7e8a6dbce0dc0c4b8d829b283b6",
}


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def predicates(native):
    from robocasa.utils import object_utils as ou
    result = {}
    for fixture_name in ("drawer", "cabinet", "microwave"):
        fixture = getattr(native, fixture_name, None)
        if fixture is not None:
            result["inside"] = bool(ou.obj_inside_of(
                env=native, obj_name=native.objects["obj"].name,
                fixture_id=fixture, partial_check=True,
            ))
            door = float(fixture.get_door_state(env=native)["door"])
            if fixture_name == "cabinet":
                result["door_correct"] = bool(
                    native.is_door_open() if native.behavior == "open" else native.is_door_closed()
                )
            else:
                result["door_correct"] = door >= 0.5 if native.behavior == "open" else door <= 0.005
            return result, {"fixture": fixture_name, "door": door}
    if getattr(native, "target_container", None):
        result = {
            "gripper_container_far": bool(ou.any_gripper_obj_far(native, obj_name="container")),
            "gripper_obj_far": bool(ou.any_gripper_obj_far(native, obj_name="obj")),
        }
        highest = (
            ou.get_highest_spawn_region(native, native.objects["container"])
            if native.target_container in ("tiered_basket", "tiered_shelf") else None
        )
        result.update({
            "obj_in_container": bool(ou.check_obj_in_receptacle(
                native, "obj", "container", spawn_regions=[highest],
            )),
            "obj_not_on_counter": not bool(ou.check_obj_fixture_contact(native, "obj", native.counter)),
            "container_upright": bool(ou.check_obj_upright(native, "container", threshold=0.8)),
        })
        inside = []
        if "distractor_obj" in native.distractor_config.get("regions", {}):
            for cfg in native.object_cfgs:
                name = cfg["name"]
                if "distractor_obj_distractor_obj" in name:
                    if ou.check_obj_in_receptacle(
                        native, name, "container", spawn_regions=[highest],
                    ):
                        inside.append(name)
        result["distractors_clear"] = not inside
        return result, {"fixture": "container", "distractors_in_container": inside}
    raise ValueError(f"Unsupported native predicate family: {type(native).__name__}")


class EpisodeDiagnostics(gym.Wrapper):
    def __init__(self, env, directory, env_index):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.env_index = int(env_index)
        self.episode = -1
        self.handle = None
        self.capture = os.environ.get("WM4A_CAPTURE_HELDOUT") == "1"
        self.bases = (120, 360)

    def _native(self):
        current = self.env
        while isinstance(current, gym.Wrapper):
            current = current.env
        return current.env

    def _record(self, observation, info, action=None):
        native = self._native()
        components, extra = predicates(native)
        recomputed = all(components.values())
        # No task is scored by this duplicate predicate. It audits the native info.
        if self.steps:
            assert recomputed == bool(info["success"]), (
                type(native).__name__, self.steps, components, info["success"])
        row = {
            "step": self.steps, "success": bool(info.get("success", False)),
            "predicate_success": recomputed, "components": components, **extra,
            "object_position": np.asarray(native.sim.data.body_xpos[native.obj_body_id["obj"]]).tolist(),
            "grasp": {side: bool(native._check_grasp(
                gripper=native.robots[0].gripper[side], object_geoms=native.objects["obj"],
            )) for side in ("left", "right")},
        }
        if "container" in native.obj_body_id:
            row["container_position"] = np.asarray(
                native.sim.data.body_xpos[native.obj_body_id["container"]]).tolist()
        if action is not None:
            row["executed_action"] = {k: np.asarray(v).tolist() for k, v in action.items()}
            assert all(np.isfinite(v).all() for v in action.values())
        self.handle.write(json.dumps(row) + "\n")
        if self.steps % 12 == 0:
            self.handle.flush()
        wanted = set(self.bases) | {step + 16 for step in self.bases}
        if self.capture and self.episode == 0 and self.steps in wanted:
            from PIL import Image
            keys = [k for k in observation if k.startswith("video.") and "ego" in k]
            preferred = "video.ego_view_bg_crop_pad_res256_freq20"
            keys = [preferred] if preferred in keys else ["video.ego_view"] if "video.ego_view" in keys else keys
            assert len(keys) == 1, keys
            pixels = np.asarray(observation[keys[0]])
            while pixels.ndim > 3 and pixels.shape[0] == 1:
                pixels = pixels[0]
            assert pixels.ndim == 3 and pixels.shape[-1] == 3, pixels.shape
            path = self.directory / f"{self.stem}_step{self.steps:04d}.png"
            Image.fromarray(pixels).resize((224, 224)).save(path)
            with (self.directory / f"{self.stem}_frames.jsonl").open("a") as handle:
                handle.write(json.dumps({
                    "step": self.steps, "path": str(path), "raw_pixel_sha256": sha(pixels),
                    "image_key": keys[0], "recorded_unix": time.time(),
                }) + "\n")

    def reset(self, **kwargs):
        if self.handle is not None:
            self.handle.close()
        observation, info = self.env.reset(**kwargs)
        self.episode += 1
        self.steps = 0
        native = self._native()
        self.stem = f"env{self.env_index}_episode{self.episode:04d}"
        self.handle = (self.directory / f"{self.stem}.jsonl").open("x")
        source = inspect.getsource(type(native)._check_success)
        source_hash = hashlib.sha256(source.encode()).hexdigest()
        assert source_hash in NATIVE_SUCCESS_HASHES, "Native success source changed; audit predicates before running"
        metadata = {
            "episode": self.episode, "seed": info["wm4a_episode_seed"],
            "native_type": type(native).__name__, "native_success_source": source,
            "native_success_source_sha256": source_hash, "predicate_schema_version": 2,
            "target_container": getattr(native, "target_container", None),
            "xml_sha256": hashlib.sha256(native.sim.model.get_xml().encode()).hexdigest(),
            "initial_state_sha256": sha(native.sim.get_state().flatten()),
            "created_unix": time.time(), "episode_metadata": native.get_ep_meta(),
            "heldout_bases": list(self.bases) if self.capture and self.episode == 0 else [],
            "observation_annotations": {k: str(v) for k, v in observation.items()
                                        if k.startswith("annotation.")},
        }
        (self.directory / f"{self.stem}.json").write_text(json.dumps(metadata, indent=2, default=str))
        self._record(observation, info)
        return observation, info

    def step(self, action):
        result = self.env.step(action)
        self.steps += 1
        self._record(result[0], result[-1], action)
        return result

    def close(self):
        try:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
        finally:
            self.env.close()
