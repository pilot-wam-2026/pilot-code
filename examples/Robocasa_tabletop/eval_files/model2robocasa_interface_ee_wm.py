from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from examples.Robocasa_tabletop.eval_files.model2robocasa_interface_ee import (
    PolicyWarper as BasePolicyWarper,
)
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
)


class PolicyWarper(BasePolicyWarper):
    """EE policy wrapper with WM future-image visualization.

    This keeps the default VLM-based eval wrapper free of WM-only response
    handling while still allowing Cosmos/Wan policy models to save predicted
    future images during simulation.
    """

    def __init__(
        self,
        *args,
        wm_future_image_dir: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        shift: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wm_future_image_dir = Path(wm_future_image_dir) if wm_future_image_dir else None
        self.wm_episode_ids = [0]
        self.wm_env_step_indices = [0]
        self.wm_video_file_stem = None
        self._wm_video_writers = {}
        self.num_inference_steps = num_inference_steps
        self.shift = shift
        if self.wm_future_image_dir is not None:
            self.wm_future_image_dir.mkdir(parents=True, exist_ok=True)

    def set_eval_context(
        self,
        episode_ids=None,
        env_step_indices=None,
        completed_episodes: int = 0,
        video_file_stem: Optional[str] = None,
    ) -> None:
        if episode_ids is not None:
            self.wm_episode_ids = list(episode_ids)
        if env_step_indices is not None:
            self.wm_env_step_indices = list(env_step_indices)
        if video_file_stem is not None:
            self.wm_video_file_stem = str(video_file_stem)

    def _handle_model_response(self, response: dict, images) -> None:
        self._save_wm_future_images(response.get("data", {}), images)

    def _update_vla_input(self, vla_input: dict) -> None:
        if self.num_inference_steps is not None:
            vla_input["num_inference_steps"] = int(self.num_inference_steps)
        if self.shift is not None:
            vla_input["shift"] = float(self.shift)
        future_cfg = dict(vla_input.get("future_image_generation", {}))
        future_cfg.setdefault("enabled", True)
        future_cfg.setdefault("return_full_video", False)
        future_cfg.setdefault("num_frames", 9)
        future_cfg.setdefault("future_frame_index", -1)
        future_cfg.setdefault("height", 224)
        future_cfg.setdefault("width", "auto")
        future_cfg.setdefault("num_latent_conditional_frames", 1)
        future_cfg.setdefault("num_inference_steps", 1)
        vla_input["future_image_generation"] = future_cfg

    def _save_wm_future_images(self, response_data: dict, obs_images) -> None:
        if self.wm_future_image_dir is None:
            return

        future_images = response_data.get("pred_future_images")
        if future_images is None:
            return

        future_batches = self._as_batch_sequences(future_images, batch_size=len(obs_images))
        for batch_idx, future_sequence in enumerate(future_batches):
            try:
                if not future_sequence:
                    continue
                episode_id = self.wm_episode_ids[batch_idx] if batch_idx < len(self.wm_episode_ids) else batch_idx
                compare_frames = []
                if batch_idx < len(obs_images) and len(obs_images[batch_idx]) > 0:
                    obs_image = self._as_uint8_rgb(obs_images[batch_idx][0])
                else:
                    obs_image = None

                if obs_image is not None and len(future_sequence) > 1:
                    first_future = self._as_uint8_rgb(future_sequence[0])
                    future_sequence[0] = cv.resize(
                        obs_image,
                        (first_future.shape[1], first_future.shape[0]),
                        interpolation=cv.INTER_AREA,
                    )

                for future_image in future_sequence:
                    future_image = self._as_uint8_rgb(future_image)
                    save_image = future_image
                    if obs_image is not None:
                        obs_panel, future_panel = self._fit_panels_to_common_canvas([obs_image, future_image])
                        if future_image.shape[1] > future_image.shape[0]:
                            save_image = np.concatenate([obs_panel, future_panel], axis=0)
                        else:
                            save_image = np.concatenate([obs_panel, future_panel], axis=1)
                    compare_frames.append(self._pad_to_even_size(save_image))

                recorder = self._get_wm_video_recorder(batch_idx, episode_id, compare_frames[0].shape)
                for save_image in compare_frames:
                    recorder.write_frame(save_image)
                latest_path = self.wm_future_image_dir / f"latest_batch_{batch_idx:02d}.png"
                cv.imwrite(str(latest_path), cv.cvtColor(compare_frames[-1], cv.COLOR_RGB2BGR))
            except Exception as exc:
                print(f"Warning: failed to save WM future video frame: {exc}")

    def _get_wm_video_recorder(self, batch_idx: int, episode_id: int, image_shape) -> VideoRecorder:
        height, width = int(image_shape[0]), int(image_shape[1])
        frame_size = (width, height)
        current = self._wm_video_writers.get(batch_idx)
        video_stem = self.wm_video_file_stem or f"episode_{int(episode_id):03d}"
        if current is not None:
            current_episode_id, current_frame_size, current_video_stem, current_recorder = current
            if (
                current_episode_id == int(episode_id)
                and current_frame_size == frame_size
                and current_video_stem == video_stem
            ):
                return current_recorder
            current_recorder.stop()
            stale_path = self.wm_future_image_dir / f"{current_video_stem}.mp4"
            if stale_path.exists():
                stale_path.unlink()

        video_path = self.wm_future_image_dir / f"{video_stem}.mp4"
        recorder = VideoRecorder.create_h264(
            fps=1,
            codec="h264",
            input_pix_fmt="rgb24",
            crf=22,
            thread_type="FRAME",
            thread_count=1,
        )
        recorder.start(str(video_path))
        self._wm_video_writers[batch_idx] = (int(episode_id), frame_size, video_stem, recorder)
        return recorder

    def finish_eval_episode(self, env_idx: int, episode_id: int, success: bool) -> None:
        current = self._wm_video_writers.pop(env_idx, None)
        if current is None or self.wm_future_image_dir is None:
            return

        _, _, video_stem, recorder = current
        recorder.stop()
        video_path = self.wm_future_image_dir / f"{video_stem}.mp4"
        final_path = self.wm_future_image_dir / f"{video_stem}_success{int(success)}.mp4"
        if video_path.exists():
            video_path.rename(final_path)
        if video_path.exists():
            video_path.unlink()

    def close_wm_videos(self) -> None:
        writers = getattr(self, '_wm_video_writers', None)
        if not writers:
            return
        for _, _, _, recorder in writers.values():
            recorder.stop()
        writers.clear()

    def __del__(self):
        self.close_wm_videos()

    @staticmethod
    def _as_uint8_rgb(image) -> np.ndarray:
        image = np.asarray(image)
        while image.ndim > 3 and image.shape[0] == 1:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = np.moveaxis(image, 0, -1)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 2 or 3 dims, got shape {image.shape}")
        if image.shape[-1] == 4:
            image = image[..., :3]
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.dtype != np.uint8:
            if image.size and np.nanmax(image) <= 1.0 and np.nanmin(image) >= 0.0:
                image = image * 255.0
            elif image.size and np.nanmin(image) < 0.0:
                image = (image + 1.0) * 127.5
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @classmethod
    def _as_frame_sequence(cls, images) -> list[np.ndarray]:
        images = np.asarray(images)
        if images.ndim == 3:
            return [cls._as_uint8_rgb(images)]
        if images.ndim != 4:
            raise ValueError(f"Expected image/video with 3 or 4 dims, got shape {images.shape}")
        return [cls._as_uint8_rgb(frame) for frame in images]

    @classmethod
    def _as_batch_sequences(cls, future_images, batch_size: int) -> list[list[np.ndarray]]:
        future_images = np.asarray(future_images)
        if future_images.ndim == 5:
            return [cls._as_frame_sequence(future_images[batch_idx]) for batch_idx in range(future_images.shape[0])]
        if future_images.ndim == 4 and future_images.shape[0] == batch_size and (
            future_images.shape[-1] in (1, 3, 4) or future_images.shape[1] in (1, 3, 4)
        ):
            return [[cls._as_uint8_rgb(future_images[batch_idx])] for batch_idx in range(future_images.shape[0])]
        if future_images.ndim in (3, 4):
            return [cls._as_frame_sequence(future_images)]
        raise ValueError(f"Unsupported pred_future_images shape {future_images.shape}")

    @staticmethod
    def _fit_panels_to_common_canvas(images: list[np.ndarray]) -> list[np.ndarray]:
        panel_height = max(image.shape[0] for image in images)
        panel_width = max(image.shape[1] for image in images)
        panels = []
        for image in images:
            height, width = image.shape[:2]
            scale = min(panel_width / width, panel_height / height)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            resized = cv.resize(image, (resized_width, resized_height), interpolation=cv.INTER_AREA)
            canvas = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
            y0 = (panel_height - resized_height) // 2
            x0 = (panel_width - resized_width) // 2
            canvas[y0:y0 + resized_height, x0:x0 + resized_width] = resized
            panels.append(canvas)
        return panels

    @staticmethod
    def _pad_to_even_size(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        pad_bottom = height % 2
        pad_right = width % 2
        if pad_bottom == 0 and pad_right == 0:
            return image
        return np.pad(
            image,
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode="edge",
        )
