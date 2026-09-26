import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from decord import VideoReader
import transformers
from omegaconf import OmegaConf
from starVLA.dataloader.qwenvl_llavajson.qwen_data_config import data_list
from starVLA.dataloader.qwenvl_llavajson.rope2d import get_rope_index_25, get_rope_index_2

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>\n"
DEFAULT_VIDEO_TOKEN = "<video>\n"

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _extract_input_ids(encoded):
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    elif hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids

    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    elif hasattr(encoded, "tolist") and not isinstance(encoded, list):
        encoded = encoded.tolist()

    if len(encoded) > 0 and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError(f"Expected a single sequence from chat template, but got batch size {len(encoded)}.")
        encoded = encoded[0]

    return list(encoded)


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw: Optional[List] = None,
    visual_type: str = "image",
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        system_ids = _extract_input_ids(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}],
                tokenize=True,
            )
        )
        input_id += system_ids
        target += [IGNORE_INDEX] * len(system_ids)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    if grid_thw is None:
                        raise ValueError(f"Found {visual_tag} in conversation, but no {visual_type} input was provided.")
                    parts = content.split(visual_tag)
                    new_parts = []
                    for i in range(len(parts) - 1):
                        if visual_replicate_index >= len(grid_thw):
                            raise ValueError(
                                f"Found more {visual_tag} placeholders than provided {visual_type} inputs."
                            )
                        new_parts.append(parts[i])
                        replacement = ("<|vision_start|>" + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index] + "<|vision_end|>")
                        new_parts.append(replacement)
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = _extract_input_ids(tokenizer.apply_chat_template(conv, tokenize=True))
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(data_args, "video_max_total_pixels", 1664 * 28 * 28)
        self.video_min_total_pixels = getattr(data_args, "video_min_total_pixels", 256 * 28 * 28)
        self.model_type = data_args.model_type
        if data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        else:
            self.get_rope_index = get_rope_index_2

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if data["data_path"] != "":
                    ann["data_path"] = data["data_path"]
                elif "raw_data" in ann.keys():
                    ann["data_path"] = ann["raw_data"]["data_root"]
            list_data_dict += annotations

        list_data_dict = self.pre_filter_long_case(list_data_dict, max_words=tokenizer.max_len_single_sentence)
        random.shuffle(list_data_dict)     # Randomly shuffle the data for training

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

        rank0_print(f"Total training samples: {len(self.list_data_dict)}")
        rank0_print("Formatting inputs...Skip in lazy mode")

        # self.data_args.image_processor.max_pixels = data_args.max_pixels
        # self.data_args.image_processor.min_pixels = data_args.min_pixels
        # self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        # self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    def pre_filter_long_case(self, list_data_dict, max_words=1024):
        """filter out conversations with total words exceeding max_words"""

        def count_total_words(convs):
            total = 0
            for entry in convs:
                value = entry.get("value", "")
                total += len(value.strip().split())
            return total

        return [item for item in list_data_dict if count_total_words(item.get("conversations", [])) <= max_words]

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        image = Image.open(image_file).convert("RGB")
        # if fix image size?
        if getattr(self.data_args, "fix_image_size", None) is not None:
            image = image.resize(
                self.data_args.fix_image_size,
                resample=Image.BICUBIC,
            )
        visual_processed = processor.preprocess(image, return_tensors="pt")
        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 4)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 8)

        target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.image_processor)
        self.data_args.video_max_frame_pixels = self.video_max_total_pixels
        self.data_args.video_min_frame_pixels = self.video_min_total_pixels
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        processor_valid_kwargs = getattr(processor, "valid_kwargs", None)
        processor_valid_keys = None
        if processor_valid_kwargs is not None:
            if isinstance(processor_valid_kwargs, dict):
                processor_valid_keys = set(processor_valid_kwargs.keys())
            elif hasattr(processor_valid_kwargs, "__annotations__"):
                processor_valid_keys = set(processor_valid_kwargs.__annotations__.keys())
            else:
                try:
                    processor_valid_keys = set(processor_valid_kwargs)
                except TypeError:
                    processor_valid_keys = None

        supports_video_kwarg = processor_valid_keys is None or "video" in processor_valid_keys
        supports_videos_kwarg = processor_valid_keys is None or "videos" in processor_valid_keys

        if supports_videos_kwarg:
            video_processed = processor.preprocess(images=None, videos=video, return_tensors="pt")
            video_tensor = video_processed["pixel_values_videos"]
            grid_thw = video_processed["video_grid_thw"][0]
        elif supports_video_kwarg:
            video_processed = processor.preprocess(images=None, video=video, return_tensors="pt")
            video_tensor = video_processed["pixel_values_videos"]
            grid_thw = video_processed["video_grid_thw"][0]
        else:
            video_processed = processor.preprocess(list(video), return_tensors="pt")
            video_tensor = video_processed["pixel_values"]
            image_grid_thw = video_processed["image_grid_thw"]
            if len(image_grid_thw) == 1:
                grid_thw = image_grid_thw[0]
            else:
                grid_thw = image_grid_thw[0].clone()
                grid_thw[0] = image_grid_thw[:, 0].sum()
        second_per_grid_ts = [self.data_args.image_processor.temporal_patch_size / fps]
        return video_tensor, grid_thw, second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                sample["_debug_dataset_index"] = i
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                sample["_debug_dataset_index"] = next_index
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            sample["_debug_dataset_index"] = i
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"     # FIXME
        video = None
        if "image" in sources[0] and len(sources[0]["image"]):
            image_folder = self.list_data_dict[i]["data_path"]
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, List):
                if len(image_file) > 1:
                    image_file = [os.path.join(image_folder, file) for file in image_file]
                    results = [self.process_image_unified(file) for file in image_file]
                    image, grid_thw = zip(*results)
                else:
                    image_file = image_file[0]
                    image_file = os.path.join(image_folder, image_file)
                    image, grid_thw = self.process_image_unified(image_file)
                    image = [image]
            else:
                image_file = os.path.join(image_folder, image_file)
                image, grid_thw = self.process_image_unified(image_file)
                image = [image]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="image")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                torch.stack(grid_thw, dim=0),     # (1,16,16)
            )
        elif "video" in sources[0] and len(sources[0]["video"]):
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.list_data_dict[i]["data_path"]
            if isinstance(video_file, List):
                if len(video_file) > 1:
                    video_file = [os.path.join(video_folder, file) for file in video_file]
                    results = [self.process_video(file) for file in video_file]
                    video, grid_thw, second_per_grid_ts = zip(*results)
                    second_per_grid_ts = list(itertools.chain.from_iterable(second_per_grid_ts))
                else:
                    video_file = video_file[0]
                    video_file = os.path.join(video_folder, video_file)
                    video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                    video = [video]
            else:
                video_file = os.path.join(video_folder, video_file)
                video, grid_thw, second_per_grid_ts = self.process_video(video_file)
                video = [video]
            grid_thw_merged = copy.deepcopy(grid_thw)
            if not isinstance(grid_thw, Sequence):
                grid_thw_merged = [grid_thw_merged]
                grid_thw = [grid_thw]
            grid_thw_merged = [merged_thw.prod() // self.data_args.image_processor.merge_size**2 for merged_thw in grid_thw_merged]
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged, visual_type="video")
            position_ids, _ = self.get_rope_index(
                self.data_args.image_processor.merge_size,
                data_dict["input_ids"],
                video_grid_thw=torch.stack(grid_thw, dim=0),
                second_per_grid_ts=second_per_grid_ts,
            )
        else:
            grid_thw_merged = None
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess_qwen_2_visual(sources, self.tokenizer, grid_thw=grid_thw_merged)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0],
                labels=data_dict["labels"][0],
                position_ids=position_ids,
            )
        if "image" in self.list_data_dict[i]:
            data_dict["pixel_values"] = image
            data_dict["image_grid_thw"] = grid_thw
        # video exist in the data
        elif "video" in self.list_data_dict[i]:
            data_dict["pixel_values_videos"] = video
            data_dict["video_grid_thw"] = grid_thw

        max_len = self.tokenizer.max_len_single_sentence
        if data_dict["input_ids"].shape[0] > max_len:
            truncated_input_ids = data_dict["input_ids"][:max_len]
            image_tokens_before = int((data_dict["input_ids"] == IMAGE_TOKEN_INDEX).sum().item())
            image_tokens_after = int((truncated_input_ids == IMAGE_TOKEN_INDEX).sum().item())
            video_tokens_before = int((data_dict["input_ids"] == VIDEO_TOKEN_INDEX).sum().item())
            video_tokens_after = int((truncated_input_ids == VIDEO_TOKEN_INDEX).sum().item())

            if image_tokens_before != image_tokens_after or video_tokens_before != video_tokens_after:
                raw_sample = self.list_data_dict[i]
                debug_message = _format_raw_sample_debug(raw_sample)
                raise ValueError(
                    "Sequence truncation would drop visual placeholder tokens: "
                    f"image {image_tokens_before}->{image_tokens_after}, "
                    f"video {video_tokens_before}->{video_tokens_after}, "
                    f"max_len={max_len}.\n"
                    "Raw sample debug:\n"
                    f"{debug_message}"
                )

            data_dict["input_ids"] = data_dict["input_ids"][:max_len]
            data_dict["labels"] = data_dict["labels"][:max_len]
            data_dict["position_ids"] = position_ids[:, :, :max_len]

        return data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
            padding_side=self.tokenizer.padding_side,
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX,
                                                 padding_side=self.tokenizer.padding_side)
        position_ids = pad_and_cat(position_ids)

        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        position_ids = position_ids[..., :self.tokenizer.model_max_length]     # 3,bs,length

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
        images = list(itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance)))
        videos = list(itertools.chain(*(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)))
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance)))
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance)))
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        batch["_debug_dataset_index"] = [instance.get("_debug_dataset_index") for instance in instances]
        return batch


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids"))

        seq_lens = torch.tensor([0] + [len(seq) for seq in input_ids], dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=0)
        labels = torch.cat(labels, dim=0)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids.unsqueeze(0),
            labels=labels.unsqueeze(0),
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance)))
        videos = list(itertools.chain(*(instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance)))
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = list(itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance)))
            grid_thw = torch.stack(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = list(itertools.chain(*(instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance)))
            video_grid_thw = torch.stack(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["_debug_dataset_index"] = [instance.get("_debug_dataset_index") for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # load training dataset
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)

    # load evaluation dataset (if specified eval dataset path)
    eval_dataset = None
    if hasattr(data_args, "eval_dataset") and data_args.eval_dataset:
        eval_data_args = copy.deepcopy(data_args)
        eval_data_args.dataset_use = data_args.eval_dataset
        eval_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=eval_data_args)

    # select appropriate collator based on whether data needs to be flattened
    if data_args.data_flatten:
        data_collator = FlattenedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def make_vlm_dataloader(cfg):
    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm,).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side=
        "left",     # flash Attention version of Qwen2.5_VL. Make sure to  call `tokenizer.padding_side  = 'left'` before tokenizing the input.
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = int(data_args.max_pixels)
    image_processor.min_pixels = int(data_args.min_pixels)
    image_processor.size["longest_edge"] = int(data_args.max_pixels)
    image_processor.size["shortest_edge"] = int(data_args.min_pixels)
    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor     # TODO later remove the logic bound to model
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
        num_workers=4,
    )

    eval_dataloader = None
    eval_dataset = data_module["eval_dataset"]
    if eval_dataset is not None:
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=cfg.datasets.vlm_data.per_device_batch_size,
            collate_fn=data_collator,
            num_workers=4,
        )

    return {
        "train_dataloader": train_dataloader,
        "eval_dataloader": eval_dataloader,
    }


from transformers import AutoTokenizer, AutoProcessor


def _safe_shape(x):
    if x is None:
        return None
    if hasattr(x, "shape"):
        return tuple(x.shape)
    if isinstance(x, (list, tuple)):
        return f"len={len(x)}"
    return type(x).__name__


def _print_sample_debug(dataset, sample_index):
    raw_sample = dataset.list_data_dict[sample_index]
    print(f"=== Raw sample[{sample_index}] ===")
    print(f"keys: {sorted(raw_sample.keys())}")
    if "conversations" in raw_sample:
        print(f"num_conversations: {len(raw_sample['conversations'])}")
        for idx, conv in enumerate(raw_sample["conversations"][:4]):
            role = conv.get("role", conv.get("from"))
            text = conv.get("content", conv.get("value", ""))
            text = text.replace("\n", "\\n")
            print(f"conv[{idx}] role={role} text[:160]={text[:160]}")
    image_files = raw_sample.get("image", raw_sample.get("images"))
    video_files = raw_sample.get("video", raw_sample.get("videos"))
    print(f"image field: {image_files}")
    print(f"video field: {video_files}")
    print(f"data_path: {raw_sample.get('data_path')}")
    print()

    try:
        sample = dataset[sample_index]
    except Exception as e:
        print(f"=== dataset[{sample_index}] failed ===")
        print(repr(e))
        return

    print(f"=== Processed sample[{sample_index}] ===")
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            child_shapes = [_safe_shape(v) for v in value[:4]]
            print(f"{key}: list_len={len(value)} first_items={child_shapes}")
        else:
            print(f"{key}: {value}")

    input_ids = sample["input_ids"]
    print(f"num_image_tokens: {int((input_ids == IMAGE_TOKEN_INDEX).sum().item())}")
    print(f"num_video_tokens: {int((input_ids == VIDEO_TOKEN_INDEX).sum().item())}")
    if "video_grid_thw" in sample:
        print(f"video_grid_thw: {sample['video_grid_thw']}")
    if "image_grid_thw" in sample:
        print(f"image_grid_thw: {sample['image_grid_thw']}")


def _format_raw_sample_debug(raw_sample):
    lines = []
    lines.append(f"keys: {sorted(raw_sample.keys())}")
    if "id" in raw_sample:
        lines.append(f"id: {raw_sample['id']}")
    lines.append(f"data_path: {raw_sample.get('data_path')}")
    lines.append(f"image field: {raw_sample.get('image', raw_sample.get('images'))}")
    lines.append(f"video field: {raw_sample.get('video', raw_sample.get('videos'))}")
    conversations = raw_sample.get("conversations", [])
    lines.append(f"num_conversations: {len(conversations)}")
    for idx, conv in enumerate(conversations[:4]):
        role = conv.get("role", conv.get("from"))
        text = conv.get("content", conv.get("value", ""))
        text = text.replace("\n", "\\n")
        lines.append(f"conv[{idx}] role={role} text[:160]={text[:160]}")
    return "\n".join(lines)


def _print_batch_debug(batch, batch_index):
    print(f"=== Batch[{batch_index}] ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"{key}: {_safe_shape(value)}")
    input_ids = batch["input_ids"]
    print(f"batch_image_tokens: {int((input_ids == IMAGE_TOKEN_INDEX).sum().item())}")
    print(f"batch_video_tokens: {int((input_ids == VIDEO_TOKEN_INDEX).sum().item())}")
    print()


def _print_batch_qa_debug(dataset, batch, batch_index, batch_size):
    sample_indices = batch.get("_debug_dataset_index")
    if sample_indices is None:
        start = batch_index * batch_size
        end = min(start + batch_size, len(dataset.list_data_dict))
        sample_indices = list(range(start, end))
    print(f"=== Batch[{batch_index}] QA ===")
    for sample_index in sample_indices:
        if sample_index is None:
            continue
        raw_sample = dataset.list_data_dict[sample_index]
        sample_id = raw_sample.get("id", "<no-id>")
        print(f"-- sample[{sample_index}] id={sample_id}")
        conversations = raw_sample.get("conversations", [])
        for conv_idx, conv in enumerate(conversations):
            role = conv.get("role", conv.get("from"))
            text = conv.get("content", conv.get("value", ""))
            text = text.replace("\n", "\\n")
            print(f"  [{conv_idx}] {role}: {text[:300]}")
        print()


def _sanitize_filename(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name))


def _wrap_text_for_width(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _save_annotated_debug_image(image, conversations, save_path):
    font = ImageFont.load_default()
    image = image.convert("RGB")
    panel_width = image.width
    padding = 12
    line_gap = 6

    dummy = Image.new("RGB", (panel_width, 10), "white")
    draw = ImageDraw.Draw(dummy)

    wrapped_lines = []
    for conv_idx, conv in enumerate(conversations):
        role = conv.get("role", conv.get("from", "unknown"))
        text = conv.get("content", conv.get("value", ""))
        text = text.replace("\n", " ")
        prefix = f"[{conv_idx}] {role}: "
        body_lines = _wrap_text_for_width(draw, text, font, panel_width - padding * 2)
        if body_lines:
            wrapped_lines.append(prefix + body_lines[0])
            wrapped_lines.extend(body_lines[1:])
        else:
            wrapped_lines.append(prefix)
        wrapped_lines.append("")

    line_height = draw.textbbox((0, 0), "Ag", font=font)[3] + line_gap
    text_height = padding * 2 + max(1, len(wrapped_lines)) * line_height
    canvas = Image.new("RGB", (panel_width, image.height + text_height), "white")
    canvas.paste(image, (0, 0))

    draw = ImageDraw.Draw(canvas)
    y = image.height + padding
    for line in wrapped_lines:
        draw.text((padding, y), line, fill="black", font=font)
        y += line_height

    canvas.save(save_path)


def _save_debug_media(dataset, batch, batch_index, batch_size, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_indices = batch.get("_debug_dataset_index")
    if sample_indices is None:
        start = batch_index * batch_size
        end = min(start + batch_size, len(dataset.list_data_dict))
        sample_indices = list(range(start, end))
    print(f"=== Batch[{batch_index}] media saved to {output_dir} ===")

    for sample_index in sample_indices:
        if sample_index is None:
            continue
        raw_sample = dataset.list_data_dict[sample_index]
        sample_id = _sanitize_filename(raw_sample.get("id", f"sample_{sample_index}"))
        data_path = raw_sample.get("data_path", "")
        conversations = raw_sample.get("conversations", [])

        image_files = raw_sample.get("image", raw_sample.get("images"))
        if image_files is not None:
            if not isinstance(image_files, (list, tuple)):
                image_files = [image_files]
            for media_idx, image_file in enumerate(image_files):
                image_path = os.path.join(data_path, image_file)
                try:
                    image = Image.open(image_path).convert("RGB")
                    annotated_path = os.path.join(output_dir, f"batch{batch_index}_sample{sample_index}_{sample_id}_image{media_idx}_qa.jpg")
                    _save_annotated_debug_image(image, conversations, annotated_path)
                    print(f"saved image+qa: {annotated_path}")
                except Exception as e:
                    print(f"failed to save image {image_path}: {e}")

        video_files = raw_sample.get("video", raw_sample.get("videos"))
        if video_files is not None:
            if not isinstance(video_files, (list, tuple)):
                video_files = [video_files]
            for media_idx, video_file in enumerate(video_files):
                video_path = os.path.join(data_path, video_file)
                try:
                    vr = VideoReader(video_path, num_threads=1)
                    frame = vr[0].asnumpy()
                    image = Image.fromarray(frame).convert("RGB")
                    annotated_path = os.path.join(output_dir, f"batch{batch_index}_sample{sample_index}_{sample_id}_video{media_idx}_frame0_qa.jpg")
                    _save_annotated_debug_image(image, conversations, annotated_path)
                    print(f"saved video preview+qa: {annotated_path}")
                except Exception as e:
                    print(f"failed to save video preview {video_path}: {e}")

if __name__ == "__main__":
    # import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml",
                        type=str,
                        default="./examples/Suqian_agibot/train_files/starvla_cotrain_robocasa_gr1.yaml",
                        help="Path to YAML config")
    parser.add_argument("--sample_index", type=int, default=None, help="Inspect a single dataset sample by index")
    parser.add_argument("--max_batches", type=int, default=100, help="Number of batches to iterate for debugging")
    parser.add_argument("--save_debug_media_dir", type=str, default=None, help="Optional directory to save raw images and video previews for each debug batch")
    args, clipargs = parser.parse_known_args()

    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    data_args = cfg.datasets.vlm_data
    image_processor = AutoProcessor.from_pretrained(cfg.framework.qwenvl.base_vlm,).image_processor

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.framework.qwenvl.base_vlm,
        model_max_length=data_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )

    # avoid processing these in dataset
    image_processor.max_pixels = data_args.max_pixels
    image_processor.min_pixels = data_args.min_pixels
    image_processor.size["longest_edge"] = data_args.max_pixels
    image_processor.size["shortest_edge"] = data_args.min_pixels

    data_args_ns = SimpleNamespace(**OmegaConf.to_container(data_args, resolve=True))
    data_args_ns.image_processor = image_processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args_ns)

    #
    train_dataset = data_module["train_dataset"]
    data_collator = data_module["data_collator"]
    from torch.utils.data import DataLoader

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.datasets.vlm_data.per_device_batch_size,
        collate_fn=data_collator,
    )
    if args.sample_index is not None:
        _print_sample_debug(train_dataset, args.sample_index)
    else:
        batchs = iter(train_dataloader)
        count = 0
        while count < args.max_batches:
            batch_samples = next(batchs)
            _print_batch_debug(batch_samples, count)
            _print_batch_qa_debug(train_dataset, batch_samples, count, cfg.datasets.vlm_data.per_device_batch_size)
            if args.save_debug_media_dir is not None:
                _save_debug_media(train_dataset, batch_samples, count, cfg.datasets.vlm_data.per_device_batch_size, args.save_debug_media_dir)
            count += 1
