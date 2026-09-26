import re

from pathlib import Path

# You can add multimodal datasets here and register a short nickname to ${data_dict}.
# The data format should follow the general multimodal VLM format, for example:
# https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/README.md

json_root = f"/path/to/workspace/jsons"
# json_root = f"/path/to/workspace/datasets/Cambrian-10M/jsons"
image_root = f"/path/to/workspace/datasets"

############# VLM Pretrain #############
CAMBRAIN_300K = {
    "annotation_path": f"/path/to/workspace/jsons/Cambrian300k_withsystemprompt.jsonl",
    "data_path": f"{image_root}/Cambrian-10M/",
}

REFSPATIAL_2D_100K = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_2d_100k.json",
    "data_path": f"{image_root}/RefSpatial/2D/image/",
}

REFSPATIAL_3D_150K = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_3d_150k.json",
    "data_path": f"{image_root}/RefSpatial/3D/image/",
}

REFSPATIAL_3D_VISUAL_50K = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_3d_visual_50k.json",
    "data_path": f"{image_root}/RefSpatial/3D/image_visual_choice/",
}

ROBOVQA_300K = {
    "annotation_path": f"/path/to/workspace/jsons/robovqa_300k.json",
    "data_path": f"/path/to/workspace/datasets/robovqa/",
}

GAL_SUBTASK_110K = {
    "annotation_path": f"/path/to/workspace/datasets/Galaxea_subtask/llava_jsons/galaxea_subtask_110k.json",
    "data_path": f"/path/to/workspace/datasets/Galaxea_subtask",
}

AGIBOT_SUBTASK = {
    "annotation_path": "/path/to/workspace/datasets/Agibot-Beta-Transformation/Subtask_QA/all_data.jsonl",
    "data_path": "/path/to/workspace/datasets/Agibot-Beta-Transformation/Subtask_QA/images"
}

SQ_SUBTASK = {
    "annotation_path": "/path/to/workspace/Suqian_subtask_qa/all_data.jsonl",
    "data_path": "/path/to/workspace/Suqian_subtask_qa/images"
}

############# VLA Pretrain #############
CAMBRAIN_300K_VLAPT = {
    "annotation_path": f"/path/to/workspace/jsons/Cambrian300k_withsystemprompt_vlapretrain.jsonl",
    "data_path": f"{image_root}/Cambrian-10M/",
}

REFSPATIAL_2D_100K_VLAPT = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_2d_100k_vlapretrain.json",
    "data_path": f"{image_root}/RefSpatial/2D/image/",
}

REFSPATIAL_3D_150K_VLAPT = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_3d_150k_vlapretrain.json",
    "data_path": f"{image_root}/RefSpatial/3D/image/",
}

REFSPATIAL_3D_VISUAL_50K_VLAPT = {
    "annotation_path": f"/path/to/workspace/jsons/refspatial/refspatial_3d_visual_50k_vlapretrain.json",
    "data_path": f"{image_root}/RefSpatial/3D/image_visual_choice/",
}

ROBOVQA_300K_VLAPT = {
    "annotation_path": f"/path/to/workspace/jsons/robovqa_300k_vlapretrain.json",
    "data_path": f"/path/to/workspace/datasets/robovqa/",
}

data_dict = {
    "cambrain_300k": CAMBRAIN_300K,
    "refspatial_2d_100k": REFSPATIAL_2D_100K,
    "refspatial_3d_150k": REFSPATIAL_3D_150K,
    "refspatial_3d_visual_50k": REFSPATIAL_3D_VISUAL_50K,
    "robovqa_300k": ROBOVQA_300K,
    "gal_subtask_110k": GAL_SUBTASK_110K,
    "cambrain_300k_vlapt": CAMBRAIN_300K_VLAPT,
    "refspatial_2d_100k_vlapt": REFSPATIAL_2D_100K_VLAPT,
    "refspatial_3d_150k_vlapt": REFSPATIAL_3D_150K_VLAPT,
    "refspatial_3d_visual_50k_vlapt": REFSPATIAL_3D_VISUAL_50K_VLAPT,
    "robovqa_300k_vlapt": ROBOVQA_300K_VLAPT,
    "agibot_subtask": AGIBOT_SUBTASK,
    "sq_subtask": SQ_SUBTASK,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    if dataset_names == ["all"]:
        dataset_names = list(data_dict.keys())
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    print(data_list)
