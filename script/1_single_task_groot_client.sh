#!/bin/bash
export PYTHONPATH=/path/to/workspace/RoboTwin_utils:$PYTHONPATH
source /path/to/workspace/envs/conda3/bin/activate robotwin2_n

# 指定使用空闲的 GPU（可根据需要修改）
export CUDA_VISIBLE_DEVICES=0

# NVIDIA 渲染库路径（解决 Vulkan 初始化问题）
export LD_LIBRARY_PATH=/tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json


export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
# 禁用 CUDA，因为 GR00T client 不需要本地 GPU（推理在服务器端）


# source .venv/bin/activate
# cd ../.. # move to root
# source /path/to/workspace/RoboTwin_utils/RoboTwin/policy/pi0/.venv/bin/activate

# source .venv/bin/activate
cd /path/to/workspace/RoboTwin_utils/RoboTwin

which python

# 注意：GR00T client 不需要本地 GPU（推理在服务器端）
# ROBOTWIN_PRINT_STEP=1 在终端打印每个step的进度

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 PYTHONWARNINGS=ignore::UserWarning \

python script/groot_simulation_client.py \
    --task_name  place_burger_fries \
    --task_config demo_clean \
    --seed 0 \
    --host localhost \
    --port 7035\
    --save_dir /path/to/workspace/GR00T_QwenVLA/outputs_robotwin/output_robotwin_easy_ckpt_3tasks/n1.5_nopretrain_finetuneALL_on_robotwin_eepose_v0.2/place_burger_fries
# adjust_bottle   place_container_plate  place_burger_fries