#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G
# 禁用 CUDA，因为 GR00T client 不需要本地 GPU（推理在服务器端）

# # 简化的参数定义
# task_name=${1}
# task_config=${2}
# seed=${3}

# GR00T server 连接参数（可选，有默认值）
# GROOT_HOST=${4:-localhost}  # 默认 localhost
# GROOT_PORT=${5:-8811}       # 默认 8811（匹配服务端端口）

# instruction_type（可选，默认是 "unseen"）
# INSTRUCTION_TYPE=${6:-unseen}

gpu_id="${GPU_ID:-0}"          # 允许用环境变量 GPU_ID 覆盖
export CUDA_VISIBLE_DEVICES="${gpu_id}"
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

# source .venv/bin/activate
# cd ../.. # move to root
source /path/to/workspace/RoboTwin_utils/RoboTwin/policy/pi0/.venv/bin/activate

# source .venv/bin/activate
cd /path/to/workspace/RoboTwin_utils/RoboTwin

which python

# 注意：GR00T client 不需要本地 GPU（推理在服务器端）

TORCH_USE_CUDA_DSA=1 CUDA_LAUNCH_BLOCKING=1 PYTHONWARNINGS=ignore::UserWarning \
python script/groot_simulation_client.py \
    --task_name adjust_bottle \
    --task_config demo_clean \
    --seed 0 \
    --host localhost \
    --port 8811 \
    --save_dir /path/to/workspace/GR00T_QwenVLA/outputs_robotwin/eval_result_eepose