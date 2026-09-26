#!/bin/bash
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# One-click script: Start server on GPU 0, run 50 tasks dynamically on GPUs 1-7

# source /path/to/workspace/envs/conda3/bin/activate robot_simulation
# bash /path/to/workspace/JoyRA/StarVLA/scripts/setup_kdl/set_kdl.sh
# bash /path/to/workspace/JoyRA/StarVLA/scripts/setup_kdl/set_pykdl.sh

set -e

#####################################################################
# ======================= Configuration ============================
#####################################################################
# export LD_LIBRARY_PATH=/tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUROBO_TORCH_COMPILE=1

# -------------------- Path Settings --------------------
STARVLA_PATH=/path/to/workspace/projects/MaskWAM
ROBOTWIN_PATH=/path/to/workspace/projects/WM4A/RoboTwin
STAR_VLA_PYTHON=/path/to/workspace/envs/conda3/envs/starVLA_1/bin/python
ROBOTWIN_PYTHON=/path/to/workspace/envs/conda3/envs/robot_simulation/bin/python

# -------------------- Model Settings --------------------

# wm4a
CKPT_SETTING="robotwin_wan22_masked_future_image"
# CKPT_PATH=/path/to/policy_run/checkpoints/model.pt
CKPT_PATH=/path/to/policy_run/checkpoints/model.pt

# joyra05
# CKPT_SETTING="robotwin_clean_random_reversed-QwenFastWanPerceiver-mn-train-motus-fast-chunk50"
# CKPT_PATH=/path/to/policy_run/checkpoints/model.pt

WAN_PATH=/path/to/workspace/models/Wan2.2-TI2V-5B-Diffusers
FAST_TOKENIZER_PATH=/path/to/workspace/models/physical-intelligence-fast
export FAST_TOKENIZER_PATH
USE_BF16=true

# -------------------- Server Settings --------------------
SERVER_BASE_PORT=5693        # Per-GPU server ports: base + (gpu_id - EVAL_GPU_START_ID)
SEED=42

# -------------------- Eval Settings --------------------
TASK_CONFIG="demo_clean"
EVAL_SEED=0
NUM_EVAL_GPUS=8              # Number of GPUs used for eval
EVAL_GPU_START_ID=0          # Set 0 to include GPU0 as eval client
SLOTS_PER_GPU=8              # Keep 1 for per-GPU single-task strategy
NUM_TRAJ_WORKERS=8           # Keep 1 so each task is bound to one GPU
TEST_NUM_PER_TASK=100        # Total trajectories per task

# -------------------- Timing Settings --------------------
SERVER_STARTUP_WAIT=240

# -------------------- Output Settings --------------------
# 统一运行时间戳：以本脚本被调用的时刻为准，作为固定值传递给所有子进程，
# 保证 log 与 视频/图片 保存在同一个时间戳目录下。
RUN_TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

# 评估结果保存根目录（与 eval_policy.py 默认值保持一致）
EVAL_SAVE_ROOT="/path/to/output/robotwin"

# 本次运行的统一输出目录：${EVAL_SAVE_ROOT}/${CKPT_SETTING}/${RUN_TIMESTAMP}
RUN_OUTPUT_DIR="${EVAL_SAVE_ROOT}/${CKPT_SETTING}/${RUN_TIMESTAMP}"
# 日志统一保存到 LOG 子目录
LOG_DIR="${RUN_OUTPUT_DIR}/LOG"
mkdir -p "${LOG_DIR}"

#####################################################################
# ======================= Helper Functions =========================
#####################################################################

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

cleanup() {
    log_warn "Cleaning up..."
    for pid in "${SERVER_PIDS[@]}"; do
        if [ ! -z "$pid" ]; then
            log_info "Stopping policy server (PID: $pid)..."
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    log_info "Cleanup complete."
}

trap cleanup EXIT INT TERM

#####################################################################
# ======================= Main Script ==============================
#####################################################################

log_step "========== StarVLA 50-Task Dynamic Evaluation =========="
log_info "Eval GPU range: ${EVAL_GPU_START_ID}-$((EVAL_GPU_START_ID + NUM_EVAL_GPUS - 1))"
log_info "Slots per GPU: ${SLOTS_PER_GPU}"
log_info "Trajectory workers per task: ${NUM_TRAJ_WORKERS}"
log_info "Trajectories per task: ${TEST_NUM_PER_TASK}"
log_info "Total Tasks: 50"
log_info "Checkpoint: ${CKPT_PATH}"
log_info "Server Base Port: ${SERVER_BASE_PORT}"
log_info "Log Directory: ${LOG_DIR}"
log_info "Server Python: ${STAR_VLA_PYTHON}"
log_info "Eval Python: ${ROBOTWIN_PYTHON}"
echo ""

# -------------------- Step 1: Start Per-GPU Policy Servers --------------------
log_step "Step 1: Starting policy servers (one per eval GPU)..."

cd ${STARVLA_PATH}
export PYTHONPATH=$(pwd):${PYTHONPATH}

BF16_FLAG=""
if [ "$USE_BF16" = true ]; then
    BF16_FLAG="--use_bf16"
fi

SERVER_PIDS=()
SERVER_PORTS=()
SERVER_PORTS_ARG=""

for ((idx=0; idx<NUM_EVAL_GPUS; idx++)); do
    gpu_id=$((EVAL_GPU_START_ID + idx))
    server_port=$((SERVER_BASE_PORT + idx))

    CUDA_VISIBLE_DEVICES=${gpu_id} ${STAR_VLA_PYTHON} \
        ${STARVLA_PATH}/deployment_robotwin/model_server/server_policy.py \
        --ckpt_path ${CKPT_PATH} \
        --port ${server_port} \
        --seed ${SEED} \
        ${BF16_FLAG} &

    server_pid=$!
    SERVER_PIDS+=("${server_pid}")
    SERVER_PORTS+=("${server_port}")

    if [ -z "${SERVER_PORTS_ARG}" ]; then
        SERVER_PORTS_ARG="${gpu_id}:${server_port}"
    else
        SERVER_PORTS_ARG="${SERVER_PORTS_ARG},${gpu_id}:${server_port}"
    fi

    log_info "Started server for GPU ${gpu_id} on port ${server_port} (PID: ${server_pid})"
done

log_info "Waiting ${SERVER_STARTUP_WAIT}s for server to initialize..."
sleep ${SERVER_STARTUP_WAIT}

for i in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$i]}"
    port="${SERVER_PORTS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
        log_error "Policy server failed to start (PID: ${pid}, port: ${port})!"
        exit 1
    fi
done
log_info "All policy servers are running."
log_info "Server mapping: ${SERVER_PORTS_ARG}"
echo ""

# -------------------- Step 2: Run Dynamic Task Scheduler --------------------
log_step "Step 2: Running Dynamic Task Scheduler (50 tasks)..."

set +e  # Don't exit on error, capture exit code

${STAR_VLA_PYTHON} ${STARVLA_PATH}/examples/Robotwin_ee/eval_files/dynamic_task_scheduler_50_submission.py \
    --mode run \
    --num_gpus ${NUM_EVAL_GPUS} \
    --gpu_start_id ${EVAL_GPU_START_ID} \
    --slots_per_gpu ${SLOTS_PER_GPU} \
    --num_traj_workers ${NUM_TRAJ_WORKERS} \
    --test_num ${TEST_NUM_PER_TASK} \
    --ckpt_path ${CKPT_PATH} \
    --server_port ${SERVER_BASE_PORT} \
    --server_ports "${SERVER_PORTS_ARG}" \
    --task_config ${TASK_CONFIG} \
    --ckpt_setting ${CKPT_SETTING} \
    --eval_seed ${EVAL_SEED} \
    --log_dir "${LOG_DIR}" \
    --eval_save_root "${EVAL_SAVE_ROOT}" \
    --eval_timestamp "${RUN_TIMESTAMP}" \
    --starvla_path ${STARVLA_PATH} \
    --robotwin_path ${ROBOTWIN_PATH} \
    --python_path ${ROBOTWIN_PYTHON} \
    --resource_log_interval 300
# CPU监测
#   --resource_log_interval 30 间隔多少s打印
#   --disable_resource_monitor 不打印CPU监测
SCHEDULER_EXIT_CODE=$?

set -e

echo ""

# -------------------- Step 3: Summary --------------------
log_step "Step 3: Evaluation Complete"

if [ "$SCHEDULER_EXIT_CODE" -eq 0 ]; then
    log_info "All 50 tasks completed successfully!"
else
    log_error "Some tasks failed. Check logs in: ${LOG_DIR}"
fi

exit $SCHEDULER_EXIT_CODE
