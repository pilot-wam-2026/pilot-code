#!/bin/bash

# ============================================================
# WM Evaluation Script
# ============================================================

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# NVIDIA H200 EGL Rendering Path
export LD_LIBRARY_PATH=/path/to/workspace/lib/nvidia_egl:\$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export MUJOCO_GL=egl
export STARVLA_WM_DEBUG=${STARVLA_WM_DEBUG:-0}
export STARVLA_WM_DEBUG_MAX=${STARVLA_WM_DEBUG_MAX:-6}
export STARVLA_ACTION_DEBUG=${STARVLA_ACTION_DEBUG:-0}
export STARVLA_ACTION_DEBUG_MAX=${STARVLA_ACTION_DEBUG_MAX:-20}

cd /path/to/workspace/code/starVLA
starVLA_PYTHON=/path/to/workspace/envs/conda3/envs/starVLA_1/bin/python
ROBOCASA_PYTHON=/path/to/workspace/envs/conda3/envs/robocasa/bin/python
export PYTHONPATH=$(pwd):${PYTHONPATH}

CKPT_DEFAULT="/path/to/policy_run/checkpoints/model.pt"
SIMULATION_SCRIPT="examples/Robocasa_tabletop/eval_files/simulation_env_wm.py"

N_ENVS_DEFAULT=1
MAX_EPISODE_STEPS_DEFAULT=720
N_ACTION_STEPS_DEFAULT=12
NUM_INFERENCE_STEPS_DEFAULT=5
SHIFT_DEFAULT=5.0

BASE_PORT=9999
NUM_GPUS=8

CKPT_PATH=${1:-$CKPT_DEFAULT}
N_ENVS=${2:-$N_ENVS_DEFAULT}
MAX_EPISODE_STEPS=${3:-$MAX_EPISODE_STEPS_DEFAULT}
N_ACTION_STEPS=${4:-$N_ACTION_STEPS_DEFAULT}
NUM_INFERENCE_STEPS=${5:-$NUM_INFERENCE_STEPS_DEFAULT}
SHIFT=${6:-$SHIFT_DEFAULT}

echo "=== WM Evaluation Configuration ==="
echo "Checkpoint Path      : ${CKPT_PATH}"
echo "Number of Envs       : ${N_ENVS}"
echo "Max Episode Steps    : ${MAX_EPISODE_STEPS}"
echo "Action Chunk Length  : ${N_ACTION_STEPS}"
echo "Inference Steps      : ${NUM_INFERENCE_STEPS}"
echo "RF Shift             : ${SHIFT}"
echo "Simulation Script    : ${SIMULATION_SCRIPT}"
echo "==================================="

SERVER_PIDS=()
EVAL_PIDS=()
CLEANED_UP=0

kill_process_tree() {
    local PID=$1
    local SIGNAL=${2:-TERM}

    if [[ -z "${PID}" ]] || ! kill -0 "${PID}" 2>/dev/null; then
        return
    fi

    local CHILD_PIDS
    CHILD_PIDS=$(pgrep -P "${PID}" 2>/dev/null || true)
    for CHILD_PID in ${CHILD_PIDS}; do
        kill_process_tree "${CHILD_PID}" "${SIGNAL}"
    done

    kill -"${SIGNAL}" "${PID}" 2>/dev/null || true
}

cleanup() {
    local EXIT_CODE=$?
    if [[ "${CLEANED_UP}" -eq 1 ]]; then
        exit "${EXIT_CODE}"
    fi
    CLEANED_UP=1

    trap - INT TERM EXIT

    echo ""
    echo "Cleaning up WM evaluation processes..."

    for PID in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
        kill_process_tree "${PID}" TERM
    done

    sleep 2

    for PID in "${EVAL_PIDS[@]}" "${SERVER_PIDS[@]}"; do
        kill_process_tree "${PID}" KILL
    done

    wait 2>/dev/null || true
    echo "Cleanup finished."
    exit "${EXIT_CODE}"
}

trap cleanup INT TERM EXIT

EvalEnv() {
    local GPU_ID=$1
    local PORT=$2
    local ENV_NAME=$3
    local CKPT_PATH=$4
    local LOG_DIR=$5
    local ROBOCASA_PYTHON=$6
    local N_ENVS=$7
    local MAX_EPISODE_STEPS=$8
    local N_ACTION_STEPS=$9
    local NUM_INFERENCE_STEPS=${10}
    local SHIFT=${11}

    local SAVE_ROOT=$(dirname "$(dirname "$CKPT_PATH")")
    local ckpt_name=$(basename "$CKPT_PATH" .pt)
    local VIDEO_OUT_PATH="${SAVE_ROOT}/videos/${ckpt_name}/$(basename "${LOG_DIR}")/${ENV_NAME}"
    mkdir -p "${VIDEO_OUT_PATH}"

    echo "Launching WM evaluation | GPU ${GPU_ID} | Port ${PORT} | Env ${ENV_NAME}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    EGL_VISIBLE_DEVICES=0 \
    MUJOCO_EGL_DEVICE_ID=0 \
    ${ROBOCASA_PYTHON} "${SIMULATION_SCRIPT}" \
        --args.env_name "${ENV_NAME}" \
        --args.port "${PORT}" \
        --args.n_episodes 50 \
        --args.n_envs "${N_ENVS}" \
        --args.max_episode_steps "${MAX_EPISODE_STEPS}" \
        --args.n_action_steps "${N_ACTION_STEPS}" \
        --args.num_inference_steps "${NUM_INFERENCE_STEPS}" \
        --args.shift "${SHIFT}" \
        --args.video_out_path "${VIDEO_OUT_PATH}" \
        --args.pretrained_path "${CKPT_PATH}" \
        > "${LOG_DIR}/eval_env_${ENV_NAME//\//_}_gpu${GPU_ID}.log" 2>&1
}

ENV_NAMES=(
  gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
  gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env
)

LOG_DIR="${CKPT_PATH}.log/wm_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=== Launching WM Multi-GPU Evaluation ==="
echo "GPUs             : ${NUM_GPUS}"
echo "Num Environments : ${#ENV_NAMES[@]}"
echo "Log Directory    : ${LOG_DIR}"

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((BASE_PORT + GPU_ID))
    echo "Starting policy server | GPU ${GPU_ID} | Port ${PORT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    ${starVLA_PYTHON} deployment/model_server/server_policy.py \
        --ckpt_path "${CKPT_PATH}" \
        --port "${PORT}" \
        --use_bf16 \
        > "${LOG_DIR}/server_gpu${GPU_ID}_port${PORT}.log" 2>&1 &

    SERVER_PIDS[$GPU_ID]=$!
    sleep 10
done

sleep 30

COUNT=0
for ENV_NAME in "${ENV_NAMES[@]}"; do
    GPU_ID=$((COUNT % NUM_GPUS))
    PORT=$((BASE_PORT + GPU_ID))

    if (( (COUNT + 1) % NUM_GPUS == 0 )); then
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" "${NUM_INFERENCE_STEPS}" "${SHIFT}" &
        EVAL_PIDS+=($!)
        wait "${EVAL_PIDS[@]}"
        EVAL_PIDS=()
    else
        EvalEnv "${GPU_ID}" "${PORT}" "${ENV_NAME}" "${CKPT_PATH}" "${LOG_DIR}" \
                "${ROBOCASA_PYTHON}" "${N_ENVS}" "${MAX_EPISODE_STEPS}" "${N_ACTION_STEPS}" "${NUM_INFERENCE_STEPS}" "${SHIFT}" &
        EVAL_PIDS+=($!)
    fi

    COUNT=$((COUNT + 1))
    sleep 2
done

while pgrep -f "${SIMULATION_SCRIPT}" > /dev/null; do
    echo "Waiting for all WM evaluation environments to finish..."
    sleep 30
done

echo ""
echo "Shutting down policy servers..."

for PID in "${SERVER_PIDS[@]}"; do
    kill_process_tree "${PID}" TERM
    echo "Killed server PID ${PID}"
done

echo "=== WM Evaluation Finished ==="

python summarize_eval_logs.py "${LOG_DIR}"
