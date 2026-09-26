#!/usr/bin/env bash
set -euo pipefail

# DreamZero Wan2.2 policy training through the starVLA trainer.
#
# This is a starVLA-native model file that uses the local Wan2.2 diffusers
# backbone and does not require the official DreamZero/Groot training stack.

if [ -d /sys/class/infiniband ]; then
  RDMA_DEVICES=$(ls /sys/class/infiniband 2>/dev/null || true)
else
  RDMA_DEVICES=""
fi

if [ -n "${RDMA_DEVICES}" ]; then
  NCCL_IB_HCA=$(echo "${RDMA_DEVICES}" | tr '\n' ',' | sed 's/,$//')
  export NCCL_IB_HCA
  echo "Detected RDMA devices: ${NCCL_IB_HCA}"
  export NCCL_IB_DISABLE=0
  export NCCL_NET_GDR_LEVEL=2
else
  echo "No RDMA devices detected; using TCP/NCCL fallback."
  export NCCL_IB_DISABLE=1
fi

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS || true
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

if [ -f /path/to/workspace/envs/conda3/bin/activate ] && [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]; then
  source /path/to/workspace/envs/conda3/bin/activate "${STARVLA_CONDA_ENV:-starVLA_1}" || true
fi

NUM_GPUS_PER_NODE=${1:-8}

Framework_name=DreamZero
base_wm=${BASE_WM:-/path/to/workspace/models/Wan2.2-TI2V-5B-Diffusers}
base_vlm=${base_wm}
freeze_module_list=${FREEZE_MODULES:-backbone.vae,backbone.text_encoder}
data_root_dir=${DATA_ROOT_DIR:-/path/to/workspace/datasets/}
data_mix=${DATA_MIX:-robocasa_teleop_ee}
wm_height=${WM_HEIGHT:-224}
wm_width=${WM_WIDTH:-224}

action_dim=${ACTION_DIM:-32}
state_dim=${STATE_DIM:-64}
future_action_window_size=${FUTURE_ACTION_WINDOW_SIZE:-15}
action_horizon=${ACTION_HORIZON:-16}

run_root_dir=${RUN_ROOT_DIR:-./outputs/dreamzero_policy_fm}
run_id=${RUN_ID:-robocasa_teleop_ee_dreamzero_wan22_fm}

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
export WANDB_MODE=disabled
export WANDB_DIR="${output_dir}/wandb"
mkdir -p "${WANDB_DIR}"
cp "$0" "${output_dir}/"

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-29500} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.world_model.extract_layers "[-1]" \
  --framework.world_model.height ${wm_height} \
  --framework.world_model.width ${wm_width} \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.future_action_window_size ${future_action_window_size} \
  --framework.action_model.action_horizon ${action_horizon} \
  --framework.action_model.past_action_window_size 0 \
  --framework.policy.train_time_distribution logitnormal \
  --framework.policy.shift 5.0 \
  --framework.policy.action_loss_weight 1.0 \
  --framework.policy.future_image_loss_weight ${FUTURE_IMAGE_LOSS_WEIGHT:-1.0} \
  --framework.policy.num_inference_steps 5 \
  --framework.policy.timestep_scale 1000.0 \
  --framework.policy.autoregressive true \
  --framework.policy.latent_clip_value ${LATENT_CLIP_VALUE:-30.0} \
  --framework.policy.prediction_clip_value ${PREDICTION_CLIP_VALUE:-30.0} \
  --framework.policy.decode_future_images true \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH_SIZE:-1} \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps ${MAX_TRAIN_STEPS:-300000} \
  --trainer.save_interval ${SAVE_INTERVAL:-10000} \
  --trainer.logging_frequency ${LOGGING_FREQUENCY:-200} \
  --trainer.eval_interval ${EVAL_INTERVAL:-200} \
  --trainer.num_warmup_steps ${NUM_WARMUP_STEPS:-1000} \
  --trainer.learning_rate.base ${LEARNING_RATE:-1e-6} \
  --trainer.scheduler_specific_kwargs.min_lr ${MIN_LR:-1e-7} \
  --trainer.optimizer.betas "[0.9,0.999]" \
  --trainer.optimizer.weight_decay ${WEIGHT_DECAY:-0.1} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
