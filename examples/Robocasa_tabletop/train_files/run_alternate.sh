#!/usr/bin/env zsh
set -euo pipefail

# Cosmos-Predict2.5 Policy training, aligned with the official RF/FM recipe:
# action/future latent frames are trained as velocity targets.

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

if [ -f /path/to/workspace/envs/conda3/bin/activate ]; then
  source /path/to/workspace/envs/conda3/bin/activate starVLA_1
fi

NUM_GPUS_PER_NODE=${1:-8}

Framework_name=CosmoPredict25Policy
base_wm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Cosmos-Predict2.5-2B-Post-Trained
base_vlm=${base_wm}
freeze_module_list=backbone.vae,backbone.text_encoder
data_root_dir=/path/to/workspace/datasets/
data_mix=robocasa_teleop_ee

action_dim=32
state_dim=64

run_root_dir=./outputs/cosmopredict25_policy_fm
run_id=robocasa_teleop_ee_official_rf_state_steps5_2gpu

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
  --framework.world_model.height 224 \
  --framework.world_model.width 224 \
  --framework.world_model.text_max_length 128 \
  --framework.world_model.conditional_frame_timestep 0.0 \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.policy.train_time_distribution logitnormal \
  --framework.policy.train_time_weight uniform \
  --framework.policy.shift 5.0 \
  --framework.policy.action_loss_weight 16.0 \
  --framework.policy.future_image_loss_weight 1.0 \
  --framework.policy.num_inference_steps 5 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 200 \
  --trainer.eval_interval 200 \
  --trainer.num_warmup_steps 1000 \
  --trainer.learning_rate.base 2e-4 \
  --trainer.scheduler_specific_kwargs.min_lr 1.2e-5 \
  --trainer.optimizer.betas "[0.9,0.999]" \
  --trainer.optimizer.weight_decay 0.1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
