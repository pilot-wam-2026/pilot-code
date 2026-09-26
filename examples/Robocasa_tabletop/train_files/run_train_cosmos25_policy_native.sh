#!/usr/bin/env zsh
set -euo pipefail

# Native Cosmos-Predict2.5 Policy training.
# This uses NVIDIA's cosmos-predict2.5 policy model directly instead of the
# diffusers Cosmos2_5_PredictBasePipeline wrapper.

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
MASTER_PORT=${2:-12341}
reason1_ckpt=${3:-}

Framework_name=CosmoPredict25PolicyNative
native_repo_path=/path/to/workspace/code/cosmos-predict2.5
native_ckpt=/path/to/workspace/models/Cosmos-Predict2.5-2B

data_root_dir=/path/to/workspace/datasets/
data_mix=robocasa_teleop_ee

action_dim=32
state_dim=64

run_root_dir=./outputs/cosmopredict25_policy_native
run_id=robocasa_teleop_ee_native_policy

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
  --master_port=${MASTER_PORT} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm "${native_ckpt}" \
  --framework.qwenvl.base_vlm "${native_ckpt}" \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.action_horizon 16 \
  --framework.policy.native_repo_path "${native_repo_path}" \
  --framework.policy.checkpoint_path "${native_ckpt}" \
  --framework.policy.experiment cosmos_predict2p5_2b_480p_robocasa_50_demos_per_task_no_s3 \
  --framework.policy.use_online_text_encoder true \
  --framework.policy.text_encoder_ckpt_path "${reason1_ckpt}" \
  --framework.policy.height 224 \
  --framework.policy.width 224 \
  --framework.policy.shift 5.0 \
  --framework.policy.num_inference_steps 5 \
  --framework.policy.mask_policy_loss true \
  --framework.policy.action_loss_multiplier 16 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 10000 \
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
