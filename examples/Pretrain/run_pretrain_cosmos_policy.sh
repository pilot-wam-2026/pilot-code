#!/usr/bin/env zsh
set -euo pipefail

# Cosmos2 EDM Policy training script:
# action/future-image latent frames are trained with a sigma-space denoising objective.

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

NUM_GPUS_PER_NODE=8

Framework_name=CosmoPredict2PolicyEDM
base_wm=/path/to/workspace/models/Cosmos-Predict2-2B-Video2World
# base_wm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Cosmos-Predict2.5-2B-Post-Trained
base_vlm=${base_wm}
freeze_module_list=backbone.vae,backbone.text_encoder
data_root_dir=/path/to/workspace/datasets/
data_mix=agi-beta_robotwin_robocasa

action_dim=32
state_dim=64

run_root_dir=./outputs/cosmos2_policy
run_id=pretrain_agi-beta_robotwin_robocasa

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
  --config_yaml ./examples/Pretrain/starvla_cotrain_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.world_model.height 224 \
  --framework.world_model.width 224 \
  --framework.world_model.text_max_length 128 \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.policy.sigma_data 1.0 \
  --framework.policy.sigma_min 0.0002 \
  --framework.policy.sigma_max 80.0 \
  --framework.policy.sigma_p_mean -1.2 \
  --framework.policy.sigma_p_std 1.2 \
  --framework.policy.hybrid_sigma_distribution true \
  --framework.policy.hybrid_sigma_lognormal_prob 0.7 \
  --framework.policy.uniform_lower 1.0 \
  --framework.policy.uniform_upper 85.0 \
  --framework.policy.rho 7.0 \
  --framework.policy.loss_weighting edm \
  --framework.policy.max_loss_weight 100.0 \
  --framework.policy.num_inference_steps 5 \
  --framework.policy.sampler 2ab \
  --framework.policy.preconditioning true \
  --framework.policy.future_image_loss_weight 1.0 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 200 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base 1e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
