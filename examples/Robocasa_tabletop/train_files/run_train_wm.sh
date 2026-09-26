#!/usr/bin/env zsh
set -euo pipefail

# Cosmos Predict2.5 Perceiver training script based on run_train.sh.

if [ -d /sys/class/infiniband ]; then
  RDMA_DEVICES=$(ls /sys/class/infiniband 2>/dev/null || true)
else
  RDMA_DEVICES=""
fi

if [ -n "${RDMA_DEVICES}" ]; then
  NCCL_IB_HCA=$(echo "${RDMA_DEVICES}" | tr '\n' ',' | sed 's/,$//')
  export NCCL_IB_HCA
  echo "Detected RDMA devices: ${NCCL_IB_HCA}"

  if command -v show_gids >/dev/null 2>&1; then
    NCCL_IB_GID_INDEX=""
    output=$(show_gids | grep v2 || true)
    while IFS= read -r line; do
      ipv4=$(echo "${line}" | awk '{print $5}')
      if [[ -n "${ipv4}" && "${ipv4}" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "${ipv4}" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
        NCCL_IB_GID_INDEX=$(echo "${line}" | awk '{print $3}')
        break
      fi
    done <<<"${output}"
    export NCCL_IB_GID_INDEX
  fi

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

if [ -f /path/to/workspace/envs/conda3/bin/activate ]; then
  source /path/to/workspace/envs/conda3/bin/activate starVLA_1
fi

NUM_GPUS_PER_NODE=${1:-8}

Framework_name=CosmoPredict25Perceiver
base_wm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Cosmos-Predict2.5-2B-Post-Trained
cosmos_revision=${COSMOS_PREDICT25_REVISION-diffusers/base/post-trained}
base_vlm=${base_wm}
freeze_module_list=''
DIT_TYPE="DiT-B"
data_root_dir=/path/to/workspace/datasets/
data_mix=robocasa_teleop_ee

action_dim=32
state_dim=64
action_hidden_dim=1024
cross_attention_dim=2048

run_root_dir=./outputs
run_id=cosmopredict25_perceiver_robocasa_teleop_ee

export WANDB_MODE=disabled
export WANDB_DIR=../../

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
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
  --framework.world_model.revision "${cosmos_revision}" \
  --framework.world_model.extract_layers "[-1]" \
  --framework.world_model.height 224 \
  --framework.world_model.width 224 \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_hidden_dim ${action_hidden_dim} \
  --framework.action_model.hidden_size ${action_hidden_dim} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${cross_attention_dim} \
  --framework.action_model.diffusion_model_cfg.output_dim ${action_hidden_dim} \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 200 \
  --trainer.eval_interval 200 \
  --trainer.learning_rate.base 1e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
