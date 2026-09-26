#!/usr/bin/env bash

RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  echo "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    break
  fi
done <<<"$output"

export NCCL_SOCKET_IFNAME="eth0"
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

# ==== 多节点配置 ====
NUM_NODES=${PET_NNODES}
NODE_RANK=${PET_NODE_RANK}
MASTER_ADDR=${PET_MASTER_ADDR}
MASTER_PORT=${PET_MASTER_PORT}
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-8}

source /path/to/workspace/envs/conda3/bin/activate starVLA_1

Framework_name=QwenPerceiver
base_vlm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list=''
DIT_TYPE="DiT-B"
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml
data_root_dir=/path/to/workspace/datasets/
data_mix=robotwin
pretrained_checkpoint=${PRETRAINED_CHECKPOINT:-"/path/to/policy_run/checkpoints/model.pt"}
action_dim=32
state_dim=64
output_dim=2560
action_horizon=50
future_action_window_size=49

run_root_dir=./outputs
run_id=post_robotwin_pre_cotrain_sq_agi-beta_egodex-f

export WANDB_MODE=disabled
export WANDB_DIR=../../

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-29500} \
  --nproc_per_node=${NUM_GPUS_PER_NODE:-8} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 1e-4 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.hidden_size ${output_dim} \
  --framework.action_model.diffusion_model_cfg.output_dim ${output_dim} \
  --framework.action_model.action_horizon ${action_horizon} \
  --framework.action_model.future_action_window_size ${future_action_window_size} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
  # --is_debug True
