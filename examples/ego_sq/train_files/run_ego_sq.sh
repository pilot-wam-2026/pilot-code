RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  log "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
# 遍历每一行
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  # 检查IPv4地址是否有值（不是空字符串且不是全0地址）
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    # 找到第一个匹配项后立即退出循环
    break
  fi
done <<<"$output"

export NCCL_SOCKET_IFNAME="eth0"

# 处理分布式训练需要使用到的相关环境变量
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
# export NCCL_DEBUG_SUBSYS=ALL
# 加速
export NCCL_DEBUG=WARN
unset NCCL_DEBUG_SUBSYS
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}

# used for check save when communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000  # timeout set to 1 hour (unit: seconds)

# ==== 多节点配置 ====
# 总节点数（根据实际情况修改）
NUM_NODES=${PET_NNODES}
# 当前节点索引（0-based，每个节点启动时需指定不同值）
NODE_RANK=${PET_NODE_RANK}
# 主节点IP地址（确保所有节点可访问）
MASTER_ADDR=${PET_MASTER_ADDR}  # 替换为实际主节点IP
# 主节点端口（保持与代码中一致）
MASTER_PORT=${PET_MASTER_PORT}
NUM_GPUS_PER_NODE=8

cd /path/to/workspace/starVLA2
source /path/to/workspace/envs/conda3/bin/activate starVLA_1

Framework_name=QwenGR00T
base_vlm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct
freeze_module_list='qwen_vl_interface.model.model' # just for fast debug, sota is under fully FT, i.g., freeze_module_list=""
DIT_TYPE="DiT-B"
data_root_dir=/path/to/workspace
data_mix=ego_dex_suqian_agi


run_root_dir=./res
run_id=test

export WANDB_MODE=disabled
export WANDB_DIR= #YOUR_WANDB

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
# mv this script to the output dir
cp $0 ${output_dir}/

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-29500} \
  --nproc_per_node=${NUM_GPUS_PER_NODE:-8} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/ego_sq/train_files/starvla_cotrain_ego_sq.yaml \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --datasets.vla_data.data_root_dir ${data_root_dir} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.learning_rate.base 1e-5 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous\
  2>&1 | tee -a "${output_dir}/train.log"
  # --is_debug True


