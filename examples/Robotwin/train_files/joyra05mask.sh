#!/usr/bin/env zsh
set -euo pipefail

# Cosmos-Predict2.5 Perceiver training with eval-time future-image visualization.
# Action training uses Cosmos hidden states + Perceiver; eval additionally runs
# the Cosmos2.5 decoder/generation path and saves pred future images.

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

cd /path/to/workspace/projects/WM4A
if [ -f /path/to/workspace/envs/conda3/bin/activate ]; then
  source /path/to/workspace/envs/conda3/bin/activate starVLA_1
fi

NUM_GPUS_PER_NODE=${1:-8}   # mytest

Framework_name=QwenWanPerceiverMask
base_vlm=/path/to/workspace/models/Qwen3-VL-4B-Instruct
base_wm=/path/to/models/Wan2.2-TI2V-5B-Diffusers
transformer_path=/path/to/workspace/models/Motus_Wan2_2_5B_pretrain
pretrained_checkpoint=/path/to/policy_run/checkpoints/model.pt
config_yaml=/path/to/workspace/projects/WM4A/examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml
DIT_TYPE="DiT-B"
data_root_dir=/path/to/workspace/datasets/
data_mix=robotwin_ee

run_root_dir=/path/to/output
run_id=robotwin             # mytest

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
export WANDB_MODE=disabled
export WANDB_DIR="${output_dir}/wandb"
mkdir -p "${WANDB_DIR}"
cp "$0" "${output_dir}/"
cp ${config_yaml} ${output_dir}

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-29500} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  starVLA/training/train_starvla_zero2.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --framework.qwenvl.is_frozen false \
  --framework.world_model.base_wm ${base_wm} \
  --framework.world_model.transformer_path ${transformer_path} \
  --framework.world_model.extract_layers "[-1]" \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.diffusion_model_cfg.output_dim 1024 \
  --framework.future_image_training.enabled true \
  --framework.future_image_training.loss_weight 1.0 \
  --framework.future_image_training.train_time_distribution logitnormal \
  --framework.future_image_training.shift 5.0 \
  --framework.future_image_training.change_mask_enabled true \
  --framework.future_image_training.change_mask_top_ratio 0.25 \
  --framework.future_image_training.change_mask_static_weight 0.05 \
  --framework.future_image_training.change_mask_soft_loss true \
  --framework.future_image_training.masked_denoising false \
  --framework.future_image_training.condition_static_regions false \
  --framework.future_image_generation.enabled true \
  --framework.future_image_generation.conditioning_mode auto \
  --framework.future_image_generation.max_samples 8 \
  --framework.future_image_generation.height 224 \
  --framework.future_image_generation.width 224 \
  --framework.future_image_generation.num_frames 9 \
  --framework.future_image_generation.num_inference_steps 20 \
  --framework.future_image_generation.guidance_scale 7.0 \
  --framework.future_image_generation.output_type pil \
  --framework.future_image_generation.future_frame_index -1 \
  --framework.future_image_generation.num_latent_conditional_frames 1 \
  --framework.future_image_generation.return_full_video false \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.future_video_delta_indices "[16]" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 2 \
  --datasets.vla_data.num_workers 8 \
  --datasets.vla_data.prefetch_factor 2 \
  --datasets.vla_data.pin_memory true \
  --datasets.vla_data.persistent_workers true \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 500 \
  --trainer.num_warmup_steps 1000 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.scheduler_specific_kwargs.min_lr 1.2e-6 \
  --trainer.optimizer.betas "[0.9,0.999]" \
  --trainer.optimizer.weight_decay 0.1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
