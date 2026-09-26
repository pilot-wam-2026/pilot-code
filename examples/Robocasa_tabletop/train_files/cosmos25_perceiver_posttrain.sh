#!/usr/bin/env zsh
set -euo pipefail

# Cosmos-Predict2.5 Perceiver training with DiT4DiT-style video-action fusion.
# Keeps the starVLA Perceiver action head, while training Cosmos video dynamics
# with a tri-timestep scheme and using fixed-timestep Cosmos hidden states as
# the action condition.

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

Framework_name=CosmoPredict25PerceiverDiT4DiT
base_wm=/path/to/workspace/code/starVLA/playground/Pretrained_models/Cosmos-Predict2.5-2B-Post-Trained
cosmos_revision=${COSMOS_PREDICT25_REVISION-diffusers/base/post-trained}
base_vlm=${base_wm}
freeze_module_list=''
DIT_TYPE="DiT-B"
data_root_dir=/path/to/workspace/datasets/
data_mix=robocasa_teleop_ee
pretrained_checkpoint="/path/to/policy_run/checkpoints/model.pt"


action_dim=32
state_dim=64
action_hidden_dim=2560
cross_attention_dim=2048

run_root_dir=./outputs1/cosmopredict25_perceiver_dit4dit/posttrain
run_id=robocasa_pre_sq_robocasa

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
  --master_port=${PET_MASTER_PORT:-29600} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.world_model.revision "${cosmos_revision}" \
  --framework.world_model.attn_implementation flash_attention_2 \
  --framework.world_model.extract_layers "[17]" \
  --framework.world_model.height 224 \
  --framework.world_model.width 224 \
  --framework.world_model.conditional_frame_timestep 0.0001 \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_hidden_dim ${action_hidden_dim} \
  --framework.action_model.hidden_size ${action_hidden_dim} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${cross_attention_dim} \
  --framework.action_model.diffusion_model_cfg.output_dim ${action_hidden_dim} \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.dit4dit_video.enabled true \
  --framework.dit4dit_video.training joint \
  --framework.dit4dit_video.future_loss_type flow_matching \
  --framework.dit4dit_video.future_loss_weight 1.0 \
  --framework.dit4dit_video.future_num_inference_steps 1 \
  --framework.dit4dit_action_head.enabled true \
  --framework.dit4dit_video.conditional_frame_timestep 0.0001 \
  --framework.dit4dit_video.height 224 \
  --framework.dit4dit_video.width auto \
  --framework.dit4dit_video.tri_timestep.video_time_distribution logit_normal \
  --framework.dit4dit_video.tri_timestep.high_sigma_ratio 0.05 \
  --framework.dit4dit_video.tri_timestep.high_sigma_min 0.98 \
  --framework.dit4dit_video.tri_timestep.feature_timestep 1.0 \
  --framework.dit4dit_video.tri_timestep.action_time_distribution beta_action_head \
  --framework.future_image_generation.enabled true \
  --framework.future_image_generation.conditioning_mode auto \
  --framework.future_image_generation.max_samples 1 \
  --framework.future_image_generation.height 224 \
  --framework.future_image_generation.width auto \
  --framework.future_image_generation.num_frames 9 \
  --framework.future_image_generation.num_inference_steps 1 \
  --framework.future_image_generation.guidance_scale 1.0 \
  --framework.future_image_generation.output_type pil \
  --framework.future_image_generation.future_frame_index -1 \
  --framework.future_image_generation.return_full_video true \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.video_backend torchvision_av \
  --datasets.vla_data.future_video_delta_indices "[2,4,6,8,10,12,14,16]" \
  --datasets.vla_data.action_video_freq_ratio 2 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 200 \
  --trainer.eval_interval 200 \
  --trainer.num_warmup_steps 5000 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.backbone 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --trainer.scheduler_specific_kwargs.min_lr 5e-7 \
  --trainer.optimizer.betas "[0.9,0.95]" \
  --trainer.optimizer.weight_decay 1e-8 \
  --trainer.pretrained_checkpoint ${pretrained_checkpoint} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"
