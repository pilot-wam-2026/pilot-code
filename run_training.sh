#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the GPUs allocated to this run.}"
: "${WM4A_DATA_ROOT:?Set WM4A_DATA_ROOT to the dataset directory.}"
: "${WM4A_COSMOS_ROOT:?Set WM4A_COSMOS_ROOT to the local Cosmos model directory.}"
: "${WM4A_VJEPA_CHECKPOINT:?Set WM4A_VJEPA_CHECKPOINT to the initialization weight.}"
: "${WM4A_RESUME_CHECKPOINT:?Set WM4A_RESUME_CHECKPOINT to the resume checkpoint.}"
WM4A_CONFIG="${WM4A_CONFIG:-${PROJECT_ROOT}/examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr2.yaml}"
WM4A_VJEPA_REPO="${WM4A_VJEPA_REPO:-${PROJECT_ROOT}/starVLA/facebookresearch_vjepa2_main}"

NUM_GPUS_PER_NODE="${1:-1}"
if [[ ! "$NUM_GPUS_PER_NODE" =~ ^[1-9][0-9]*$ || "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
  echo "Use a positive process count and an explicit GPU allocation." >&2
  exit 2
fi
IFS=, read -r -a VISIBLE_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
if (( NUM_GPUS_PER_NODE > ${#VISIBLE_GPU_IDS[@]} )); then
  echo "Requested processes exceed CUDA_VISIBLE_DEVICES." >&2
  exit 2
fi
for required in "$WM4A_DATA_ROOT" "$WM4A_COSMOS_ROOT" "$WM4A_VJEPA_REPO"; do
  [[ -d "$required" ]] || { echo "Missing directory: $required" >&2; exit 2; }
done
for required in "$WM4A_CONFIG" "$WM4A_VJEPA_CHECKPOINT" "$WM4A_RESUME_CHECKPOINT"; do
  [[ -f "$required" ]] || { echo "Missing file: $required" >&2; exit 2; }
done

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
Framework_name=CosmoPredict25PerceiverVJEPA2AC
base_wm="$WM4A_COSMOS_ROOT"
base_vlm="$base_wm"
cosmos_revision="${COSMOS_PREDICT25_REVISION:-diffusers/base/post-trained}"
freeze_module_list=''
DIT_TYPE="DiT-B"
data_root_dir="$WM4A_DATA_ROOT"
data_mix=robocasa_teleop_ee
action_dim=32
state_dim=64
action_hidden_dim=1024
cross_attention_dim=2048
run_root_dir="${WM4A_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/wm4a}"
run_id="robocasa_$(date -u +%Y%m%dT%H%M%SZ)"
resume_checkpoint="$WM4A_RESUME_CHECKPOINT"
output_dir="${run_root_dir}/${run_id}"
mkdir -p "$output_dir"
export WANDB_DIR="${output_dir}/wandb"
mkdir -p "$WANDB_DIR"
cp "$0" "$output_dir/"

env IS_TORCHRUN=1 torchrun \
  --nnodes=${PET_NNODES:-1} \
  --node_rank=${PET_NODE_RANK:-0} \
  --master_addr=${PET_MASTER_ADDR:-127.0.0.1} \
  --master_port=${PET_MASTER_PORT:-38500} \
  --nproc_per_node=${NUM_GPUS_PER_NODE} \
  "${PROJECT_ROOT}/starVLA/training/train_starvla.py" \
  --config_yaml "${WM4A_CONFIG}" \
  --framework.name ${Framework_name} \
  --framework.world_model.base_wm "${base_wm}" \
  --framework.world_model.revision "${cosmos_revision}" \
  --framework.world_model.extract_layers "[-1]" \
  --framework.world_model.height 224 \
  --framework.world_model.width 224 \
  --framework.world_model.conditional_frame_timestep 0.1 \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.qwenvl.vl_hidden_dim 2048 \
  --framework.action_model.action_model_type ${DIT_TYPE} \
  --framework.action_model.action_dim ${action_dim} \
  --framework.action_model.state_dim ${state_dim} \
  --framework.action_model.action_hidden_dim ${action_hidden_dim} \
  --framework.action_model.hidden_size ${action_hidden_dim} \
  --framework.action_model.diffusion_model_cfg.cross_attention_dim ${cross_attention_dim} \
  --framework.action_model.diffusion_model_cfg.output_dim ${action_hidden_dim} \
  --framework.vjepa2_ac.repo_dir "${WM4A_VJEPA_REPO}" \
  --framework.vjepa2_ac.pretrained "${WM4A_VJEPA_CHECKPOINT}" \
  --framework.future_latent_training.enabled true \
  --framework.future_latent_training.loss_weight 0.01 \
  --framework.future_latent_training.train_time_distribution logitnormal \
  --framework.future_latent_training.shift 5.0 \
  --framework.future_image_training.enabled true \
  --framework.future_image_training.loss_weight 1.0 \
  --framework.future_image_training.train_time_distribution logitnormal \
  --framework.future_image_training.shift 5.0 \
  --framework.future_image_generation.enabled true \
  --framework.future_image_generation.conditioning_mode auto \
  --framework.future_image_generation.max_samples 1 \
  --framework.future_image_generation.height 224 \
  --framework.future_image_generation.width 224 \
  --framework.future_image_generation.num_frames 93 \
  --framework.future_image_generation.num_inference_steps 20 \
  --framework.future_image_generation.guidance_scale 7.0 \
  --framework.future_image_generation.output_type pil \
  --framework.future_image_generation.future_frame_index -1 \
  --framework.future_image_generation.return_full_video false \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 14 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.is_resume true \
  --trainer.pretrained_checkpoint "${resume_checkpoint}" \
  --trainer.max_train_steps 300000 \
  --trainer.save_interval 20000 \
  --trainer.logging_frequency 500 \
  --trainer.eval_interval 500 \
  --trainer.num_warmup_steps 2000 \
  --trainer.learning_rate.base 1e-5 \
  --trainer.learning_rate.action_model 1.5e-5 \
  --trainer.learning_rate.vjepa_predictor 1.5e-5 \
  --trainer.learning_rate.state_projector 1.5e-5 \
  --trainer.learning_rate.action_projector 1.5e-5 \
  --trainer.scheduler_specific_kwargs.min_lr 1e-7 \
  --trainer.optimizer.betas "[0.9,0.999]" \
  --trainer.optimizer.weight_decay 0.1 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project wm4a_anonymous \
  --wandb_entity anonymous \
  2>&1 | tee -a "${output_dir}/train.log"