#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${CUDA_VISIBLE_DEVICES:?Set the allocated physical GPU ID explicitly.}"
: "${PILOT_DATA_ROOT:?Set the root of the converted 24-task LeRobot dataset.}"
: "${PILOT_OUTPUT_ROOT:?Choose a NEW output directory outside the release.}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* || "$CUDA_VISIBLE_DEVICES" == "-1" ]]; then
  echo "The exact-resume release recipe supports one GPU, batch 1, workers 0, accumulation 1." >&2
  exit 2
fi
export PILOT_ROOT="$ROOT"
export PILOT_COSMOS_ROOT="${PILOT_COSMOS_ROOT:-$ROOT/.runtime/backbones/Cosmos-Predict2.5-2B-Post-Trained}"
export PILOT_VJEPA_CHECKPOINT="${PILOT_VJEPA_CHECKPOINT:-$ROOT/.runtime/backbones/vjepa2-ac.pt}"
export PYTHONPATH="$ROOT" PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled CUBLAS_WORKSPACE_CONFIG=:4096:8
PYTHON="${POLICY_PYTHON:-python}"
CONFIG="$ROOT/configs/pilot_340000.yaml"
INIT="${PILOT_INITIAL_CHECKPOINT:-$ROOT/.runtime/run/checkpoints/model.pt}"
RUN_ID="${PILOT_RUN_ID:-pilot_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ -d "$PILOT_DATA_ROOT" && -d "$PILOT_COSMOS_ROOT" && -f "$PILOT_VJEPA_CHECKPOINT" ]]
[[ ! -e "$PILOT_OUTPUT_ROOT/$RUN_ID" ]] || { echo "Output run already exists." >&2; exit 2; }
ARGS=(--trainer.is_resume false --trainer.pretrained_checkpoint "$INIT")
if [[ -n "${PILOT_RESUME_STATE:-}" ]]; then
  CONFIG="$PILOT_RESUME_STATE/config.full.yaml"
  [[ -f "$CONFIG" ]]
  ARGS=(--trainer.is_resume false --trainer.pretrained_checkpoint null
        --trainer.resume_training_state "$PILOT_RESUME_STATE")
elif [[ "${PILOT_FROM_PRETRAINED:-0}" == 1 ]]; then
  # Supply ORIGINAL pretrained Cosmos/VJEPA directories, not materialized 340000 components.
  ARGS=(--trainer.is_resume false --trainer.pretrained_checkpoint null)
else
  [[ -f "$INIT" ]] || { echo "Run release.prepare before weight-only fine-tuning." >&2; exit 2; }
fi
mkdir -p "$PILOT_OUTPUT_ROOT/$RUN_ID"
cp "$0" "$PILOT_OUTPUT_ROOT/$RUN_ID/run_training.sh"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
  "$ROOT/starVLA/training/train_starvla.py" --config_yaml "$CONFIG" \
  --run_root_dir "$PILOT_OUTPUT_ROOT" --run_id "$RUN_ID" \
  --datasets.vla_data.per_device_batch_size 1 \
  --datasets.vla_data.num_workers 0 \
  --trainer.gradient_accumulation_steps 1 \
  "${ARGS[@]}" "$@"
