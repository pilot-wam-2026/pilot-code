# Training And Resume

## Scope

This release retains the model and data path for the selected
`CosmoPredict25PerceiverVJEPA2AC` RoboCasa-GR1 artifact. Two ancestor classes
remain as implementation helpers, but the public factory rejects other
architectures. Unrelated benchmark launchers and models are excluded.

`configs/pilot_340000.yaml` is derived from the selected run's saved
configuration, with private paths parameterized and explicit latent/temporal
contracts added. It is not a claim that all paper experiments use this
one configuration.

## Data

Training demonstrations are not bundled. Supply a legally obtained,
converted LeRobot-format 24-task RoboCasa-GR1 mixture. The ordered paths
are defined in `starVLA/dataloader/gr00t_lerobot/mixtures.py`.

The interface uses the ego-view RGB stream, current robot state,
`annotation.human.coarse_action`, 16 future action steps, and a future
observation at offset 16. The GR1 modality configuration and transforms
are retained with min/max normalization, padded 64-D state and 32-D
action tensors.

Do not substitute a merged dataset with different episode order,
normalization statistics, image decoding, or EE/joint conventions and
call it an exact resume. The release includes the selected checkpoint's
`configs/dataset_statistics.json` for evaluation, not a substitute for
training data.

## Artifact Configuration

### Dataset Layout Check

`PILOT_DATA_ROOT` is the directory **above**
`PhysicalAI-Robotics-GR00T-Teleop-Sim`, not an individual task directory:

```text
$PILOT_DATA_ROOT/
  PhysicalAI-Robotics-GR00T-Teleop-Sim/
    LeRobot_eepose/
      gr1_unified.PnPBottleToCabinetClose_ee/
        meta/modality.json
        meta/info.json
        meta/episodes.jsonl
        meta/tasks.jsonl
        data/<chunk>/<episode>.parquet
        videos/...
      ... (all 24 task directories)
```

The loader reads modality mappings and video paths from metadata; directory
names alone do not validate action semantics. This lightweight check lists
all expected tasks and rejects missing roots/required metadata:

```bash
"$POLICY_PYTHON" - <<'PY'
import os
from pathlib import Path
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
root = Path(os.environ["PILOT_DATA_ROOT"])
missing = []
for name, weight, embodiment in DATASET_NAMED_MIXTURES["robocasa_teleop_ee"]:
    path = root / name
    print(name)
    for relative in ("meta/modality.json", "meta/info.json"):
        if not (path / relative).is_file():
            missing.append(str(path / relative))
if missing:
    raise SystemExit("Missing dataset files:\n" + "\n".join(missing))
print("24 dataset roots found; decoding and EE/state semantics still need validation.")
PY
```

Do not treat unconverted joint-space demonstrations as the required EE-pose
data. This release does not include a verified raw-data conversion pipeline,
dataset redistribution permission, or all training demonstrations.

### Selected Hyperparameters

| Parameter | Archived selected-run value |
|---|---|
| Seed | 42 |
| Planned training budget | 400000 updates; this release selects update 340000 |
| Per-device batch | 14 in the archived run configuration |
| Action steps / horizon | 20 / 16 |
| Motion query count | 64 |
| Base learning rate | `1e-5` |
| Action/CDE/projector learning rates | `5e-5` |
| Optimizer | AdamW, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0.1` |
| Warmup | 2000 updates |
| LR schedule | Cosine with minimum LR `2e-7` |
| Action-loss weight | 1 |
| Future-frame-loss weight | 1 |
| RD/future-representation-loss weight | **0.1** |
| Gradient accumulation / clipping | 1 / 1.0 |

The supplied manuscript appendix states RD weight `0.01`. The actual
archived configuration for the selected checkpoint says `0.1`; the
release does not silently replace one with the other.

The historical visualization requested 93 frames. The repaired artifact
uses **5 frames and one conditioning latent frame**, matching its
supervised temporal construction. The `legacy` coordinate system is
mandatory for the released weights.

## Conservative Fine-Tuning

The launcher defaults to one rank, batch 1, zero workers, accumulation 1,
native ZeRO-2, deterministic mode, and disabled external W&B reporting.
These are safety/replay-validation settings, **not the historical batch-14
training run**.

```bash
export POLICY_PYTHON=/absolute/path/to/policy/bin/python
"$POLICY_PYTHON" -m pip install -r environment/training.requirements.txt
"$POLICY_PYTHON" -m release.prepare
export CUDA_VISIBLE_DEVICES=0
export PILOT_DATA_ROOT=/absolute/path/to/converted-datasets
export PILOT_OUTPUT_ROOT=/absolute/new/pilot-training
unset PILOT_RESUME_STATE PILOT_FROM_PRETRAINED
export PILOT_RUN_ID=pilot_finetune_1000
set -o pipefail
bash run_training.sh --trainer.max_train_steps 1000 \
  --trainer.save_interval 500 --trainer.logging_frequency 50 \
  2>&1 | tee "$PILOT_OUTPUT_ROOT.console.log"
```

The original 340000 checkpoint initializes the full model, while the
optimizer and update counter start a **new experiment**. A `.pt` file
contains weights, not a historical optimizer/RNG/data cursor.

For initialization from the original pretrained backbones instead,
provide their locally available, licensed initialization files:

```bash
export PILOT_FROM_PRETRAINED=1
unset PILOT_RESUME_STATE
export PILOT_COSMOS_ROOT=/absolute/path/to/original/Cosmos-Predict2.5-2B-Post-Trained
export PILOT_VJEPA_CHECKPOINT=/absolute/path/to/original/vjepa2-ac-vitg.pt
bash run_training.sh
```

Do not point these variables at the components materialized from 340000
and describe that as training from the original pretrained initialization.
No unattended long-running training job is started by downloading the release.

## Initialization Modes

| Mode | Input | Optimizer/counter |
|---|---|---|
| Default fine-tuning | Prepared 340000 weight export | New optimizer; counter starts at zero |
| Original-pretrained initialization | Licensed original Cosmos and VJEPA files, `PILOT_FROM_PRETRAINED=1` | New optimizer; counter starts at zero |
| Exact repaired-trainer resume | Complete `*_training_state` directory | Restore optimizer, scheduler, RNG and data progress |

Choose exactly one mode and use a new `PILOT_RUN_ID`/output location.
These examples do not run the historical 340000 updates again.
The example keeps the archived 2000-step warmup, so a 1000-update smoke run
stays within warmup. A different schedule is a new training experiment,
not an exact continuation of the archived run.

## Full-State Resume

The repaired trainer writes separate `steps_*_training_state` directories
containing model/optimizer/scheduler/RNG state, data progress, configuration,
and a completion manifest.

```bash
export PILOT_RESUME_STATE=/absolute/path/to/previous-run/checkpoints/steps_500_training_state
export PILOT_OUTPUT_ROOT=/absolute/new/resumed-run-root
unset PILOT_FROM_PRETRAINED PILOT_INITIAL_CHECKPOINT
export PILOT_RUN_ID=pilot_resume
bash run_training.sh
```

The launcher uses the saved full configuration. Do not pass a weight-only
initialization at the same time. Incomplete saves, changed training semantics,
and changed dependency versions are rejected. Exact data replay is supported
only by the validated single-rank/stateful-loader configuration; unsupported
worker and multi-rank settings must not be described as exact continuation.

## Outputs And Checkpoints

The run lives at `$PILOT_OUTPUT_ROOT/$PILOT_RUN_ID`. It retains the launcher,
configuration, normalization statistics, weight exports and separate
`steps_*_training_state` directories. Full-state saves include a completion
manifest; do not copy or resume a directory while it is still being written.
The example explicitly saves at updates 500 and 1000. The archived default
`save_interval=10000` would not produce a scheduled checkpoint in a
1000-update smoke run.

Console output is captured separately in the example above. Choose an
existing writable parent for that console-log path and a fresh filename
for every experiment. W&B reporting is disabled by the launcher.
For the source checkpoint's original training metrics through update
340000, see [the published CSV and extraction scope](LOGS.md#training-metrics).

The strict `release.prepare`/`release.verify` commands intentionally target
the original benchmark checkpoint hash. Do not overwrite its file or
manifest with newly trained weights. Evaluation can accept an explicit
`--checkpoint /absolute/run/checkpoints/steps_N_pytorch_model.pt` from a
compatible repaired run, with its colocated configuration/statistics.
A new checkpoint requires its own smoke test and benchmark; it does not
inherit the released model's 59.75% score.

## Validation Limits

The source environment completed 1000 unique updates of the full model,
with 2,525,075,872 trainable parameters, using the real 24-task data mixture.
Fresh-process resume reproduced updates 501-505 exactly across 2596 model
entries, optimizer, scheduler, RNG, sampled data, and metrics.

That test used batch 1, workers 0, accumulation 1, and bounded 256-update
epoch windows. It did not traverse the entire corpus, restore the original
historical optimizer, prove multi-rank behavior, or validate hundreds of
thousands of updates. Its new export is not the model with the 59.75%
benchmark score and is not included in this release.
