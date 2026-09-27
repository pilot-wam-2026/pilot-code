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
bash run_training.sh --trainer.max_train_steps 1000
```

The original 340000 checkpoint initializes the full model, while the
optimizer and update counter start a **new experiment**. A `.pt` file
contains weights, not a historical optimizer/RNG/data cursor.

For initialization from the original pretrained backbones instead,
provide their locally available, licensed initialization files:

```bash
export PILOT_FROM_PRETRAINED=1
export PILOT_COSMOS_ROOT=/absolute/path/to/original/Cosmos-Predict2.5-2B-Post-Trained
export PILOT_VJEPA_CHECKPOINT=/absolute/path/to/original/vjepa2-ac-vitg.pt
bash run_training.sh
```

Do not point these variables at the components materialized from 340000
and describe that as training from the original pretrained initialization.
No unattended long-running training job is started by downloading the release.

## Full-State Resume

The repaired trainer writes separate `steps_*_training_state` directories
containing model/optimizer/scheduler/RNG state, data progress, configuration,
and a completion manifest.

```bash
export PILOT_RESUME_STATE=/absolute/path/to/previous-run/checkpoints/steps_500_training_state
export PILOT_OUTPUT_ROOT=/absolute/new/resumed-run-root
bash run_training.sh
```

The launcher uses the saved full configuration. Do not pass a weight-only
initialization at the same time. Incomplete saves, changed training semantics,
and changed dependency versions are rejected. Exact data replay is supported
only by the validated single-rank/stateful-loader configuration; unsupported
worker and multi-rank settings must not be described as exact continuation.

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
