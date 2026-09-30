# PILOT

**Physical Inference for Latent Optimized Trajectories**

Built on NVIDIA Cosmos.

Code for **Motion Chain of Thought: Disentangling Motion Semantics from Visual
Appearance for Robotic Manipulation**, an anonymous ICLR 2027 submission
currently under review.

[Project website](https://pilot-wam-2026.github.io/) |
[Model and simulation resources](https://huggingface.co/mxk1998/WM4A) |
[Evaluation guide](docs/EVALUATION.md) |
[Training guide](docs/TRAINING.md) |
[Evaluation logs](docs/LOGS.md) |
[Validation status](docs/VALIDATION.md)

**Release status (September 30, 2026):** code and sanitized logs are public.
The original 340000 checkpoint and customized RoboCasa resources are uploaded
to a **private** Hugging Face repository. Authorized range downloads passed;
anonymous downloads are not available. See [access and integrity](docs/DOWNLOAD.md).

## Start Here

| Goal | Entry point |
|---|---|
| Inspect and verify the recorded score without a GPU | [Logs and CPU evidence audit](docs/LOGS.md) |
| Download weights and the exact simulator | [Download, permissions and checksums](docs/DOWNLOAD.md) |
| Build policy/simulation environments | [Linux installation and variables](environment/README.md) |
| Run one episode or all 24 tasks | [Evaluation protocol and outputs](docs/EVALUATION.md) |
| Fine-tune, initialize from pretrained components, or resume | [Training/data guide](docs/TRAINING.md) |
| Understand what is and is not validated | [Validation status](docs/VALIDATION.md) |

![PILOT architecture from the supplied manuscript](docs/assets/method.svg)

## Overview

PILOT separates motion-semantic representation learning from fine-grained
trajectory generation:

1. **World-model context.** A Cosmos-Predict2.5 backbone encodes the current
   RGB observation and instruction.
2. **Motion-CoT and action decoding.** A Perceiver-style action head learns
   motion-semantic query tokens and predicts action chunks with flow matching.
   Asymmetric attention prevents noised action tokens from entering the
   query pathway.
3. **Representational Deduction.** During training, a Causal Dynamics Engine
   predicts future VJEPA2-AC representations from current features, motion
   queries, and robot state. A frozen encoder supplies the targets.
4. **Auxiliary future-frame supervision.** A shared world-model generation
   branch adds visual supervision. Future-image decoding is optional at
   inference and is **not an input required by the action head**.

The manuscript covers multiple benchmarks. **This release is specifically
the RoboCasa-GR1 340000-step checkpoint and its training/evaluation path.**
It does not bundle LIBERO, LIBERO-Plus, real-robot checkpoints, private
demonstration datasets, or the later training-validation export.

## Release At A Glance

| Item | This artifact |
|---|---|
| Checkpoint | `checkpoints/steps_340000_pytorch_model.pt` |
| Architecture identifier | `CosmoPredict25PerceiverVJEPA2AC` |
| Stored state entries | 2596 tensor entries |
| Checkpoint size | 23,913,608,021 bytes |
| Input | RGB observation, instruction, proprioceptive state |
| State/action interface | Padded 64-D state, 32-D action, horizon 16 |
| Executed action chunk | 12 steps |
| Action integration | 20 steps |
| Latent coordinates | Explicit `legacy` mode |
| Optional generated clip | 5 frames, not the old 93-frame diagnostic |
| Execution platform | Linux x86_64, NVIDIA CUDA and EGL |
| macOS | Download, inspect, and archive only; no native execution claim |

The internal `starVLA` package and architecture identifiers are retained for
checkpoint compatibility. They are not alternative release names.

## Download

### Source Code

```bash
git clone https://github.com/pilot-wam-2026/pilot-code.git
cd pilot-code
python3 -m release.audit_logs  # No Torch, GPU, checkpoint or HF login needed.
```

### Complete Model And Simulator Bundle

Download the resource repository into a separate directory:

```bash
python -m pip install huggingface_hub
hf auth login  # Required while the resource repository is private.
hf download mxk1998/WM4A --local-dir PILOT
cd PILOT
```

If `archives/simulation_assets.tar.gz` is present, unpack and verify it:

```bash
python -m release.unpack_assets
python -m release.verify --assets
```

The Hugging Face repository ID remains **`mxk1998/WM4A`**; its project name
and released artifact are **PILOT**. Repository visibility is not changed
by the release scripts. Login requires an account already granted access.
For downloading only weights/assets into a GitHub checkout without replacing
its code or documentation, use [Download Option B](docs/DOWNLOAD.md#option-b-github-code-plus-only-the-large-resources).

Checkpoint SHA-256:

```text
34bf115c1c99a33b12c79e1dca20e596cd510f006060687c2f145fecea58e60e
```

## Installation

Policy and simulation use **separate Python 3.10 environments**. Do not
replace the supplied RoboCasa/RoboSuite implementations with an arbitrary
newer `pip install robocasa`.

```bash
bash environment/create_linux_envs.sh /absolute/new/pilot-envs
export POLICY_PYTHON=/absolute/new/pilot-envs/policy/bin/python
export SIM_PYTHON=/absolute/new/pilot-envs/simulation/bin/python
source environment/activate_paths.sh

"$POLICY_PYTHON" -m release.doctor --role policy
"$SIM_PYTHON" -m release.doctor --role simulation
"$POLICY_PYTHON" -m release.prepare
```

`release.prepare` verifies the checkpoint and reconstructs the frozen and
trainable component files required by the original constructors. It uses
the **supplied checkpoint**, not newly downloaded backbone weights, and
does not convert its tensor values or dtypes. The additional files go to
`.runtime/backbones`; the original checkpoint remains unchanged.

Allow at least 120 GiB of free storage for environments, assets, weights,
materialized components, and outputs; 128 GiB host RAM is recommended.
The validated source evaluation used H200 GPUs. Smaller-device operation
and a fresh-machine installation of the portable package require separate
validation. See [environment/README.md](environment/README.md).

## Evaluate

Inspect your assigned resources first:

```bash
nvidia-smi
```

One-task smoke run:

```bash
"$POLICY_PYTHON" -m release.evaluate \
  --gpus 0 --render-gpu 0 \
  --policy-python "$POLICY_PYTHON" --sim-python "$SIM_PYTHON" \
  --episodes 1 --seed 9000 \
  --task gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
  --output /absolute/new/pilot-smoke
```

Full 24-task, 1200-episode protocol with two allocated GPUs:

```bash
"$POLICY_PYTHON" -m release.evaluate \
  --gpus 0,1 --render-gpu 1 \
  --policy-python "$POLICY_PYTHON" --sim-python "$SIM_PYTHON" \
  --episodes 50 --seed 9000 --workers-per-policy 4 \
  --output /absolute/new/pilot-evaluation
```

GPU IDs above are examples, **not reservations**. The launcher checks
inherited allocations and current headroom, records existing processes,
binds policy services to loopback, and stops only its own child processes.
Use a new output directory for each run. The full protocol, expected files,
scoring semantics, and troubleshooting are in
[docs/EVALUATION.md](docs/EVALUATION.md).

## Results And Scope

| Measurement | Success | Meaning |
|---|---:|---|
| Manuscript RoboCasa-GR1 table | 58.3% | Reported manuscript experiment |
| Original 340000 checkpoint, repaired-protocol source evaluation | **717/1200 = 59.75%** | 24 tasks, 50 episodes each, seeds 9000-9049, any-physical-step success |
| Same trajectories, block-end-only rescoring | 712/1200 = 59.33% | Not an independent rerun |
| Same trajectories, final-step-only rescoring | 637/1200 = 53.08% | Not an independent rerun |

**These measurements are not interchangeable.** The source evaluation is
not a reproduction of every number in the paper, nor proof that a new
installation of this portable package has already reproduced the score.
Per-task counts and the exact contract are in
[results/robocasa_340000.json](results/robocasa_340000.json).

The validated source run audited all 1200 episodes, their initial seeds
and IK-cache resets, and all 72000 action-request seeds and hashes. It did
not terminate successful episodes early.

The [published evaluation archive](results/evaluation_340000_logs.tar.gz)
contains all 24 sanitized simulator logs and task results, 1200 episode
diagnostics, and a source provenance report. Run `python3 -m release.audit_logs`
to verify its hashes and recorded counts without a GPU. The archive does
not include the raw per-request action streams; the source provenance check
and the public consistency check are different verification levels.
See [logs, redactions and omissions](docs/LOGS.md) and the
[historical training metrics](results/training_metrics_340000.csv).

## Train Or Fine-Tune

Install the additional training dependencies:

```bash
"$POLICY_PYTHON" -m pip install -r environment/training.requirements.txt
```

Example **weight-only fine-tuning** from the supplied checkpoint:

```bash
export CUDA_VISIBLE_DEVICES=0
export PILOT_DATA_ROOT=/absolute/path/to/converted-datasets
export PILOT_OUTPUT_ROOT=/absolute/new/training-runs
bash run_training.sh --trainer.max_train_steps 1000 \
  --trainer.save_interval 500 --trainer.logging_frequency 50
```

This initializes a **new optimizer**. It is not recovery of the historical
340000-step optimizer. A full-state resume uses a `*_training_state`
directory produced by the repaired trainer, not the `.pt` weight export.
See [docs/TRAINING.md](docs/TRAINING.md) for initialization, data layout,
full-state resume, and the distinction between the archived recipe and
the conservative single-GPU validation configuration.

## Reproducibility And Known Limits

- **Latent scale is checkpoint-specific.** Use `legacy` for this artifact.
  A Diffusers attribute-semantic change previously caused training and
  generation to disagree on coordinates; selecting a mode by package
  version or guessing from a filename is unsafe.
- **Five-frame visualization is not long-video prediction.** The auxiliary
  target is a future observation at offset 16. A 93-frame rollout exceeds
  the supervised setting.
- **Future-image quality remains limited.** On 48 online-rollout held-out
  pairs, matched coordinates improve over the wrong scale on all pairs,
  but outperform copying the current frame on only 13/48 by full-image MSE.
  These are not unseen-expert-data or long-video generalization claims.
- **Training validation is bounded.** Full-model 1000-update stability,
  fresh-process exact resume, and export/reload were checked in the source
  environment at one rank, batch 1, workers 0, accumulation 1. This is not
  hundreds-of-thousands-step convergence or multi-rank validation.
- **Manuscript/configuration differences are disclosed.** The archived
  configuration of this checkpoint uses RD loss weight `0.1`; the supplied
  manuscript appendix describes `0.01`. This release follows the artifact's
  archived configuration and does not silently claim the recipes coincide.
- Code anonymization does not anonymize repository ownership or existing
  Git history. Third-party legal attribution is intentionally retained.

## Repository Map

```text
configs/             340000-compatible configuration and normalization statistics
starVLA/             Model, required backbone helpers, dataset and training code
deployment/          WebSocket policy transport
examples/            RoboCasa-GR1 interface, IK, seeding and physical-step scoring
release/             Preparation, preflight, evaluation and verification commands
environment/         Pinned environments, KDL build and portable environment variables
results/             Evaluation logs, 1200-episode audit and historical training metrics
docs/                Evaluation, training and release documentation
checkpoints/         Full checkpoint (Hugging Face/local bundle only)
third_party/         Simulator sources/assets (Hugging Face/local bundle only)
runtime_assets/      GR1 retarget URDF (Hugging Face/local bundle only)
```

Unsanitized logs, private paths, credentials, dataset contents, and unrelated
experiments are excluded. Published log redactions and omitted fields are
documented rather than represented as untouched raw evidence.

## Attribution And License

This implementation builds on StarVLA, NVIDIA Cosmos and GR00T/RoboCasa,
RoboSuite, VJEPA2, and Orocos KDL/PyKDL. Original copyright notices and
licenses are retained. Model weights, code, and robot/object assets are
not assumed to share one blanket license. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [LICENSE.md](LICENSE.md).

## Citation

```bibtex
@unpublished{pilot2026,
  title = {Motion Chain of Thought: Disentangling Motion Semantics from
           Visual Appearance for Robotic Manipulation},
  author = {{Anonymous Authors}},
  year = {2026},
  note = {ICLR 2027 submission, under review}
}
```
