# RoboCasa-GR1 Evaluation

## Fixed Contract

| Setting | Value |
|---|---|
| Checkpoint | Original 340000-step export |
| Contract identifier | `wm4a_contract_v2_any_physical_step` |
| Latent coordinates | `legacy` |
| Instruction format | `Task: <instruction>.` |
| Tasks | Ordered 24-task list in `release_manifest.json` |
| Full-run episodes | 50 per task |
| Scene seeds | 9000 through 9049 within every task |
| Policy randomness | Seeded per episode and action chunk |
| Action-sampling steps | 20 |
| Predicted horizon / executed chunk | 16 / 12 |
| Maximum physical steps | 720 |
| Environment process | One independent simulator environment per process |
| Parallelism | Four clients per policy server; one or two policy GPUs |
| Rendering | Explicit physical EGL GPU index |
| Success | Native predicate at any executed physical step |
| Stop on success | No |
| IK state | Cleared per environment at episode boundaries |
| Future-image generation | Off by default |

The wrapper evaluates success within each executed action chunk. Testing
only its last step misses transient success. An initially successful reset
is not a substitute for an executed successful transition.

## Setup

Follow the root README to download resources, create the two environments,
run `release.doctor`, verify the checkpoint/assets, and run `release.prepare`.
Preserve the directory layout: the IK wrapper resolves the GR1 URDF relative
to the release root.

Services bind to `127.0.0.1`. Simulation and policy processes therefore run
on the same machine; no public unauthenticated policy endpoint is exposed.

```bash
nvidia-smi
"$POLICY_PYTHON" -m release.evaluate \
  --gpus 0,1 --render-gpu 1 \
  --policy-python "$POLICY_PYTHON" --sim-python "$SIM_PYTHON" \
  --episodes 50 --seed 9000 --workers-per-policy 4 \
  --output /absolute/new/pilot-evaluation
```

The launcher uses round-robin task assignment across policy GPUs, matching
the source evaluation when two GPUs are supplied. It records GPU headroom
and pre-existing process information before launch, checks inherited
allocations, and does not signal any process it did not start.

The default policy cap is 40 GiB. Admission additionally requires 8 GiB
headroom on policy-only GPUs and 16 GiB on the rendering GPU. These checks
are not reservations and cannot guarantee that another process will not
grow later. Reduce concurrency or use another allocated device if needed.

For a smoke test, use `--episodes 1` and one `--task` from the manifest.
Smoke tests are not full benchmark measurements.

## Outputs

```text
gpu_admission.json
server_gpu<N>.log
load_gpu<N>.json
<task_slug>/simulation.log
<task_slug>/rollouts/result.json
<task_slug>/rollouts/episodes.jsonl
<task_slug>/rollouts/action_requests.jsonl
<task_slug>/rollouts/physical_steps/
<task_slug>/rollouts/videos/
summary.json
```

`summary.json` must contain `complete: true` and the expected total episode
count before reporting a full success rate. Failed, interrupted, and
missing tasks are not silently counted as completed failures or successes.
Action records include normalized float32 values, request seeds, and
SHA-256 hashes for replay checks.

Raw runtime logs may contain local paths and process information. Review
them before public redistribution. The bundled `results/` directory
contains only sanitized aggregate numerical evidence.

## Optional Future Visualization

Add `--visualize-every 10` to generate a diagnostic prediction every ten
action chunks. This invokes additional video-model denoising and decoding.
It is off in the full action-only benchmark commands.

The model uses a five-frame construction for a future observation at
offset 16, not a demonstrated continuous long-video rollout. Do not
restore the historical 93-frame visualization setting. Visualization
randomness is isolated from action sampling.

## Interpreting Results

The completed source-snapshot benchmark produced 717/1200 successes
(59.75%). On those same trajectories, block-end-only scoring gives
712/1200, and final-step-only scoring gives 637/1200.
Those latter values are rescoring results, not additional model runs.

The paper's 58.3% is a separately reported experiment. No matched-protocol
proof equating it with this audit is claimed. The source score is also not
a guarantee of bitwise identical trajectories across different GPUs,
CUDA kernels, simulator builds, or a new installation.

## Troubleshooting

**Noise-like predicted images or unexpected action behavior**

Verify the checkpoint hash, `legacy` coordinates, pinned dependencies,
normalization statistics, and five-frame temporal setting first. Do not
change min/max statistics, IK conventions, or prompts as an undocumented fix.

**Missing GR1 assets or PyKDL**

Run `release.unpack_assets` if needed, source
`environment/activate_paths.sh`, and run the simulation preflight.
Use `environment/build_kdl.sh` in a new simulation environment. Do not
fall back silently from end-effector retargeting to joint-space actions.

**EGL initialization failure**

Check the host's NVIDIA EGL installation, `MUJOCO_GL`,
`PYOPENGL_PLATFORM`, and the physical `--render-gpu` index. Use
`EGL_LIBRARY_PATH` only for a verified directory on that machine.
CUDA-visible numbering and EGL physical indexing are not interchangeable.

**Out of memory**

Check existing jobs before starting. Lower `--workers-per-policy`, use a
single allocated GPU where safe, or choose a larger allocation. Never kill
other users' processes or use artificial workloads to reserve memory.

**Resume/configuration mismatch**

`release.prepare` expects this release's exact checkpoint. An old 280000
cache or another run's statistics must not be reused. Use a new `--cache`
directory and explicitly pass its prepared checkpoint to evaluation.
