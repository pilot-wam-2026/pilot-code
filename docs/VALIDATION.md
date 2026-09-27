# Release Validation

## Checked On September 27, 2026

The local portable source passed 19 CPU tests using Python 3.10.21 and
PyTorch 2.8.0 on macOS. The tests cover:

- The two native Diffusers latent conventions and explicit checkpoint coordinates.
- Conflicting checkpoint/configuration metadata and unknown coordinate rejection.
- The five-frame supervised horizon and explicit temporal-extrapolation opt-in.
- Per-request seeded sampling without changing the surrounding Torch RNG.
- Evaluation-mode and Python/NumPy/Torch RNG restoration, including exceptions.
- The selected checkpoint metadata, 24-task list, and published numerical counts.
- GPU allocation/headroom admission and owned-process cleanup.
- Complete task-result protocol validation, including seed and request count.
- Rejection of same-size corruption in materialized component caches.
- Safe archive paths, parent symlinks, overwrite refusal, and executable modes.
- Python source syntax and the absence of private machine bindings in launchers.
- Separate Hub ignore rules that retain model weights and simulator archives.

The selected framework, policy transport, resume validator, and
`Cosmos2_5_PredictBasePipeline` also imported successfully in the local CPU
environment. No complete model was constructed by that check.

The full 23,913,608,021-byte checkpoint passed SHA-256 verification against
the source file. All 45,853 simulator/robot resource files matched the
source hashes before the two documented local path relocations.
`release.prepare` then completed on the local CPU and reconstructed the
four constructor component files directly from the selected checkpoint.
These checks do not constitute a GPU model forward pass.

The policy, simulation, and training requirement sets resolved against
PyPI metadata for a Linux x86_64 / Python 3.10 target. Resolving the
CUDA-specific wheel indexes was blocked by a local network timeout.
Dependency resolution is not installation or runtime verification.

Whitespace cleanup was checked against each file's Python AST. It did
not change the parsed computation.

## Not Established By These Checks

- A fresh Linux/CUDA installation reproducing policy inference and EGL rendering.
- The portable bundle reproducing the source-environment 1200-episode score.
- A full-model forward/backward pass on macOS, MPS, or smaller GPUs.
- Multi-rank exact resume or hundreds-of-thousands-update training convergence.
- General-purpose video generation or superiority to a copy-current-frame baseline.
- Unrestricted redistribution rights or production safety certification for every asset.

The completed source-environment benchmark and bounded training checks
are described separately in the evaluation/training guides. They must
not be relabeled as fresh-install validation of this portable package.

## Rerun

After installing the policy environment, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=-1 "$POLICY_PYTHON" -m unittest discover -s tests -v
```

Without Torch, six tensor tests are explicitly skipped rather than
reported as passed. Resource and checkpoint hashes must also be checked
using `python -m release.verify --assets` after the bundle is downloaded.
