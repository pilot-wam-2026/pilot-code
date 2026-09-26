# WM4A Anonymous Source Submission

This package contains research source code and configuration examples.
It does not include past training or evaluation logs, experiment results,
checkpoint weights, or a submitter identity.

See [environment/README.md](environment/README.md) for environment setup and
[README_starVLA.md](README_starVLA.md) for the standalone StarVLA installation.
Third-party attribution is explained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Local Resources

Paths under `/path/to/` are placeholders. Supply datasets, model weights,
simulation dependencies, and local paths before running the examples.
The root training launcher accepts these through `WM4A_*` variables and
requires an explicit `CUDA_VISIBLE_DEVICES` allocation.

The release preparation helper requires locally supplied inputs:

```bash
python -m release.prepare \
  --checkpoint /path/to/policy_run/checkpoints/model.pt \
  --config /path/to/policy_run/config.yaml \
  --dataset-statistics /path/to/policy_run/dataset_statistics.json \
  --checkpoint-sha256 CHECKPOINT_SHA256_FROM_A_TRUSTED_SOURCE
```

The release manifest describes model components and an evaluation recipe,
not measured results or a particular previous training run.
Some simulation assets and provenance manifests referenced by the release
utilities are not bundled. These utilities require additional local setup;
this source-only archive is not a complete runnable evaluation environment.
