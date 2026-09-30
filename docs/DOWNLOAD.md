# Download, Access And Integrity

## Resource Locations

| Resource | Location | Access |
|---|---|---|
| Project page | https://pilot-wam-2026.github.io/ | Public |
| Training/evaluation code and evidence | https://github.com/pilot-wam-2026/pilot-code | Public; no model download needed to inspect logs |
| Original 340000 weights, simulator and framework | https://huggingface.co/mxk1998/WM4A | **Private; an authorized account is required** |

The resource repository ID is `mxk1998/WM4A`; PILOT is the release name.
An HF login alone does not grant access to a private repository.
There is no promise of anonymous access or automatic approval.

On **September 30, 2026**, both large artifacts passed authenticated
4096-byte range downloads (HTTP 206), and published LFS sizes/hashes were
verified. An earlier check that day returned
`403: Private repository storage limit reached`. The later successful
sample is not a full re-download, a future availability guarantee, or a
change in repository visibility.

## Option A: Complete Bundle

Use a new destination directory and an account with repository access:

```bash
python -m pip install huggingface_hub
hf auth login
hf download mxk1998/WM4A --local-dir PILOT
cd PILOT
```

Authenticate interactively or through your platform's protected secret
mechanism. Do not put tokens in shell commands, source code, or issue logs.
Do not use `git clone` to fetch this large resource bundle unless you have
deliberately configured its large-file tooling.

## Option B: GitHub Code Plus Only The Large Resources

This avoids overwriting newer GitHub documentation with a resource snapshot:

```bash
git clone https://github.com/pilot-wam-2026/pilot-code.git
cd pilot-code
python -m pip install huggingface_hub
hf auth login
hf download mxk1998/WM4A \
  --revision a25598f06ce07b1d7a284b168244a48e6b5dc8ca \
  --include "checkpoints/*" "archives/*" "environment/resource_manifest.json" \
  --local-dir .
```

That immutable resource revision contains the same weights and simulator
as the documentation update. Keep code, `model_metadata/`, and
`configs/dataset_statistics.json` from this release together.
To resume an interrupted download, run the same command with the same
destination rather than deleting completed files.

## Unpack And Check

From the bundle or supplemented code root, with Python 3.10:

```bash
python -m release.unpack_assets
python -m release.verify --assets
python -m release.audit_logs
```

The unpacker refuses unsafe paths, symlinks, and conflicting existing files.
It checks resource hashes. `release.verify --assets` hashes the original
checkpoint and all unpacked resources; expect substantial disk I/O.
The evidence audit needs only the Python standard library, not CUDA, Torch,
the checkpoint, or simulator assets.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `checkpoints/steps_340000_pytorch_model.pt` | 23,913,608,021 | `34bf115c1c99a33b12c79e1dca20e596cd510f006060687c2f145fecea58e60e` |
| `archives/simulation_assets.tar.gz` | 4,526,090,391 | `411c6fcedb435e0b9bbfd2df3ece274a4bdd94838dcd0bed3fb5c4d43ca2a0a7` |

The environment archive contains **45,853 files** and expands to
9,577,418,261 bytes. It includes customized RoboCasa/RoboSuite source,
robot/object assets, retargeting URDF, and KDL sources, not a credentials
archive or a ready-to-run copy of another machine's activated environment.
Linux environments must still be built using
[the environment guide](../environment/README.md).

Plan for at least 120 GiB free disk, including the component cache, two
environments and outputs. Do not unpack large resources into Git history.

## Access Troubleshooting

| Symptom | What to check |
|---|---|
| Private repository / 401 / repository not found | Confirm the exact repo ID and that the logged-in account has access. |
| `Private repository storage limit reached` | Account-side storage restriction, not a model corruption error. The owner must resolve the quota. Re-uploading weights is not a fix. |
| Incomplete download / SHA mismatch | Resume the same download, then re-run verification. Never use a partially downloaded checkpoint. |
| Missing `third_party` or URDF | Run the archive unpacker from the bundle root. |
| macOS model execution fails | Execution is Linux x86_64/CUDA/EGL; a Mac can store and audit the bundle. |

Do not bypass access restrictions or assume that deleting a current file
pointer frees all historical storage. No quota upgrade, history rewrite,
or visibility change is performed by the supplied tools.
