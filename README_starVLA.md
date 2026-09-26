# StarVLA Environment Installation

For the separate WM4A policy and simulation environments, use
[the WM4A environment instructions](environment/README.md).
The commands below describe the standalone StarVLA environment.

Run from the repository root:

```bash
conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

FlashAttention requires compatible PyTorch, CUDA toolkit, and compiler
versions. Inspect the installed versions before building the extension:

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```
