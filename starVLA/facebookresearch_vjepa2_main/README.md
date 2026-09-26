# V-JEPA Environment Installation

Use a separate environment for the bundled standalone V-JEPA package.
Run from this directory:

```bash
conda create -n vjepa2-312 python=3.12
conda activate vjepa2-312
pip install .
```

For an editable installation, use `pip install -e .` instead.
Dependencies are listed in `requirements.txt` and the package setup files.
Keep this environment separate from the WM4A Python 3.10 environments.

Package licensing is provided in [LICENSE](LICENSE).
