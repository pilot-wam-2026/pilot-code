# DINOv3 Environment Installation

Use a separate Linux environment for the bundled DINOv3 package.
Its package metadata requires Python 3.11 or later; the environment
definition and requirement files specify its remaining dependencies.

From this directory:

```bash
micromamba env create -f conda.yaml
micromamba activate dinov3
```

The dependency list is in `requirements.txt`; development dependencies
are in `requirements-dev.txt`. Keep this environment separate from the
WM4A Python 3.10 environments.

Package licensing is provided in [LICENSE.md](LICENSE.md).
