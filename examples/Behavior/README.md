<!-- # Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
-->
# BEHAVIOR Environment Installation

Create the BEHAVIOR environment separately from the StarVLA environment:

```bash
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
conda create -n behavior python=3.10 -y
conda activate behavior
cd BEHAVIOR-1K
pip install "setuptools<=79"
./setup.sh --omnigibson --bddl --joylo --dataset
conda install -c conda-forge libglu
pip install rich omegaconf hydra-core msgpack websockets av pandas google-auth
```

Install the communication dependency in the StarVLA environment as well:

```bash
conda activate starVLA
pip install websockets
```
