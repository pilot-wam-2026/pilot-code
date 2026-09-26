# SimplerEnv Environment Installation

Install the base environment using the
[SimplerEnv installation instructions](https://github.com/simpler-env/SimplerEnv).

Install the additional dependencies in that environment:

```bash
conda activate simpler_env
pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4
```

Provide the Vulkan runtime libraries required by the simulator.
Keep the SimplerEnv environment separate from the StarVLA environment.
