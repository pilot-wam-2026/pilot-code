"""Serve a supplied policy with a loopback-only, explicitly scoped GPU."""

import argparse
import json
import logging
import os
from pathlib import Path
import random
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--max-vram-gib", type=float, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before launching the policy.")
    import numpy as np
    import torch
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from starVLA.model.framework.base_framework import baseframework

    torch.set_num_threads(1)
    total_memory = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(args.max_vram_gib * 1024**3 / total_memory, 0)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    started = time.time()
    policy = baseframework.from_pretrained(args.checkpoint)
    policy = policy.to(torch.bfloat16).to("cuda").eval()
    torch.cuda.synchronize()
    evidence = {
        "strict_checkpoint_load": True,
        "framework_class": f"{type(policy).__module__}.{type(policy).__name__}",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "state_dict_keys": len(policy.state_dict()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "allocated_bytes_after_load": torch.cuda.memory_allocated(),
        "peak_allocated_bytes_after_load": torch.cuda.max_memory_allocated(),
        "load_seconds": time.time() - started,
        "seed": args.seed,
    }
    args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence), flush=True)
    WebsocketPolicyServer(
        policy=policy, host="127.0.0.1", port=args.port,
        idle_timeout=-1, metadata={"env": "robocasa", "release": True},
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
