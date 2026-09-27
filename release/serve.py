"""Loopback-only PILOT policy service, with a fixed checkpoint coordinate system."""
import argparse
import json
import logging
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-vram-gib", type=float, default=40)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise ValueError("Expose exactly one explicitly allocated GPU per policy process.")
    import torch
    from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
    from starVLA.model.framework.base_framework import baseframework

    torch.set_num_threads(4)
    total = torch.cuda.get_device_properties(0).total_memory
    limit = args.max_vram_gib * 1024**3
    if not 0 < limit < total:
        raise ValueError("The policy memory cap must be below device capacity.")
    torch.cuda.set_per_process_memory_fraction(limit / total)
    policy = baseframework.from_pretrained(args.checkpoint, latent_normalization="legacy")
    policy = policy.to(torch.bfloat16).cuda().eval()
    metadata = {
        "protocol": "wm4a_contract_v2_any_physical_step",
        "checkpoint": args.checkpoint, "latent_normalization": "legacy",
        "state_keys": len(policy.state_dict()), "num_frames": 5,
        "action_steps": policy.action_model.num_inference_timesteps,
        "physical_policy_gpu": visible,
        "native_convention": policy.backbone.native_latent_convention,
        "pid": os.getpid(),
    }
    if metadata["action_steps"] != 20:
        raise ValueError("The release protocol requires 20 action sampling steps.")
    args.output.write_text(json.dumps(metadata, indent=2) + "\n")
    WebsocketPolicyServer(policy, host="127.0.0.1", port=args.port,
                          idle_timeout=-1, metadata=metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
