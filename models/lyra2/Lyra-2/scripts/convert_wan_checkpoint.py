# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the public Wan 2.1 safetensor shards to Lyra training initialization weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing diffusion_pytorch_model-*-of-*.safetensors",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination .pth file")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output file")
    return parser.parse_args()

def convert(input_dir: Path, output: Path, force: bool = False) -> None:
    shards = sorted(input_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No Wan diffusion checkpoint shards found in {input_dir}")
    if output.suffix != ".pth":
        raise ValueError(f"Output must end in .pth: {output}")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output} (pass --force to overwrite it)")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shards:
        print(f"Loading {shard.name}", flush=True)
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in state_dict:
                    raise ValueError(f"Duplicate checkpoint key: {key}")
                tensor = checkpoint.get_tensor(key)
                if key == "patch_embedding.weight" and tensor.ndim == 5:
                    # Public Wan uses Conv3d patchification. This Lyra experiment
                    # performs the identical projection with a Linear layer over
                    # patches flattened in (channel, time, height, width) order.
                    tensor = tensor.flatten(1)
                if torch.is_floating_point(tensor):
                    tensor = tensor.to(torch.bfloat16)
                state_dict[key] = tensor.contiguous()

    required_keys = {"patch_embedding.weight", "head.head.weight"}
    missing = required_keys.difference(state_dict)
    if missing:
        raise ValueError(f"Input does not look like a Wan I2V diffusion checkpoint; missing {sorted(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(state_dict)} tensors to {output}", flush=True)
    torch.save(state_dict, output)

def main() -> None:
    args = parse_args()
    convert(args.input_dir.expanduser(), args.output.expanduser(), args.force)

if __name__ == "__main__":
    main()
