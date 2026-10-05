"""Direct A/B: base policy (CausVid) vs a trained `new` checkpoint, same seeds.

Samples k rollouts per scene under the `ref` adapter and under `new` (loaded from
--ckpt), writes both to a rollout store, then the reward CLI scores everything.

    python -m rl.inference.ab_eval --cond-root ... --scenes s1 s2 --ckpt nft_new_stepNNNN.safetensors \
        --out /tmp/ab --k 4 --device cuda:0
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.loop import adapters
from rl.data.cond_bundle import load_cond_bundle
from rl.loop.rollout_store import RolloutStoreWriter
from rl.loop.sampling import decode_latents, sample_rollout
from rl.loop.schedule import build_rl_schedule

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--base-seed", type=int, default=424242,
                    help="seed j uses base_seed+j; pass the loop's --val-base-seed (777000) "
                         "to reproduce the exact clips the held-out validation scored")
    ap.add_argument("--arms", nargs="+", default=["base", "new"],
                    help="which arms to sample; base rollouts are seed-identical across "
                         "checkpoints, so they can be reused from an earlier A/B store")
    ap.add_argument("--transformer_path", default="./checkpoints/UniView")
    ap.add_argument("--model_name", default="./checkpoints/Wan2.1-VACE-14B-diffusers")
    ap.add_argument("--causvid-lora",
                    default="./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors")
    args = ap.parse_args()
    device = args.device

    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from model.pipeline_uniview import WanVACEPipeline
    from model.uniview_transformer import WanVACETransformer3DModel
    from rl.loop.schedule import SCHEDULER_CONFIG

    tr = WanVACETransformer3DModel.from_pretrained(args.transformer_path,
                                                   torch_dtype=torch.bfloat16).to(device)
    vae = AutoencoderKLWan.from_pretrained(args.model_name, subfolder="vae",
                                           torch_dtype=torch.float32).to(device)
    pipe = WanVACEPipeline(tokenizer=None, text_encoder=None, transformer=tr, vae=vae,
                           scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG))
    adapters.load_rl_adapters(pipe, args.causvid_lora)
    n = adapters.load_new_adapter_inplace(pipe, args.ckpt)
    print(f"[ab] loaded ckpt into new ({n} params)", flush=True)
    sched = build_rl_schedule(8, device=device)
    tr.eval()

    writer = RolloutStoreWriter(args.out)
    for scene in args.scenes:
        b = load_cond_bundle(os.path.join(args.cond_root, scene, "cond.pt"), device)
        cond_path = os.path.join(args.cond_root, scene, "cond.pt")
        for arm, adapter in (("base", "ref"), ("new", "new")):
            if arm not in args.arms:
                continue
            with adapters.adapter_ctx(tr, adapter), torch.no_grad():
                for j in range(args.k):
                    seed = args.base_seed + j
                    x0 = sample_rollout(tr, b, sched, seed, guidance_scale=args.guidance,
                                        device=device)
                    frames = decode_latents(vae, x0)
                    # encode arm in the scene key so rewards group cleanly
                    writer.write(f"{scene[:12]}__{arm}", j, x0, frames, cond_path, seed)
            print(f"[ab] {scene[:12]} {arm}: {args.k} rollouts", flush=True)
    print("[ab] sampling done; score with rl.scoring.nft_score_cli", flush=True)

if __name__ == "__main__":
    main()
