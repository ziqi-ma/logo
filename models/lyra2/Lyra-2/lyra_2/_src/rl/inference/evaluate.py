# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone evaluator for a trained adapter (no training).

Samples K rollouts per scene with a given adapter as the sampling policy, scores a
combo reward on each rollout's full sequence, and reports per-scene +
overall mean. Scene-parallel across ranks (rank r evaluates scenes[r::world]), so an
N-GPU node evaluates N scenes at once.

Use it to A/B any checkpoint against the DMD baseline on matched seeds: run once with
``--adapter <dmd lora>`` and once with ``--adapter <trained adapter>``, same
``--base-seed`` and ``--scenes``, and compare the printed per-scene reward.

    torchrun --standalone --nproc_per_node=8 -m lyra_2._src.rl.inference.evaluate \
        --checkpoint_dir checkpoints/model --experiment lyra2 \
        --adapter /tmp/adapter.pt --scenes-root /inputs/scenes \
        --scenes "0000 0001 ... 0007" --combo reproj_rgbd --k-rollouts 8 --num_frames 241
"""

from __future__ import annotations

import json
import math
import os
import shutil

import torch

from lyra_2._src.rl.loop.nft_loop import _dist

def _normalized_adapter(path: str) -> str:
    """Return a loadable adapter path. Adapters saved before the save_new_adapter fix
    contain "._checkpoint_wrapped_module" in their keys (from selective-checkpoint
    wrapping), which load_lora_weights can't map. Strip it on the fly so the eval runs
    with any checkpoint, old or new. .safetensors (e.g. the DMD LoRA) is always clean."""
    if not path.endswith(".pt"):
        return path
    sd = torch.load(path, map_location="cpu", weights_only=False)
    needs = any("_checkpoint_wrapped_module" in k or k.endswith(".lora_B.bias") for k in sd)
    if not needs:
        return path
    fixed = {}
    for k, v in sd.items():
        k = k.replace("._checkpoint_wrapped_module", "")
        # LoRA bias must be "diff_b" for load_lora_weights, not PEFT's "lora_B.bias".
        if k.endswith(".lora_B.bias"):
            k = k[: -len(".lora_B.bias")] + ".diff_b"
        fixed[k] = v
    tmp = path + ".normalized.pt"
    torch.save(fixed, tmp)
    return tmp

def run_eval(args) -> None:
    import torch.distributed as dist

    from lyra_2._ext.imaginaire.utils import log
    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._src.rl.loop.sampler import RolloutStoreWriter, run_nft_sampling, save_rollout_videos
    from lyra_2._src.rl.scoring.nft_score import score_epoch
    from lyra_2._src.rl.inference.sample import _build_args, build_inference_model, build_scene_data_batch

    rank, world_size, local_rank = _dist()
    dev = f"cuda:{local_rank}"

    # Plain inference model with the adapter activated as the sampling policy.
    # Normalize keys so any checkpoint (incl. pre-fix ones) loads without manual surgery.
    # The NFT model already carries new/old/ref initialised from the DMD policy, so the
    # base model is this with no adapter. A trained checkpoint overwrites `new` through the
    # loop's own loader: load_lora_weights consumes the DMD layout and silently drops a
    # checkpoint written by save_new_adapter (-96 of 1845 params loaded, the rest rejected),
    # which made every adapter evaluate as the base model.
    # Plain inference model (--experiment lyra2) with the adapter activated as the sampling
    # policy. Normalize keys so any checkpoint (incl. pre-fix ones) loads without manual
    # surgery. The 3-adapter training experiment (lyra2_nft) must NOT be used here: it wraps
    # the base weights in __init__, so the base checkpoint then maps onto nothing and the
    # net stays at init, which renders as noise.
    model, _ = build_inference_model(args.checkpoint_dir, args.experiment,
                                     _normalized_adapter(args.adapter))
    model.eval()
    inf_args = _build_args(args.checkpoint_dir, args.num_frames, args.resolution,
                           args.pose_scale, args.guidance, args.shift, args.base_seed)
    inf_args.offload = False
    da3_model = load_da3_model(da3_model_name=inf_args.da3_model_name,
                               da3_model_path_custom=inf_args.da3_model_path_custom, device=dev)
    da3_model.eval()
    neg_t5 = torch.load("checkpoints/text_encoder/negative_prompt.pt", map_location="cpu",
                        weights_only=False)["t5_text_embeddings"]

    all_scenes = args.scenes.split()
    my_scenes = all_scenes[rank::world_size]
    target_hw = tuple(int(x) for x in args.resolution.split(","))
    log.info(f"[nft-eval] rank {rank}/{world_size} evaluating {my_scenes} "
             f"(adapter={os.path.basename(args.adapter)}, K={args.k_rollouts})", rank0_only=False)

    # Build every scene's conditioning while T5 is resident, then evict T5: the
    # VGGT scorer runs in a subprocess that needs GPU memory the main process
    # can't share while it holds it (mirrors the resident loop).
    batches = {
        sc: build_scene_data_batch(
            model, da3_model,
            image_path=os.path.join(args.scenes_root, sc, "image.png"),
            traj_file=os.path.join(args.scenes_root, sc, "lyra2_traj.npz"),
            caption=args.prompt, neg_t5=neg_t5, num_frames=int(args.num_frames),
            target_hw=target_hw, pose_scale=float(args.pose_scale),
        )
        for sc in my_scenes
    }
    import lyra_2._src.inference.get_t5_emb as _t5mod
    if _t5mod.t5_encoder is not None:
        _t5mod.t5_encoder.model.to("cpu")
        _t5mod.t5_encoder = None
    torch.cuda.empty_cache()

    results: dict = {}
    for sc in my_scenes:
        db = batches[sc]
        root = os.path.join(args.work_dir, f"eval_{sc}_rank{rank}")
        rollout_root = os.path.join(root, "rollout")
        rewards_jsonl = os.path.join(root, "rewards.jsonl")
        os.makedirs(rollout_root, exist_ok=True)
        writer = RolloutStoreWriter(rollout_root, append=False)
        model.net.to(dev)
        da3_model.to(dev)
        with torch.no_grad():
            run_nft_sampling(model, db, inf_args, writer, sc,
                             k_rollouts=args.k_rollouts, base_seed=args.base_seed, da3_model=da3_model,
                             strip_init_prefix=not args.untrimmed)
        # Eval never trains, so the 14B net + depth model are dead weight during
        # scoring. Evict both (and the allocator cache) so the VGGT subprocess
        # gets the whole card; otherwise it CUDA-OOMs and every score is nan.
        model.net.to("cpu")
        da3_model.to("cpu")
        torch.cuda.empty_cache()
        score_epoch(rollout_root, rewards_jsonl, combo_name=args.combo, gpu_id=local_rank)
        torch.cuda.empty_cache()
        # One full-sequence reward per rollout (broadcast to its chunks) -> dedup by rollout.
        # Also collect the RAW per-metric values (vggt_mse/vggt_depth_mae) so the eval reports the
        # un-normalized reconstruction error, not just the z-scored combo.
        import statistics as _st

        by_roll, by_mse, by_dmae = {}, {}, {}
        if os.path.exists(rewards_jsonl):
            for line in open(rewards_jsonl):
                rec = json.loads(line)
                by_roll[rec["rollout"]] = float(rec["R"])
                m = rec.get("metrics") or {}
                if m.get("vggt_mse") is not None:
                    by_mse[rec["rollout"]] = float(m["vggt_mse"])
                if m.get("vggt_depth_mae") is not None:
                    by_dmae[rec["rollout"]] = float(m["vggt_depth_mae"])
        finite = [v for v in by_roll.values() if math.isfinite(v)]

        def _ms(d):
            vals = [v for v in d.values() if math.isfinite(v)]
            return ((sum(vals) / len(vals)) if vals else float("nan"),
                    _st.pstdev(vals) if len(vals) > 1 else 0.0)

        mse_mean, mse_std = _ms(by_mse)
        dmae_mean, dmae_std = _ms(by_dmae)
        results[sc] = {"mean": (sum(finite) / len(finite)) if finite else float("nan"),
                       "n_finite": len(finite), "n_total": len(by_roll),
                       "vggt_mse": mse_mean, "vggt_mse_std": mse_std,
                       "vggt_depth_mae": dmae_mean, "vggt_depth_mae_std": dmae_std,
                       # per-rollout values so callers can compute best-of-K / spread, not just means
                       "roll_R": {str(k): v for k, v in by_roll.items()},
                       "roll_vggt_mse": {str(k): v for k, v in by_mse.items()},
                       "roll_vggt_depth_mae": {str(k): v for k, v in by_dmae.items()}}
        log.info(f"[nft-eval] scene {sc}: {args.combo}={results[sc]['mean']:.4f} "
                 f"({len(finite)}/{len(by_roll)} finite)  raw vggt_mse={mse_mean:.5f}±{mse_std:.5f} "
                 f"vggt_depth_mae={dmae_mean:.5f}±{dmae_std:.5f}", rank0_only=False)
        if args.videos_uri:
            nv = save_rollout_videos(rollout_root, sc, args.videos_uri)
            log.info(f"[nft-eval] uploaded {nv} rollout videos for {sc}", rank0_only=False)
        shutil.rmtree(root, ignore_errors=True)

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, results)
    else:
        gathered = [results]
    merged = {}
    for g in gathered:
        merged.update(g)

    if rank == 0:
        means = [v["mean"] for v in merged.values() if math.isfinite(v["mean"])]
        overall = sum(means) / len(means) if means else float("nan")
        log.info(f"[nft-eval] === {args.combo} eval: adapter={args.adapter} ===", rank0_only=False)
        for sc in sorted(merged):
            v = merged[sc]
            log.info(f"[nft-eval]   {sc}: R={v['mean']:.4f}  "
                     f"vggt_mse={v.get('vggt_mse', float('nan')):.5f}  "
                     f"vggt_depth_mae={v.get('vggt_depth_mae', float('nan')):.5f}  "
                     f"({v['n_finite']}/{v['n_total']} finite)", rank0_only=False)
        log.info(f"[nft-eval]   OVERALL {args.combo} = {overall:.4f} over {len(means)} scenes",
                 rank0_only=False)
        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w") as f:
                json.dump({"adapter": args.adapter, "combo": args.combo, "overall": overall,
                           "per_scene": merged}, f, indent=2)
            log.info(f"[nft-eval] wrote {args.out_json}", rank0_only=False)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Standalone DMD-RL adapter evaluator (sample + score)")
    ap.add_argument("--checkpoint_dir", default="checkpoints/model")
    ap.add_argument("--experiment", default="lyra2")
    ap.add_argument("--adapter", required=True,
                    help="adapter (LoRA) to evaluate as the sampling policy; the base-model\n                         baseline is this with the DMD-distillation LoRA")
    ap.add_argument("--scenes-root", required=True, help="dir with <scene>/{image.png,lyra2_traj.npz}")
    ap.add_argument("--scenes", required=True, help="space-separated scene ids")
    ap.add_argument("--combo", default="reproj_rgbd")
    ap.add_argument("--k-rollouts", type=int, default=8)
    ap.add_argument("--work-dir", default="/outputs/eval")
    ap.add_argument("--out-json", default=None, help="path to write the per-scene results json")
    ap.add_argument("--videos-uri", default=None, help="object-store prefix to upload rollout mp4s to")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--base-seed", type=int, default=0, help="match a training run's base seed for a clean A/B")
    ap.add_argument("--num_frames", type=int, default=241)
    ap.add_argument("--resolution", default="480,832")
    ap.add_argument("--pose_scale", type=float, default=0.35)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--untrimmed", action="store_true",
                    help="score the full buffer incl. the static init-history prefix -- "
                         "reproduces exactly what a pre-fix reward run scored")
    args = ap.parse_args()
    run_eval(args)

if __name__ == "__main__":
    _main()
