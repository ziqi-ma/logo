"""Measure within-scene reward statistics for NORM calibration.

Lyra's NORM mu/sigma are calibrated on 704x1280 lyra rollouts; at lingbot's
480x832 they mis-weight multi-term combos (sigma sets each term's effective
weight after z-scoring). This generates K base-model rollouts for each scene,
scores the raw metrics, and prints per-metric pooled mean + WITHIN-SCENE std
(the spread the per-scene advantage normalizes over — see the hpsv3_vid note
in nft_score.NORM) as ready-to-paste NORM entries.

Runs inside the NFT image (needs the reward envs):
    torchrun --standalone --nproc_per_node=8 -m wan.rl.scoring.calibrate_norm \
        --checkpoint-dir /weights/lingbot-world-v2-14b-causal-fast \
        --scenes-root /scenes --scenes @scenes.txt --k-rollouts 8 \
        --metrics hpsv3_vid,vggt_mse,vggt_depth_mae --work-dir /tmp/calib

Set --num-frames / --local-attn-size / --sink-size to the training arm's values, and
--k-rollouts to its K_ROLLOUTS: the printed within-scene std is the spread over one
advantage group, so a different K measures a different quantity. Also export
NFT_REPROJ_FRAMES / NFT_HPSV3_KEYFRAMES to the arm's values so each metric is measured on
the same evidence the run will score on.
"""
import argparse
import json
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
from PIL import Image

from wan.configs import WAN_CONFIGS
from wan.image2video import WanI2VCausal
from wan.rl.loop import rollout_store
from wan.rl.scoring import nft_score
from wan.rl.loop.nft_loop import _dist, _parse_scenes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--prompt", default="a camera walkthrough of a realistic 3D scene")
    ap.add_argument("--k-rollouts", type=int, default=8)
    ap.add_argument("--metrics", default="hpsv3_vid,vggt_mse,vggt_depth_mae")
    ap.add_argument("--combo", default="hpsv3_reproj_r23",
                    help="scored combo; its terms must cover --metrics")
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--chunk-size", type=int, default=4)
    # Must match the training arm being calibrated, for two reasons. Correctness: sigma is
    # length- and regime-specific, so it has to be measured under the same attention window
    # the run will use. Feasibility: with the default -1 the KV cache is
    # frame_seqlen * lat_f (image2video._prepare_causal_fast), i.e. ~1.28 GB per latent
    # frame, so a long-horizon calibration asks for ~100 GB and OOMs. 18/6 caps it at
    # frame_seqlen * 18 regardless of length.
    ap.add_argument("--local-attn-size", type=int, default=-1)
    ap.add_argument("--sink-size", type=int, default=0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    rank, world, local_rank = _dist()
    device = torch.device(f"cuda:{local_rank}")
    scenes = _parse_scenes(args.scenes)
    metrics_wanted = [m.strip() for m in args.metrics.split(",") if m.strip()]

    pipe = WanI2VCausal(
        config=WAN_CONFIGS['i2v-A14B'], checkpoint_dir=args.checkpoint_dir,
        device_id=local_rank, rank=rank, t5_cpu=True, init_on_cpu=False,
        local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        infer_mode="causal_fast")

    # (scene, k) tasks flat across ranks; base model = the policy, no adapters.
    tasks = [(s, k) for s in scenes for k in range(args.k_rollouts)][rank::world]
    root = os.path.join(args.work_dir, f"rank_{rank:03d}")
    writer = rollout_store.RolloutStoreWriter(root)
    for scene, k in tasks:
        sdir = os.path.join(args.scenes_root, scene)
        img_path = os.path.join(sdir, "image.jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(sdir, "image.png")
        img = Image.open(img_path).convert("RGB")
        ppath = os.path.join(sdir, "prompt.txt")
        prompt = open(ppath).read().strip() if os.path.exists(ppath) else args.prompt
        seed = rollout_store.rollout_seed(scene, args.base_seed, k, args.k_rollouts)
        x0, video, state = rollout_store.sample_rollout(
            pipe, prompt, img, sdir, seed, frame_num=args.num_frames,
            chunk_size=args.chunk_size, shift=args.shift)
        writer.write_rollout(scene, k, x0, state, video)
        del x0, video, state

    pipe.model.cpu()
    torch.cuda.empty_cache()
    rewards_jsonl = os.path.join(root, "rewards.jsonl")
    nft_score.score_epoch(writer.manifest_path, combo_name=args.combo,
                          work_dir=os.path.join(root, "reward"),
                          gpu_id=local_rank, out_jsonl=rewards_jsonl)

    local = [json.loads(l) for l in open(rewards_jsonl)]
    gathered = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(gathered, local)
    else:
        gathered = [local]
    if rank != 0:
        return

    recs = [r for part in gathered for r in part]
    per_scene = defaultdict(lambda: defaultdict(list))
    pooled = defaultdict(list)
    for r in recs:
        for m, v in (r.get("metrics") or {}).items():
            if m in metrics_wanted and v is not None and math.isfinite(float(v)):
                per_scene[r["scene"]][m].append(float(v))
                pooled[m].append(float(v))

    print(f"\n# NORM calibration over {len(recs)} rollouts, "
          f"{len(per_scene)} scenes, K={args.k_rollouts}")
    out = {}
    for m in metrics_wanted:
        vals = pooled.get(m, [])
        if len(vals) < 2:
            print(f"# {m}: insufficient finite values ({len(vals)})")
            continue
        mu = sum(vals) / len(vals)
        within = []
        for sc, mm in per_scene.items():
            v = mm.get(m, [])
            if len(v) >= 2:
                sm = sum(v) / len(v)
                within.append(math.sqrt(sum((x - sm) ** 2 for x in v) / (len(v) - 1)))
        ws = sum(within) / len(within) if within else float("nan")
        pooled_std = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
        sign = nft_score.NORM[m][2] if m in nft_score.NORM else 1
        out[m] = {"mean": mu, "within_scene_std": ws, "pooled_std": pooled_std,
                  "sign": sign, "n": len(vals)}

    # PERSIST before FORMATTING. Everything above is hours of generation and scoring; the
    # console block below is a convenience. With that order reversed, a
    # formatting bug destroyed the whole run: NORM stores sign as a FLOAT (-1.0), `{sign:+d}`
    # raised ValueError on the first metric, and the job died before writing anything.
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"# wrote {args.out_json}")

    for m, e in out.items():
        print(f'    "{m}": ({e["mean"]:.5f}, {e["within_scene_std"]:.5f}, '
              f'{int(e["sign"]):+d}),'
              f'  # within-scene std; pooled std {e["pooled_std"]:.5f}, n={e["n"]}')

if __name__ == "__main__":
    main()
