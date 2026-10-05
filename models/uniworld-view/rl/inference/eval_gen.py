"""Trajbench generation stage for UniWorld-View (lingbotsub-style prefixes).

Consumes a worldscore-eval prefix layout on the object store:
    <ws_prefix>/inputs/<sid>/{<sid>.png, trajectory.npz[, captions.json]}
    <ws_prefix>/uniworld_captions.json          (optional: sid -> prompt, precomputed BLIP2)
and writes
    <ws_prefix>/videos/<sid>/video/<sid>.mp4

Frame policy (per the vibe check): 241-pose trajectories are subsampled at
stride 3 -> 81 poses (full trajectory, 3x camera speed); 81-pose trajectories
run natively. One-shot 81-frame generation, CausVid fused, 8 steps, CFG 4.0,
translation anchor 0.5 * fg-median MoGe depth.

Resumable: sids whose output mp4 already exists are skipped.

    python -m rl.inference.eval_gen --ws_prefix s3://<bucket>/<prefix>/pem_medium \
        --start 0 --end 25 [--num_frames 81] [--device cuda:0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.data.gcs_util import download_file, exists as _exists, upload_file
from rl.data.traj_adapter import load_trajectory_npz
from rl.inference.vibe_gen import build_opts, nvs_custom_traj

# _exists is gcs_util.exists, a HEAD. It was a local len(list_keys(uri)) > 0 -- a LIST, which
# the object store answers inconsistently for fresh writes, so a clip generated minutes earlier could read
# as absent and be regenerated. One-directional (wasted GPU, never a wrong skip), but it bit
# hardest on exactly the gap-fill resubmits this skip check exists to make cheap.

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws_prefix", default="", help="run prefix (s3://) "
                    "(single-cell mode; use --plan + --ws-template for many cells)")
    ap.add_argument("--start", type=int, default=-1)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--num_frames", type=int, default=81,
                    help="output frame count; source poses are strided down to this")
    ap.add_argument("--pose-scale", type=float, default=1.0)
    ap.add_argument("--trans-scale", type=float, default=-1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--rl-ckpt", default="",
                    help="nft_new_stepNNNN.safetensors: evaluate the trained policy instead "
                         "of the base model (replaces the demo's fused CausVid with the "
                         "ref/old/new adapter set and activates `new`)")
    ap.add_argument("--causvid-for-rl", default="",
                    help="CausVid LoRA path for the RL adapter init (default: --lora_path)")
    ap.add_argument("--plan", default="",
                    help="MULTI-CELL plan: 'cell:pose_scale:sid,sid,...;cell:pose:sids;...'. The "
                         "pipeline is built ONCE and every group is generated in the same process. "
                         "Requires --ws-template. Without this, one process handles one prefix -- "
                         "which cost ~35%% of an f25 run's wall clock in per-cell model reloads "
                         "(measured: 4-5 min warm reload x 6-8 cells per rank).")
    ap.add_argument("--out_prefix", default="",
                    help="write videos/ under this prefix instead of --ws_prefix. Lets "
                         "inputs live in ONE shared, model-agnostic prefix while each "
                         "checkpoint's outputs go to its own namespace (no per-ckpt "
                         "input staging). Empty = write under --ws_prefix (legacy).")
    ap.add_argument("--out-template", default="",
                    help="plan-mode analogue of --out_prefix: output prefix template "
                         "containing {cell}. Empty = outputs under --ws-template.")
    ap.add_argument("--ws-template", default="",
                    help="prefix template containing {cell}, e.g. s3://<bucket>/<prefix>/f25v1_NAME_{cell}")
    ap.add_argument("--sids", default="",
                    help="explicit comma-separated clip indices instead of [--start,--end). "
                         "Lets ONE job dispatch all 498 f25 clips across N GPUs by striding "
                         "a global index, which contiguous ranges cannot express")
    ap.add_argument("--seed", type=int, default=42,
                    help="sampling seed (the demo hardcodes 42). Vary it to draw a rollout GROUP "
                         "for one clip -- what the group-relative reward actually compares")
    ap.add_argument("--rollout-tag", default="",
                    help="when set, write to videos/<sid>/rollouts/<tag>.mp4 instead of "
                         "videos/<sid>/video/<sid>.mp4, so K seeds of one clip coexist")
    ap.add_argument("--no-text", action="store_true",
                    help="sample with an EMPTY prompt. NOTE this does NOT match training: "
                         "rl/scene_prep encodes a BLIP2 caption + refine suffix into every cond "
                         "bundle, so the policy is trained and validated WITH text at CFG 4.0. "
                         "Only for ablating the text conditioning; leave unset to match training")
    for k, v in (("blip_path", "./checkpoints/blip2-opt-2.7b"),
                 ("transformer_path", "./checkpoints/UniView"),
                 ("model_name", "./checkpoints/Wan2.1-VACE-14B-diffusers"),
                 ("lora_path", "./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"),
                 ("stream3r_path", "./checkpoints/STream3R"),
                 ("moge_path", "./checkpoints/moge/model.pt"),
                 ("segnet_path", "./checkpoints/tracer_b7.pth")):
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()
    # groups: [(cell, ws_prefix, pose_scale, [sid, ...]), ...] -- one entry per cell. The cell
    # name is carried, never re-derived from the prefix: rsplit("_") yields "easy" for pem_easy and
    # collides hard_photo with ext_hard_photo (the same trap as the old cut -c1-7 job names).
    groups = []
    if args.plan:
        if "{cell}" not in args.ws_template:
            raise SystemExit("--plan needs --ws-template containing {cell}")
        for grp in args.plan.split(";"):
            grp = grp.strip()
            if not grp:
                continue
            cell, pose, sids = grp.split(":")
            ws = args.ws_template.format(cell=cell).rstrip("/")
            out = (args.out_template.format(cell=cell).rstrip("/")
                   if args.out_template else ws)
            groups.append((cell, ws, out, float(pose),
                           [int(x) for x in sids.split(",") if x.strip() != ""]))
    else:
        if not args.ws_prefix:
            raise SystemExit("give --ws_prefix (single cell) or --plan + --ws-template")
        if args.sids:
            # zero-padded to at least three digits, so a bare index keeps the old meaning
            # ("1" -> "001") while a dataset's own wider ids survive ("0046" -> "0046")
            todo = [x.strip().zfill(3) for x in args.sids.split(",") if x.strip() != ""]
        elif args.start >= 0 and args.end >= 0:
            todo = [f"{i:03d}" for i in range(args.start, args.end)]
        else:
            raise SystemExit("give --sids or both --start/--end")
        ws = args.ws_prefix.rstrip("/")
        groups.append((ws.rsplit("/", 1)[-1], ws,
                       (args.out_prefix.rstrip("/") or ws), float(args.pose_scale), todo))
    n_clips = sum(len(g[4]) for g in groups)
    print(f"[eval_gen] {len(groups)} group(s), {n_clips} clip(s) total; ONE pipeline load",
          flush=True)

    opts = build_opts(args)
    opts.render_only = False
    opts.trans_scale = float(args.trans_scale)
    from demo import UniScene
    pvd = UniScene(opts)

    if args.rl_ckpt:
        # Evaluate a trained policy on the same harness as the base model. The demo path
        # FUSES CausVid into the base weights; the RL adapters must stay unfused (3 named
        # adapters over a frozen base), so replace the fused LoRA with ref/old/new and load
        # the checkpoint into `new`. Sampling then runs with `new` active, exactly as the
        # training loop's rollouts do.
        from rl.loop import adapters
        tr = pvd.pipeline.transformer
        # unload_lora_weights() does not reverse a fusion -- demo.py calls fuse_lora(), which
        # bakes CausVid*0.95 into the base weights, and diffusers' unload only drops unfused
        # adapters (verified in 0.35.0: its body never touches fusion state). Without the
        # unfuse below the policy samples on base+CausVid while it was TRAINED on base alone;
        # measured effect was +26% mean saturation on trajbench vs +2% on the val path.
        try:
            pvd.pipeline.unfuse_lora()
            print("[eval_gen] unfused demo CausVid LoRA", flush=True)
        except Exception as e:  # noqa: BLE001 - nothing fused in some builds
            print(f"[eval_gen] unfuse_lora: {type(e).__name__}: {e}", flush=True)
        try:
            pvd.pipeline.unload_lora_weights()
            print("[eval_gen] unloaded demo CausVid LoRA", flush=True)
        except Exception as e:  # noqa: BLE001 - some builds have nothing to unload
            print(f"[eval_gen] unload_lora_weights: {type(e).__name__}: {e}", flush=True)
        adapters.load_rl_adapters(pvd.pipeline, args.causvid_for_rl or args.lora_path)
        n = adapters.load_new_adapter_inplace(pvd.pipeline, args.rl_ckpt)
        adapters.set_active_adapter(tr, "new")
        print(f"[eval_gen] RL policy active: {args.rl_ckpt} ({n} params in `new`)", flush=True)

    work = os.environ.get("EVAL_GEN_WORK", "/tmp/eval_gen")
    os.makedirs(work, exist_ok=True)
    n_done = n_skip = n_fail = 0
    for cellname, ws, out, pose_scale, todo in groups:
      captions = {}
      with tempfile.TemporaryDirectory() as td:
        try:
            download_file(f"{ws}/uniworld_captions.json", f"{td}/cap.json")
            captions = json.load(open(f"{td}/cap.json"))
            print(f"[eval_gen] {cellname}: {len(captions)} precomputed captions", flush=True)
        except Exception:  # noqa: BLE001
            print(f"[eval_gen] {cellname}: no uniworld_captions.json; BLIP2 captions in-process",
                  flush=True)
      print(f"[eval_gen] {cellname}: pose_scale={pose_scale} {len(todo)} clip(s): "
            f"{todo[:8]}{'...' if len(todo) > 8 else ''}", flush=True)
      for i in todo:
        sid = str(i)
        out_uri = (f"{out}/videos/{sid}/rollouts/{args.rollout_tag}.mp4" if args.rollout_tag
                   else f"{out}/videos/{sid}/video/{sid}.mp4")
        if _exists(out_uri):
            print(f"[eval_gen] {sid}: exists, skip", flush=True)
            n_skip += 1
            continue
        try:
            sdir = os.path.join(work, sid)
            os.makedirs(sdir, exist_ok=True)
            download_file(f"{ws}/inputs/{sid}/{sid}.png", f"{sdir}/{sid}.png")
            download_file(f"{ws}/inputs/{sid}/trajectory.npz", f"{sdir}/trajectory.npz")

            # the GROUP pose_scale, not args: hard_* and ext_* are 1.5 while easy/medium are 1.0,
            # and one process now spans several cells
            w2c, K, src_hw = load_trajectory_npz(f"{sdir}/trajectory.npz", pose_scale=pose_scale)
            n = w2c.shape[0]
            stride = max(1, round((n - 1) / (args.num_frames - 1)))
            idx = torch.arange(0, n, stride)[: args.num_frames]
            print(f"[eval_gen] {sid}: {n} poses -> {len(idx)} frames (stride {stride})", flush=True)

            # Default (None) -> the demo captions with BLIP2, which is exactly what
            # rl/scene_prep bakes into the training cond bundles (BLIP2 caption + refine suffix,
            # CFG 4.0). So the captioned path is the one that MATCHES training; --no-text is an
            # ablation, not a correction. The board's "empty caption" wording is wrong.
            prompt = "" if args.no_text else captions.get(sid)
            if prompt is not None:
                pvd._eval_prompt_override = prompt

            meta = nvs_custom_traj(
                pvd, f"{sdir}/{sid}.png", w2c[idx], K[idx], src_hw, sdir, fps=args.fps)
            upload_file(os.path.join(sdir, "diffusion.mp4"), out_uri)
            upload_file(os.path.join(sdir, "render.mp4"), f"{out}/videos/{sid}/render.mp4")
            upload_file(os.path.join(sdir, "meta.json"), f"{out}/videos/{sid}/meta.json")
            print(f"[eval_gen] {sid} done: {meta}", flush=True)
            n_done += 1
            import shutil
            shutil.rmtree(sdir, ignore_errors=True)
        except Exception as e:  # noqa: BLE001 - one bad clip must not kill the shard
            print(f"[eval_gen] {cellname}/{sid} FAILED: {type(e).__name__}: {e}", flush=True)
            n_fail += 1
    print(f"[eval_gen] {n_clips} clips over {len(groups)} group(s): "
          f"done={n_done} skip={n_skip} fail={n_fail}", flush=True)
    if n_fail:
        sys.exit(1)

if __name__ == "__main__":
    main()
