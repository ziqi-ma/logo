"""Lyra-2 generation for trajbench cells (worldscore_eval contract).

The same I/O layout the other two models' trajbench generators use, so one benchmark
cell can be generated for every model and scored by ``eval/score_videos.py``:

    reads   <ws_prefix>/inputs/<sid>/{<sid>.png, trajectory.npz, captions.json}
    writes  <ws_prefix>/videos/<sid>/video/<sid>.mp4

sids are zero-padded 3-digit indices; a shard covers [--sid-start, --sid-end).
Resumable: sids whose video already exists are skipped. The per-clip prompt is
captions.json["0"] when present, else empty (which is what the policies trained with
no caption expect).

    torchrun --standalone --nproc_per_node=1 -m lyra_2._src.rl.inference.gen_trajbench \
        --ws_prefix s3://<bucket>/<prefix> --sid-start 0 --sid-end 20 \
        --checkpoint_dir checkpoints/model --adapter <nft_new_stepNNNN.pt>

Pass the DMD-distillation LoRA as --adapter for the base-model clips. One clip per sid
at --base-seed; evaluate remains the K-rollout evaluator.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import tempfile

import torch

from lyra_2._ext.imaginaire.utils import log
from lyra_2._src.rl.data.gcs_util import download_file, list_keys, upload_file


def _fetch_inputs(ws: str, sid: str, dest_dir: str, use_captions: bool = False):
    """Pull one clip's conditioning image, trajectory and caption. Returns (image, traj, prompt)."""
    os.makedirs(dest_dir, exist_ok=True)
    img = os.path.join(dest_dir, "image.png")
    traj = os.path.join(dest_dir, "lyra2_traj.npz")
    download_file(f"{ws}/inputs/{sid}/{sid}.png", img)
    download_file(f"{ws}/inputs/{sid}/trajectory.npz", traj)
    # Empty by default: the policies train with no caption and evaluate evaluates the same
    # way, so a caption here would put the clip off that distribution. --use-captions opts in.
    prompt = os.environ.get("NFT_EVAL_PROMPT", "")
    if use_captions:
        caps = os.path.join(dest_dir, "captions.json")
        try:
            download_file(f"{ws}/inputs/{sid}/captions.json", caps)
            prompt = json.load(open(caps)).get("0", prompt)
        except Exception:  # noqa: BLE001 - captions are optional
            pass
    return img, traj, prompt


def _encode_and_upload(rollout_root: str, scene: str, out_uri: str, fps: int) -> bool:
    """Encode the rollout's accumulated frames to H.264 and upload to the worldscore path."""
    rolls = sorted(glob.glob(os.path.join(rollout_root, f"scene_{scene}", "rollout_*")))
    if not rolls:
        return False
    chunks = sorted(glob.glob(os.path.join(rolls[0], "chunk_*")))
    frames_dir = os.path.join(chunks[-1], "frames") if chunks else None
    if not frames_dir or not sorted(glob.glob(os.path.join(frames_dir, "*.png"))):
        return False
    local = os.path.join(rollout_root, f"{scene}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-pattern_type", "glob", "-i", os.path.join(frames_dir, "*.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", local],
        check=True,
    )
    upload_file(local, out_uri)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws_prefix", required=True, help="worldscore-eval prefix (s3://)")
    # torchrun owns --start* and --end*, so the shard bounds are named --sid-start/--sid-end.
    ap.add_argument("--sid-start", dest="start", type=int, default=0)
    ap.add_argument("--sid-end", dest="end", type=int, default=0)
    ap.add_argument("--ids", default="",
                    help="comma-separated clip ids, used verbatim (e.g. the four-digit "
                         "TrajectoryBench ids). Overrides --sid-start/--sid-end.")
    ap.add_argument("--checkpoint_dir", default="checkpoints/model")
    ap.add_argument("--experiment", default="lyra2")
    ap.add_argument("--adapter", required=True,
                    help="adapter to sample with; pass the DMD-distillation LoRA for the base")
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--resolution", default="480,832")
    ap.add_argument("--pose_scale", type=float, default=1.0)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--base-seed", dest="base_seed", type=int, default=1)
    ap.add_argument("--use-captions", dest="use_captions", action="store_true",
                    help="condition on captions.json[\"0\"]; default is the empty caption")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--work-dir", default="/tmp/gen_trajbench_lyra")
    args = ap.parse_args()
    if not args.ids and not args.end:
        ap.error("give --ids or --sid-end")

    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._src.rl.inference.evaluate import _normalized_adapter
    from lyra_2._src.rl.inference.sample import (
        _build_args, build_inference_model, build_scene_data_batch,
    )
    from lyra_2._src.rl.loop.sampler import RolloutStoreWriter, run_nft_sampling

    ws = args.ws_prefix.rstrip("/")
    # Resume: skip sids whose mp4 is already there. gcs_util has no HEAD helper, so this
    # lists the videos prefix once rather than once per sid.
    try:
        done = {k.rsplit("/", 1)[-1][:-4] for k in list_keys(f"{ws}/videos/") if k.endswith(".mp4")}
    except Exception:  # noqa: BLE001 - nothing generated yet
        done = set()
    want = ([t.strip().zfill(3) for t in args.ids.split(",") if t.strip()] if args.ids
            else [f"{i:03d}" for i in range(args.start, args.end)])
    todo = [s for s in want if s not in done]
    log.info(f"[gen_trajbench] {len(todo)} of {len(want)} clip(s) to generate",
             rank0_only=False)
    if not todo:
        return

    neg_path = os.path.join(os.path.dirname(args.checkpoint_dir.rstrip("/")),
                            "text_encoder", "negative_prompt.pt")
    neg_t5 = torch.load(neg_path, map_location="cpu", weights_only=False)["t5_text_embeddings"]

    # --experiment lyra2 (the plain inference model) with the adapter as the sampling
    # policy, exactly as evaluate does. lyra2_nft wraps the base weights in __init__ and
    # the base checkpoint then loads into nothing, which renders as noise.
    model, _ = build_inference_model(args.checkpoint_dir, args.experiment,
                                     _normalized_adapter(args.adapter))
    model.eval()

    inf_args = _build_args(args.checkpoint_dir, args.num_frames, args.resolution,
                           args.pose_scale, args.guidance, args.shift, args.base_seed)
    inf_args.offload = False
    da3_model = load_da3_model(da3_model_name=inf_args.da3_model_name,
                               da3_model_path_custom=inf_args.da3_model_path_custom,
                               device="cuda")
    target_hw = tuple(int(x) for x in args.resolution.split(","))

    ok = fail = 0
    for sid in todo:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                img, traj, prompt = _fetch_inputs(ws, sid, tmp, args.use_captions)
                root = os.path.join(args.work_dir, f"clip_{sid}")
                rollout_root = os.path.join(root, "rollout")
                os.makedirs(rollout_root, exist_ok=True)
                writer = RolloutStoreWriter(rollout_root, append=False)
                model.net.to("cuda")
                da3_model.to("cuda")
                batch = build_scene_data_batch(
                    model, da3_model, image_path=img, traj_file=traj, caption=prompt,
                    neg_t5=neg_t5, num_frames=int(args.num_frames), target_hw=target_hw,
                    pose_scale=float(args.pose_scale),
                )
                with torch.no_grad():
                    run_nft_sampling(model, batch, inf_args, writer, sid,
                                     k_rollouts=1, base_seed=args.base_seed,
                                     da3_model=da3_model)
                if _encode_and_upload(rollout_root, sid,
                                      f"{ws}/videos/{sid}/video/{sid}.mp4", args.fps):
                    ok += 1
                    log.info(f"[gen_trajbench] {sid} done", rank0_only=False)
                else:
                    fail += 1
                    log.warning(f"[gen_trajbench] {sid}: no frames to encode", rank0_only=False)
        except Exception as e:  # noqa: BLE001 - one bad clip must not end the shard
            fail += 1
            log.warning(f"[gen_trajbench] {sid} FAILED: {type(e).__name__}: {e}", rank0_only=False)
        torch.cuda.empty_cache()
    log.info(f"[gen_trajbench] shard done ok={ok} fail={fail}", rank0_only=False)


if __name__ == "__main__":
    main()
