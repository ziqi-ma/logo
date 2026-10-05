"""Batch lingbot generation for trajbench cells (worldscore_eval contract).

Same resident-pipe pattern as gen_scenes, with the trajbench I/O layout:

    reads   <ws_prefix>/inputs/<sid>/{<sid>.png, trajectory.npz, captions.json}
    writes  <ws_prefix>/videos/<sid>/video/<sid>.mp4

sids are zero-padded 3-digit indices; a shard covers [--start, --end)
(--end defaults to --total).
Resumable: sids whose video already exists are skipped. The per-clip prompt
is captions.json["0"] when present.

--pose_scale is accepted for CLI parity with the other models but IGNORED:
lingbot normalizes camera translation per clip internally
(compute_relative_poses), so there is no user pose scale to apply.

    python -m wan.rl.inference.gen_trajbench --ws_prefix s3://<bucket>/<prefix> \
        --start 0 --end 20 --num_frames 241 [--lora_path s3://<bucket>/nft_new_stepNNNN.pt]
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time

from PIL import Image
import torch

from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.rl.data import gcs_util
from wan.rl.inference import convert_trajectory
from wan.rl.loop.generate_with_adapter import add_pipe_args, build_pipe
from wan.utils.utils import save_video

DEFAULT_PROMPT = "A camera walkthrough of a realistic 3D scene"

log = logging.getLogger("gen_trajbench")

def _download_inputs(ws, sid, tmp):
    """Fetch the clip's conditioning image (canonical <sid>.png, tolerating a
    first_frame.png variant) and trajectory; captions are optional."""
    gcs_util.download_file(f"{ws}/inputs/{sid}/trajectory.npz", f"{tmp}/trajectory.npz")
    img_path = f"{tmp}/{sid}.png"
    try:
        gcs_util.download_file(f"{ws}/inputs/{sid}/{sid}.png", img_path)
    except Exception:  # noqa: BLE001 -- alternate staging layout
        gcs_util.download_file(f"{ws}/inputs/{sid}/first_frame.png", img_path)
    # Prompt resolution, in order:
    #   1. NFT_EVAL_PROMPT env, if set -- wins outright, and MAY be the empty string. This is
    #      the knob to match a run trained without text: nft_loop's `--prompt` defaults to ""
    #      and no dl3dv scene ships a prompt.txt, so those policies never saw any caption.
    #      Evaluating them on DEFAULT_PROMPT conditions them off-distribution.
    #   2. captions.json["0"] when the key is present -- used verbatim, including "". Earlier
    #      this was `.get("0") or DEFAULT_PROMPT`, so a staged-but-empty caption (which is what
    #      the f25 v1 staging holds: {"0": ""}) silently became DEFAULT_PROMPT.
    #   3. DEFAULT_PROMPT only when there is genuinely no caption file / no "0" key.
    env_prompt = os.environ.get("NFT_EVAL_PROMPT")
    if env_prompt is not None:
        return img_path, env_prompt
    prompt = DEFAULT_PROMPT
    try:
        gcs_util.download_file(f"{ws}/inputs/{sid}/captions.json", f"{tmp}/captions.json")
        caps = json.load(open(f"{tmp}/captions.json"))
        if "0" in caps and caps["0"] is not None:
            prompt = caps["0"]
    except Exception:  # noqa: BLE001 -- captions are optional
        pass
    return img_path, prompt

def _peek_n_poses(ws: str, sid: str, workdir: str):
    """Pose count of one clip's trajectory, to size the attention window before the
    pipeline is built. Returns None if it cannot be read (the loop reports the real error)."""
    import numpy as np

    dest = os.path.join(workdir, "peek_trajectory.npz")
    try:
        gcs_util.download_file(f"{ws.rstrip('/')}/inputs/{sid}/trajectory.npz", dest)
        with np.load(dest) as z:
            key = "w2c" if "w2c" in z.files else z.files[0]
            return int(np.asarray(z[key]).shape[0])
    except Exception as e:  # noqa: BLE001 - absent or unreadable; fall back to --num_frames
        log.warning("could not read %s trajectory to size the attention window: %s", sid, e)
        return None

def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    parser = argparse.ArgumentParser(
        description="lingbot batch generation for trajbench (worldscore layout)")
    add_pipe_args(parser)
    # T5 on CPU so the 14B DiT + long KV cache fit a single GPU.
    parser.set_defaults(t5_cpu=True)
    parser.add_argument("--ws_prefix", required=True,
                        help="s3:// run prefix (inputs read, videos written)")
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None,
                        help="exclusive shard end (default: --total)")
    parser.add_argument("--num_frames", type=int, default=241,
                        help="81 easy / 241 medium-hard; must match trajectory length")
    parser.add_argument("--pose_scale", type=float, default=1.0,
                        help="accepted for CLI parity with the other models; IGNORED by lingbot")
    parser.add_argument("--transition_fov_scale", type=float, default=6.0,
                        help="OFFICIAL ext camera conditioning: divide the conditioning fx by "
                             "this factor for transition (door) cells only. fy is NEVER divided, "
                             "so the vertical FOV stays true and yaw stays rigid. Default 6.0 IS "
                             "the standard -- pre-6.0 ext results are superseded, not an "
                             "alternative (their board cells are quarantined in "
                             "trajbench_master.INVALID_prefov6.json). Applies ONLY to ext cells; "
                             "single-space cells always use the true intrinsics. This is a LENS "
                             "change, not a scene rescale: it works because a wide-angle camera "
                             "turns the same translation into larger peripheral flow, so the "
                             "model renders each commanded step bigger. Set 1.0 only to "
                             "reproduce pre-fov6 numbers.")
    parser.add_argument("--base_seed", type=int, default=1,
                        help="generation seed (1 = the published eval seed)")
    parser.add_argument("--ids", default="",
                        help="comma-separated clip ids to generate, used verbatim (e.g. the "
                             "four-digit TrajectoryBench ids). Overrides --start/--end/--total.")
    args = parser.parse_args()

    end = args.end if args.end is not None else args.total
    if args.ids:
        sids = [t.strip().zfill(3) for t in args.ids.split(",") if t.strip()]
    else:
        sids = [f"{i:03d}" for i in range(args.start, end)]
    ws = args.ws_prefix.rstrip("/")
    log.info("pose_scale=%s is ignored: lingbot normalizes translation per clip "
             "(compute_relative_poses); logged once for the record.", args.pose_scale)
    # Gate on this job's own group rather than trusting the caller to pass the flag per
    # group: ws_prefix always ends in the group, so the job is self-describing and a
    # single-space group can never be given the ext lens by a caller's mistake. Keyed on the
    # clip being a transition one at all, not on its difficulty, so it holds however the
    # difficulty is spelled -- both `<row>_ext_hard_photo` as the published rows were named
    # and a directory named for the group itself.
    _group = ws.rsplit("/", 1)[-1]
    is_transition = _group.startswith("transition_") or "_transition_" in _group
    fov_scale = args.transition_fov_scale if is_transition else 1.0
    log.info("group=%s is_transition=%s -> fov_scale=%s (fy untouched); official transition scale is %s",
             _group, is_transition, fov_scale, 6.0)
    if is_transition and args.transition_fov_scale != 6.0:
        log.warning("transition_fov_scale=%s is NOT the official 6.0 -- these transition videos are not "
                    "comparable with board rows and must not be aggregated as if they were.",
                    args.transition_fov_scale)

    # Per-sid existence check rather than one prefix listing, so the same code resumes
    # against an object store or a plain directory.
    todo = [s for s in sids
            if gcs_util.object_size(f"{ws}/videos/{s}/video/{s}.mp4") is None]
    log.info("%d clip(s) requested: %d to generate (%d already done)",
             len(sids), len(todo), len(sids) - len(todo))
    if not todo:
        return

    workdir = tempfile.mkdtemp(prefix="gen_trajbench_")
    if args.lora_path and "://" in args.lora_path:
        local = os.path.join(workdir, os.path.basename(args.lora_path))
        gcs_util.download_file(args.lora_path, local)
        args.lora_path = local

    # Long-horizon KV window for 241-frame clips (local_attn_size 18 / sink 6); keep
    # the repo's full-attention default for 81-frame clips. Explicit CLI
    # flags win over this heuristic.
    #
    # The window follows the length actually generated, which is
    # min(--num_frames, poses in the trajectory) -- not --num_frames. Keying it on the
    # request instead put an 81-pose trajbench clip on the windowed path whenever the
    # caller passed the 241-frame default, and the clip then diverges from a full-attention
    # one after the window starts dropping history.
    n_poses0 = _peek_n_poses(args.ws_prefix, todo[0], workdir)
    eff_frames = min(args.num_frames, n_poses0) if n_poses0 else args.num_frames
    if eff_frames > 81 and args.local_attn_size == -1 and args.sink_size == 0:
        args.local_attn_size, args.sink_size = 18, 6
        log.info("effective frames=%d: using local_attn_size=18 sink_size=6", eff_frames)
    else:
        log.info("effective frames=%d: full attention (local_attn_size=%d sink_size=%d)",
                 eff_frames, args.local_attn_size, args.sink_size)

    cfg = WAN_CONFIGS[args.task]
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    pipe = build_pipe(args, cfg)
    h, w = SIZE_CONFIGS[args.size]

    ok = fail = 0
    logged_prompt = False
    for sid in todo:
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory(dir=workdir) as tmp:
                img_path, prompt = _download_inputs(ws, sid, tmp)
                if not logged_prompt:
                    # Log the text conditioning once per shard. Without this, a run that
                    # silently substituted DEFAULT_PROMPT for a staged empty caption produced
                    # 996 clips with nothing in the logs to show which prompt was used --
                    # exactly how the f25 step-55 rows ended up off-distribution against
                    # policies trained with no caption at all.
                    log.info("text conditioning: %r (empty = matches a run trained with no "
                             "prompt; set NFT_EVAL_PROMPT to override)", prompt)
                    logged_prompt = True
                n_poses = convert_trajectory(f"{tmp}/trajectory.npz", tmp)
                if n_poses != args.num_frames:
                    log.warning("%s: trajectory has %d poses, num_frames=%d",
                                sid, n_poses, args.num_frames)
                img = Image.open(img_path).convert("RGB")
                img = img.resize((w, h), Image.BICUBIC)
                video = pipe.generate(
                    prompt, img, action_path=tmp,
                    chunk_size=args.chunk_size,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=min(args.num_frames, n_poses),
                    shift=args.sample_shift,
                    seed=args.base_seed,
                    offload_model=False,
                    max_attention_size=args.max_attention_size,
                    fov_scale=fov_scale, fov_scale_y=1.0)
                mp4 = f"{tmp}/{sid}.mp4"
                save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                           nrow=1, normalize=True, value_range=(-1, 1))
                gcs_util.upload_file(mp4, f"{ws}/videos/{sid}/video/{sid}.mp4")
                del video
                torch.cuda.empty_cache()
            ok += 1
            log.info("OK %s (%ds)", sid, round(time.time() - t0))
        except Exception as e:  # noqa: BLE001 -- keep the shard alive; rerun picks up failures
            fail += 1
            log.error("FAIL %s: %s", sid, str(e)[-200:])
    log.info("TRAJBENCH SHARD DONE start=%d end=%d ok=%d fail=%d",
             args.start, end, ok, fail)

if __name__ == "__main__":
    main()
