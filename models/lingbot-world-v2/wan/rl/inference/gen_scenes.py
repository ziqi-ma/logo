"""Batch lingbot generation over a directory of scenes (resident pipe).

Loads WanI2VCausal once, then loops the scenes: download
``first_frame.png`` + ``trajectory.npz`` from the staged inputs prefix,
convert the trajectory to lingbot ``poses.npy``/``intrinsics.npy``, generate,
and upload ``<videos_base>/<hash>/seed1.mp4`` — the layout ``eval/score_videos.py``
scans (``<videos_base>/<scene>/*.mp4``).
Resumable: scenes whose seed1.mp4 already exists are skipped.

Canonical eval config is first-half trajectory / 241 frames; for 241 frames
the repo's long-horizon KV setting (local_attn_size 18, sink_size 6) is the
default here. Note lingbot rounds latent frames down to a chunk_size multiple,
so a 241-frame request produces 237 frames; the scorer is length-agnostic.

    python -m wan.rl.inference.gen_scenes --input_base <scene root> --videos_base <out> \
        [--lora_path s3://<bucket>/nft_new_step0150.pt] [--num_shards 10 --shard 0]
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

# Scene root: <input_base>/<scene>/{first_frame.png,trajectory.npz}.
DEFAULT_INPUT_BASE = ""
# Eval scene list. Empty (the default) derives the set by listing the staged inputs
# prefix; pass a local path or an s3:// URI to pin an explicit subset.
DEFAULT_SCENES_JSON = os.environ.get("SCENES_JSON", "")
DEFAULT_PROMPT = "A camera walkthrough of a realistic 3D scene"

log = logging.getLogger("gen_scenes")

def _load_scenes(scenes_json, input_base):
    """Scene hashes from the eval-set json (local path or ``s3://`` URI); with no
    json (the default) the set is the staged inputs prefix's scene listing."""
    if not scenes_json:
        scenes_json = ""
    if "://" in scenes_json:
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "scenes.json")
            gcs_util.download_file(scenes_json, local)
            return json.load(open(local))
    if os.path.exists(scenes_json):
        return json.load(open(scenes_json))
    log.warning("scenes_json %s not found; deriving scene list from %s",
                scenes_json, input_base)
    _, bucket, pfx = gcs_util._split(input_base.rstrip("/") + "/")
    scenes = sorted({k[len(pfx):].split("/", 1)[0]
                     for k in gcs_util._list_keys(bucket, pfx)
                     if k[len(pfx):].count("/") == 1
                     and k.endswith("/trajectory.npz")})
    return scenes

def _maybe_fetch_lora(args, workdir):
    if args.lora_path and "://" in args.lora_path:
        local = os.path.join(workdir, os.path.basename(args.lora_path))
        gcs_util.download_file(args.lora_path, local)
        args.lora_path = local

def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    parser = argparse.ArgumentParser(
        description="lingbot batch generation over a directory of scenes")
    add_pipe_args(parser)
    # 241-frame long-horizon defaults (local_attn_size 18 / sink 6); T5 on CPU so the
    # 14B DiT + 241-frame KV cache fit a single GPU (lingbot_batch pattern).
    parser.set_defaults(local_attn_size=18, sink_size=6, t5_cpu=True)
    parser.add_argument("--input_base", default=DEFAULT_INPUT_BASE, required=not DEFAULT_INPUT_BASE,
                        help="staged inputs prefix: <base>/<hash>/{first_frame.png,trajectory.npz}")
    parser.add_argument("--scenes_json", default=DEFAULT_SCENES_JSON,
                        help="eval scene list JSON (local path or s3:// URI); empty = list --input_base")
    parser.add_argument("--videos_base", required=True,
                        help="output prefix; writes <videos_base>/<hash>/seed1.mp4")
    parser.add_argument("--frame_num", type=int, default=241)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--base_seed", type=int, default=1,
                        help="seed 1 to match the seed1.mp4 naming the scorer scans")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0,
                        help="generate at most N scenes of this shard (smoke tests)")
    args = parser.parse_args()

    scenes = sorted(_load_scenes(args.scenes_json, args.input_base))
    scenes = scenes[args.shard::args.num_shards]
    if args.limit:
        scenes = scenes[: args.limit]
    log.info("shard %d/%d: %d scenes", args.shard, args.num_shards, len(scenes))

    in_base = args.input_base.rstrip("/")
    out_base = args.videos_base.rstrip("/")
    _, bucket, vpfx = gcs_util._split(out_base + "/")
    done_keys = set(gcs_util._list_keys(bucket, vpfx))
    # The clip name follows --base_seed: seed1 (the default) keeps the existing layout, and a
    # second seed lands beside it as seed2.mp4 instead of colliding with seed1 -- which is what
    # used to happen, and worse, the skip-existing check then saw seed1.mp4 present and generated
    # nothing, so a "seed 2" run silently re-published seed 1's numbers.
    _clip = f"seed{args.base_seed}.mp4"
    todo = [s for s in scenes if f"{vpfx}{s}/{_clip}" not in done_keys]
    log.info("%d to generate (%d already done)", len(todo), len(scenes) - len(todo))
    if not todo:
        return

    workdir = tempfile.mkdtemp(prefix="gen_scenes_")
    _maybe_fetch_lora(args, workdir)
    cfg = WAN_CONFIGS[args.task]
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    pipe = build_pipe(args, cfg)
    h, w = SIZE_CONFIGS[args.size]

    ok = fail = 0
    for scene in todo:
        t0 = time.time()
        try:
            with tempfile.TemporaryDirectory(dir=workdir) as tmp:
                gcs_util.download_file(f"{in_base}/{scene}/first_frame.png",
                                       f"{tmp}/first_frame.png")
                gcs_util.download_file(f"{in_base}/{scene}/trajectory.npz",
                                       f"{tmp}/trajectory.npz")
                n_poses = convert_trajectory(f"{tmp}/trajectory.npz", tmp)
                # Resize to the generation resolution: the staged intrinsics
                # are normalized to 480x832 by convert_trajectory, so a plain
                # (anisotropic) resize keeps image and K consistent.
                img = Image.open(f"{tmp}/first_frame.png").convert("RGB")
                img = img.resize((w, h), Image.BICUBIC)
                video = pipe.generate(
                    args.prompt, img, action_path=tmp,
                    chunk_size=args.chunk_size,
                    max_area=MAX_AREA_CONFIGS[args.size],
                    frame_num=min(args.frame_num, n_poses),
                    shift=args.sample_shift,
                    seed=args.base_seed,
                    offload_model=False,
                    max_attention_size=args.max_attention_size)
                mp4 = f"{tmp}/{_clip}"
                save_video(tensor=video[None], save_file=mp4, fps=cfg.sample_fps,
                           nrow=1, normalize=True, value_range=(-1, 1))
                gcs_util.upload_file(mp4, f"{out_base}/{scene}/{_clip}")
                del video
                torch.cuda.empty_cache()
            ok += 1
            log.info("OK %s (%ds)", scene, round(time.time() - t0))
        except Exception as e:  # noqa: BLE001 -- keep the shard alive; rerun picks up failures
            fail += 1
            log.error("FAIL %s: %s", scene, str(e)[-200:])
    log.info("SHARD DONE shard=%d/%d ok=%d fail=%d",
             args.shard, args.num_shards, ok, fail)

if __name__ == "__main__":
    main()
