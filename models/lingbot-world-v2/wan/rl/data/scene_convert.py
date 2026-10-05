"""Convert lyra scene inputs to the lingbot camera format.

Each input scene dir holds ``image.png`` (or ``image.jpg``) and ``lyra2_traj.npz``
with keys ``w2c [F,4,4]`` (world-to-camera, OpenCV), ``intrinsics [F,3,3]`` and the
resolution those intrinsics refer to, as either ``image_wh`` or ``image_height`` +
``image_width``. Each output scene dir gets:

    image.jpg        first-frame conditioning image
    poses.npy        c2w [F,4,4] float32 = inv(w2c)   (OpenCV convention)
    intrinsics.npy   [F,4] float32 (fx, fy, cx, cy) rescaled to 480x832

Single scene:
    python -m wan.rl.data.scene_convert --in_dir /scenes/abc --out_dir /out/abc
Batch (local roots or object-store prefixes via wan.rl.data.gcs_util):
    python -m wan.rl.data.scene_convert --in_prefix s3://b/scenes/ --out_prefix s3://b/converted/ [--limit 4]
"""

from __future__ import annotations

import argparse
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

REF_HEIGHT = 480
REF_WIDTH = 832
OUT_FILES = ("image.jpg", "poses.npy", "intrinsics.npy")

log = logging.getLogger("scene_convert")

def _find_image(in_dir: str) -> str:
    for name in ("image.png", "image.jpg"):
        p = os.path.join(in_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no image.png/image.jpg in {in_dir}")

def convert_scene(in_dir: str, out_dir: str) -> None:
    """Convert one local scene dir; writes image.jpg, poses.npy, intrinsics.npy."""
    traj = np.load(os.path.join(in_dir, "lyra2_traj.npz"))
    w2c = traj["w2c"].astype(np.float32)          # [F, 4, 4]
    K = traj["intrinsics"].astype(np.float32)     # [F, 3, 3]
    # Either size schema: ``image_wh`` (width, height) as TrajectoryBench writes it, or the
    # separate ``image_height``/``image_width`` scalars.
    if "image_wh" in traj:
        width, height = (float(v) for v in np.asarray(traj["image_wh"]).ravel())
    else:
        height = float(traj["image_height"])
        width = float(traj["image_width"])

    poses = np.linalg.inv(w2c).astype(np.float32)  # c2w, stays OpenCV

    sx = REF_WIDTH / width
    sy = REF_HEIGHT / height
    intrinsics = np.stack(
        [K[:, 0, 0] * sx, K[:, 1, 1] * sy, K[:, 0, 2] * sx, K[:, 1, 2] * sy], axis=1
    ).astype(np.float32)  # [F, 4] fx, fy, cx, cy at 480x832

    os.makedirs(out_dir, exist_ok=True)
    Image.open(_find_image(in_dir)).convert("RGB").save(os.path.join(out_dir, "image.jpg"), quality=95)
    np.save(os.path.join(out_dir, "poses.npy"), poses)
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics)

def _convert_local_batch(in_root: str, out_root: str, limit: int = 0) -> None:
    scenes = sorted(
        d for d in os.listdir(in_root)
        if os.path.exists(os.path.join(in_root, d, "lyra2_traj.npz"))
    )
    if limit:
        scenes = scenes[:limit]
    done = 0
    for scene in scenes:
        out_dir = os.path.join(out_root, scene)
        if all(os.path.exists(os.path.join(out_dir, f)) for f in OUT_FILES):
            continue
        convert_scene(os.path.join(in_root, scene), out_dir)
        done += 1
    log.info("converted %d/%d scenes %s -> %s", done, len(scenes), in_root, out_root)

def _convert_remote_batch(in_prefix: str, out_prefix: str, limit: int = 0, workers: int = 16) -> None:
    from wan.rl.data import gcs_util

    in_prefix = in_prefix.rstrip("/") + "/"
    out_prefix = out_prefix.rstrip("/") + "/"
    in_scheme, in_bucket, in_pfx = gcs_util._split(in_prefix)
    out_scheme, out_bucket, out_pfx = gcs_util._split(out_prefix)

    by_scene = {}
    for key in gcs_util._list_keys(in_bucket, in_pfx):
        rel = key[len(in_pfx):]
        if "/" in rel:
            scene, name = rel.split("/", 1)
            by_scene.setdefault(scene, {})[name] = key
    scenes = sorted(s for s, fs in by_scene.items()
                    if "lyra2_traj.npz" in fs and ("image.png" in fs or "image.jpg" in fs))
    if limit:
        scenes = scenes[:limit]

    done_keys = set(gcs_util._list_keys(out_bucket, out_pfx))
    todo = [s for s in scenes
            if not all(f"{out_pfx}{s}/{f}" in done_keys for f in OUT_FILES)]
    log.info("%d scenes, %d to convert (%d already done)", len(scenes), len(todo), len(scenes) - len(todo))

    def _one(scene: str) -> None:
        files = by_scene[scene]
        with tempfile.TemporaryDirectory() as tmp:
            in_dir = os.path.join(tmp, "in")
            out_dir = os.path.join(tmp, "out")
            img = "image.png" if "image.png" in files else "image.jpg"
            for name in (img, "lyra2_traj.npz"):
                gcs_util.download_file(f"{in_scheme}://{in_bucket}/{files[name]}",
                                       os.path.join(in_dir, name))
            convert_scene(in_dir, out_dir)
            for name in OUT_FILES:
                gcs_util.upload_file(os.path.join(out_dir, name),
                                     f"{out_scheme}://{out_bucket}/{out_pfx}{scene}/{name}")
        log.info("converted %s", scene)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_one, todo))

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", help="single scene: input dir")
    ap.add_argument("--out_dir", help="single scene: output dir")
    ap.add_argument("--in_prefix", help="batch: input root (local dir or s3://bucket/prefix)")
    ap.add_argument("--out_prefix", help="batch: output root (local dir or s3://bucket/prefix)")
    ap.add_argument("--limit", type=int, default=0, help="batch: convert at most N scenes")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    if args.in_dir and args.out_dir:
        convert_scene(args.in_dir, args.out_dir)
    elif args.in_prefix and args.out_prefix:
        if "://" in args.in_prefix:
            _convert_remote_batch(args.in_prefix, args.out_prefix, args.limit, args.workers)
        else:
            _convert_local_batch(args.in_prefix, args.out_prefix, args.limit)
    else:
        ap.error("need --in_dir/--out_dir or --in_prefix/--out_prefix")

if __name__ == "__main__":
    main()
