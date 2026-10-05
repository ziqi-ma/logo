#!/usr/bin/env python3
"""Build the training inputs from a local DL3DV-10K checkout.

Per scene, reads the native COLMAP poses in ``transforms.json`` (nerfstudio export),
converts them to OpenCV, resamples the first ``--traj-fraction`` of the path to
``--num-frames`` poses, relativizes ``w2c`` to frame 0, and writes the layout the
training loops read:

    <out>/<scene_hash>/image.png          frame 0, the conditioning image
    <out>/<scene_hash>/lyra2_traj.npz     w2c [F,4,4] f32, intrinsics [F,3,3] f32,
                                          image_height, image_width

Covering a fraction of the path lowers the per-frame camera speed geometrically rather than
rescaling translations, so no pose scale is applied here; the loaders scale translation at
load time and the published runs use 1.0.

    python3 build_trajectories.py --dl3dv-root /path/to/DL3DV-10K --out /inputs/scenes \
        [--scenes scene_sets/dl3dv.json]

Needs numpy, scipy and pillow; ``--frame0-from video`` also needs ffmpeg on PATH.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

IMAGE_DIRS = ("images_4", "images_8", "images")


def resample(c2w: np.ndarray, intr: np.ndarray, num_frames: int, fraction: float):
    """Resample to num_frames poses -- position lerp, rotation SLERP, nearest intrinsics."""
    from scipy.spatial.transform import Rotation, Slerp

    n = c2w.shape[0]
    last = (n - 1) * max(0.0, min(1.0, fraction))
    t = np.linspace(0.0, last, num_frames)
    src = np.arange(n)
    pos = np.stack([np.interp(t, src, c2w[:, d, 3]) for d in range(3)], axis=1)
    rot = Slerp(src, Rotation.from_matrix(c2w[:, :3, :3]))(t).as_matrix()
    out = np.tile(np.eye(4), (num_frames, 1, 1))
    out[:, :3, :3] = rot
    out[:, :3, 3] = pos
    return out, intr[np.clip(t.round().astype(np.int64), 0, n - 1)]


def cameras(transforms: dict, wh: tuple) -> tuple:
    """transforms.json -> (c2w [N,4,4] f64 OpenCV, intrinsics [N,3,3] f32 at wh).

    DL3DV poses are nerfstudio (camera +X right, +Y up, +Z back); flip y/z for OpenCV.
    The global intrinsics are given at full resolution and scaled to the frame size used.
    """
    w, h = wh
    sx, sy = w / transforms["w"], h / transforms["h"]
    frames = transforms["frames"]
    c2w = np.stack([np.asarray(f["transform_matrix"], np.float64) for f in frames])
    c2w = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    k = np.array([[transforms["fl_x"] * sx, 0.0, transforms["cx"] * sx],
                  [0.0, transforms["fl_y"] * sy, transforms["cy"] * sy],
                  [0.0, 0.0, 1.0]], np.float32)
    return c2w, np.tile(k, (len(frames), 1, 1))


def frame0(scene: Path, transforms: dict, source: str, dst: Path) -> None:
    """Write frame 0 -- the pose-0 view -- to dst."""
    name = Path(transforms["frames"][0]["file_path"]).name
    if source != "video":
        for d in IMAGE_DIRS:
            for cand in (scene / d / name, *sorted((scene / d).glob("*"))[:1]):
                if cand.is_file():
                    from PIL import Image
                    Image.open(cand).convert("RGB").save(dst)
                    return
        raise FileNotFoundError(f"no {'/'.join(IMAGE_DIRS)} frame in {scene}")
    vids = sorted(scene.glob("**/*.mp4"))
    if not vids:
        raise FileNotFoundError(f"no mp4 in {scene}")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vids[0]),
                    "-frames:v", "1", str(dst)], check=True)


def build_scene(scene: Path, out: Path, num_frames: int, fraction: float,
                max_motion: float, source: str) -> float:
    """Write one scene's image.png + lyra2_traj.npz; returns max ||t - t0||."""
    transforms = json.loads((scene / "transforms.json").read_text())
    n = len(transforms["frames"])
    if n < num_frames + 4:
        raise ValueError(f"only {n} posed frames (< {num_frames} + 4)")

    out.mkdir(parents=True, exist_ok=True)
    frame0(scene, transforms, source, out / "image.png")
    from PIL import Image
    iw, ih = Image.open(out / "image.png").size

    c2w, intr = cameras(transforms, (iw, ih))
    c2w, intr = resample(c2w, intr, num_frames, fraction)
    w2c = (np.linalg.inv(c2w) @ c2w[0]).astype(np.float32)
    motion = float(np.linalg.norm(w2c[:, :3, 3] - w2c[0, :3, 3], axis=1).max())
    # Broken COLMAP reconstructions produce absurd translations; normal scenes sit at 8-14.
    if motion > max_motion:
        (out / "image.png").unlink()
        raise ValueError(f"motion {motion:.1f} > {max_motion}")

    np.savez(out / "lyra2_traj.npz", w2c=w2c, intrinsics=intr.astype(np.float32),
             image_height=np.int64(ih), image_width=np.int64(iw))
    return motion


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl3dv-root", required=True, help="local DL3DV-10K checkout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", help="scene_sets JSON; default = every scene under --dl3dv-root")
    ap.add_argument("--num-frames", type=int, default=241)
    ap.add_argument("--traj-fraction", type=float, default=0.5,
                    help="fraction of the native path covered over --num-frames")
    ap.add_argument("--max-motion", type=float, default=50.0,
                    help="drop scenes whose max ||t - t0|| exceeds this")
    ap.add_argument("--frame0-from", choices=("images", "video"), default="images",
                    help="conditioning frame from an images_* dir or the scene video")
    a = ap.parse_args()

    root, out = Path(a.dl3dv_root), Path(a.out)
    if a.scenes:
        split = json.loads(Path(a.scenes).read_text())
        hashes = sorted(set(split.get("train", [])) | set(split.get("val", [])))
    else:
        hashes = sorted(p.parent.name for p in root.glob("**/transforms.json"))

    built, motions, dropped = 0, [], 0
    for h in hashes:
        cands = [root / h, *root.glob(f"*/{h}")]
        scene = next((c for c in cands if (c / "transforms.json").is_file()), None)
        if scene is None:
            print(f"DROP {h[:12]} no transforms.json under --dl3dv-root")
            dropped += 1
            continue
        try:
            motions.append(build_scene(scene, out / h, a.num_frames, a.traj_fraction,
                                       a.max_motion, a.frame0_from))
            built += 1
        except Exception as e:
            print(f"DROP {h[:12]} {e}")
            dropped += 1

    print(f"built {built}/{len(hashes)} scenes -> {out} (dropped {dropped})")
    if motions:
        m = np.array(motions)
        print(f"||t - t0||  p10={np.percentile(m, 10):.2f}  med={np.median(m):.2f}  "
              f"p90={np.percentile(m, 90):.2f}  max={m.max():.2f}")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main())
