"""Render a reconstructed Gaussian scene (.ply) from 8 still views.

The 8 views are:
  - 4 sampled from the recon's own trajectory poses, spread by farthest-point
    sampling on (position, view direction) so they are visually diverse.
  - 4 novel views: a LEVEL fan around the scene core. The four differ by yaw
    (horizontal pan) plus a gentle pitch, so they show clearly different parts of
    the scene while the horizon stays level (no ceiling/floor stare) and the
    handedness matches the cameras (no left-right mirror).

Inputs are a recon directory containing ``reconstructed_scene.ply`` and
``vipe_predictions.npz`` (keys ``w2c_vipe`` (N,4,4), ``intrinsics_vipe`` (N,3,3)).
Writes 8 PNGs + a 2x4 montage to ``outdir`` and returns their paths.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from lyra_2._src.inference.vipe_da3_gs_recon import _load_gaussian_ply_to_gaussians
from depth_anything_3.model.utils.gs_renderer import render_3dgs

# Novel-view fan geometry.
YAW_DEG = 22.0  # horizontal pan of the outer views
PITCH_DEG = 8.0  # gentle up/down; small so the horizon stays level
BASE_FRAC = 0.25  # lateral eye offset as a fraction of camera->core distance
CORE_PCTL = 55  # keep points within this distance percentile of the path centroid (drop floaters)

def _lookat_w2c(eye: np.ndarray, target: np.ndarray, cam_down: np.ndarray) -> np.ndarray:
    """World-to-camera for an OpenCV camera (x=right, y=down, z=forward) at ``eye``
    looking at ``target``. ``cam_down`` (the world "down" direction) fixes the roll
    and the right-handedness, so views are not mirrored."""
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(cam_down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1)
    m[:3, 3] = eye
    return np.linalg.inv(m)

def _trajectory_views(w2c: np.ndarray, n: int = 4) -> tuple[np.ndarray, list[int]]:
    """Pick ``n`` trajectory poses by farthest-point sampling on (position, forward)."""
    c2w = np.linalg.inv(w2c)
    cc = c2w[:, :3, 3]
    fwd = c2w[:, :3, 2]
    pos = cc - cc.mean(0)
    feat = np.concatenate([pos / (np.linalg.norm(pos, axis=1).max() + 1e-9), fwd], axis=1)
    chosen = [0]
    for _ in range(n - 1):
        d = np.min([np.linalg.norm(feat - feat[c], axis=1) for c in chosen], axis=0)
        chosen.append(int(np.argmax(d)))
    chosen = sorted(chosen)
    return w2c[chosen], chosen

def _novel_views(w2c: np.ndarray, means: np.ndarray) -> np.ndarray:
    """4 level fan views around the floater-filtered scene core."""
    c2w = np.linalg.inv(w2c)
    cc = c2w[:, :3, 3]
    Pc = cc.mean(0)
    core = means[np.linalg.norm(means - Pc, axis=1) < np.percentile(np.linalg.norm(means - Pc, axis=1), CORE_PCTL)]
    C = core.mean(0)
    up0 = -c2w[:, :3, 1].mean(0)
    up0 /= np.linalg.norm(up0)
    v = C - Pc
    v -= (v @ up0) * up0  # level base forward
    dist = float(np.linalg.norm(v))
    v /= dist
    rt = c2w[:, :3, 0].mean(0)
    rt -= (rt @ v) * v
    rt -= (rt @ up0) * up0  # camera right, projected horizontal
    rt /= np.linalg.norm(rt)
    down0 = -up0
    yaw, pitch, base = np.radians(YAW_DEG), np.radians(PITCH_DEG), BASE_FRAC * dist
    nov = []
    for sy, sp in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:  # TL, TR, BR, BL
        fp = np.cos(sp * pitch) * (np.cos(sy * yaw) * v + np.sin(sy * yaw) * rt) + np.sin(sp * pitch) * up0
        fp /= np.linalg.norm(fp)
        eye = Pc + sy * base * rt
        nov.append(_lookat_w2c(eye, eye + fp * dist, down0))
    return np.stack(nov)

def compute_camera_set(recondir: str) -> tuple[np.ndarray, list[str]]:
    """Derive the 8 view w2c (4 trajectory + 4 novel) and their labels from a recon.

    Use this once per scene on a reference recon, then pass the result to
    ``render_with`` for every seed to get an identical camera set across seeds.
    """
    pred = np.load(f"{recondir}/vipe_predictions.npz")
    w2c = pred["w2c_vipe"].astype(np.float64)
    traj, chosen = _trajectory_views(w2c)
    # the novel fan only needs the point cloud for the scene core
    import plyfile  # local import; only needed when computing a fresh set

    v = plyfile.PlyData.read(f"{recondir}/reconstructed_scene.ply")["vertex"].data
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    allw = np.concatenate([traj, _novel_views(w2c, means)], 0)
    labs = [f"traj{c}" for c in chosen] + ["core_TL", "core_TR", "core_BR", "core_BL"]
    return allw, labs

def render_with(recondir: str, allw2c: np.ndarray, labels: list[str], outdir: str,
                image_hw: tuple[int, int] = (480, 832)) -> list[str]:
    """Render ``recondir``'s gaussians from the given 8 w2c (intrinsics from this recon)."""
    dev = torch.device("cuda")
    g = _load_gaussian_ply_to_gaussians(f"{recondir}/reconstructed_scene.ply", dev)
    Kpix = np.load(f"{recondir}/vipe_predictions.npz")["intrinsics_vipe"][0].astype(np.float64)
    H, W = image_hw
    extr = torch.from_numpy(allw2c.astype(np.float64)).float().to(dev)
    Kn = np.array([[Kpix[0, 0] / W, 0, Kpix[0, 2] / W], [0, Kpix[1, 1] / H, Kpix[1, 2] / H], [0, 0, 1]])
    intr = torch.from_numpy(Kn).float().to(dev).unsqueeze(0).repeat(len(allw2c), 1, 1)
    with torch.no_grad():
        color, _ = render_3dgs(extrinsics=extr, intrinsics=intr, image_shape=(H, W), gaussian=g,
                               num_view=len(allw2c), use_sh=True)
    imgs = (color.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        p = f"{outdir}/view{i}_{lab}.png"
        Image.fromarray(im).save(p)
        paths.append(p)
    mont = np.zeros((2 * H, 4 * W, 3), np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, 4)
        mont[r * H:(r + 1) * H, c * W:(c + 1) * W] = im
    Image.fromarray(mont).save(f"{outdir}/montage.png")
    paths.append(f"{outdir}/montage.png")
    return paths

def render_8_views(recondir: str, outdir: str, image_hw: tuple[int, int] = (480, 832)) -> list[str]:
    """Render the 8 views for ``recondir`` using its own per-recon camera set."""
    allw, labs = compute_camera_set(recondir)
    return render_with(recondir, allw, labs, outdir, image_hw)

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render 8 views (4 trajectory + 4 novel) from a recon dir.")
    ap.add_argument("recon_dir", help="dir with reconstructed_scene.ply + vipe_predictions.npz")
    ap.add_argument("out_dir", help="output dir for the 8 PNGs + montage.png")
    ap.add_argument("--views_npz", default=None,
                    help="load a fixed camera set (w2c + labels) from this .npz instead of computing "
                         "it from recon_dir; pass the same file for every seed to fix the cameras.")
    ap.add_argument("--save_views", default=None,
                    help="compute the camera set from recon_dir and save it to this .npz (for reuse).")
    args = ap.parse_args()
    if args.views_npz:
        d = np.load(args.views_npz, allow_pickle=True)
        allw, labs = d["w2c"], list(d["labels"])
    else:
        allw, labs = compute_camera_set(args.recon_dir)
        if args.save_views:
            np.savez(args.save_views, w2c=allw, labels=np.array(labs))
    out = render_with(args.recon_dir, allw, labs, args.out_dir)
    print(f"saved {len(out)} files -> {args.out_dir}")
