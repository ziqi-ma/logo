"""Full VideoGPA metric suite for the DL3DV benchmark (GPU, reference-free).

Reproduces VideoGPA's `replicate_scorer.py` / `pipelines/process_video.py` scoring:
sample 10 uniform frames from a generated video, run a reconstruction backbone to
estimate geometry, unproject depth to a colored point cloud, reproject that cloud into
every estimated view, and compute the VideoGPA metrics on the result. All metrics are
reference-free (the "ground truth" is the generated video's own frames; "rep" is the
reprojection), so the score measures 3D self-consistency, not fidelity to real DL3DV.

Two reconstruction backbones exist: VGGT-Omega (`vggt_*`, 's recon stack) and
DepthAnything3 (`da3_*`, VideoGPA's default). Only VGGT-Omega runs by default — the DA3
arm is deprecated and costs a second recon forward pass per clip for metrics nothing
reports. Set VIDEOGPA_BACKBONES=da3,vggt (or =da3) to bring it back. Per video and per
active backbone it returns mse/psnr/ssim/lpips/mvcs/consistency_score/motion/
dropout_psnr/coverage, plus one shared, recon-independent `epipolar`. Reuses
scorers.videogpa (DA3 reprojection), scorers.mvcs, and scorers.vggt (VGGT-Omega
install + checkpoint); metric classes live under scorers/video_gpa/metrics/. DA3,
vggt_omega, lpips, lightglue import lazily in _init so this module imports anywhere.

num_frames=10 and the lightglue epipolar match VideoGPA's defaults.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from . import videogpa
from .mvcs import per_pair_mvcs

NUM_FRAMES = 10
BACKBONES = ("da3", "vggt")
# DA3 is off by default. Read at import because METRICS (the reported column set)
# has to be fixed before the first video is scored.
_WANTED = os.environ.get("VIDEOGPA_BACKBONES", "vggt").split(",")
ACTIVE_BACKBONES = tuple(b for b in BACKBONES if b in _WANTED)
# Recon-dependent metrics, computed once per backbone (prefixed da3_/vggt_).
_PER_BACKBONE = ["mse", "psnr", "ssim", "lpips", "mvcs", "consistency_score", "motion",
                 "dropout_psnr", "coverage"]
# epipolar is recon-independent (runs on the raw frames), so it is reported once.
METRICS = ["epipolar"] + [f"{b}_{m}" for b in ACTIVE_BACKBONES for m in _PER_BACKBONE]
# Where to find the VGGT-Omega backbone when the caller does not pass one. Obtain
# vggt_omega_1b_512.pt from https://huggingface.co/facebook/VGGT-Omega (gated: accept the terms
# once), or from the archive. VGGT_CHECKPOINT wins (the launchers set it), then
# VGGT_CKPT_URI, then the path lingbot's stage.sh downloads into. No internal location is baked
# in: the original default was an unreadable internal bucket, and that failure surfaced as a
# corrupt checkpoint mid-reward rather than as a missing file, so it read like a scorer bug.
DEFAULT_VGGT_CKPT = (
    os.environ.get("VGGT_CHECKPOINT")
    or os.environ.get("VGGT_CKPT_URI")
    or "./weights/vggt/vggt_omega_1b_512.pt"
)

def _decode_uniform(video_path, n_frames: int = NUM_FRAMES) -> np.ndarray:
    """Decode `n_frames` uniformly-spaced frames, center-cropped + resized to 518x518.

    Mirrors VideoGPA/utils/video_utils.sample_uniform_frames (linspace indices, center
    crop, resize 518). Returns [T, 518, 518, 3] uint8 RGB.
    """
    cap = cv2.VideoCapture(str(video_path))
    raw = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw.append(frame)
    cap.release()
    if not raw:
        raise RuntimeError(f"No frames decoded from {video_path}")

    n_eff = min(n_frames, len(raw))
    indices = np.linspace(0, len(raw) - 1, n_eff).astype(int)
    frames = []
    for idx in indices:
        img = cv2.cvtColor(raw[idx], cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
        img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
    return np.stack(frames, axis=0)

def _load_vggt(args):
    """Load VGGT-Omega (reuses scorers.vggt for install + checkpoint download)."""
    import torch
    from .vggt import download_vggt_checkpoint
    from vggt_omega.models import VGGTOmega

    ckpt_uri = getattr(args, "vggt_checkpoint", None) or DEFAULT_VGGT_CKPT
    local = download_vggt_checkpoint(ckpt_uri, Path("/tmp/vggt_omega"))
    model = VGGTOmega().eval().to(args.device)
    model.load_state_dict(torch.load(str(local), map_location="cpu"))
    return model

def _init(args):
    """Load the active backbones, LPIPS, and the metric objects.

    DA3 comes from videogpa._init (model + Consistency_Score scorer, which owns the
    MSE/LPIPS metrics); VGGT-Omega is loaded via _load_vggt. The metric comparators
    (Consistency_Score, PSNR, SSIM, Epipolar) are stateless and shared across both
    backbones. With DA3 inactive `da3_model` is None and the Consistency_Score scorer is
    built here instead, so the DA3 weights are never downloaded. DA3, vggt_omega,
    lightglue import lazily so this imports anywhere.
    """
    if "da3" in ACTIVE_BACKBONES:
        da3 = videogpa._init(args)  # {"model", "scorer", "device"}
        device, da3_model, scorer = da3["device"], da3["model"], da3["scorer"]
    else:
        import lpips

        from .video_gpa.metrics.consistency_score import Consistency_Score

        device, da3_model = args.device, None
        scorer = Consistency_Score(lpips_net=lpips.LPIPS(net="vgg").to(device).eval(),
                                   device=device)

    from .video_gpa.metrics.epipolar import EpipolarMetric
    from .video_gpa.metrics.mse import PSNRMetric, SSIMMetric

    return {
        "device": device,
        "da3_model": da3_model,
        "vggt_model": _load_vggt(args) if "vggt" in ACTIVE_BACKBONES else None,
        "scorer": scorer,
        "psnr": PSNRMetric(device=device),
        "ssim": SSIMMetric(device=device),
        "epipolar": EpipolarMetric(descriptor_type="lightglue", device=device),
    }

def reproject_vggt(ctx, frames_np):
    """VGGT-Omega counterpart of videogpa.reproject; returns the same 5-tuple.

    Runs VGGT-Omega on the decoded frames, unprojects its depth to a colored point
    cloud, and reprojects into every estimated view — mirroring the DA3 path so the
    metric code is backbone-agnostic. extrinsics are padded to W2C [T,4,4] and depth
    squeezed to [T,H,W] for the geometry metrics.
    """
    import tempfile

    import torch
    from depth_anything_3.utils.geometry import affine_inverse, unproject_depth
    from PIL import Image
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    from .video_gpa.utils.pointcloud_utils import get_colored_pointcloud
    from .video_gpa.utils.projection_utils import batch_reproject

    device, model = ctx["device"], ctx["model"]
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, frame in enumerate(frames_np):
            p = f"{td}/f{i:04d}.png"
            Image.fromarray(frame, "RGB").save(p)
            paths.append(p)
        images = load_and_preprocess_images(paths, image_resolution=512).to(device)

    with torch.inference_mode():
        preds = model(images)

    extr, intr = encoding_to_camera(preds["pose_enc"], preds["images"].shape[-2:])
    intrinsics = intr[0].float()  # [T, 3, 3]
    depths = preds["depth"][0].float()[..., 0]  # [T, H, W]
    conf = preds["depth_conf"][0].float()
    imgs = preds["images"][0].float()  # [T, C, H, W]
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    n = extr.shape[1]
    w2c = torch.eye(4, device=device).repeat(n, 1, 1)
    w2c[:, :3, :4] = extr[0].float()  # pad [T,3,4] -> [T,4,4]

    c2w = affine_inverse(w2c)
    world = unproject_depth(
        depths.unsqueeze(0).unsqueeze(-1), intrinsics.unsqueeze(0), c2w.unsqueeze(0)
    ).squeeze(0)
    preds_d = {
        "world_points_from_depth": world, "depth_conf": conf, "images": imgs,
        "extrinsic": w2c, "intrinsic": intrinsics, "depth": depths,
    }
    h, w = imgs.shape[-2:]
    vertices, colors = get_colored_pointcloud(preds_d, mode="depth", conf_thres=0)
    reprojected = batch_reproject(vertices, colors, intrinsics, w2c, h, w)
    return imgs, reprojected, w2c, depths, intrinsics

def _to_w2c44(extrinsics):
    """Pad [T,3,4] extrinsics to [T,4,4] (DA3 returns 3x4; MVCS's inv needs square)."""
    if extrinsics.shape[-2:] == (4, 4):
        return extrinsics
    import torch
    t = extrinsics.shape[0]
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsics.device)
    return torch.cat([extrinsics, row.view(1, 1, 4).expand(t, 1, 4)], dim=1)

def _dropout_visible_psnr(images, depths, w2c, intr):
    """Frame-dropout consistency: reproject each view from the OTHER frames only (exclude its
    own points) and PSNR over the COVERED pixels (holes excluded, since holes are disocclusion,
    which just penalizes camera travel, not consistency). Returns (mean masked PSNR, mean coverage).
    Isolates cross-view geometric error, which the all-frames metric hides via self-reproduction."""
    import torch
    from depth_anything_3.utils.geometry import affine_inverse, unproject_depth

    from .video_gpa.utils.pointcloud_utils import get_colored_pointcloud
    from .video_gpa.utils.projection_utils import batch_reproject

    T = w2c.shape[0]
    if T < 2:
        return float("nan"), float("nan")
    H, W = depths.shape[-2:]
    world = unproject_depth(depths.unsqueeze(0).unsqueeze(-1), intr.unsqueeze(0),
                            affine_inverse(w2c).unsqueeze(0)).squeeze(0)
    conf = torch.ones_like(depths)
    psnrs, covs = [], []
    for k in range(T):
        keep = [j for j in range(T) if j != k]
        pd = {"world_points_from_depth": world[keep], "depth_conf": conf[keep], "images": images[keep],
              "extrinsic": w2c[keep], "intrinsic": intr[keep], "depth": depths[keep]}
        v, c = get_colored_pointcloud(pd, mode="depth", conf_thres=0)
        rep = batch_reproject(v, c, intr[k:k + 1], w2c[k:k + 1], H, W)
        rp = (rep[0].permute(1, 2, 0) + 1) / 2  # [-1,1] -> [0,1]
        gt = images[k].permute(1, 2, 0)
        cov = rp.amax(-1) > 0.02  # holes render as exactly black
        covs.append(float(cov.float().mean()))
        if cov.any():
            m = float(((gt - rp) ** 2)[cov].mean())
            psnrs.append(10 * np.log10(1.0 / m) if m > 0 else 99.0)
    return (float(np.mean(psnrs)) if psnrs else float("nan")), float(np.mean(covs))

def score_frames(ctx, frames_np) -> dict[str, float]:
    """Run each active backbone's reprojection + metrics on decoded frames.

    Returns the METRICS dict: one shared, recon-independent epipolar plus
    mse/psnr/ssim/lpips/mvcs/consistency_score/motion under each active backbone's
    prefix. Each backbone's MVCS uses its own depths/intrinsics and the W2C extrinsics
    from the same reprojection pass, reduced as exp(-mean) over consecutive pairs with a
    valid overlap (VideoGPA skips empty pairs; 0.0 if none).
    """
    dev, sc = ctx["device"], ctx["scorer"]  # Consistency_Score owns MSE + LPIPS
    reproject = {"da3": videogpa.reproject, "vggt": reproject_vggt}
    models = {"da3": ctx["da3_model"], "vggt": ctx["vggt_model"]}

    # Epipolar is recon-independent: run once on the raw sampled frames.
    out = {"epipolar": float(ctx["epipolar"].compute(gt=frames_np, rep=frames_np))}
    for b in ACTIVE_BACKBONES:
        images, reprojected, extrinsics, depths, intr = reproject[b](
            {"model": models[b], "device": dev}, frames_np
        )
        consistency, motion = sc.compute(
            gt=images, rep=reprojected, extrinsics=extrinsics
        )
        per_pair = per_pair_mvcs(depths, w2c=_to_w2c44(extrinsics), intrinsics=intr)
        valid = [v for v in per_pair if not np.isnan(v)]
        out[f"{b}_mvcs"] = float(np.exp(-np.mean(valid))) if valid else 0.0
        out[f"{b}_mse"] = float(sc.mse_metric.compute(gt=images, rep=reprojected))
        out[f"{b}_psnr"] = float(ctx["psnr"].compute(gt=images, rep=reprojected))
        out[f"{b}_ssim"] = float(ctx["ssim"].compute(gt=images, rep=reprojected))
        out[f"{b}_lpips"] = float(sc.lpips_metric.compute(gt=images, rep=reprojected))
        out[f"{b}_consistency_score"] = float(consistency)
        out[f"{b}_motion"] = float(motion)
        dpsnr, dcov = _dropout_visible_psnr(images, depths, _to_w2c44(extrinsics), intr)
        out[f"{b}_dropout_psnr"] = dpsnr
        out[f"{b}_coverage"] = dcov
    return out

def score_video(ctx, video_path, n_frames: int = NUM_FRAMES) -> dict[str, float]:
    """Decode a video file and score it. video_path is a local path."""
    return score_frames(ctx, _decode_uniform(video_path, n_frames))
