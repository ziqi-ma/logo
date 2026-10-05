"""VideoGPA RGB-reprojection consistency reward (GPU; lower is better).

Ported from lyra/Lyra-2/run_videogpa_for_lyra.py. For each seed, decodes the
rollout video (<scene>/seed<N>.mp4), runs DepthAnything3 to estimate geometry,
unprojects depth to a colored 3D point cloud, reprojects that cloud into every
view, and scores RGB consistency (MSE + LPIPS) of the reprojection against the
original frames. Depth drives the geometry; the metric compares RGB.

VideoGPA's metric/reprojection code is vendored under scorers/video_gpa/.
DepthAnything3 and lpips are imported lazily, so this module imports anywhere; the DA3
backbone only runs where both are present (it is off by default -- see
``VIDEOGPA_BACKBONES`` in dl3dv_videogpa).
"""

import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

def _decode_frames(video_path: Path, stride: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode every `stride`-th frame, center-cropped and resized to 518x518.

    Returns (frames [T,518,518,3], original frame indices [T]).
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

    frames, sampled = [], []
    for idx in range(0, len(raw), stride) or [0]:
        img = cv2.cvtColor(raw[idx], cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        side = min(h, w)
        img = img[(h - side) // 2:(h + side) // 2, (w - side) // 2:(w + side) // 2]
        img = cv2.resize(img, (518, 518), interpolation=cv2.INTER_LINEAR)
        frames.append(img)
        sampled.append(idx)
    return np.stack(frames, axis=0), np.asarray(sampled, dtype=np.int64)

def _ensure_da3() -> None:
    """Check that depth_anything_3 is importable, and say what to do if it is not.

    DA3_SRC_DIR prepends a local checkout to sys.path; otherwise DA3 must already be
    installed (the training images have it, and models/lingbot-world-v2 vendors it under
    wan/third_party/depth_anything_3).
    """
    try:
        import depth_anything_3  # noqa: F401
        return
    except ImportError:
        pass
    src = os.environ.get("DA3_SRC_DIR", "").strip()
    if src and src not in sys.path:
        sys.path.insert(0, src)
    try:
        import depth_anything_3  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "the da3 backbone needs depth_anything_3 importable; install it, or set "
            "DA3_SRC_DIR to a checkout. The default backbone is vggt, which does not "
            "need it (VIDEOGPA_BACKBONES)."
        ) from e

def _init(args):
    """Load the DA3 model and the LPIPS-based consistency scorer once."""
    _ensure_da3()
    import lpips
    from depth_anything_3.api import DepthAnything3

    from .video_gpa.metrics.consistency_score import Consistency_Score

    device = args.device
    try:
        model = DepthAnything3.from_pretrained("depth-anything/DA3-Large", local_files_only=True)
    except Exception:
        model = DepthAnything3.from_pretrained("depth-anything/DA3-Large")
    model = model.to(device).eval()

    lpips_net = lpips.LPIPS(net="vgg").to(device).eval()
    scorer = Consistency_Score(lpips_net=lpips_net, device=device)
    return {"model": model, "scorer": scorer, "device": device}

def reproject(ctx, frames_np):
    """Run DA3 inference + colored-point-cloud reprojection on decoded frames.

    Returns (images, reprojected, extrinsics, depths, intrinsics): images (gt)
    [T,C,H,W] in [0,1], reprojected [T,C,H,W] in [-1,1], extrinsics (W2C) [T,4,4],
    depths [T,H,W], intrinsics (3x3) [T,3,3]. depths/intrinsics feed the geometry
    metrics (MVCS). Factored out so the scalar reward (_score), the per-frame eval
    harness, and the full VideoGPA scorer share one reprojection path.
    """
    from depth_anything_3.utils.geometry import affine_inverse, unproject_depth

    from .video_gpa.utils.pointcloud_utils import get_colored_pointcloud
    from .video_gpa.utils.projection_utils import batch_reproject

    model, device = ctx["model"], ctx["device"]
    prediction = model.inference([frames_np[i] for i in range(len(frames_np))])

    images = torch.from_numpy(prediction.processed_images).float().to(device)
    if images.max() > 1.0:
        images = images / 255.0
    images = images.permute(0, 3, 1, 2).contiguous()  # [T, C, H, W]

    extrinsics = torch.from_numpy(prediction.extrinsics).float().to(device)
    intrinsics = torch.from_numpy(prediction.intrinsics).float().to(device)
    depths = torch.from_numpy(prediction.depth).float().to(device)
    confidences = (
        torch.from_numpy(prediction.conf).float().to(device)
        if prediction.conf is not None
        else torch.ones_like(depths)
    )

    c2w = affine_inverse(extrinsics)
    world_points = unproject_depth(
        depths.unsqueeze(0).unsqueeze(-1),
        intrinsics.unsqueeze(0),
        c2w.unsqueeze(0),
    ).squeeze(0)

    preds = {
        "world_points_from_depth": world_points,
        "depth_conf": confidences,
        "images": images,
        "extrinsic": extrinsics,
        "intrinsic": intrinsics,
        "depth": depths,
    }

    height, width = images.shape[-2:]
    vertices_3d, colors_rgb = get_colored_pointcloud(preds, mode="depth", conf_thres=0)
    reprojected = batch_reproject(vertices_3d, colors_rgb, intrinsics, extrinsics, height, width)
    return images, reprojected, extrinsics, depths, intrinsics

def per_frame(ctx, frames_np) -> list[float]:
    """Per-decoded-frame reprojection consistency (MSE + LPIPS), one value per frame.

    Shares reproject() with _score and mirrors Consistency_Score.compute (ratio=1), but
    keeps the per-frame terms instead of averaging — for the windowed eval harness.
    """
    import torch.nn.functional as F

    images, reprojected, _, _, _ = reproject(ctx, frames_np)
    sc = ctx["scorer"]
    g01, r01 = sc.mse_metric._to_tensor_01(images), sc.mse_metric._to_tensor_01(reprojected)
    if g01.shape[-2:] != r01.shape[-2:]:
        r01 = F.interpolate(r01, size=g01.shape[-2:], mode="bilinear", align_corners=False)
    mse_pf = (g01 - r01).pow(2).mean(dim=(1, 2, 3))
    gl, rl = sc.lpips_metric._to_tensor_neg1_pos1(images), sc.lpips_metric._to_tensor_neg1_pos1(reprojected)
    if gl.shape[-2:] != rl.shape[-2:]:
        rl = F.interpolate(rl, size=gl.shape[-2:], mode="bilinear", align_corners=False)
    with torch.no_grad():
        lp_pf = sc.lpips_metric.lpips(gl, rl).flatten()
    return (mse_pf + lp_pf).detach().cpu().tolist()
