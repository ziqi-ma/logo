"""Load benchmark camera trajectories (lyra2_traj/trajectory .npz) into UniWorld cameras.

npz contract (trajbench / Lyra NFT scene inputs):
    w2c          (N,4,4) float, world-to-camera, OpenCV, relativized to frame 0 (w2c[0]=I)
    intrinsics   (N,3,3) float, pixel-space K at (image_height, image_width)
    image_height, image_width  int scalars

UniWorld's world frame is anchored by ``set_initial_camera`` (frame-0 camera at
distance ``depth_avg`` from the world origin), so an npz pose relative to frame 0
composes as ``w2c_i_uni = w2c_npz[i] @ w2c_0_uni`` — the same alignment the dynamic
path applies to STream3R cameras (demo.py:1107-1110).
"""
from __future__ import annotations

import numpy as np
import torch

def load_trajectory_npz(path: str, pose_scale: float = 1.0):
    """Return (w2c [N,4,4] f32 tensor, K [N,3,3] f32 tensor, (H, W) the K refers to).

    ``pose_scale`` scales only the translation column (Lyra convention).
    """
    data = np.load(path)
    w2c = torch.from_numpy(data["w2c"].astype(np.float32))
    K = torch.from_numpy(data["intrinsics"].astype(np.float32))
    if pose_scale != 1.0:
        w2c = w2c.clone()
        w2c[:, :3, 3] *= pose_scale
    if "image_height" in data.files:
        hw = (int(data["image_height"]), int(data["image_width"]))
    else:  # hard-tier (Pexels) staging stores image_wh = [W, H]
        w, h = (int(x) for x in data["image_wh"])
        hw = (h, w)
    return w2c, K, hw

def rescale_intrinsics(K: torch.Tensor, src_hw, dst_hw) -> torch.Tensor:
    if tuple(src_hw) == tuple(dst_hw):
        return K
    sy = dst_hw[0] / src_hw[0]
    sx = dst_hw[1] / src_hw[1]
    K = K.clone()
    K[:, 0, 0] *= sx
    K[:, 0, 2] *= sx
    K[:, 1, 1] *= sy
    K[:, 1, 2] *= sy
    return K

def slice_setting(w2c: torch.Tensor, K: torch.Tensor, setting: str):
    """Vibe-check settings over a 241-pose trajectory (241 = 3*80+1 = 2*80+81).

    241f-native : all 241 poses, 1x speed, full trajectory
    81f-3x      : stride 3 -> 81 poses, 3x speed, full trajectory
    81f-2x      : stride 2 over first 161 -> 81 poses, 2x speed, first 2/3
    """
    n = w2c.shape[0]
    if setting == "241f-native":
        idx = torch.arange(n)
    elif setting == "81f-3x":
        idx = torch.arange(0, n, 3)
    elif setting == "81f-2x":
        idx = torch.arange(0, min(161, n), 2)
    else:
        raise ValueError(f"unknown setting {setting!r}")
    return w2c[idx], K[idx]

def align_to_world(w2c_rel: torch.Tensor, w2c_0_uni: torch.Tensor):
    """Compose frame-0-relative npz poses with UniWorld's initial camera.

    Returns (w2cs [N,4,4], c2ws [N,4,4]) on w2c_rel's device/dtype.
    """
    w2cs = w2c_rel @ w2c_0_uni.to(w2c_rel).unsqueeze(0)
    c2ws = torch.linalg.inv(w2cs)
    return w2cs, c2ws
