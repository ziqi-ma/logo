"""VideoGPA MVCS reward (GPU; higher is better as exp(-mse)).

Multi-View Consistency Score: project frame i's depth into frame i+1 and measure the
squared error between the re-projected depth and frame i+1's sampled depth, over every
consecutive pair. Ported from VideoGPA metrics/mvcs.py (MSE, consecutive frames, mask =
in-bounds + positive projected depth, no occlusion threshold). Reuses the VGGT recon's
depth + cameras, the same input as the mae reward.
"""

import torch
import torch.nn.functional as F

def per_pair_mvcs(depths, w2c, intrinsics) -> list[float]:
    """Reprojection-depth MSE per consecutive (i, i+1) pair (NaN where no valid pixel).

    Element p corresponds to source frame p, so it can be binned by source frame for
    per-chunk aggregation. VideoGPA's scalar MVCS is exp(-mean(per_pair)).
    """
    device = depths.device
    T, H, W = depths.shape
    n = T - 1
    if n <= 0:
        return []

    i_idx = torch.arange(n, device=device)
    j_idx = i_idx + 1

    di = depths[i_idx]
    dj = depths[j_idx]
    Ki = intrinsics[i_idx]
    Kj = intrinsics[j_idx]
    c2w_i = torch.linalg.inv(w2c[i_idx])
    w2c_j = w2c[j_idx]

    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    X = (xs - Ki[:, 0, 2, None, None]) / Ki[:, 0, 0, None, None] * di
    Y = (ys - Ki[:, 1, 2, None, None]) / Ki[:, 1, 1, None, None] * di
    pts_i = torch.stack([X, Y, di, torch.ones_like(di)], dim=1).reshape(n, 4, -1)

    pts_j = torch.bmm(w2c_j, torch.bmm(c2w_i, pts_i))

    Xj = pts_j[:, 0].reshape(n, H, W)
    Yj = pts_j[:, 1].reshape(n, H, W)
    Zj = pts_j[:, 2].reshape(n, H, W)  # re-projected (theoretical) depth in frame j

    uj = (Xj / Zj.clamp(min=1e-8)) * Kj[:, 0, 0, None, None] + Kj[:, 0, 2, None, None]
    vj = (Yj / Zj.clamp(min=1e-8)) * Kj[:, 1, 1, None, None] + Kj[:, 1, 2, None, None]

    mask = (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H) & (Zj > 0)

    un = uj / (W - 1) * 2 - 1
    vn = vj / (H - 1) * 2 - 1
    sampled = F.grid_sample(
        dj.unsqueeze(1), torch.stack([un, vn], dim=-1),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    ).squeeze(1)

    per_pair = []
    for p in range(n):
        m = mask[p]
        per_pair.append((sampled[p][m] - Zj[p][m]).pow(2).mean().item() if m.any() else float("nan"))
    return per_pair
