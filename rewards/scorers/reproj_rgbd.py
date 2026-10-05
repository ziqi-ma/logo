"""Global-point-cloud reprojection reward: RGB MSE + depth MAE.

Runs VGGT-Omega on N uniformly-sampled rollout frames, fuses their unprojected depths
into one world point cloud, and reprojects that cloud into each view:
  * vggt_mse       -- RGB reprojection MSE (input vs reprojected color), the videogpa metric.
  * vggt_depth_mae -- depth reprojection MAE: render the fused cloud's nearest-z depth into
                      each view and compare to that view's own VGGT depth, masked exactly like
                      scorers.mae.per_pair_mae (occluded pixels dropped via the same threshold).
Both are lower-better. Reuses dl3dv_videogpa._load_vggt / reproject_vggt so the VGGT recon and
the RGB metric match the videogpa suite. VGGT is the only model (no DA3/lpips/lightglue)."""
import numpy as np
import torch

from . import dl3dv_videogpa as _vg

_THR = 0.8

def load(vggt_checkpoint=None, device="cuda"):
    """Load VGGT-Omega + the videogpa MSE metric once; returns a ctx for :func:`score`."""
    import types

    from .video_gpa.metrics.mse import MSEMetric
    model = _vg._load_vggt(types.SimpleNamespace(device=device, vggt_checkpoint=vggt_checkpoint))
    return {"model": model, "device": device, "mse_metric": MSEMetric()}

def _proj_depth(pc, K, E, H, W):
    """Nearest-z depth buffer of world points ``pc`` [N,3] in camera (K,E). 0 where uncovered."""
    R, t = E[:3, :3], E[:3, 3]
    pr = (pc @ R.T + t) @ K.T
    z = pr[:, 2]
    u = (pr[:, 0] / (z + 1e-8)).round().long()
    v = (pr[:, 1] / (z + 1e-8)).round().long()
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    dep = torch.zeros(H, W, device=pc.device)
    u, v, z = u[ok], v[ok], z[ok]
    if u.numel():
        o = torch.argsort(z, descending=True)   # nearest (small z) written last -> wins
        dep[v[o], u[o]] = z[o]
    return dep

def score(ctx, frames_np, per_frame=False) -> dict:
    """{vggt_mse, vggt_depth_mae} for the decoded rollout frames (a [T,...] frame array).

    ``per_frame`` additionally returns ``vggt_mse_pf`` / ``vggt_depth_mae_pf`` -- the
    per-frame arrays (length T) whose (nan)means ARE the scalars -- so the windowed
    reward can bin them per generated latent frame. depth-MAE is NaN for a frame with no
    unoccluded reprojected pixel."""
    import torch.nn.functional as F
    dev = ctx["device"]
    imgs, reproj_rgb, w2c, depths, K = _vg.reproject_vggt(ctx, frames_np)
    # Per-frame RGB MSE with the same [0,1] normalization MSEMetric uses (imgs and reproj
    # live in different ranges); the mean over frames equals the videogpa vggt_mse scalar.
    mm = ctx["mse_metric"]
    gt_t, rep_t = mm._to_tensor_01(imgs), mm._to_tensor_01(reproj_rgb)
    if gt_t.shape[-2:] != rep_t.shape[-2:]:
        rep_t = F.interpolate(rep_t, size=gt_t.shape[-2:], mode="bilinear", align_corners=False)
    mse_pf = (gt_t - rep_t).pow(2).mean(dim=(1, 2, 3))   # [T]
    vggt_mse = float(mse_pf.mean().item())

    depths = depths.to(dev).float(); w2c = w2c.to(dev).float(); K = K.to(dev).float()
    T, H, W = depths.shape
    ys, xs = torch.meshgrid(
        torch.arange(H, device=dev, dtype=torch.float32),
        torch.arange(W, device=dev, dtype=torch.float32), indexing="ij")
    c2w = torch.linalg.inv(w2c)
    pts = []
    for i in range(T):
        di, Ki = depths[i], K[i]
        X = (xs - Ki[0, 2]) / Ki[0, 0] * di
        Y = (ys - Ki[1, 2]) / Ki[1, 1] * di
        p = torch.stack([X, Y, di, torch.ones_like(di)], -1).reshape(-1, 4)
        pts.append((c2w[i] @ p.T).T[:, :3])
    pc = torch.cat(pts, 0)   # fused world cloud

    dmae_pf = []
    for i in range(T):
        rd = _proj_depth(pc, K[i], w2c[i], H, W)
        pred = depths[i]
        m = (rd > 0) & ~(pred < rd * _THR)   # per_pair_mae occlusion mask
        dmae_pf.append((pred[m] - rd[m]).abs().mean().item() if m.any() else float("nan"))
    valid = [v for v in dmae_pf if v == v]
    out = {"vggt_mse": vggt_mse,
           "vggt_depth_mae": float(np.mean(valid)) if valid else float("nan")}
    if per_frame:
        out["vggt_mse_pf"] = mse_pf.detach().cpu().tolist()
        out["vggt_depth_mae_pf"] = dmae_pf
    return out
