# The single shared copy of the voxel scorer, loaded as "scorers.reproj_voxel" by each model's
# nft_voxel._import_reproj_voxel so its relative imports resolve against REWARDS_DIR's scorers
# package. There is no second copy to re-sync: a stale copy used to fail as NaN rewards at
# runtime rather than at import, which is exactly the failure this consolidation removes.
"""Voxel-pooled reprojection reward: per-voxel RGB/depth reprojection error tables.

Same VGGT recon and error definitions as reproj_rgbd (fused world cloud from
unprojected depths; per-pixel RGB reprojection MSE; per-pixel depth MAE against the
fused cloud's nearest-z render under the per_pair_mae occlusion mask), but errors are
pooled per VOXEL of a coarse world grid instead of per frame:

  * every pixel of every frame back-projects (via its own depth) to a point whose
    voxel receives that pixel's RGB and depth errors (observing-pixel attribution);
  * a voxel's error is the mean over all its observations across frames -- the
    multi-view consistency of that physical region within one rollout.

All K rollouts of a scene share the conditioning image, and VGGT anchors its world
frame at the first camera, so after normalizing scale by the first frame's p90
depth the voxel keys correspond across rollouts. The training loop z-scores each
voxel's error across the K rollouts (the per-voxel advantage) and paints frames from
each pixel's own voxel; this module only produces the per-rollout tables.

score_voxel returns:
  vggt_mse / vggt_depth_mae   -- the reproj_rgbd scalars (identical math), for
                                 logging/NORM continuity.
  voxel_keys                  -- [V, 3] int grid coordinates (shared across rollouts).
  voxel_stats                 -- [V, 4] floats: rgb_sum, rgb_cnt, depth_sum, depth_cnt
                                 (means = sum/cnt; depth_cnt counts unoccluded obs).
  patch_voxels                -- [T][gh][gw] lists of [voxel_index, pixel_count]: the
                                 top-M voxels the patch's own pixels fall in.
  alpha / grid                -- echo of the voxel edge and patch grid used.
"""
import numpy as np
import torch

def _fused_cloud(depths, w2c, K, dev):
    """Back-project every pixel of every frame; point index = t*H*W + y*W + x."""
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
    return torch.cat(pts, 0)

def pool_voxels(e_rgb, depths, w2c, K, alpha, patch_grid, top_m=8, depth_cap=4.0,
                latent_mode=False, csr=False):
    """Pool per-pixel errors into voxels + per-patch voxel membership (recon-agnostic).

    ``e_rgb`` [T,H,W] per-pixel RGB reprojection error aligned with ``depths``
    [T,H,W]; ``w2c`` [T,4,4]; ``K`` [T,3,3] -- all on the same device. Returns
    (voxel_keys [V,3] long, voxel_stats [V,4] float, patch_voxels, voxel_stats_t,
    dmae_pf): ``voxel_stats_t`` holds per-FRAME sparse rows
    [voxel_index, rgb_sum, rgb_cnt, depth_sum, depth_cnt] (the time-resolved cells
    the advantage compares); ``voxel_stats`` is their sum over frames; dmae_pf is the
    reproj_rgbd per-frame depth MAE (its nanmean IS vggt_depth_mae).

    ``depth_cap`` (x anchor; 0/None = off): pixels whose own depth exceeds it are
    excluded from voxel accumulation and patch membership -- far-tail/sky geometry
    otherwise shatters into thousands of barely-observed cells on deep scenes. The
    full cloud is still rendered for the depth-MAE term, so the reproj_rgbd scalars
    are unaffected; capped-out pixels simply paint neutral."""
    from .reproj_rgbd import _THR, _proj_depth

    dev = depths.device
    T, H, W = depths.shape
    pc = _fused_cloud(depths, w2c, K, dev)                     # [T*H*W, 3]

    # Shared voxel keys: scale-normalize by the first frame's p90 depth (the frame is
    # the conditioning image, identical across the scene's K rollouts, so the keys
    # land on one grid without cross-rollout alignment). p90, not median: a close-up
    # foreground drags the median far below the scene scale (4x is not unusual) and
    # would shatter the grid; p90 tracks the scene behind it.
    s = depths[0].reshape(-1).quantile(0.9).clamp_min(1e-6)
    keys3 = torch.floor(pc / (alpha * s)).long()               # [N,3]
    uniq, inv = torch.unique(keys3, dim=0, return_inverse=True)
    V = uniq.shape[0]

    ok = ((depths <= depth_cap * s).reshape(-1) if depth_cap
          else torch.ones(T * H * W, dtype=torch.bool, device=dev))

    gh, gw = patch_grid
    py = (torch.arange(H, device=dev) * gh) // H
    px = (torch.arange(W, device=dev) * gw) // W
    pid = (py[:, None] * gw + px[None, :]).reshape(-1)         # [H*W]

    # Errors are accumulated per (voxel, frame): the voxel keeps its 3D identity
    # (shared keys across rollouts) but its error is time-resolved, so a frame-local
    # artifact is not diluted by the cell's clean observations from other frames --
    # a near cell can hold 100k+ lifetime observations under perspective, ~1000x a
    # far cell, which made whole-rollout voxel means blind to localized damage.
    def _patch_mean(vals, sel):
        ssum = torch.zeros(gh * gw, device=dev).index_add_(0, pid[sel], vals[sel])
        scnt = torch.zeros(gh * gw, device=dev).index_add_(
            0, pid[sel], torch.ones(int(sel.sum()), device=dev))
        return torch.where(scnt > 0, ssum / scnt,
                           torch.full_like(ssum, float("nan"))).reshape(gh, gw)

    rgb_sum_t = torch.zeros(T, V, device=dev)
    rgb_cnt_t = torch.zeros(T, V, device=dev)
    d_sum_t = torch.zeros(T, V, device=dev)
    d_cnt_t = torch.zeros(T, V, device=dev)
    dmae_pf, patch_voxels, patch_errors, csr_rows = [], [], [], []
    for i in range(T):
        rd = _proj_depth(pc, K[i], w2c[i], H, W)
        pred = depths[i]
        m = (rd > 0) & ~(pred < rd * _THR)
        dmae_pf.append((pred[m] - rd[m]).abs().mean().item() if m.any()
                       else float("nan"))
        fok = ok[i * H * W:(i + 1) * H * W]
        inv_i = inv[i * H * W:(i + 1) * H * W]
        rgb_sum_t[i].index_add_(0, inv_i[fok], e_rgb[i].reshape(-1)[fok])
        rgb_cnt_t[i].index_add_(0, inv_i[fok], torch.ones(int(fok.sum()), device=dev))
        sel = m.reshape(-1) & fok
        if sel.any():
            idx = inv_i[sel]
            d_sum_t[i].index_add_(0, idx, (pred - rd).abs().reshape(-1)[sel])
            d_cnt_t[i].index_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))
        pe_rgb = _patch_mean(e_rgb[i].reshape(-1), fok)
        pe_d = _patch_mean((pred - rd).abs().reshape(-1), sel)
        patch_errors.append(torch.stack([pe_rgb, pe_d], dim=-1).cpu().tolist())
        comb = (pid * V + inv_i)[fok]
        uc, cnt = torch.unique(comb, return_counts=True)
        if csr:
            # Untruncated membership, CSR: one row per (frame, cell), rows in exactly the
            # order the nested lists are walked (frame-major, then row-major over cells).
            # `uc` is sorted and comb = pid*V + inv, so each cell's voxels already form one
            # ascending run; only the empty cells have to be filled in, which scatter_add on
            # a dense per-cell length vector does. Nothing is dropped here -- the nested
            # lists are the top-`top_m`-by-count view of these same rows, derived from them
            # in score_voxel so the two cannot disagree.
            cell_i = (uc // V).long()
            lens_i = torch.zeros(gh * gw, dtype=torch.long, device=dev)
            lens_i.scatter_add_(0, cell_i, torch.ones_like(cell_i))
            csr_rows.append((lens_i.cpu(), (uc % V).long().cpu(), cnt.long().cpu()))
            continue
        if latent_mode:
            # one voxel per cell, no averaging: keep the voxel that owns the most of the cell's
            # pixels. Vectorized -- pack (count, voxel) into one int so a single
            # scatter_reduce(amax) picks the max-count voxel deterministically.
            cell = (uc // V).long()
            vox = (uc % V).long()
            packed = cnt.long() * V + vox
            best = torch.full((gh * gw,), -1, dtype=torch.long, device=dev)
            best.scatter_reduce_(0, cell, packed, reduce="amax", include_self=True)
            vmap = torch.where(best >= 0, best % V, torch.full_like(best, -1))
            patch_voxels.append(vmap.reshape(gh, gw).cpu().tolist())
            continue
        # torch.unique returns SORTED output and comb = pid*V + inv, so uc is ordered by bin and
        # every bin's entries form one CONTIGUOUS run. The original code instead did
        # `(p_of == p).nonzero()` inside `for p in range(gh*gw)`, i.e. a full scan of the pair
        # array per bin -- O(pairs * bins). That is fine at an 8x8 grid (64 bins) and quadratic
        # nonsense at the latent grid (6,240 bins) for exactly the same data. unique_consecutive
        # + split gets the same groups in one pass, O(pairs).
        p_of, v_of = (uc // V).cpu(), (uc % V).cpu()
        cnt = cnt.cpu()
        frame = [[[] for _ in range(gw)] for _ in range(gh)]
        if p_of.numel():
            bins, lens = torch.unique_consecutive(p_of, return_counts=True)
            lens_l = lens.tolist()
            offs = torch.cumsum(torch.tensor([0] + lens_l[:-1]), 0).tolist()
            v_l, c_l = v_of.tolist(), cnt.tolist()
            for b, off, ln in zip(bins.tolist(), offs, lens_l):
                if ln > top_m:
                    # only the rare oversubscribed bin pays for a sort; at the latent grid a bin
                    # covers ~8x8 pixels so this branch almost never fires (at 8x8 it fired for
                    # every bin, silently discarding all but the 8 largest voxels).
                    sl = torch.argsort(cnt[off:off + ln], descending=True)[:top_m].tolist()
                    frame[b // gw][b % gw] = [[v_l[off + j], c_l[off + j]] for j in sl]
                else:
                    frame[b // gw][b % gw] = [[v_l[off + j], c_l[off + j]] for j in range(ln)]
        patch_voxels.append(frame)

    if csr:
        # ptr[r]:ptr[r+1] slices row r out of vox/cnt; len(ptr) == T * ncell + 1.
        lens = torch.cat([l for l, _, _ in csr_rows]) if csr_rows \
            else torch.zeros(0, dtype=torch.long)
        ptr = torch.zeros(lens.numel() + 1, dtype=torch.long)
        if lens.numel():
            torch.cumsum(lens, 0, out=ptr[1:])
        patch_voxels = {
            "ptr": ptr.numpy(),
            "vox": (torch.cat([v for _, v, _ in csr_rows]) if csr_rows
                    else torch.zeros(0, dtype=torch.long)).numpy(),
            "cnt": (torch.cat([c for _, _, c in csr_rows]) if csr_rows
                    else torch.zeros(0, dtype=torch.long)).numpy(),
        }

    stats = torch.stack([rgb_sum_t.sum(0), rgb_cnt_t.sum(0),
                         d_sum_t.sum(0), d_cnt_t.sum(0)], dim=1)
    # sparse per-frame rows [voxel_index, rgb_sum, rgb_cnt, d_sum, d_cnt]
    stats_t = []
    for i in range(T):
        live = (rgb_cnt_t[i] > 0).nonzero(as_tuple=True)[0]
        rows = torch.stack([live.float(), rgb_sum_t[i][live], rgb_cnt_t[i][live],
                            d_sum_t[i][live], d_cnt_t[i][live]], dim=1)
        stats_t.append(rows.cpu().tolist())
    return uniq, stats, patch_voxels, stats_t, patch_errors, dmae_pf

def nested_from_csr(c, T, gh, gw, top_m=8):
    """The top-`top_m`-by-count per-cell membership, derived from the untruncated CSR.

    Same shape and meaning as pool_voxels' nested `patch_voxels`: [T][gh][gw] lists of
    [voxel_index, pixel_count]. Deriving it from the CSR rather than recomputing keeps the
    two views consistent by construction -- the failure mode being that a cell's truncated
    counts disagree with the full ones, which is invisible until the painted reward is wrong.
    """
    ptr, vox, cnt = c["ptr"], c["vox"], c["cnt"]
    out, r = [], 0
    for _ in range(T):
        frame = [[[] for _ in range(gw)] for _ in range(gh)]
        for i in range(gh):
            for j in range(gw):
                a, b = int(ptr[r]), int(ptr[r + 1])
                r += 1
                pairs = list(zip(vox[a:b].tolist(), cnt[a:b].tolist()))
                if len(pairs) > top_m:
                    pairs = sorted(pairs, key=lambda kv: kv[1], reverse=True)[:top_m]
                frame[i][j] = [[int(v), int(n)] for v, n in pairs]
        out.append(frame)
    return out

def score_voxel(ctx, frames_np, alpha=0.5, patch_grid=(8, 8), top_m=8,
                depth_cap=4.0, latent_div=8, csr=False) -> dict:
    """Per-voxel error tables + voxel membership for one rollout clip.

    ``patch_grid=None`` selects LATENT resolution: the grid becomes
    (H // latent_div, W // latent_div) -- the VAE's spatial factor -- and each
    cell carries the single voxel owning most of its pixels, so the reward maps
    voxel -> latent with no patch aggregation in between. An explicit (gh, gw)
    keeps the per-patch top-M membership byte-for-byte.

    ``csr=True`` additionally returns ``voxel_csr`` -- the UNtruncated per-cell
    membership as flat ptr/vox/cnt arrays -- for direct per-voxel painting, where the
    top-`top_m` nested lists throw away the pixel evidence that painting needs. It is
    far too large for a reward record, so the caller writes it to a sidecar (np.savez)
    and passes only the path. ``patch_voxels`` is still returned, derived from the CSR.
    """
    import torch.nn.functional as F

    from . import dl3dv_videogpa as _vg

    dev = ctx["device"]
    imgs, reproj_rgb, w2c, depths, K = _vg.reproject_vggt(ctx, frames_np)

    # Per-pixel RGB reprojection error with MSEMetric's [0,1] normalization; the
    # global mean reproduces the reproj_rgbd/videogpa vggt_mse exactly.
    mm = ctx["mse_metric"]
    gt_t, rep_t = mm._to_tensor_01(imgs), mm._to_tensor_01(reproj_rgb)
    if gt_t.shape[-2:] != rep_t.shape[-2:]:
        rep_t = F.interpolate(rep_t, size=gt_t.shape[-2:], mode="bilinear",
                              align_corners=False)
    e_rgb = (gt_t - rep_t).pow(2).mean(dim=1).to(dev)          # [T,H,W]
    vggt_mse = float(e_rgb.mean().item())

    depths = depths.to(dev).float()
    w2c, K = w2c.to(dev).float(), K.to(dev).float()
    if e_rgb.shape[-2:] != depths.shape[-2:]:  # same VGGT preds; guard anyway
        e_rgb = F.interpolate(e_rgb.unsqueeze(1), size=depths.shape[-2:],
                              mode="bilinear", align_corners=False).squeeze(1)

    # patch_grid=None -> latent resolution (VAE factor `latent_div`), one voxel per cell.
    latent_mode = patch_grid is None
    if latent_mode:
        Hd, Wd = depths.shape[-2:]
        patch_grid = (max(1, int(Hd) // latent_div), max(1, int(Wd) // latent_div))
    if csr and latent_mode:
        raise ValueError(
            "csr=True needs an explicit patch_grid: the CSR is per-cell membership for "
            "painting, while latent mode already collapses each cell to one voxel.")
    uniq, stats, patch_voxels, stats_t, patch_errors, dmae_pf = pool_voxels(
        e_rgb, depths, w2c, K, alpha, patch_grid, top_m=top_m, depth_cap=depth_cap,
        latent_mode=latent_mode, csr=csr)
    voxel_csr = None
    if csr:
        voxel_csr, patch_voxels = patch_voxels, nested_from_csr(
            patch_voxels, len(dmae_pf), patch_grid[0], patch_grid[1], top_m)
    valid = [v for v in dmae_pf if v == v]
    out = {
        "vggt_mse": vggt_mse,
        "vggt_depth_mae": float(np.mean(valid)) if valid else float("nan"),
        "voxel_keys": uniq.cpu().tolist(),
        "voxel_stats": stats.cpu().tolist(),
        "voxel_stats_t": stats_t,
        "patch_errors": patch_errors,
        "alpha": float(alpha),
        "grid": [patch_grid[0], patch_grid[1]],
        "n_frames": int(len(dmae_pf)),
    }
    if voxel_csr is not None:
        out["voxel_csr"] = voxel_csr
    # Exactly one of these keys is present, so a consumer can tell the two payload
    # shapes apart without a version field (patch payloads keep working unchanged).
    out["latent_voxel" if latent_mode else "patch_voxels"] = patch_voxels
    return out
