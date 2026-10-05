"""Voxel DiffusionNFT reward (NFT_REWARD_VOXEL).

Spatial counterpart of the windowed per-frame reward: each rollout's fused VGGT cloud
is pooled into coarse voxels by the ``scorers.reproj_voxel`` scorer (per-voxel RGB
reproj MSE + depth reproj MAE over all observing pixels of all frames), the loop
z-scores each voxel ACROSS the scene's K rollouts (all rollouts share the conditioning
image, and VGGT anchors its world frame at the first camera, so voxel keys correspond
across rollouts without alignment), and each frame's gh x gw patch grid is painted
from the voxels its own pixels back-project into. The result is r in [0,1] per
(latent frame, patch), which ``compute_nft_loss`` upsamples to the latent grid.

Env:
  NFT_REWARD_VOXEL  -- voxel edge as a fraction of the first frame's p90 depth
                       (e.g. 0.5). unset/0/off => disabled; on/true => 0.5.
                       Yields to NFT_EVAL_GLOBAL_DEPTH (validation scores globally).
  NFT_VOXEL_PATCH   -- patch grid "GHxGW", default 8x8.
  NFT_VOXEL_DEPTH_CAP -- exclude pixels deeper than this multiple of the scale anchor
                       from voxel pooling/painting (far-tail shatter guard; they paint
                       neutral). Default 4.0; 0 disables.
  NFT_VOXEL_LOCAL   -- weight of the within-rollout spatial-contrast term added to the
                       group advantage (penalizes a rollout's artifact regions more
                       than its surroundings). Default 0.5; 0 disables.
  NFT_VOXEL_PAINT   -- "voxel" (default): depth-aware cells (each pixel's own
                       back-projected voxel, per frame). "patch": image-space cells
                       (frame x 8x8 patch mean errors) -- the bin-geometry ablation:
                       same measurement and advantage, rectangular bins instead of
                       depth-aware ones.
  NFT_VOXEL_TEMPORAL -- "rollout" (default): a cell is a voxel, and every pixel that
                       observes it contributes regardless of which frame it came from.
                       "frame": split each voxel into one cell per frame.

hpsv3 has no 3D localization: it stays the per-frame guard (z-scored across rollouts
per NFT_REWARD_WINDOW window) and is blended into r by its combo weight.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

_GEOM_METRICS = ("vggt_mse", "vggt_depth_mae")
_MIN_OBS = 10          # min observations for a voxel's term to count (noise guard)
_DEFAULT_ALPHA = 0.5
_VAE_SPATIAL = 8       # Wan2.1 VAE spatial compression (pixel HxW -> latent HxW)

def _find_shared_rewards() -> str:
    """The single shared scorers tree at ``<repo>/rewards`` (see rewards/README.md).
    Located by walking up from this file rather than a hard-coded depth, so it holds
    wherever the tree is checked out."""
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, "rewards", "scorers")):
        d = os.path.dirname(d)
    return os.path.join(d, "rewards")

def rewards_dir() -> str:
    """Directory holding the ``scorers`` package and the shared runner scripts:
    REWARDS_DIR if set, else the repo's shared rewards/ tree."""
    return os.environ.get("REWARDS_DIR", "").strip() or _find_shared_rewards()

def _rewards_root():
    return rewards_dir()

def _voxel_common():
    """The pieces of the voxel reward that are identical for every model, loaded from
    the shared ``rewards/`` tree so there is one copy (see rewards/voxel_common.py)."""
    global _VC
    if _VC is None:
        import importlib.util
        path = os.path.join(_rewards_root(), "voxel_common.py")
        spec = importlib.util.spec_from_file_location("logo_voxel_common", path)
        _VC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_VC)
    return _VC

_VC = None

def voxel_alpha() -> float:
    return _voxel_common().voxel_alpha()

def _decode_uniform_full(path: str, n: int):
    """Uncropped frames for the voxel reward (``rewards/scorers/decode.py``)."""
    global _DEC
    if _DEC is None:
        import importlib.util
        p = os.path.join(_rewards_root(), "scorers", "decode.py")
        spec = importlib.util.spec_from_file_location("logo_decode", p)
        _DEC = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_DEC)
    return _DEC.decode_uniform(path, n)

_DEC = None


def voxel_enabled() -> bool:
    """Voxel reward on, except during validation's forced global scoring."""
    if os.environ.get("NFT_EVAL_GLOBAL_DEPTH", "").strip() in ("1", "true", "True"):
        return False
    return voxel_alpha() > 0

def patch_grid() -> Tuple[int, int]:
    """Grid the voxel r is painted onto. ``latent`` paints straight onto the latent
    grid, so the loss weight lands at the resolution it is applied at; ``GHxGW`` keeps
    the coarse image grid, whose cells span many latents and blur the voxel r across
    them."""
    g = os.environ.get("NFT_VOXEL_PATCH", "latent").strip().lower()
    if g == "latent":
        h, w = (int(v) for v in os.environ.get("RESOLUTION", "480,832").split(","))
        return h // _VAE_SPATIAL, w // _VAE_SPATIAL
    gh, gw = g.split("x")
    return int(gh), int(gw)

def direct_paint() -> bool:
    return os.environ.get("NFT_VOXEL_PATCH", "8x8").strip().lower() == "latent"


def score_reproj_voxel(frames_dir, work, gpu_id: int, spec=None) -> Optional[Dict]:
    """Voxel-mode replacement for ``nft_score._score_reproj_rgbd``.

    Returns the reproj_rgbd-compatible scalar metrics (vggt_mse / vggt_depth_mae, and
    hpsv3_vid when the combo needs it) plus ``voxel_payload`` -- the per-voxel error
    table + per-patch voxel membership that :func:`global_pervoxel_r` consumes. None
    on failure (caller treats the rollout as -inf). Shares nft_score's VGGT ctx cache
    so voxel scoring and validation's global scoring load VGGT once."""
    import importlib

    from lyra_2._src.rl.scoring import nft_score as ns

    mp4 = Path(work) / "reproj_clip.mp4"
    if not ns._frames_to_mp4(Path(frames_dir), mp4):
        print(f"[nft_voxel] no frames in {frames_dir}", flush=True)
        return None
    try:
        rr = ns._import_reproj_rgbd()
        rv = importlib.import_module("scorers.reproj_voxel")
        if direct_paint() and os.environ.get("NFT_VOXEL_PAINT", "voxel").strip().lower() == "patch":
            raise RuntimeError(
                "NFT_VOXEL_PAINT=patch z-scores on image cells, which is only meaningful on "
                "the coarse grid; it has no meaning per latent. Pick one.")
        if ns._REPROJ_RGBD_CTX is None:
            ns._REPROJ_RGBD_CTX = rr.load(
                vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"), device="cuda")
        fpl = 4  # frames_per_latent (framepack); matches the windowed reward's grid
        nf = int(os.environ.get("NUM_FRAMES", "81"))
        n = max(1, -(-(nf - 1) // fpl))
        frames = _decode_uniform_full(str(mp4), n)
        if frames is None:
            return None
        res = rv.score_voxel(ns._REPROJ_RGBD_CTX, frames,
                             alpha=voxel_alpha(), patch_grid=patch_grid(),
                             depth_cap=float(os.environ.get("NFT_VOXEL_DEPTH_CAP", "4.0")),
                             csr=direct_paint())
        payload = {k: res.pop(k)
                   for k in ("voxel_keys", "voxel_stats", "patch_voxels",
                             "voxel_stats_t", "alpha", "grid", "n_frames")}
        # The CSR membership is far too large for the reward jsonl and is never needed off
        # this rank (only the rollout's own r is painted from it), so it goes to a sidecar
        # and only the path travels -- keeping the record and the all_gather the size they
        # are at 8x8, whatever the grid.
        csr = res.pop("voxel_csr", None)
        if csr is not None:
            import numpy as np

            p = Path(work) / "voxel_csr.npz"
            np.savez(p, **csr)
            payload["csr_path"] = str(p)
        # est_w2c is an ARRAY: it must ride the payload, never `res` (whose leftovers become the
        # metrics dict and get float()d -- that is the "stray list kills every rank" trap).
        if "est_w2c" in res:
            payload["est_w2c"] = res.pop("est_w2c")
        # patch cells ride the payload only when the patch painting is on; anything
        # left in `res` lands in the jsonl metrics dict, whose values the loop
        # float()s -- a stray list kills every rank.
        pe = res.pop("patch_errors", None)
        if os.environ.get("NFT_VOXEL_PAINT", "voxel").strip().lower() == "patch":
            payload["patch_errors"] = pe
        if spec is not None and any(t.metric == "hpsv3_vid" for t in spec):
            hp = ns._score_hpsv3_perframe(Path(frames_dir), Path(work), gpu_id, fpl)
            if not hp:
                return None
            payload["hpsv3_pf"] = hp
            res["hpsv3_vid"] = sum(hp) / len(hp)
        res["voxel_payload"] = payload
        return res
    except Exception as e:  # noqa: BLE001 -- isolate scorer faults -> -inf for the rollout
        print(f"[nft_voxel] score failed for {frames_dir}: {type(e).__name__}: {e}", flush=True)
        return None

def _voxel_G(vp, gterms) -> Dict[tuple, float]:
    """{voxel key: combo-weighted geometry z} for one rollout; a voxel counts only if
    every geometry term in the combo has >= _MIN_OBS observations (strict, so the
    cross-rollout z compares like with like)."""
    out = {}
    for key, (rs, rc, ds, dc) in zip(vp["voxel_keys"], vp["voxel_stats"]):
        vals = {"vggt_mse": (rs / rc) if rc >= _MIN_OBS else None,
                "vggt_depth_mae": (ds / dc) if dc >= _MIN_OBS else None}
        g = 0.0
        for t in gterms:
            v = vals[t.metric]
            if v is None or not math.isfinite(v):
                g = None
                break
            g += t.sign * t.weight * (v - t.mean) / t.std
        if g is not None:
            out[tuple(key)] = g
    return out

def _voxel_G_t(vp, gterms, L):
    """{(voxel key, t): combo-weighted geometry z} for one rollout -- the voxel's
    error observed AT frame t (>= _MIN_OBS pixels for every term in that frame)."""
    out = {}
    keys = vp["voxel_keys"]
    for t, rows in enumerate(vp.get("voxel_stats_t") or []):
        if t >= L:
            break
        for vi, rs, rc, ds, dc in rows:
            vals = {"vggt_mse": (rs / rc) if rc >= _MIN_OBS else None,
                    "vggt_depth_mae": (ds / dc) if dc >= _MIN_OBS else None}
            g = 0.0
            for term in gterms:
                v = vals[term.metric]
                if v is None or not math.isfinite(v):
                    g = None
                    break
                g += term.sign * term.weight * (v - term.mean) / term.std
            if g is not None:
                out[(tuple(keys[int(vi)]), t)] = g
    return out

def _patch_G(vp, gterms, L):
    """{(t,i,j): combo-weighted z of the patch's mean errors} -- image-space cells
    with uniform pixel evidence (the bin-geometry ablation of _voxel_G_t). NaN terms
    are skipped; a cell is absent if no term is valid."""
    out = {}
    pe = vp.get("patch_errors")
    if not pe:
        return out
    for t in range(min(L, len(pe))):
        for i, row in enumerate(pe[t]):
            for j, (rgb, dep) in enumerate(row):
                tot, used = 0.0, 0
                for term in gterms:
                    v = rgb if term.metric == "vggt_mse" else dep
                    if v is not None and math.isfinite(v):
                        tot += term.sign * term.weight * (v - term.mean) / term.std
                        used += 1
                if used:
                    out[(t, i, j)] = tot
    return out

def _membership(vp, L, ncell):
    """CSR (ptr, vox, cnt) over the first ``L`` frames' cells, from the sidecar when the
    paint is direct and from the nested lists otherwise. None when the membership is not
    on this rank (the gather carries other ranks' payloads, but not their sidecars)."""
    import numpy as np

    path = vp.get("csr_path")
    if path:
        with np.load(path) as z:
            ptr = z["ptr"][:L * ncell + 1]
            return ptr, z["vox"][:ptr[-1]], z["cnt"][:ptr[-1]]
    pv = vp.get("patch_voxels")
    if pv is None:
        return None
    vox, cnt, ptr = [], [], [0]
    for t in range(L):
        for row in pv[t]:
            for cell in row:
                for v, c in cell:
                    vox.append(v)
                    cnt.append(c)
                ptr.append(len(vox))
    return (np.asarray(ptr, dtype=np.int64), np.asarray(vox, dtype=np.int64),
            np.asarray(cnt, dtype=np.float64))

def _paint(vp, rv, L, gh, gw, per_frame):
    """[L, gh, gw] of each cell's pixel-count-weighted mean r over the voxels its own
    pixels fall in; NaN where the cell resolved no voxel. The counts are full-resolution
    pixel counts, so this IS the mean of the cell's per-pixel r."""
    import numpy as np

    ncell = gh * gw
    mem = _membership(vp, L, ncell)
    if mem is None:
        return None
    ptr, vox, cnt = mem
    keys = vp["voxel_keys"]
    idx_of = {tuple(k): i for i, k in enumerate(keys)}
    rv_arr = np.full((len(keys), L), np.nan, dtype=np.float32)
    for cell, val in rv.items():
        key, t = cell if per_frame else (cell, None)
        i = idx_of.get(key)
        if i is None:
            continue
        if t is None:
            rv_arr[i, :] = val
        elif t < L:
            rv_arr[i, t] = val

    rows = np.repeat(np.arange(len(ptr) - 1), np.diff(ptr))
    vals = rv_arr[vox, rows // ncell]
    ok = np.isfinite(vals)
    w = cnt[ok]
    num = np.bincount(rows[ok], weights=w * vals[ok], minlength=L * ncell)
    den = np.bincount(rows[ok], weights=w, minlength=L * ncell)
    out = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return out.reshape(L, gh, gw).astype(np.float32)

def _hpsv3_frame_r(entries, spec, adv_clip_max, L, std_eps=1e-4, window=None):
    """Per-frame hpsv3 r in [0,1] per rollout, z-scored across rollouts per
    ``window`` pixel frames (whole sequence when None). None if unavailable."""
    if not any(t.metric == "hpsv3_vid" for t in spec):
        return None
    series = {(rk, idx): (vp.get("hpsv3_pf") or []) for rk, idx, vp in entries}
    if any(len(s) < L for s in series.values()):
        return None
    w_lat = max(1, round(int(window) / 4)) if window else L
    out = {k: [0.5] * L for k in series}
    for ws in range(0, L, w_lat):
        we = min(ws + w_lat, L)
        aggs = {k: sum(s[ws:we]) / (we - ws) for k, s in series.items()}
        vals = list(aggs.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
        if len(vals) < 2 or std < std_eps:
            continue
        for k, a in aggs.items():
            adv = max(-adv_clip_max, min(adv_clip_max, (a - mean) / (std + std_eps)))
            for f in range(ws, we):
                out[k][f] = adv / adv_clip_max / 2.0 + 0.5
    return out

def pervoxel_advantage(payloads, spec, adv_clip_max=2.0, lam=0.5, temporal="rollout",
                       hpsv3_window=None, std_eps=1e-4):
    """Pure single-group voxel advantage: {rollout_id: voxel_payload} -> per-patch r.

    The env-free core shared by the training loop (which wraps it with the
    distributed gather) and the standalone :class:`~lyra_2._src.rl.scoring.voxel_reward.
    VoxelReward` facade. Per (voxel, frame) cell -- or per pooled voxel when
    ``temporal="rollout"`` -- the combo-weighted geometry error is mean-centered
    across the rollouts observing it, scaled by the cell's own std (group z), plus
    ``lam`` x the within-rollout median/MAD contrast of the consensus deviation (the
    term that localizes artifacts). Clipped at ``adv_clip_max``, mapped to [0, 1],
    painted per cell via each pixel's own voxel; per-frame hpsv3 (payload key
    ``hpsv3_pf``, z-scored per ``hpsv3_window`` pixel frames, whole clip when None)
    blends in by its combo weight. Returns ({rollout_id: [T,gh,gw] array},
    diagnostics); rollouts whose membership is not on this rank are absent.
    """
    gterms = [t for t in spec if t.metric in _GEOM_METRICS]
    wh = sum(t.weight for t in spec if t.metric == "hpsv3_vid")
    wh = wh / (wh + sum(t.weight for t in gterms)) if wh > 0 else 0.0
    entries = [(0, rid, vp) for rid, vp in payloads.items()]

    L = min(vp.get("n_frames") or len(vp["patch_voxels"]) for _, _, vp in entries)
    per_frame = temporal != "rollout"
    paint_patch = os.environ.get("NFT_VOXEL_PAINT", "voxel").strip().lower() == "patch"
    if paint_patch:
        G = {(rk, idx): _patch_G(vp, gterms, L) for rk, idx, vp in entries}
    else:
        G = {(rk, idx): (_voxel_G_t(vp, gterms, L) if per_frame
                         else _voxel_G(vp, gterms)) for rk, idx, vp in entries}

    by_key = {}
    for k_ent, g in G.items():
        for key, val in g.items():
            by_key.setdefault(key, {})[k_ent] = val
    # Per-cell z across the K rollouts: mean subtraction cancels region difficulty,
    # the cell's own std cancels region noise scale. At small K this is a rank
    # statistic (worst-of-K caps at -sqrt(K-1)); the lam-local term below carries
    # the within-rollout severity structure.
    adv_vox = {k: {} for k in G}
    means = {}
    vox_stds = []
    for key, obs in by_key.items():
        if len(obs) < 2:
            continue
        vals = list(obs.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
        vox_stds.append(std)
        if std < std_eps:
            continue
        means[key] = mean
        for k_ent, g in obs.items():
            adv_vox[k_ent][key] = (g - mean) / (std + std_eps)
    # Within-rollout spatial contrast: the consensus deviation d = G - group_mean
    # (difficulty stays cancelled), centered/scaled over the ROLLOUT'S own cells with
    # median/MAD -- a scale a spatial outlier cannot own, so severity survives.
    if lam > 0:
        for k_ent, g in G.items():
            d = {key: g[key] - means[key] for key in g if key in means}
            if len(d) < 8:
                continue
            vals = sorted(d.values())
            med = vals[len(vals) // 2]
            mad = sorted(abs(x - med) for x in vals)[len(vals) // 2] * 1.4826
            if mad < std_eps:
                continue
            for key, dv in d.items():
                if key in adv_vox[k_ent]:
                    adv_vox[k_ent][key] += lam * (dv - med) / mad
    r_vox = {k_ent: {key: max(-adv_clip_max, min(adv_clip_max, a))
                     / adv_clip_max / 2.0 + 0.5 for key, a in advs.items()}
             for k_ent, advs in adv_vox.items()}

    hp_r = _hpsv3_frame_r(entries, spec, adv_clip_max, L, std_eps,
                          window=hpsv3_window)
    # Appearance-only combos (NFT_COMBO_SCHEDULE's hpsv3_only phase) have no geometry terms,
    # so wh == 1 and the whole reward IS hp_r. If hpsv3_pf is missing/short, hp_r is None and
    # every cell falls back to 0.5 -> a flat reward, zero advantage, a silently wasted step.
    # Fail loudly instead: this is exactly the class of bug that hides behind a healthy log.
    if not gterms and hp_r is None:
        raise RuntimeError(
            "voxel reward: appearance-only combo but per-frame hpsv3 (hpsv3_pf) is "
            "unavailable or shorter than the latent length -- the reward would be flat "
            "(r=0.5 everywhere, zero advantage). Check the hpsv3 runner emits per-frame scores.")
    import numpy as np

    r_out = {}
    match_fracs = []
    for rk, idx, vp in entries:
        rv = r_vox[(rk, idx)]
        gh, gw = vp["grid"]
        if paint_patch:
            arr = np.array([[[rv.get((t, i, j), np.nan) for j in range(gw)]
                             for i in range(gh)] for t in range(L)], dtype=np.float32)
        else:
            arr = _paint(vp, rv, L, gh, gw, per_frame)
            if arr is None:
                continue
        ok = np.isfinite(arr)
        match_fracs.append(float(ok.mean()))
        arr = np.where(ok, arr, 0.5).astype(np.float32)
        if hp_r is not None:
            arr = (1.0 - wh) * arr + wh * np.asarray(
                hp_r[(rk, idx)][:L], dtype=np.float32).reshape(L, 1, 1)
        r_out[idx] = arr
    diag = {
        "voxel_std": (sum(vox_stds) / len(vox_stds)) if vox_stds else float("nan"),
        "matched_frac": (sum(match_fracs) / len(match_fracs)) if match_fracs
                        else float("nan"),
    }
    return r_out, diag

def global_pervoxel_r(local_payloads, scene, rank, world_size, adv_clip_max, spec,
                      std_eps=1e-4):
    """Distributed wrapper for :func:`pervoxel_advantage`: all_gather the per-rank
    payloads, run the advantage per scene group (env-configured lam/temporal/window),
    and return this rank's ({rollout_idx: [T][gh][gw]}, pooled diagnostics)."""
    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "VP": local_payloads}
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene: Dict[str, dict] = {}
    for p in gathered:
        # A sidecar path from another rank names a file on another node -- and on this node
        # it could name a live one belonging to a different rank. Drop it so those payloads
        # contribute their voxel stats to the cross-rollout z and are simply not painted.
        strip_csr = p["rank"] != rank
        for idx, vp in (p.get("VP") or {}).items():
            if strip_csr and "csr_path" in vp:
                vp = {k: v for k, v in vp.items() if k != "csr_path"}
            by_scene.setdefault(p["scene"], {})[(p["rank"], int(idx))] = vp

    lam = float(os.environ.get("NFT_VOXEL_LOCAL", "0") or 0)
    temporal = os.environ.get("NFT_VOXEL_TEMPORAL", "rollout").strip().lower()
    win = os.environ.get("NFT_REWARD_WINDOW", "full").strip().lower()
    hpsv3_window = int(win) if win not in ("", "full", "0") else None

    r_global, stds, fracs = {}, [], []
    for sc, group in by_scene.items():
        r_grp, diag = pervoxel_advantage(group, spec, adv_clip_max=adv_clip_max,
                                         lam=lam, temporal=temporal,
                                         hpsv3_window=hpsv3_window, std_eps=std_eps)
        r_global.update(r_grp)
        if math.isfinite(diag["voxel_std"]):
            stds.append(diag["voxel_std"])
        if math.isfinite(diag["matched_frac"]):
            fracs.append(diag["matched_frac"])

    r_by_idx = {idx: r_global[(rank, int(idx))]
                for idx in (local_payloads or {}) if (rank, int(idx)) in r_global}
    diag = {
        "voxel_std": (sum(stds) / len(stds)) if stds else float("nan"),
        "matched_frac": (sum(fracs) / len(fracs)) if fracs else float("nan"),
    }
    return r_by_idx, diag

def expand_r_to_latent(r, H: int, W: int):
    """[B, T, gh, gw] -> [B, 1, T, H, W] by nearest upsampling: latent pixel (y, x)
    reads cell (floor(y*gh/H), floor(x*gw/W)) -- both grids partition the same image
    plane uniformly, so nearest-neighbor IS the index map. A no-op under the direct
    paint, where the two grids are the same."""
    import torch.nn.functional as F

    if direct_paint() and tuple(r.shape[-2:]) != (H, W):
        raise ValueError(
            f"[nft_voxel] direct paint asked for the latent grid but got {tuple(r.shape[-2:])} "
            f"for a {H}x{W} latent; RESOLUTION disagrees with the sampler. Upsampling here "
            f"would silently coarsen the reward, which is the whole thing this mode removes.")
    return F.interpolate(r, size=(H, W), mode="nearest").unsqueeze(1)
