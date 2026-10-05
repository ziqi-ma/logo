"""Voxel DiffusionNFT reward (NFT_REWARD_VOXEL).

Spatial counterpart of the windowed per-frame reward: each rollout's fused VGGT cloud
is pooled into coarse voxels by the ``scorers.reproj_voxel`` scorer (per-voxel RGB
reproj MSE + depth reproj MAE over the observing pixels, time-resolved per frame), the
loop z-scores each (voxel, frame) cell ACROSS the scene's K rollouts (all rollouts
share the conditioning image, and VGGT anchors its world frame at the first camera, so
voxel keys correspond across rollouts without alignment), and each frame's gh x gw
patch grid is painted from the voxels its own pixels back-project into. The result is
r in [0,1] per (latent frame, patch), which ``compute_nft_loss`` upsamples to the
latent grid. ``<repo>/rewards/README.md`` documents the formulation.

The spatial map is built from the combo's GEOMETRY terms only (vggt_mse /
vggt_depth_mae); any non-geometry term stays in the scalar R, with no spatial
localization.

Env, read by the loop. The scorer side is gated by the ``voxel_frames`` parameter,
which only nft_loop's training score_epoch call passes, so validation always scores
the plain global way:
  NFT_REWARD_VOXEL  -- voxel edge as a fraction of the first frame's p90 depth
                       (e.g. 0.5). unset/0/off => disabled; on/true => 0.5.
  NFT_VOXEL_PATCH   -- patch grid "GHxGW". UNSET (the default) = latent resolution,
                       one voxel per latent cell; "8x8" selects a coarse grid
                       (see :func:`patch_grid`).
  NFT_VOXEL_DEPTH_CAP -- exclude pixels deeper than this multiple of the scale anchor
                       from voxel pooling/painting (far-tail shatter guard; they paint
                       neutral). Default 4.0; 0 disables.
  NFT_REWARD_MIX    -- weight on the LOCAL r map in a convex blend with the rollout's
                       own scalar advantage (see :func:`reward_mix`). Default 1.0 =
                       the local map replaces the scalar.
  NFT_VOXEL_LOCAL   -- weight of the within-rollout spatial-contrast term added to the
                       group advantage (penalizes a rollout's artifact regions more
                       than its surroundings). Default 0.5; 0 disables.
  NFT_VOXEL_TEMPORAL -- "rollout" (default): one pooled error per voxel. "frame": per
                       (voxel, frame) cell, so a frame-local artifact is compared
                       against the siblings AT that frame instead of drowning in the
                       voxel's whole-rollout mean (near cells collect 100k+ lifetime
                       observations under perspective). "rollout": one pooled error
                       per voxel.

Module-level imports are stdlib-only: nft_loop imports this module unconditionally in
the training env, which has no cv2 or scorer dependencies.
"""
from __future__ import annotations

import math
import os
from typing import Dict, Tuple

_GEOM_METRICS = ("vggt_mse", "vggt_depth_mae")
_MIN_OBS = 10          # min observations for a voxel's term to count (noise guard)
_DEFAULT_ALPHA = 0.5

# The scorer implementations the reward calls (scorers/reproj_rgbd.py, dl3dv_videogpa.py,
# reproj_voxel.py, base.py, ...) live once at <repo>/rewards/scorers, shared by all three
# models. This path is the PARENT of `scorers`, matching how _ensure_scorers_namespace joins
# "scorers" onto it.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "rewards", "scorers")):
    _d = os.path.dirname(_d)
_SHARED_REWARDS = os.path.join(_d, "rewards")
del _d

def rewards_dir() -> str:
    """Directory holding the ``scorers`` package: REWARDS_DIR if set, else the repo's
    shared rewards/ tree."""
    env = os.environ.get("REWARDS_DIR", "").strip()
    return env or _SHARED_REWARDS

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


def patch_grid():
    """``None`` (the DEFAULT) = latent resolution: one voxel per latent cell, no patch
    aggregation. An explicit ``NFT_VOXEL_PATCH="GHxGW"`` (e.g. "8x8") selects the legacy
    coarse grid instead of one voxel per latent cell."""
    v = os.environ.get("NFT_VOXEL_PATCH", "").strip().lower()
    if not v or v in ("latent", "none", "auto"):
        return None
    g = v.split("x")
    return int(g[0]), int(g[1])

def latent_credit() -> bool:
    """NFT_REWARD_LATENT: per-LATENT-CELL credit assignment (image space), no voxels.

    The third variant, sitting between the windowed per-frame reward and the
    voxel reward:

      global      one r per rollout
      window/framewise   r per (window of) latent FRAME  -- temporal only, full frame each
      LATENT (this)      r per (latent frame, latent cell) keyed in IMAGE space
      voxel              r per (world voxel, frame), painted back onto latent cells

    Versus voxel: the cross-rollout comparison is made at the same IMAGE cell rather than the
    same WORLD voxel. All rollouts of a scene share the conditioning image and the camera
    trajectory, so cell (t, i, j) is the same view of the same region across rollouts -- no
    3D pooling, no depth cap, no backprojection painting, and nothing to match (voxel's
    matched_frac failure mode cannot occur). The signal is the scorer's own per-cell reproj
    error table, which it already computes at latent resolution and used to discard.
    """
    v = os.environ.get("NFT_REWARD_LATENT", "").strip().lower()
    return v not in ("", "0", "off", "false", "no")

def depth_cap() -> float:
    v = os.environ.get("NFT_VOXEL_DEPTH_CAP", "").strip()
    return float(v) if v else 4.0

def local_lambda() -> float:
    v = os.environ.get("NFT_VOXEL_LOCAL", "").strip()
    return float(v) if v else 0.0

def reward_mix() -> float:
    """``NFT_REWARD_MIX``: weight on the local r map when the voxel reward is active.

    ``r = mix * r_local + (1 - mix) * R_adv`` per cell, where ``R_adv`` is the rollout's
    own scalar group advantage -- the number the global reward would have trained on.
    So 1.0 is "the local map replaces the scalar", 0.0 is the plain global reward, and
    0.5, the default, grades every cell half on its own key's contrast and half on how
    the whole rollout placed.

    Both operands are z-scores, so the blend keeps the reward in the same units -- but
    not the same variance: the mix is a convex combination of two imperfectly correlated
    z-scores, so its std is <= 1 and the effective step size drops with it. Read
    ``r_std`` in the step log before comparing a mixed run's LoRA drift against a pure
    one; this is the same calibration trap that made an uncalibrated voxel run look like
    a recipe change rather than a gain change.

    Rollouts the voxel scorer gave no map still train on the pure scalar, mix or no
    mix, and hpsv3-only phases have no spatial decomposition at all, so they are global
    by construction.
    """
    v = os.environ.get("NFT_REWARD_MIX", "").strip()
    if not v:
        return 0.5
    try:
        m = float(v)
    except ValueError:
        raise ValueError(f"NFT_REWARD_MIX must be a float in [0,1], got {v!r}") from None
    if not 0.0 <= m <= 1.0:
        raise ValueError(f"NFT_REWARD_MIX must be in [0,1], got {m}")
    return m

def blend_local_global(r_local, r_global: float, mix: float):
    """Convex blend of a nested local r map (``[T][gh][gw]``) with a scalar advantage.

    Returns ``r_local`` untouched at ``mix >= 1.0`` so the default path is bit-identical
    to the pre-knob behaviour (and allocates nothing).
    """
    if mix >= 1.0:
        return r_local
    g = float(r_global)
    if isinstance(r_local, (int, float)):
        return mix * float(r_local) + (1.0 - mix) * g
    return [blend_local_global(x, g, mix) for x in r_local]


def _ensure_scorers_namespace() -> None:
    """Create the synthetic ``scorers`` package namespace over the scorers tree
    without executing ``scorers/__init__`` (which eagerly imports hpsv3/moge and
    fails outside the scorer env). Identical to what nft_score's
    ``_import_reproj_rgbd`` sets up (a no-op when that already ran, as in the
    reproj_runner child); duplicated here so this module stays loadable without
    the ``wan`` package (tests load rl modules by file path)."""
    import sys
    import types

    if "scorers" in sys.modules:
        return
    rewards = rewards_dir()
    pkg = types.ModuleType("scorers")
    pkg.__path__ = [os.path.join(rewards, "scorers")]
    sys.modules["scorers"] = pkg

def _import_reproj_voxel():
    """Load the reproj_voxel scorer as ``scorers.reproj_voxel``.

    Prefers the tree via the synthetic ``scorers`` package; the baked 
    checkout lacks the file, so the normal path is the vendored copy in
    wan/rl/scorers, executed under the same dotted name so its relative
    imports (reproj_rgbd, dl3dv_videogpa) resolve against the same package."""
    import importlib
    import importlib.util
    import sys

    _ensure_scorers_namespace()
    try:
        return importlib.import_module("scorers.reproj_voxel")
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scorers", "reproj_voxel.py")
    spec = importlib.util.spec_from_file_location("scorers.reproj_voxel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scorers.reproj_voxel"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("scorers.reproj_voxel", None)
        raise
    return mod

def score_voxel_clip(ctx, mp4_path: str, n_frames: int,
                     alpha: float, grid: Tuple[int, int], cap: float):
    """Voxel error tables for one rollout clip -> (scalars, voxel_payload).

    ``ctx`` is the caller's loaded reproj_rgbd ctx (one VGGT load serves both the
    scalar reward and this). ``n_frames`` is the latent count -- the loop passes
    latent_T so the payload's t axis matches x0.shape[1] exactly (a mismatch makes
    the trainer's shape guard pad/neutralize frames). The returned scalars are
    telemetry only -- the reward's scalars come from the reproj_rgbd path, which scores
    a different frame count.

    Voxel config comes in as explicit arguments (not env) so the sacrificial
    scorer child needs no env contract beyond the manifest."""
    rv = _import_reproj_voxel()
    frames = _decode_uniform_full(mp4_path, n_frames)
    if frames is None:
        raise ValueError(f"no frames decoded from {mp4_path}")
    res = rv.score_voxel(ctx, frames, alpha=alpha,
                         patch_grid=(None if grid is None else tuple(grid)),
                         depth_cap=cap)
    _keys = ["voxel_keys", "voxel_stats", "voxel_stats_t", "alpha", "grid", "n_frames"]
    _keys.append("latent_voxel" if "latent_voxel" in res else "patch_voxels")
    payload = {k: res.pop(k) for k in _keys}
    # viz-only field; anything left in `res` lands in the jsonl metrics dict,
    # whose values downstream aggregation treats as floats.
    if latent_credit():
        # [T][gh][gw][2] = per-cell mean (rgb reproj err, depth err); at latent resolution
        # this grid IS the latent grid. The voxel path discards it as viz-only; it is the
        # entire signal for per-latent credit, so keep it (and nothing else changes).
        payload["cell_errors"] = res.get("patch_errors")
    res.pop("patch_errors", None)
    return res, payload

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

def pervoxel_advantage(payloads, spec, adv_clip_max=2.0, lam=0.5, temporal="rollout",
                       std_eps=1e-4):
    """Pure single-group voxel advantage: {rollout_id: voxel_payload} -> per-patch r.

    The env-free core wrapped by :func:`global_pervoxel_r`. Per (voxel, frame) cell
    -- or per pooled voxel when ``temporal="rollout"`` -- the combo-weighted geometry
    error is mean-centered across the rollouts observing it, scaled by the cell's own
    std (group z), plus ``lam`` x the within-rollout median/MAD contrast of the
    consensus deviation (the term that localizes artifacts). Clipped at
    ``adv_clip_max``, mapped to [0, 1], painted per patch via each pixel's own voxel
    (unmatched patches paint neutral 0.5). Returns ({rollout_id: [T][gh][gw]},
    diagnostics)."""
    gterms = [t for t in spec if t.metric in _GEOM_METRICS]
    entries = [(0, rid, vp) for rid, vp in payloads.items()]

    # Frame count comes from voxel_stats_t, not the membership map: global_pervoxel_r
    # strips `latent_voxel` before the all_gather (it is megabytes and only needed to paint
    # a rollout's own r), so other ranks' entries carry no map. Taking min() over the map
    # therefore yielded L=0 -> the painting loop ran range(0) -> matched/total = 0/0 ->
    # matched_frac NaN -> the metric silently vanished and every rollout fell back to
    # scalar r. voxel_stats_t is always present and always gathered.
    L = min(len(vp.get("voxel_stats_t") or []) for _, _, vp in entries)
    per_frame = temporal != "rollout"
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

    r_out = {}
    match_fracs = []
    for rk, idx, vp in entries:
        keys = vp["voxel_keys"]
        rv = r_vox[(rk, idx)]
        gh, gw = vp["grid"]
        lat = vp.get("latent_voxel")
        if lat is None and vp.get("patch_voxels") is None:
            continue          # map stripped before the gather (another rank's rollout)
        r_seq, matched, total = [], 0, 0
        for t in range(L):
            frame = []
            if lat is not None:
                # Latent resolution: cell -> its single owning voxel, no averaging.
                # -1 means no pixel of that cell survived the depth cap -> neutral.
                for i in range(gh):
                    row = []
                    for j in range(gw):
                        vi = lat[t][i][j]
                        if vi < 0:
                            row.append(0.5)
                            continue
                        total += 1
                        cell = ((tuple(keys[vi]), t) if per_frame else tuple(keys[vi]))
                        rval = rv.get(cell)
                        if rval is None:
                            row.append(0.5)
                        else:
                            matched += 1
                            row.append(rval)
                    frame.append(row)
            else:
                for i in range(gh):
                    row = []
                    for j in range(gw):
                        wsum, num = 0.0, 0.0
                        for vi, cnt in vp["patch_voxels"][t][i][j]:
                            total += 1
                            cell = ((tuple(keys[vi]), t) if per_frame
                                    else tuple(keys[vi]))
                            rval = rv.get(cell)
                            if rval is not None:
                                matched += 1
                                wsum += cnt * rval
                                num += cnt
                        row.append((wsum / num) if num > 0 else 0.5)
                    frame.append(row)
            r_seq.append(frame)
        r_out[idx] = r_seq
        if total:
            match_fracs.append(matched / total)
    diag = {
        "voxel_std": (sum(vox_stds) / len(vox_stds)) if vox_stds else float("nan"),
        "matched_frac": (sum(match_fracs) / len(match_fracs)) if match_fracs
                        else float("nan"),
    }
    return r_out, diag

def global_pervoxel_r(local_payloads, scene, rank, world_size, adv_clip_max, spec,
                      std_eps=1e-4):
    """Distributed wrapper for :func:`pervoxel_advantage`: all_gather the per-rank
    payloads, run the advantage per scene group (env-configured lam/temporal), and
    return this rank's ({rollout_idx: [T][gh][gw]}, pooled diagnostics).

    Contains a collective -- every rank of the world must call it, even with empty
    ``local_payloads``, or siblings hang in the gather."""
    import torch.distributed as dist

    # The latent map is ~T*gh*gw ints per rollout (megabytes at latent resolution) and
    # is only needed to paint a rollout's own r -- the cross-rollout z-score needs just
    # voxel_keys/voxel_stats_t. Strip it before the gather, re-attach locally after, so
    # going from 8x8 to latent resolution costs the collective nothing.
    _light = {idx: {k: v for k, v in vp.items() if k != "latent_voxel"}
              for idx, vp in (local_payloads or {}).items()}
    # Drop the (voxel, frame) rows the advantage would discard anyway. _voxel_G_t keeps a cell
    # only when every geometry term in `spec` cleared _MIN_OBS, so filtering on exactly those
    # terms here changes no reward -- it only stops the collective carrying rows that are
    # about to be thrown away. It is what makes small voxels affordable: at alpha 0.02 the
    # gather was 3.5 GB/step and killed the c10d transport (sendBytes, Utils.hpp:653) at step
    # 2, and 57% of its cells were below the floor. New lists, so the caller-owned payload
    # (whose latent_voxel painting still indexes the ORIGINAL rows) is untouched.
    _need_rgb = any(t.metric == "vggt_mse" for t in spec)
    _need_dep = any(t.metric == "vggt_depth_mae" for t in spec)
    if _need_rgb or _need_dep:
        for _vp in _light.values():
            _st = _vp.get("voxel_stats_t")
            if not _st:
                continue
            _vp["voxel_stats_t"] = [
                [r for r in rows
                 if (not _need_rgb or r[2] >= _MIN_OBS) and (not _need_dep or r[4] >= _MIN_OBS)]
                for rows in _st]
    payload = {"rank": rank, "scene": scene, "VP": _light}
    if world_size > 1 and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene: Dict[str, dict] = {}
    for p in gathered:
        for idx, vp in (p.get("VP") or {}).items():
            by_scene.setdefault(p["scene"], {})[(p["rank"], int(idx))] = vp

    for idx, vp in (local_payloads or {}).items():
        if "latent_voxel" in vp:
            by_scene.setdefault(scene, {})[(rank, int(idx))] = vp   # full, with the map

    lam = local_lambda()
    temporal = os.environ.get("NFT_VOXEL_TEMPORAL", "rollout").strip().lower()

    r_global, stds, fracs = {}, [], []
    for sc, group in by_scene.items():
        r_grp, diag = pervoxel_advantage(group, spec, adv_clip_max=adv_clip_max,
                                         lam=lam, temporal=temporal, std_eps=std_eps)
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

def perlatent_advantage(payloads, spec, adv_clip_max=2.0, lam=0.0, std_eps=1e-4):
    """Per-LATENT-CELL advantage: {rollout_id: payload with cell_errors} -> {id: [T][gh][gw]}.

    Identical statistics to :func:`pervoxel_advantage` -- the combo-weighted geometry error is
    mean-centered across the rollouts at that key, divided by the key's own std, optionally plus
    ``lam`` x the within-rollout contrast, clipped at ``adv_clip_max`` and mapped to [0, 1]. The
    only difference is the key: (t, i, j) in image space instead of (world voxel, t). A cell with
    no valid pixels in a rollout (NaN error) is simply absent for that rollout, and a cell whose
    group std collapses paints neutral 0.5 -- same conventions as the voxel path, so the two are
    comparable variants of the same reward.
    """
    gterms = [t for t in spec if t.metric in _GEOM_METRICS]
    if not gterms:
        return {}, {"cell_std": float("nan"), "matched_frac": float("nan")}

    # {rollout: {(t,i,j): combo-weighted z}} from the scorer's per-cell error table
    G, shapes = {}, {}
    for rid, vp in payloads.items():
        ce = vp.get("cell_errors")
        if not ce:
            continue
        g = {}
        T = len(ce)
        gh = len(ce[0]) if T else 0
        gw = len(ce[0][0]) if gh else 0
        shapes[rid] = (T, gh, gw)
        for t, rows in enumerate(ce):
            for i, row in enumerate(rows):
                for j, pair in enumerate(row):
                    vals = {"vggt_mse": pair[0], "vggt_depth_mae": pair[1]}
                    tot = 0.0
                    ok = True
                    for term in gterms:
                        v = vals.get(term.metric)
                        if v is None or not (v == v) or v in (float("inf"), float("-inf")):
                            ok = False
                            break
                        tot += term.sign * term.weight * (v - term.mean) / term.std
                    if ok:
                        g[(t, i, j)] = tot
        G[rid] = g
    if not G:
        return {}, {"cell_std": float("nan"), "matched_frac": float("nan")}

    # cross-rollout z per cell
    keys = {}
    for rid, g in G.items():
        for k in g:
            keys.setdefault(k, []).append(rid)
    adv, stds = {rid: {} for rid in G}, []
    for k, rids in keys.items():
        if len(rids) < 2:
            continue
        vals = [G[r][k] for r in rids]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))
        stds.append(std)
        if std < std_eps:
            continue
        for r in rids:
            adv[r][k] = (G[r][k] - mean) / (std + std_eps)

    # optional within-rollout spatial contrast on the consensus deviation (lam=0 by default:
    # this baseline isolates the KEYING change, and voxel's lam=0.5 would confound it)
    if lam:
        for rid, a in adv.items():
            if not a:
                continue
            vs = sorted(a.values())
            med = vs[len(vs) // 2]
            mad = sorted(abs(v - med) for v in a.values())[len(a) // 2] or std_eps
            for k in list(a):
                a[k] = a[k] + lam * (a[k] - med) / mad

    out, matched, total = {}, 0, 0
    for rid, (T, gh, gw) in shapes.items():
        a = adv.get(rid) or {}
        grid = [[[0.5] * gw for _ in range(gh)] for _ in range(T)]
        for (t, i, j), z in a.items():
            # NO negation: term.sign in the combo spec already makes G reward-like (a LOW
            # error yields a HIGH g), so the advantage maps straight through exactly as
            # pervoxel_advantage does. Negating here inverted the reward -- a synthetic group
            # gave the lowest-error rollout r=0.029 and the worst r=0.971.
            zc = max(-adv_clip_max, min(adv_clip_max, z))
            grid[t][i][j] = zc / (2.0 * adv_clip_max) + 0.5
        matched += len(a)
        total += T * gh * gw
        out[rid] = grid
    return out, {"cell_std": (sum(stds) / len(stds)) if stds else float("nan"),
                 "matched_frac": (matched / total) if total else float("nan")}

def global_perlatent_r(local_payloads, scene, rank, world_size, adv_clip_max, spec,
                       std_eps=1e-4):
    """Distributed wrapper for :func:`perlatent_advantage`. Mirrors global_pervoxel_r,
    including the invariant that every rank must call it (it contains a collective).

    Unlike the voxel wrapper there is nothing to strip: cell_errors IS the gathered signal.
    That makes the collective heavier than voxel's -- T*gh*gw*2 floats per rollout -- which is
    the price of comparing at image-cell resolution.
    """
    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "VP": local_payloads or {}}
    if world_size > 1 and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene: Dict[str, dict] = {}
    for p in gathered:
        for idx, vp in (p.get("VP") or {}).items():
            by_scene.setdefault(p["scene"], {})[(p["rank"], int(idx))] = vp

    lam = float(os.environ.get("NFT_LATENT_LOCAL", "0") or 0)
    r_global, stds, fracs = {}, [], []
    for sc, group in by_scene.items():
        r_grp, diag = perlatent_advantage(group, spec, adv_clip_max=adv_clip_max,
                                         lam=lam, std_eps=std_eps)
        r_global.update(r_grp)
        if math.isfinite(diag["cell_std"]):
            stds.append(diag["cell_std"])
        if math.isfinite(diag["matched_frac"]):
            fracs.append(diag["matched_frac"])
    r_by_idx = {idx: r_global[(rank, int(idx))]
                for idx in (local_payloads or {}) if (rank, int(idx)) in r_global}
    return r_by_idx, {"voxel_std": (sum(stds) / len(stds)) if stds else float("nan"),
                      "matched_frac": (sum(fracs) / len(fracs)) if fracs else float("nan")}

def expand_r_to_latent(r, H: int, W: int):
    """[B, T, gh, gw] -> [B, 1, T, H, W] by nearest upsampling: latent pixel (y, x)
    reads patch (floor(y*gh/H), floor(x*gw/W)) -- both grids partition the same image
    plane uniformly, so nearest-neighbor IS the index map."""
    import torch.nn.functional as F

    return F.interpolate(r, size=(H, W), mode="nearest").unsqueeze(1)
