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

Env (read by the loop and inherited by the scorer subprocess; the scorer itself is
gated by the ``--voxel N`` CLI flag, which only the TRAIN score command passes --
so validation always scores the plain global way):
  NFT_REWARD_VOXEL  -- voxel edge as a fraction of the first frame's p90 depth
                       (e.g. 0.5). unset/0/off => disabled; on/true => 0.5.
  NFT_VOXEL_PATCH   -- patch grid "GHxGW", default 8x8.
  NFT_VOXEL_DEPTH_CAP -- exclude pixels deeper than this multiple of the scale anchor
                       from voxel pooling/painting (far-tail shatter guard; they paint
                       neutral). Default 4.0; 0 disables.
  NFT_VOXEL_LOCAL   -- weight of the within-rollout spatial-contrast term added to the
                       group advantage (penalizes a rollout's artifact regions more
                       than its surroundings). Default 0.5; 0 disables.
  NFT_REWARD_MIX    -- weight of the per-voxel GRID when blended with the scene's
                       GLOBAL scalar r: r = (1-mix)*scalar + mix*grid, per cell.
                       Default 1.0 = the historical behaviour, grid supersedes the
                       scalar. 0.5 = half global / half voxel. Only reproj phases
                       produce a grid, so camera/hpsv3 steps are unaffected.
  NFT_VOXEL_TEMPORAL -- "rollout" (default): one pooled error per voxel. "frame": per
                       (voxel, frame) cell, so a frame-local artifact is compared
                       against the siblings AT that frame instead of drowning in the
                       voxel's whole-rollout mean (near cells collect 100k+ lifetime
                       observations under perspective). "rollout": one pooled error
                       per voxel.

Module-level imports are stdlib-only: nft_loop imports this module
unconditionally in the training env, which has no cv2/scorer deps.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

_GEOM_METRICS = ("vggt_mse", "vggt_depth_mae")
_MIN_OBS = 10          # min observations for a voxel's term to count (noise guard)
_DEFAULT_ALPHA = 0.5

def _rewards_root():
    """REWARDS_DIR if set, else the repo's shared rewards/ tree (found by walking up)."""
    env = os.environ.get("REWARDS_DIR", "").strip()
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    while d != os.path.dirname(d) and not os.path.isdir(os.path.join(d, "rewards", "scorers")):
        d = os.path.dirname(d)
    return os.path.join(d, "rewards")

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


def patch_grid() -> Tuple[int, int]:
    """Grid the voxel r is painted onto. ``latent`` (the default) paints straight onto the
    latent grid, so the weight lands at the resolution the loss applies it at; ``GHxGW``
    keeps a coarse image grid whose cells span many latents."""
    g = os.environ.get("NFT_VOXEL_PATCH", "latent").strip().lower()
    if g == "latent":
        h, w = (int(v) for v in os.environ.get("RESOLUTION", "480,832").split(","))
        return h // 8, w // 8          # Wan2.1 VAE spatial compression
    gh, gw = g.split("x")
    return int(gh), int(gw)

def depth_cap() -> float:
    v = os.environ.get("NFT_VOXEL_DEPTH_CAP", "").strip()
    return float(v) if v else 4.0

def local_lambda() -> float:
    v = os.environ.get("NFT_VOXEL_LOCAL", "").strip()
    return float(v) if v else 0.0

def mix_lambda() -> float:
    """Weight of the voxel grid when blended with the global scalar r.

    1.0 reproduces the original behaviour, where a usable grid replaced the scalar
    outright. Values < 1 keep a share of the scene-level scalar in every cell, which is
    what a global+voxel mix run wants; the default is 0.5.
    """
    v = os.environ.get("NFT_REWARD_MIX", "").strip()
    if not v:
        return 0.5
    return min(1.0, max(0.0, float(v)))

@dataclass(frozen=True)
class Term:
    metric: str
    weight: float
    mean: float
    std: float
    sign: float

def combo_spec():
    """The reward combo as Term objects, from nft_score_cli's UniWorld-calibrated
    COMBO/NORM (mu cancels in the per-cell z; sigma sets the real mse:depth weight)."""
    from rl.scoring.nft_score_cli import COMBO, NORM

    return [Term(m, w, *NORM[m]) for m, w in COMBO]


def _import_reproj_voxel():
    """Load the reproj_voxel scorer as ``scorers.reproj_voxel``.

    Prefers the tree via the synthetic ``scorers`` package (created by
    nft_score_cli._import_scorers); the baked checkout lacks the file, so the
    normal path is the vendored copy in rl/scorers_ext, executed under the same
    dotted name so its relative imports (reproj_rgbd, dl3dv_videogpa) resolve
    against the package."""
    import importlib
    import importlib.util
    import sys

    try:
        return importlib.import_module("scorers.reproj_voxel")
    except ImportError:
        pass
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "scorers_ext", "reproj_voxel.py")
    spec = importlib.util.spec_from_file_location("scorers.reproj_voxel", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["scorers.reproj_voxel"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop("scorers.reproj_voxel", None)
        raise
    return mod

def score_voxel_clip(ctx, clip_path: str, n_frames: int):
    """Voxel error tables for one rollout clip -> (scalars, voxel_payload).

    ``ctx`` is the caller's loaded reproj_rgbd ctx (one VGGT load serves both the
    scalar reward and this). ``n_frames`` is the latent count -- the loop passes
    latent_T so the payload's t axis matches x0.shape[2] exactly (a mismatch makes
    the loop's shape guard silently downgrade the rollout to scalar r). The returned
    scalars are telemetry only -- the reward's scalars are computed separately by the
    caller, which scores a different frame count.

    Precondition: nft_score_cli._import_scorers() has run (it creates the synthetic
    ``scorers`` package the scorer's relative imports resolve against)."""
    rv = _import_reproj_voxel()
    frames = _decode_uniform_full(clip_path, n_frames)
    if frames is None:
        raise ValueError(f"no frames decoded from {clip_path}")
    res = rv.score_voxel(ctx, frames, alpha=voxel_alpha(), patch_grid=patch_grid(),
                         depth_cap=depth_cap())
    payload = {k: res.pop(k)
               for k in ("voxel_keys", "voxel_stats", "patch_voxels",
                         "voxel_stats_t", "alpha", "grid", "n_frames")}
    # viz-only field; anything left in `res` lands in the jsonl metrics dict,
    # whose values downstream aggregation treats as floats.
    res.pop("patch_errors", None)
    return res, payload

def _term_z(term, v):
    """One term's signed, weighted z contribution: sign * weight * (v - mu) / sigma."""
    return term.sign * term.weight * (v - term.mean) / term.std

_CAMERA_METRICS = ("rpe_rot", "rpe_trans")

def _camera_vals(vp):
    """Camera terms as they enter a voxel cell: the CLIP scalar, broadcast unchanged.

    RPE is defined over frame PAIRS, so its error belongs jointly to two frames and localizes to
    neither a voxel nor a frame -- the same situation as hpsv3's clip mean in _voxel_raw. The
    resulting grid is spatially and temporally uniform, which makes a camera phase mathematically
    identical to a per-rollout scalar reward; the voxel path merely carries it so the two phases
    share one code path."""
    return {m: (float(vp[m]) if isinstance(vp.get(m), (int, float)) else None)
            for m in _CAMERA_METRICS}

def _voxel_raw(vp, terms):
    """{voxel key: {metric: RAW value}} for one rollout -- no normalization at all.

    Normalization is deliberately not done here: pervoxel_advantage z-scores each term across
    the scene's K rollouts first, then weights and sums, then renormalizes the sum per cell.
    hpsv3 has no voxel localization, so its clip mean is broadcast to every voxel.
    """
    out = {}
    hp = vp.get("hpsv3_pf") or []
    hp_mean = (sum(hp) / len(hp)) if hp else None
    cam = _camera_vals(vp)
    for key, (rs, rc, ds, dc) in zip(vp["voxel_keys"], vp["voxel_stats"]):
        vals = {"vggt_mse": (rs / rc) if rc >= _MIN_OBS else None,
                "vggt_depth_mae": (ds / dc) if dc >= _MIN_OBS else None,
                "hpsv3_vid": hp_mean, **cam}
        got = {}
        for t in terms:
            v = vals.get(t.metric)
            if v is None or not math.isfinite(v):
                got = None
                break
            got[t.metric] = float(v)
        if got:
            out[tuple(key)] = got
    return out

def _voxel_raw_t(vp, terms, L):
    """{(voxel key, t): {metric: RAW value}} -- per (voxel, frame) cell. hpsv3 uses that frame's
    own score from ``hpsv3_pf`` (clip mean if the series is short)."""
    out = {}
    keys = vp["voxel_keys"]
    from rl.scoring.nft_score_cli import hps_at_latent
    hp = vp.get("hpsv3_pf") or []
    hp_idx = vp.get("hpsv3_pf_idx") or []
    cam = _camera_vals(vp)          # one clip scalar per camera term, same in every (voxel, t)
    for t, rows in enumerate(vp.get("voxel_stats_t") or []):
        if t >= L:
            break
        # latent t -> nearest SCORED pixel frame (~4t), not hp[t]
        hp_t = hps_at_latent(hp, hp_idx, t)
        for vi, rs, rc, ds, dc in rows:
            vals = {"vggt_mse": (rs / rc) if rc >= _MIN_OBS else None,
                    "vggt_depth_mae": (ds / dc) if dc >= _MIN_OBS else None,
                    "hpsv3_vid": hp_t, **cam}
            got = {}
            for term in terms:
                v = vals.get(term.metric)
                if v is None or not math.isfinite(v):
                    got = None
                    break
                got[term.metric] = float(v)
            if got:
                out[(tuple(keys[int(vi)]), t)] = got
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
    # every combo term is summed into G -- hpsv3 included. The old code filtered to
    # _GEOM_METRICS here and blended hpsv3 after the z-score, which erased its scale and made the
    # nominal 0.1 weight meaningless. Sum first, z-score the sum.
    terms = list(spec)
    entries = [(0, rid, vp) for rid, vp in payloads.items()]
    # A term with no source in the payloads would void every cell: _voxel_raw* requires all terms
    # per cell, so one missing series drops the whole grid and matched_frac goes to 0 with no error
    # (hpsv3 in COMBO but hpsv3_pf absent -> matched_frac 0.0, no training). Drop the
    # unusable term loudly instead, so the run degrades to the terms it actually has.
    # GENERIC source check. The old version knew only about hpsv3_vid, so when camera_only arrived
    # rpe_rot fell straight through it: no payload could source the term, _voxel_raw* voided every
    # cell, and the grid painted neutral 0.5 -- which then SUPERSEDED a working scalar R, so three
    # camera phases would train on a constant with matched_frac=0 and no error anywhere.
    # Any term whose source is missing from every payload is now dropped loudly.
    _SOURCES = {"vggt_mse": ("voxel_keys",), "vggt_depth_mae": ("voxel_keys",),
                "hpsv3_vid": ("hpsv3_pf",), "rpe_rot": ("rpe_rot",), "rpe_trans": ("rpe_trans",)}
    kept = []
    for t in terms:
        srcs = _SOURCES.get(t.metric)
        if srcs and not any(any(vp.get(s) not in (None, [], {}) for s in srcs)
                            for _r, _i, vp in entries):
            print(f"[nft_voxel] {t.metric} is in the combo but no payload carries "
                  f"{'/'.join(srcs)}; dropping it from the VOXEL reward -- the remaining terms "
                  f"carry the phase", flush=True)
            continue
        kept.append(t)
    terms = kept
    if not terms:
        # every term unusable: return nothing so the caller falls back to the scalar R, rather
        # than a full grid of neutral 0.5 that outranks it.
        print("[nft_voxel] no combo term has a per-cell source; falling back to scalar R",
              flush=True)
        return {}, {"voxel_std": float("nan"), "matched_frac": float("nan")}

    L = min(len(vp["patch_voxels"]) for _, _, vp in entries)
    per_frame = temporal != "rollout"
    # RAW per-term values per cell per rollout -- no normalization yet.
    RAW = {(rk, idx): (_voxel_raw_t(vp, terms, L) if per_frame else _voxel_raw(vp, terms))
           for rk, idx, vp in entries}

    # (cell, metric) -> {rollout: raw value}
    by_key_metric = {}
    for k_ent, cells in RAW.items():
        for key, mvals in cells.items():
            for m, v in mvals.items():
                by_key_metric.setdefault((key, m), {})[k_ent] = v

    # ORDER OF OPERATIONS (this is the contract):
    #   1. z-score each TERM across the scene's K rollouts for that cell, using the mean and the
    #      POPULATION sd of that term alone. Each term therefore enters unit-variance, which is
    #      what makes a nominal 0.45/0.45/0.1 the real weighting.
    #   2. weighted sum of those per-term z's (sign applied).
    #   3. optional lam within-rollout spatial contrast.
    #   4. then clip to +/-adv_clip_max and map to [0, 1].
    # NORM's mu/sigma are not used here -- the per-cell empirical mean/sd replace them, so a
    # miscalibrated sigma cannot skew the voxel reward (it still matters for the scalar R).
    zt = {}                       # (cell, metric) -> {rollout: z}
    contrib = {}                  # metric -> summed |w*z| before the sum, for the share panel
    per_term_std = {}
    for (key, m), obs in by_key_metric.items():
        if len(obs) < 2:
            continue
        vals = list(obs.values())
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals))   # population sd
        per_term_std.setdefault(m, []).append(std)
        if std < std_eps:
            continue
        zt[(key, m)] = {k: (v - mean) / (std + std_eps) for k, v in obs.items()}

    tw = {t.metric: (t.sign, t.weight) for t in terms}
    G = {k_ent: {} for k_ent in RAW}
    for (key, m), zs in zt.items():
        sign, w = tw[m]
        for k_ent, z in zs.items():
            G[k_ent][key] = G[k_ent].get(key, 0.0) + sign * w * z
            contrib[m] = contrib.get(m, 0.0) + abs(w * z)
            contrib["_n_" + m] = contrib.get("_n_" + m, 0) + 1

    # a cell counts only if every combo term contributed to it
    nterm = len(terms)
    cell_terms = {}
    for (key, m) in zt:
        cell_terms[key] = cell_terms.get(key, 0) + 1
    for k_ent in G:
        G[k_ent] = {k: v for k, v in G[k_ent].items() if cell_terms.get(k, 0) == nterm}

    # 3. PER-SCENE (per-cell, across-rollout) normalization of the summed value. Each term was
    #    already z-scored, but their weighted sum is not unit-variance (the terms are correlated
    #    and weights differ), so the group z is still what makes cells comparable before clipping.
    #    This is the step whose removal would let a high-variance cell saturate the clip on its own.
    by_key = {}
    for k_ent, cells in G.items():
        for key, val in cells.items():
            by_key.setdefault(key, {})[k_ent] = val
    adv_vox = {k_ent: {} for k_ent in G}
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
        r_seq, matched, total = [], 0, 0
        for t in range(L):
            frame = []
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
    # mean |contribution| per term before the cross-rollout z-score, plus each term's share of
    # the summed magnitude -- the real 0.45/0.45/0.1 as opposed to the nominal one.
    pre = {}
    for m in {k for k in contrib if not k.startswith("_n_")}:
        n = contrib.get("_n_" + m, 0)
        if n:
            pre[f"pre_contrib_{m}"] = contrib[m] / n
    tot = sum(pre.values())
    if tot > 0:
        for m in list(pre):
            pre[f"pre_share_{m.replace('pre_contrib_', '')}"] = pre[m] / tot

    diag = {
        **pre,
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

    payload = {"rank": rank, "scene": scene, "VP": local_payloads}
    if world_size > 1 and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene: Dict[str, dict] = {}
    for p in gathered:
        for idx, vp in (p.get("VP") or {}).items():
            by_scene.setdefault(p["scene"], {})[(p["rank"], int(idx))] = vp

    lam = local_lambda()
    temporal = os.environ.get("NFT_VOXEL_TEMPORAL", "rollout").strip().lower()

    r_global, stds, fracs = {}, [], []
    group_diags = []
    for sc, group in by_scene.items():
        r_grp, diag = pervoxel_advantage(group, spec, adv_clip_max=adv_clip_max,
                                         lam=lam, temporal=temporal,
                                         std_eps=std_eps)
        r_global.update(r_grp)
        group_diags.append(diag)
        if math.isfinite(diag["voxel_std"]):
            stds.append(diag["voxel_std"])
        if math.isfinite(diag["matched_frac"]):
            fracs.append(diag["matched_frac"])

    r_by_idx = {idx: r_global[(rank, int(idx))]
                for idx in (local_payloads or {}) if (rank, int(idx)) in r_global}
    # Pre-z contributions come from each scene group's own diag (pervoxel_advantage computes them
    # where `contrib` lives). This block previously referenced `contrib` directly -- a NameError
    # here, because the edit that added it matched both `diag = {` sites. Average the per-group
    # pre_contrib_* across groups, then recompute the shares from those means so they still sum
    # to 1 rather than being averaged ratios.
    pre = {}
    contribs = {}
    for d in group_diags:
        for k, v in d.items():
            if k.startswith("pre_contrib_") and isinstance(v, (int, float)) and math.isfinite(v):
                contribs.setdefault(k, []).append(v)
    for k, vals in contribs.items():
        pre[k] = sum(vals) / len(vals)
    tot = sum(pre.values())
    if tot > 0:
        for k in [k for k in pre if k.startswith("pre_contrib_")]:
            pre[f"pre_share_{k[len('pre_contrib_'):]}"] = pre[k] / tot

    diag = {
        **pre,
        "voxel_std": (sum(stds) / len(stds)) if stds else float("nan"),
        "matched_frac": (sum(fracs) / len(fracs)) if fracs else float("nan"),
    }
    return r_by_idx, diag

def expand_r_to_latent(r, H: int, W: int):
    """[B, T, gh, gw] -> [B, 1, T, H, W] by nearest upsampling: latent pixel (y, x)
    reads patch (floor(y*gh/H), floor(x*gw/W)) -- both grids partition the same image
    plane uniformly, so nearest-neighbor IS the index map."""
    import torch.nn.functional as F

    return F.interpolate(r, size=(H, W), mode="nearest").unsqueeze(1)
