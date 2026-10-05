"""Reward scoring CLI for the NFT loop (runs in the SCORER env, not the training env).

Combo `reproj_rgbd` (user decision: reproj only, no HPSv3):
    R = -0.5*z(vggt_mse) - 0.5*z(vggt_depth_mae)
with Lyra's calibrated mu/sigma (nft_score.NORM). Uses the reproj_rgbd scorer
(VGGT-Omega global-point-cloud reprojection) via the namespace-package trick.

    <scorer-python> -m rl.scoring.nft_score_cli --root <rollout_root> --out rewards.jsonl \
        [--device cuda:0] [--n-frames 16]

Reads root/manifest.jsonl, writes one reward row per rollout. Any scoring failure
scores that rollout -inf (dropped from its advantage group).
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The single shared scorers tree at <repo>/rewards (see rewards/README.md). Located by walking
# up from this file instead of a hard-coded depth, so it holds wherever the tree is checked out.
_d = os.path.dirname(os.path.abspath(__file__))
while _d != os.path.dirname(_d) and not os.path.isdir(os.path.join(_d, "rewards", "scorers")):
    _d = os.path.dirname(_d)
_SHARED_REWARDS = os.path.join(_d, "rewards")
del _d
# REWARDS_DIR points at the tree holding scorers/; it defaults
# to the shared tree, which all three models load. Override it to use a checkout instead.
REWARDS_DIR = os.environ.get(
    "REWARDS_DIR", os.environ.get("REWARDS_DIR", _SHARED_REWARDS))
VGGT_CKPT = os.environ.get("VGGT_CHECKPOINT", "/tmp/vggt_omega/vggt_omega_1b_512.pt")

# Per-metric normalizers (mu, sigma, sign). The values live in norm.json beside this
# module, not here: they are calibration data, they differ per model, and a wrong sigma
# silently reweights its term. NFT_NORM_FILE points at a different file.
def _load_norm() -> dict:
    path = os.environ.get("NFT_NORM_FILE", "").strip() or \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm.json")
    with open(path) as f:
        doc = json.load(f)
    return {m: (float(v["mu"]), float(v["sigma"]), float(v["sign"]))
            for m, v in doc["metrics"].items()}

NORM: dict = _load_norm()
# Named combos. `COMBO` stays the default so every existing caller is unchanged; select another
# with --combo / NFT_COMBO. Weights are NOMINAL -- what the advantage keys on is each term's
# within-scene spread, so check reward/eff_share_* (rl/reward_diag) rather than trusting these.
COMBOS = {
    "reproj_rgbd":        [("vggt_mse", 0.5), ("vggt_depth_mae", 0.5)],
    "reproj_rgbd_hps0.1": [("vggt_mse", 0.45), ("vggt_depth_mae", 0.45), ("hpsv3_vid", 0.1)],
    "reproj_rgbd_hps0.2": [("vggt_mse", 0.40), ("vggt_depth_mae", 0.40), ("hpsv3_vid", 0.2)],
    # 0.7 geometry / 0.3 hpsv3 -- the strongest perceptual counterweight tried so far. Motivation:
    # the geometry terms are reference-free self-consistency, whose argmax is a textureless video,
    # so they need an aesthetic term with real within-scene spread to oppose flattening.
    "reproj_rgbd_hps0.3": [("vggt_mse", 0.35), ("vggt_depth_mae", 0.35), ("hpsv3_vid", 0.3)],
    # Single-objective combos, for ALTERNATING training (see nft_loop --alt-schedule) instead of
    # a weighted sum. Rationale: in a weighted sum the terms compete inside one gradient and the
    # small-weight term is dominated (measured: a nominal 0.1 hpsv3 left val HPSv3 *worse* than
    # no-hpsv3 at all). Alternating gives each objective its own undiluted update.
    "hpsv3_only": [("hpsv3_vid", 1.0)],
    # camera-only phase: RPE rotation of VGGT's estimated path vs the trajectory the sampler was
    # given (score_camera below). NORM's mu/sigma are placeholders ON PURPOSE and do not matter
    # here: a single-term combo's R is z-scored per scene across the K rollouts in
    # advantage.global_scene_advantage, so mu and sigma cancel exactly and only the SIGN survives.
    # (That is why this term has sat unused as a placeholder in Lyra's weighted vq_hps_cam combo.)
    "camera_only": [("rpe_rot", 0.5), ("rpe_trans", 0.5)],
    # rotation-weighted camera phase (70/30). Both NORM sigmas are the calibrated
    # within-scene sds, so these nominal weights ARE the effective ones -- in the scalar path
    # via (v-mu)/sigma, and in the voxel path because each term is z-scored to unit variance
    # per cell before the sign*weight sum.
    "camera_rot70": [("rpe_rot", 0.7), ("rpe_trans", 0.3)],
    # rotation only -- the parallel arm to camera_rot70. Single term, so mu/sigma cancel
    # exactly in the per-scene advantage z-score and only the sign (-1: lower is better)
    # survives; the calibrated NORM is irrelevant here.
    "camera_rot_only": [("rpe_rot", 1.0)],
}
COMBO = COMBOS[os.environ.get("NFT_COMBO", "reproj_rgbd")]

_DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
             "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
             "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_USE_AGENT_STORE", "ROLE_RANK",
             "ROLE_NAME", "ROLE_WORLD_SIZE")

def _subproc_env():
    """Env for the hpsv3 child without torchrun's rendezvous vars.

    HPSv3RewardInferencer initialises torch.distributed from the environment. Inheriting
    MASTER_ADDR/MASTER_PORT=29500 + RANK makes GLOBAL RANK 0 try to BIND the port torchrun's
    TCPStore already owns -> "EADDRINUSE: address already in use": the batch dies, hpsv3_vid is
    absent for that rank's rollouts, combine() returns -inf, and train_pass's ReduceOp.MIN then
    skips the step on all ranks. Other ranks only CONNECT, so this was rank-0-only and silent --
    it silently drops the hpsv3 term on every hpsv3 step of an alternating run.
    """
    return {k: v for k, v in os.environ.items() if k not in _DIST_ENV}

def _need_camera() -> bool:
    """True when any COMBO term is a camera metric, or NFT_LOG_CAMERA=1 asks for calibration data
    (logged without weight -- how within-scene sigma gets measured before a weight is attached)."""
    if os.environ.get("NFT_LOG_CAMERA", "").strip() in ("1", "true", "True"):
        return True
    cams = {"rpe_rot", "rpe_trans"}
    if any(m in cams for m, _w in COMBO):
        return True
    # under an alternating schedule the scorer runs before the phase is applied, so it must emit
    # camera metrics on every step if ANY phase needs them
    for name in os.environ.get("NFT_ALT_COMBOS", "").split(","):
        name = name.strip()
        if name and name in COMBOS and any(m in cams for m, _w in COMBOS[name]):
            return True
    return False

def set_combo(name: str) -> list:
    """Rebind the module-level COMBO at runtime (alternating schedule).

    nft_voxel.combo_spec() and combine() both read this module attribute when called, so
    switching it here changes the reward for every downstream consumer on the next call.
    """
    global COMBO
    if name not in COMBOS:
        raise KeyError(f"unknown combo {name!r}; have {sorted(COMBOS)}")
    COMBO = COMBOS[name]
    return COMBO

# --------------------------------------------------------------------------- #
# Camera adherence (RPE). The math is the shared rewards/scorers/camera_rpe.py (one copy
# for all three models); below is how this scorer sources the reference trajectory.
#
# Reference poses: <NFT_TRAJ_URI>/<scene>/lyra2_traj.npz, the same file rl/scene_prep read when it
# built the cond bundle (the bundle bakes the path into VACE control latents and keeps no raw w2c).
# Fetched lazily per scene and cached, like rl/cond_bundle._ensure_local.
# --------------------------------------------------------------------------- #
_TRAJ_CACHE = {}

def _camera_rpe():
    """The shared RPE implementation (``scorers.camera_rpe``), loaded by path."""
    import importlib

    _import_scorers()              # sets up the synthetic 'scorers' namespace
    return importlib.import_module("scorers.camera_rpe")

def _ref_w2c(scene):
    """Reference w2c for `scene`, from a local scene root or lazily from NFT_TRAJ_URI."""
    import numpy as np
    if scene in _TRAJ_CACHE:
        return _TRAJ_CACHE[scene]
    path = None
    root = os.environ.get("NFT_SCENES_ROOT", "").strip()
    if root:
        cand = os.path.join(root, scene, "lyra2_traj.npz")
        if os.path.exists(cand):
            path = cand
    if path is None:
        remote = os.environ.get("NFT_TRAJ_URI", "").strip().rstrip("/")
        if not remote:
            _TRAJ_CACHE[scene] = None
            return None
        cache = os.path.join(os.environ.get("NFT_TRAJ_CACHE", "/tmp/nft_traj"), scene)
        os.makedirs(cache, exist_ok=True)
        local = os.path.join(cache, "lyra2_traj.npz")
        if not os.path.exists(local):
            try:
                from rl.data.gcs_util import download_file
                tmp = local + ".part"
                download_file(f"{remote}/{scene}/lyra2_traj.npz", tmp)
                os.replace(tmp, local)          # atomic: concurrent ranks share the cache
            except Exception as e:              # noqa: BLE001
                print(f"[nft_score] traj fetch failed {scene}: {type(e).__name__}: {e}", flush=True)
                _TRAJ_CACHE[scene] = None
                return None
        path = local
    try:
        _TRAJ_CACHE[scene] = np.asarray(np.load(path)["w2c"], dtype=np.float64)
    except Exception as e:                      # noqa: BLE001
        print(f"[nft_score] traj load failed {scene}: {type(e).__name__}: {e}", flush=True)
        _TRAJ_CACHE[scene] = None
    return _TRAJ_CACHE[scene]

def score_camera(est_w2c, scene, num_frames: int = 81) -> dict:
    """RPE of the recon's estimated camera path against the trajectory the sampler was given.

    Returns {} (never -inf) when unavailable, so a missing trajectory degrades to 'no camera
    term' rather than poisoning the reward for the whole rollout.
    """
    import numpy as np
    import torch
    ref_w2c = _ref_w2c(scene)
    if ref_w2c is None or est_w2c is None:
        return {}
    try:
        # reproject_vggt returns w2c as a CUDA tensor; np.asarray on it raises TypeError, which the
        # caller's broad except swallowed -> the camera term silently did not exist for ANY rollout
        # of ANY step, so the camera phases train on nothing.
        if hasattr(est_w2c, "detach"):
            est_w2c = est_w2c.detach().cpu().numpy()
        est_w2c = np.asarray(est_w2c, dtype=np.float64)
        ref_w2c = ref_w2c[:num_frames]
        if len(est_w2c) < 2 or len(ref_w2c) < 2:
            return {}
        # the recon subsamples frames; align the reference to the estimate's count
        idx = np.linspace(0, len(ref_w2c) - 1, len(est_w2c)).round().astype(int)
        ref = torch.tensor(np.linalg.inv(ref_w2c[idx]), dtype=torch.float32)   # -> c2w
        est = torch.tensor(np.linalg.inv(est_w2c), dtype=torch.float32)        # -> c2w
        rot, trans = _camera_rpe().adherence(pred_c2w=est, target_c2w=ref)
        return {"rpe_rot": float(rot), "rpe_trans": float(trans)}
    except Exception as e:                      # noqa: BLE001
        print(f"[nft_score] camera RPE failed {scene}: {type(e).__name__}: {e}", flush=True)
        return {}

def combine(metrics: dict, allow_missing=()) -> float:
    """Weighted sum of signed z terms. Metrics in ``allow_missing`` may be absent: their weight is
    redistributed over the terms that ARE present, so one unavailable scorer degrades the reward
    instead of returning -inf (which silently drops every sample from training)."""
    total, wsum = 0.0, 0.0
    for key, w in COMBO:
        v = metrics.get(key)
        if v is None or not math.isfinite(v):
            if key in allow_missing:
                continue
            return float("-inf")
        mu, sigma, sign = NORM[key]
        total += sign * w * (float(v) - mu) / sigma
        wsum += w
    if wsum <= 0:
        return float("-inf")
    return total * (sum(w for _k, w in COMBO) / wsum)

# VAE temporal compression: 81 pixel frames -> 21 latents, so latent t covers pixel ~PIX_PER_LAT*t.
PIX_PER_LAT = 4

def hps_at_latent(hps_pf, hps_idx, t):
    """hpsv3 score for LATENT frame ``t``, from the per-sampled-PIXEL-frame series.

    rl/hpsv3_cli scores every --stride-th pixel frame (9 of 81 at the default stride 10), while
    both per-frame reward paths are indexed by latent (21 of them). Indexing hpsv3_pf by the
    latent directly is wrong twice over: it reads the score of pixel frame stride*t instead of
    ~4*t, and every latent past len(hps_pf) silently falls back to the clip mean. Map through the
    emitted pixel indices and take the nearest sampled frame.
    """
    if not hps_pf:
        return None
    if not hps_idx or len(hps_idx) != len(hps_pf):
        return sum(hps_pf) / len(hps_pf)          # no index info: clip mean, explicitly
    target = PIX_PER_LAT * int(t)
    best = min(range(len(hps_idx)), key=lambda i: abs(hps_idx[i] - target))
    return hps_pf[best]

def combine_perframe(metrics_pf: dict, hps_pf=None, hps_idx=None) -> list:
    """Per-frame combo from reproj_rgbd's `*_pf` arrays, same NORM as the scalar reward.

    Returns [] if either geometry component is missing. A frame whose components are non-finite
    (e.g. no valid depth pixels) yields None -- the windowed advantage skips those rather than
    letting a NaN poison a whole window.

    ``hps_pf``/``hps_idx`` supply hpsv3 per latent frame (see :func:`hps_at_latent`). This used to
    build ``vals`` from two hard-coded keys and then iterate COMBO, so any third term raised
    KeyError inside the per-rollout try -> R = -inf for every rollout, i.e. a whole run silently
    producing nothing. A term with no per-frame source is now skipped with a warning instead.
    """
    mse_pf = metrics_pf.get("vggt_mse_pf")
    dep_pf = metrics_pf.get("vggt_depth_mae_pf")
    if mse_pf is None or dep_pf is None:
        return []
    missing = {k for k, _w in COMBO} - {"vggt_mse", "vggt_depth_mae", "hpsv3_vid"}
    if missing:
        print(f"[nft_score] combine_perframe: no per-frame source for {sorted(missing)}; "
              f"those terms are DROPPED from the windowed reward", flush=True)
    out = []
    for i in range(min(len(mse_pf), len(dep_pf))):
        vals = {"vggt_mse": float(mse_pf[i]), "vggt_depth_mae": float(dep_pf[i])}
        h = hps_at_latent(hps_pf, hps_idx, i)
        if h is not None:
            vals["hpsv3_vid"] = float(h)
        total, ok = 0.0, True
        for key, w in COMBO:
            v = vals.get(key)
            if v is None or not math.isfinite(v):
                if key in vals:          # present but non-finite -> skip this frame
                    ok = False
                    break
                continue                 # no per-frame source -> term dropped (warned above)
            mu, sigma, sign = NORM[key]
            total += sign * w * (v - mu) / sigma
        out.append(total if ok else None)
    return out

def _import_scorers():
    if "scorers" not in sys.modules:
        pkg = types.ModuleType("scorers")
        pkg.__path__ = [os.path.join(REWARDS_DIR, "scorers")]
        sys.modules["scorers"] = pkg
    vg = importlib.import_module("scorers.dl3dv_videogpa")
    rr = importlib.import_module("scorers.reproj_rgbd")
    return vg, rr

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n-frames", type=int, default=int(os.environ.get("NFT_REPROJ_FRAMES", 16)))
    ap.add_argument("--per-frame", type=int, default=0, metavar="N",
                    help="also emit R_perframe: N frames (one per latent, so N = latent T) "
                         "decoded and scored per-frame, combined with the same NORM. Drives "
                         "the windowed reward. 0 = off. The scalar R always stays on the "
                         "16-frame decode so it is comparable across windowed/global runs.")
    ap.add_argument("--voxel", type=int, default=0, metavar="N",
                    help="also emit voxel_payload: N frames (one per latent, so N = "
                         "latent T) decoded FULL-FRAME and pooled into per-(voxel, "
                         "frame) error tables (rl.scoring.nft_voxel). Drives the voxel-"
                         "voxel reward; alpha/patch/depth-cap come from the "
                         "NFT_VOXEL_* env. 0 = off. Train-only -- the val command "
                         "never passes it, so validation stays global. The scalar R "
                         "keeps the 16-frame crop decode, untouched.")
    ap.add_argument("--num-frames", type=int, default=81,
                    help="pixel frames the rollout was generated at; truncates the reference "
                         "trajectory for the camera RPE term")
    ap.add_argument("--hpsv3", action="store_true",
                    help="also score each clip with HPSv3 (7B Qwen2-VL perceptual reward, "
                         "higher better). Validation only -- runs in HPSV3_PY, a separate "
                         "venv, because hpsv3 pins transformers==4.45.2.")
    ap.add_argument("--videogpa", action="store_true",
                    help="also run the full VideoGPA suite (both backbones: da3_*/vggt_* "
                         "psnr/ssim/lpips/mvcs/consistency/mse/coverage/dropout + epipolar). "
                         "Validation only -- ~10x the cost of the reproj reward alone.")
    args = ap.parse_args()

    from rl.loop.rollout_store import read_manifest
    recs = read_manifest(args.root)
    print(f"[nft_score] {len(recs)} rollouts", flush=True)
    vg, rr = _import_scorers()
    dec = importlib.import_module("scorers.decode")
    ctx = rr.load(vggt_checkpoint=VGGT_CKPT, device=args.device)
    gpa_ctx = None
    if args.videogpa:
        import types as _t
        gpa_ctx = vg._init(_t.SimpleNamespace(device=args.device, vggt_checkpoint=VGGT_CKPT))
        print("[nft_score] videogpa suite loaded (da3 + vggt backbones)", flush=True)

    with open(args.out, "w") as out:
        # --- hpsv3 in one subprocess for every rollout of this step ---------------------
        # The per-record path below spawns a fresh interpreter per rollout, each loading the 7B
        # model. That is fine for 4 val clips and unaffordable for a training step (k_local
        # rollouts x a model load). When hpsv3 is a REWARD term we batch instead.
        hps_batch = {}
        if args.hpsv3:
            import subprocess as _sp
            clips = [r["clip_path"] for r in recs if r.get("clip_path")]
            hp = _sp.run([os.environ.get("HPSV3_PY", "/opt/hpsv3-venv/bin/python"),
                          "-m", "rl.scoring.hpsv3_cli", "--clips", ",".join(clips),
                          "--device", args.device],
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         env=_subproc_env(),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
            try:
                line = next(l for l in reversed(hp.stdout.strip().splitlines())
                            if l.lstrip().startswith("{"))
                hps_batch = json.loads(line).get("batch") or {}
                print(f"[nft_score] hpsv3 batch: {len(hps_batch)}/{len(clips)} clips", flush=True)
            except Exception:  # noqa: BLE001 - fall back to the per-record path below
                print(f"[nft_score] hpsv3 BATCH FAILED: {hp.stdout[-300:]}{hp.stderr[-300:]}",
                      flush=True)

        for rec in recs:
            voxel_payload = None
            try:
                frames = dec.decode_uniform(rec["clip_path"], args.n_frames)
                metrics = rr.score(ctx, frames)
                # CAMERA term. reproj_rgbd.score computes VGGT's estimated w2c internally but does
                # not return it, and that file is shared code -- so re-derive the poses from the
                # already-loaded VGGT ctx via the same helper reproj_rgbd calls (no second model
                # load). Merged before R for the same reason hpsv3 is: combine() returns -inf when a
                # COMBO metric is missing.
                if _need_camera():
                    try:
                        _est = vg.reproject_vggt(ctx, frames)[2]        # (imgs, reproj, w2c, ...)
                        metrics.update(score_camera(_est, rec.get("scene"), args.num_frames))
                    except Exception as e:  # noqa: BLE001 - never kill a rollout over the camera
                        print(f"[nft_score] camera term skipped {rec.get('scene')}: "
                              f"{type(e).__name__}: {e}", flush=True)
                # merge hpsv3 before computing R: combine() returns -inf when any COMBO metric is
                # absent, so a merge further down the loop poisoned every reward while leaving the
                # logged metrics dict looking complete.
                if args.hpsv3:
                    _b = hps_batch.get(rec.get("clip_path")) or {}
                    if not _b or "error" in _b:
                        # batch missed this clip -> score it alone, still before R
                        import subprocess as _sp2
                        _hp = _sp2.run(
                            [os.environ.get("HPSV3_PY", "/opt/hpsv3-venv/bin/python"),
                             "-m", "rl.scoring.hpsv3_cli", "--clip", rec["clip_path"],
                             "--device", args.device],
                            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            env=_subproc_env(),
                            capture_output=True, text=True, encoding="utf-8", errors="replace")
                        try:
                            _l = next(l for l in reversed(_hp.stdout.strip().splitlines())
                                      if l.lstrip().startswith("{"))
                            _b = json.loads(_l)
                        except Exception:  # noqa: BLE001
                            print(f"[nft_score] hpsv3 single FAILED {rec['scene']}/"
                                  f"r{rec['rollout']}: {_hp.stdout[-200:]}{_hp.stderr[-200:]}",
                                  flush=True)
                            _b = {}
                    if _b and "error" not in _b:
                        metrics.update({k: v for k, v in _b.items() if k != "n_frames"})
                        hps_batch[rec["clip_path"]] = _b     # so the voxel payload can read it
                    else:
                        # never let a missing hpsv3 poison R: drop the term for this rollout
                        # instead of returning -inf, which would void every rollout.
                        # PRINT THE ERROR. hpsv3_cli returns per-clip failures as
                        # results[clip] = {"error": ...}; this branch used to swallow it, so
                        # rank0's hpsv3 has been silently absent on every hpsv3 step of every
                        # alternating run.
                        print(f"[nft_score] hpsv3 unavailable for {rec['scene']}/r{rec['rollout']}"
                              f": {(_b or {}).get('error', 'no batch entry')}"
                              f"; R falls back to geometry-only terms", flush=True)
                R = combine(metrics, allow_missing=("hpsv3_vid",))
                if args.voxel:
                    # a voxel failure degrades this rollout to its scalar r, not -inf
                    try:
                        from rl.scoring.nft_voxel import score_voxel_clip
                        vres, voxel_payload = score_voxel_clip(
                            ctx, rec["clip_path"], args.voxel)
                        # full-frame telemetry under distinct keys: never shadow the
                        # crop scalars that feed R / NORM / raw-metric logging
                        metrics["voxel_vggt_mse"] = vres["vggt_mse"]
                        metrics["voxel_vggt_dmae"] = vres["vggt_depth_mae"]
                        # hpsv3 has no 3D localization, so the voxel advantage blends it as a
                        # per-FRAME guard -- it reads vp["hpsv3_pf"]. Inject it here (batch result
                        # lands in `metrics` above); absent hpsv3 leaves the payload unchanged.
                        # read the BATCH, not `metrics`: the hpsv3 merge happens further down
                        # the record loop, so metrics["hpsv3_pf"] is still empty right here.
                        # camera is a GLOBAL metric: one RPE per clip, no spatial or temporal
                        # localization at all. Carry the scalars into the payload so the voxel
                        # advantage can broadcast them to every cell (same treatment hpsv3's clip
                        # mean gets in _voxel_raw). Without this the camera terms have no per-cell
                        # source, every cell is voided, and the grid silently paints neutral 0.5 --
                        # which then supersedes a perfectly good scalar R.
                        for _ck in ("rpe_rot", "rpe_trans"):
                            if isinstance(metrics.get(_ck), (int, float)):
                                voxel_payload[_ck] = float(metrics[_ck])
                        _hb = hps_batch.get(rec.get("clip_path")) or {}
                        if _hb.get("hpsv3_pf"):
                            voxel_payload["hpsv3_pf"] = _hb["hpsv3_pf"]
                            voxel_payload["hpsv3_pf_idx"] = _hb.get("hpsv3_pf_idx") or []
                    except Exception as e:  # noqa: BLE001 - voxel must not void R
                        print(f"[nft_score] voxel FAILED {rec['scene']}/"
                              f"r{rec['rollout']}: {type(e).__name__}: {e}", flush=True)
                if args.per_frame:
                    # one decoded frame per latent -> per-latent reward vector. Separate
                    # decode from the scalar R on purpose: R stays a 16-frame global score
                    # so windowed and global runs remain comparable.
                    pf = rr.score(ctx, dec.decode_uniform(rec["clip_path"], args.per_frame),
                                  per_frame=True)
                    _hb = hps_batch.get(rec.get("clip_path")) or {}
                    row_pf = combine_perframe(pf, _hb.get("hpsv3_pf"), _hb.get("hpsv3_pf_idx"))
                    if row_pf:
                        metrics["R_perframe"] = row_pf
                if args.hpsv3 and hps_batch.get(rec.get("clip_path")):
                    pass                      # already merged above, before R
                elif args.hpsv3:
                    # separate interpreter: hpsv3's transformers pin conflicts with ours
                    import subprocess
                    hp = subprocess.run(
                        [os.environ.get("HPSV3_PY", "/opt/hpsv3-venv/bin/python"),
                         "-m", "rl.scoring.hpsv3_cli", "--clip", rec["clip_path"],
                         "--device", args.device],
                        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        env=_subproc_env(),
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
                    try:
                        # hpsv3 prints banners to stdout ("Flash Attention is not
                        # installed..."), so take the last JSON object, not the last line
                        line = next(l for l in reversed(hp.stdout.strip().splitlines())
                                    if l.lstrip().startswith("{"))
                        metrics.update({k: v for k, v in json.loads(line).items()
                                        if k != "n_frames"})
                    except Exception:  # noqa: BLE001 - telemetry only
                        print(f"[nft_score] hpsv3 FAILED {rec['scene']}/r{rec['rollout']}: "
                              f"{hp.stdout[-300:]}{hp.stderr[-300:]}", flush=True)
                if gpa_ctx is not None:
                    # full suite scores the clip itself (its own uniform decode); merged
                    # alongside the reward metrics so callers log them as val/<metric>
                    try:
                        metrics.update(vg.score_video(gpa_ctx, rec["clip_path"]))
                    except Exception as e:  # noqa: BLE001 - suite must not void the reward
                        print(f"[nft_score] videogpa FAILED {rec['scene']}/r{rec['rollout']}: "
                              f"{type(e).__name__}: {e}", flush=True)
            except Exception as e:  # noqa: BLE001 - one bad rollout must not abort the step
                print(f"[nft_score] {rec['scene']}/r{rec['rollout']} FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                metrics, R = {}, float("-inf")
            row = {**rec, "R": R, "metrics": metrics}
            if voxel_payload is not None:
                # top-level, never inside metrics: local_metrics rides a second
                # all_gather in global_scene_advantage and feeds per-seed float rows
                row["voxel_payload"] = voxel_payload
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[nft_score] {rec['scene']}/r{rec['rollout']}: R={R:.4f} {metrics}", flush=True)
    print(f"[nft_score] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
