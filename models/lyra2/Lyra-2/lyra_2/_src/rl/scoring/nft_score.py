# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the DiffusionNFT pipeline: offline reward scoring.

Reads the rollout-store manifest the sampler produced, runs the
reward jobs over each sample's decoded clip, combines them into the calibrated
z-score combo, and writes ``rewards_epoch_{E}.jsonl`` keyed by sample.

Every metric comes from the shared ``<repo>/rewards`` tree, loaded by path under a
synthetic ``scorers`` package, so the reward and the eval report one implementation.
HPSv3 runs in its own interpreter (``HPSV3_PY``) because its ``transformers`` pin
conflicts with the training env; we only orchestrate.

This stage is fully decoupled: it consumes Stage 1 output on disk and produces a
rewards file Stage 3 joins against. Each sample is independent, so scoring is
embarrassingly parallel (one job per sample / GPU / the scheduler task).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Combo specs: weights + the calibrated per-metric mean/std
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Term:
    metric: str   # key in the raw-metrics dict
    weight: float
    mean: float
    std: float
    sign: float   # +1 (higher better) or -1 (lower better)

# Per-metric normalizers (mu, sigma, sign). The values live in norm.json beside this
# module, not here: they are calibration data, they differ per model, and a wrong sigma
# silently reweights its term. NFT_NORM_FILE points at a different file.
def _load_norm() -> Dict[str, tuple]:
    path = os.environ.get("NFT_NORM_FILE", "").strip() or \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "norm.json")
    with open(path) as f:
        doc = json.load(f)
    return {m: (float(v["mu"]), float(v["sigma"]), float(v["sign"]))
            for m, v in doc["metrics"].items()}

NORM: Dict[str, tuple] = _load_norm()

# Calibration is baked into the image, which made every recalibration cost a full rebuild -- and
# twice now a run has silently used stale sigma because a mutable tag was reused. NFT_NORM_OVERRIDE
# lets a payload set mu/sigma per metric at launch instead:
#   NFT_NORM_OVERRIDE='{"rpe_rot": [8.9, 1.7], "rpe_trans": [0.025, 0.006]}'
# A 2-list is (mu, sigma) keeping the table's sign; a 3-list sets sign too. This must run before
# COMBOS is built, because _term() snapshots mu/sigma into each Term at import time.
def _apply_norm_override() -> None:
    raw = os.environ.get("NFT_NORM_OVERRIDE", "").strip()
    if not raw:
        return
    try:
        over = json.loads(raw)
    except Exception as e:
        raise ValueError(f"NFT_NORM_OVERRIDE is not valid JSON: {e}") from e
    for metric, vals in over.items():
        if metric not in NORM:
            raise KeyError(f"NFT_NORM_OVERRIDE: unknown metric {metric!r}; known={sorted(NORM)}")
        if len(vals) == 2:
            mu, sd = float(vals[0]), float(vals[1])
            sign = NORM[metric][2]
        elif len(vals) == 3:
            mu, sd, sign = float(vals[0]), float(vals[1]), float(vals[2])
        else:
            raise ValueError(f"NFT_NORM_OVERRIDE[{metric}] must be [mu, sigma] or [mu, sigma, sign]")
        if not (sd > 0):
            raise ValueError(f"NFT_NORM_OVERRIDE[{metric}]: sigma must be > 0, got {sd}")
        old = NORM[metric]
        NORM[metric] = (mu, sd, sign)
        print(f"[nft_score] NORM override {metric}: {old} -> {NORM[metric]}", flush=True)

_apply_norm_override()

def _term(metric: str, weight: float) -> Term:
    mu, sd, sign = NORM[metric]
    return Term(metric, weight, mu, sd, sign)

# Rewards from reward_definitions.md (R = weighted sum of signed z-scores).
COMBOS: Dict[str, List[Term]] = {
    # The LoGo reward: fused-cloud geometry, read out globally or per voxel (nft_voxel).
    "reproj_rgbd":     [_term("vggt_mse", 0.5), _term("vggt_depth_mae", 0.5)],
    # geometry-heavy variant: depth reproj localizes artifacts (RGB reproj error is
    # diffuse on degraded clips), so weight it 0.7/0.3.
    "reproj_rgbd_g70": [_term("vggt_mse", 0.3), _term("vggt_depth_mae", 0.7)],
    # Single-objective terms the alternating schedule (NFT_COMBO_SCHEDULE) cycles through.
    "hpsv3_only":      [_term("hpsv3_vid", 1.0)],
    "camera_only":     [_term("rpe_rot", 0.5), _term("rpe_trans", 0.5)],
}

# ---------------------------------------------------------------------------
# ALTERNATING REWARD SCHEDULE
#
# A weighted sum blends objectives inside one advantage: every rollout is ranked by
# w1*z1 + w2*z2, so a term with a small weight barely moves the ranking (measured: the
# 0.10 hpsv3 term supplied ~8% of the top-vs-bottom-quartile selection pressure). An
# alternating schedule instead gives each objective *whole steps* where it is the only
# thing ranked -- the per-scene advantage is z-scored within that step's reward, so each
# objective gets an undiluted gradient and no cross-metric sigma calibration is needed.
#
#   NFT_COMBO_SCHEDULE="reproj_rgbd:7,hpsv3_only:3"
#     steps 0-6 -> reproj_rgbd, steps 7-9 -> hpsv3_only, steps 10-16 -> reproj_rgbd, ...
#
# Applies to training steps only; validation stays on the fixed --combo so val R remains
# comparable across steps (otherwise the val curve would jump between two different scales).
# ---------------------------------------------------------------------------
def parse_combo_schedule(spec: str) -> List[Tuple[str, int]]:
    """'a:7,b:3' -> [('a',7),('b',3)]. Raises on unknown combo or non-positive length."""
    out: List[Tuple[str, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"NFT_COMBO_SCHEDULE entry {part!r} must be '<combo>:<n_steps>'")
        name, n = part.rsplit(":", 1)
        name = name.strip()
        if name not in COMBOS:
            raise KeyError(f"NFT_COMBO_SCHEDULE: unknown combo {name!r}; known={sorted(COMBOS)}")
        n = int(n)
        if n <= 0:
            raise ValueError(f"NFT_COMBO_SCHEDULE: {name!r} needs a positive step count, got {n}")
        out.append((name, n))
    if not out:
        raise ValueError("NFT_COMBO_SCHEDULE is empty")
    return out

def combo_for_step(step: int, default: str) -> str:
    """Which combo trains `step`. Returns `default` when no schedule is configured."""
    spec = os.environ.get("NFT_COMBO_SCHEDULE", "").strip()
    if not spec:
        return default
    sched = parse_combo_schedule(spec)
    period = sum(n for _, n in sched)
    pos = step % period
    for name, n in sched:
        if pos < n:
            return name
        pos -= n
    return default  # unreachable

def describe_combo_schedule() -> str:
    spec = os.environ.get("NFT_COMBO_SCHEDULE", "").strip()
    if not spec:
        return ""
    sched = parse_combo_schedule(spec)
    period = sum(n for _, n in sched)
    return " -> ".join(f"{n}x {name}" for name, n in sched) + f"  (period {period})"

_PERCEPTUAL_METRICS = ("hpsv3_vid",)

def combine(metrics: Dict[str, float], spec: List[Term]) -> float:
    """Combine per-metric signed z-scores into one reward. Returns -inf if any required
    metric is missing or non-finite: the rollout drops out of its advantage group.

    NFT_COMBINE_MODE selects the structure:
      "sum" (default) -- weighted sum ``Σ sign·weight·z``. One axis can compensate for another,
        so a large gain on geometry can absorb a perceptual penalty.
      "min"/"product" -- conjunctive. Group terms into perceptual vs geometry objectives (each
        the weight-normalized mean of its z-scores, ~unit variance and comparable), then take the
        floor (min) or softplus-product. A low objective can't be bought back by a high one, so
        the reward can only rise by improving both -- the honest direction.
    """
    import math

    z = {}
    for t in spec:
        v = metrics.get(t.metric, None)
        if v is None or not math.isfinite(float(v)):
            return float("-inf")
        z[t.metric] = (t.sign, t.weight, (float(v) - t.mean) / t.std)

    mode = os.environ.get("NFT_COMBINE_MODE", "sum")
    if mode == "sum":
        return sum(s * w * zz for s, w, zz in z.values())

    def objective(perceptual: bool):
        g = [(w, s * zz) for m, (s, w, zz) in z.items()
             if (m in _PERCEPTUAL_METRICS) == perceptual]
        if not g:
            return None
        wsum = sum(w for w, _ in g)
        return sum(w * val for w, val in g) / wsum if wsum > 0 else sum(v for _, v in g) / len(g)

    parts = [o for o in (objective(True), objective(False)) if o is not None]
    if not parts:
        return float("-inf")
    if len(parts) == 1:
        return parts[0]
    if mode == "min":
        return min(parts)
    if mode == "product":
        sp = [math.log1p(math.exp(min(p, 30.0))) for p in parts]  # softplus -> positive factors
        return sp[0] * sp[1]
    return sum(s * w * zz for s, w, zz in z.values())

def _count_frames(frames_dir: Path) -> int:
    return len(list(Path(frames_dir).glob("*.png")))

# --------------------------------------------------------------------------- #
# MAE from the rewards scorer (in-process), incl. per-frame for densified reward
# --------------------------------------------------------------------------- #
_REPROJ_RGBD_MOD = None    # scorers.reproj_rgbd module (lazy)
_REPROJ_RGBD_CTX = None    # loaded VGGT-Omega ctx (once per process)

# HPSv3 runs out-of-process (its transformers pin conflicts with the training env), so the
# reward shells out to the runner script with that env's interpreter.
def _default_hpsv3_runner() -> Path:
    """The shared runner under ``rewards/`` (REWARDS_DIR if set, else the repo tree)."""
    from lyra_2._src.rl.scoring.nft_voxel import rewards_dir
    return Path(rewards_dir()) / "scorers/hpsv3.py"

HPSV3_RUNNER = Path(os.environ["HPSV3_RUNNER"]) if os.environ.get("HPSV3_RUNNER") \
    else _default_hpsv3_runner()
HPSV3_PY = os.environ.get("HPSV3_PY") or sys.executable

# --------------------------------------------------------------------------- #
# Camera adherence: relative pose error against the trajectory the sampler was given.
# The RPE math itself is the shared rewards/scorers/camera_rpe.py (one copy for all
# three models); everything below is how this loop sources the reference trajectory.
# --------------------------------------------------------------------------- #
def _camera_rpe():
    """The shared RPE implementation (``scorers.camera_rpe``)."""
    import importlib

    _import_reproj_rgbd()          # sets up the synthetic 'scorers' namespace
    return importlib.import_module("scorers.camera_rpe")

def _camera_logging() -> bool:
    """NFT_LOG_CAMERA=1 computes rpe_rot/rpe_trans and puts them in the metrics dict without
    them being in the combo -- combine() only reads metrics named in the spec, while
    seed_metrics logs the whole dict. That yields within-scene calibration data for these two
    from a normal training run, before any weight is attached to them."""
    return os.environ.get("NFT_LOG_CAMERA", "").strip() in ("1", "true", "True")

def _score_camera(est_w2c, scene: Optional[str], num_frames: int) -> Dict[str, float]:
    """RPE of the recon's estimated camera path against the trajectory the sampler was given.

    Both sides already exist: cameras.npz holds VGGT's estimated w2c, and the conditioning
    lyra2_traj.npz is on disk under NFT_SCENES_ROOT (set by nft_loop). Errors are scale-invariant
    and path-normalized by pose_control -- do not divide by trajectory length again.
    Returns {} (not -inf) when unavailable, so a missing trajectory degrades to 'no camera term'
    rather than poisoning the whole reward.
    """
    import numpy as np
    import torch

    root = os.environ.get("NFT_SCENES_ROOT", "").strip()
    if not root or not scene or est_w2c is None:
        return {}
    tf = Path(root) / scene / "lyra2_traj.npz"
    if not tf.exists():
        return {}
    try:
        est_w2c = np.asarray(est_w2c, dtype=np.float64)
        ref_w2c = np.asarray(np.load(tf)["w2c"], dtype=np.float64)[:num_frames]
        if len(est_w2c) < 2 or len(ref_w2c) < 2:
            return {}
        # the recon subsamples frames; align the reference to the estimate's count
        idx = np.linspace(0, len(ref_w2c) - 1, len(est_w2c)).round().astype(int)
        ref = torch.tensor(np.linalg.inv(ref_w2c[idx]), dtype=torch.float32)   # -> c2w
        est = torch.tensor(np.linalg.inv(est_w2c), dtype=torch.float32)        # -> c2w
        rot, trans = _camera_rpe().adherence(pred_c2w=est, target_c2w=ref)
        return {"rpe_rot": float(rot), "rpe_trans": float(trans)}
    except Exception as e:  # noqa: BLE001 -- a camera failure must not kill the rollout
        print(f"[nft_score] camera RPE failed for {scene}: {type(e).__name__}: {e}", flush=True)
        return {}

def _nanmean(xs) -> float:
    import math as _m
    v = [float(x) for x in xs if x is not None and _m.isfinite(float(x))]
    return sum(v) / len(v) if v else float("nan")

def _combine_perframe(vals: Dict[str, float], spec: List[Term]) -> float:
    """Per-frame combine that SKIPS NaN/missing terms (e.g. a frame with no valid depth
    instead of returning -inf; NaN if none available. NFT_COMBINE_MODE mirrors combine():
    "sum" (default) = weighted sum of available signed z; "min"/"product" = conjunctive over
    the perceptual vs geometry objectives (so a windowed reward can be conjunctive per frame)."""
    import math as _m

    zs = []
    for t in spec:
        v = vals.get(t.metric)
        if v is None or not _m.isfinite(float(v)):
            continue
        zs.append((t.metric, t.sign, t.weight, (float(v) - t.mean) / t.std))
    if not zs:
        return float("nan")
    mode = os.environ.get("NFT_COMBINE_MODE", "sum")
    if mode == "sum":
        return sum(s * w * z for _, s, w, z in zs)

    def obj(perceptual):
        g = [(w, s * z) for m, s, w, z in zs if (m in _PERCEPTUAL_METRICS) == perceptual]
        if not g:
            return None
        ws = sum(w for w, _ in g)
        return sum(w * val for w, val in g) / ws if ws > 0 else sum(v for _, v in g) / len(g)

    parts = [o for o in (obj(True), obj(False)) if o is not None]
    if not parts:
        return float("nan")
    if len(parts) == 1:
        return parts[0]
    if mode == "min":
        return min(parts)
    if mode == "product":
        sp = [_m.log1p(_m.exp(min(p, 30.0))) for p in parts]
        return sp[0] * sp[1]
    return sum(s * w * z for _, s, w, z in zs)

def _frames_to_mp4(frames_dir: Path, mp4: Path, fps: int = 16) -> bool:
    """Encode sorted %05d.png frames to an mp4 (matches the recon worker's --fps 16)."""
    import subprocess

    if not list(frames_dir.glob("*.png")):
        return False
    rc = subprocess.run(
        ["ffmpeg", "-y", "-r", str(fps), "-i", str(frames_dir / "%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode
    return rc == 0

# --------------------------------------------------------------------------- #
# HPSv3 directly on the raw rollout video (c8hv). No GS/DA3 recon: encode the
# generated frames to mp4, then score ~N evenly-spaced keyframes with HPSv3 in its
# own env (HPSV3_PY). The number of keyframes is NFT_HPSV3_KEYFRAMES (default 15).
# --------------------------------------------------------------------------- #
def _score_hpsv3_video(frames_dir: Path, work: Path, gpu_id: int,
                       target_end: Optional[int] = None) -> Optional[float]:
    """Mean HPSv3 over ~NFT_HPSV3_KEYFRAMES keyframes of the raw rollout clip.
    Returns the score, or None on failure (drives combine() -> -inf)."""
    import subprocess

    n_kf = int(os.environ.get("NFT_HPSV3_KEYFRAMES", "15"))
    nframes = _count_frames(frames_dir)
    if nframes == 0:
        print(f"[nft_score] hpsv3_vid: no frames in {frames_dir}", flush=True)
        return None
    stride = max(1, round(nframes / max(1, n_kf)))

    out_dir = work / "hpsv3_vid"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "hpsv3.json"
    log_path = out_dir / "hpsv3.log"
    if not out_json.exists():
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # scorers/hpsv3.py reads the PNG dir directly (sorted by int stem) and
        # scores every `stride`-th frame -> ~n_kf evenly-spaced keyframes of the clip.
        cmd = [HPSV3_PY, str(HPSV3_RUNNER), str(frames_dir), str(out_json),
               "--stride", str(stride)]
        if target_end is not None:
            cmd += ["--target-end", str(target_end)]
        with open(log_path, "w") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=f).returncode
        if rc != 0 or not out_json.exists():
            tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-30:]) \
                if log_path.exists() else "(no log)"
            print(f"[nft_score] hpsv3_vid FAILED (rc={rc}) {frames_dir}\n--- tail ---\n{tail}\n---",
                  flush=True)
            return None
    try:
        return float(json.loads(out_json.read_text())["hpsv3_score"])
    except Exception as e:  # noqa: BLE001
        print(f"[nft_score] hpsv3_vid parse failed: {type(e).__name__}: {e}", flush=True)
        return None

def _batch_hpsv3_video(items: List[tuple], gpu_id: int) -> None:
    """Pre-score hpsv3_vid for many rollouts in one subprocess (hpsv3 model loaded once),
    writing each rollout's ``<work>/hpsv3_vid/hpsv3.json``. ``score_clip``'s
    ``_score_hpsv3_video`` then finds that json and skips its own per-rollout model reload.
    ``items`` = [(frames_dir, work_dir)]. Best-effort: on failure the per-rollout path just
    reloads (the json simply won't exist)."""
    import subprocess

    n_kf = int(os.environ.get("NFT_HPSV3_KEYFRAMES", "15"))
    manifest = []
    for frames_dir, work in items:
        frames_dir, work = Path(frames_dir), Path(work)
        n = _count_frames(frames_dir)
        out_json = work / "hpsv3_vid" / "hpsv3.json"
        if n == 0 or out_json.exists():
            continue
        out_json.parent.mkdir(parents=True, exist_ok=True)
        manifest.append({"input": str(frames_dir), "output": str(out_json),
                         "stride": max(1, round(n / max(1, n_kf))), "target_end": n})
    if not manifest:
        return
    work0 = Path(items[0][1]); work0.mkdir(parents=True, exist_ok=True)
    mfile = work0 / "hpsv3_batch_manifest.json"
    mfile.write_text(json.dumps(manifest))
    log = work0 / "hpsv3_batch.log"
    env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    with open(log, "w") as f:
        rc = subprocess.run([HPSV3_PY, str(HPSV3_RUNNER), "--manifest", str(mfile)],
                            env=env, stdout=f, stderr=f).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:]) if log.exists() else ""
        print(f"[nft_score] hpsv3_vid batch rc={rc} (per-rollout will reload)\n{tail}", flush=True)

def _score_hpsv3_perframe(frames_dir: Path, work: Path, gpu_id: int, stride: int) -> Optional[List[float]]:
    """Per-frame HPSv3 over frames_dir at ``stride`` -- for the WINDOWED reward, where hpsv3 must
    localize per window (not a per-rollout scalar). Reuses the batched hpsv3_vid json if it already
    carries per-frame scores (set NFT_HPSV3_KEYFRAMES so its stride matches frames_per_latent);
    otherwise scores this rollout in its own subprocess. Returns the per-frame list or None."""
    import subprocess

    batched = Path(work) / "hpsv3_vid" / "hpsv3.json"
    if batched.exists():
        try:
            sc = json.loads(batched.read_text()).get("hpsv3_scores")
            if sc:
                return [float(x) for x in sc]
        except Exception:  # noqa: BLE001 -- fall through to fresh scoring
            pass
    if _count_frames(frames_dir) == 0:
        return None
    out_dir = Path(work) / "hpsv3_pf"; out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "hpsv3_pf.json"; log_path = out_dir / "hpsv3_pf.log"
    if not out_json.exists():
        env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [HPSV3_PY, str(HPSV3_RUNNER), str(frames_dir), str(out_json), "--stride", str(stride)]
        with open(log_path, "w") as f:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=f).returncode
        if rc != 0 or not out_json.exists():
            print(f"[nft_score] hpsv3 per-frame FAILED (rc={rc}) {frames_dir}", flush=True)
            return None
    try:
        return [float(x) for x in (json.loads(out_json.read_text()).get("hpsv3_scores") or [])]
    except Exception as e:  # noqa: BLE001
        print(f"[nft_score] hpsv3 per-frame parse failed: {type(e).__name__}: {e}", flush=True)
        return None

def _import_reproj_rgbd():
    """Load scorers.reproj_rgbd under the synthetic 'scorers' namespace (no scorers/__init__,
    which eagerly imports hpsv3/moge and fails in the training env)."""
    global _REPROJ_RGBD_MOD
    if _REPROJ_RGBD_MOD is None:
        import importlib
        import types

        from lyra_2._src.rl.scoring.nft_voxel import rewards_dir
        sdir = os.path.join(rewards_dir(), "scorers")
        if "scorers" not in sys.modules:
            pkg = types.ModuleType("scorers")
            pkg.__path__ = [sdir]  # namespace only -- do not execute scorers/__init__
            sys.modules["scorers"] = pkg
        _REPROJ_RGBD_MOD = importlib.import_module("scorers.reproj_rgbd")
    return _REPROJ_RGBD_MOD

def _score_reproj_rgbd(frames_dir: Path, work: Path, gpu_id: int,
                       spec: Optional[List[Term]] = None,
                       want_perframe: bool = False) -> Optional[Dict[str, float]]:
    """{vggt_mse, vggt_depth_mae} from the fused global VGGT cloud (reproj_rgbd combo).
    Encodes the clip to mp4, decodes uniform frames, runs the recon + RGB/depth
    reprojection. VGGT-Omega loads once per process (cached). None on failure.

    When ``want_perframe`` (windowed reward), decodes one frame per generated latent
    (``ceil((NUM_FRAMES-1)/frames_per_latent)`` uniform frames ~ one per latent) so the
    scorer's per-frame errors map directly onto the latent grid, and returns
    ``R_perframe`` = the per-latent combo reward built from those arrays via
    ``_combine_perframe`` (mirrors ``_score_mae``)."""
    global _REPROJ_RGBD_CTX
    import importlib

    mp4 = Path(work) / "reproj_clip.mp4"
    if not _frames_to_mp4(Path(frames_dir), mp4):
        print(f"[nft_score] reproj_rgbd: no frames in {frames_dir}", flush=True)
        return None
    try:
        rr = _import_reproj_rgbd()
        vg = importlib.import_module("scorers.dl3dv_videogpa")
        if _REPROJ_RGBD_CTX is None:
            _REPROJ_RGBD_CTX = rr.load(vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"),
                                       device="cuda")
        n = int(os.environ.get("NFT_REPROJ_FRAMES", "16"))
        if want_perframe:
            fpl = 4  # frames_per_latent (framepack); matches _score_mae's default
            nf = int(os.environ.get("NUM_FRAMES", "81"))
            n = max(1, -(-(nf - 1) // fpl))          # ceil((nf-1)/fpl) generated latents
        frames = importlib.import_module("scorers.decode").decode_uniform(str(mp4), n)
        res = rr.score(_REPROJ_RGBD_CTX, frames, per_frame=want_perframe)
        if want_perframe and spec is not None:
            mse_pf = res.pop("vggt_mse_pf", []) or []
            dm_pf = res.pop("vggt_depth_mae_pf", []) or []
            # A windowed reward localizes per frame, so hpsv3 must be per-frame too -- not the
            # per-rollout scalar broadcast (that guards uniformly and can't tell which windows
            # degraded). Score each latent frame (stride=fpl aligns hpsv3's grid to the reproj
            # per-latent grid; set NFT_HPSV3_KEYFRAMES so the batched json matches). If hpsv3
            # isn't in the spec, or per-frame scoring fails, we fall back to reproj-only frames.
            hp_pf = None
            if any(t.metric == "hpsv3_vid" for t in spec):
                hp_pf = _score_hpsv3_perframe(Path(frames_dir), Path(work), gpu_id, fpl)
                # NFT_HPSV3_WINDOWED_MODE: "perframe" (default) = each window's own hpsv3
                # (localized guard); "scalar" = apply hpsv3 the GLOBAL way -- one per-rollout
                # mean broadcast to every frame (uniform guard on top of windowed reproj).
                if hp_pf and os.environ.get("NFT_HPSV3_WINDOWED_MODE", "perframe") == "scalar":
                    _mean = sum(hp_pf) / len(hp_pf)
                    hp_pf = [_mean] * len(hp_pf)

            def _rpf(i, m, d):
                vals = {"vggt_mse": m, "vggt_depth_mae": d}
                if hp_pf and i < len(hp_pf):
                    vals["hpsv3_vid"] = hp_pf[i]
                return _combine_perframe(vals, spec)

            res["R_perframe"] = [_rpf(i, m, d) for i, (m, d) in enumerate(zip(mse_pf, dm_pf))]
        return res
    except Exception as e:  # noqa: BLE001 -- isolate scorer faults -> -inf for the rollout
        print(f"[nft_score] reproj_rgbd failed for {frames_dir}: {type(e).__name__}: {e}", flush=True)
        return None

def score_clip(
    frames_dir: str,
    work_dir: str,
    combo_name: str,
    gpu_id: int = 0,
    target_end: Optional[int] = None,
    scene: Optional[str] = None,
    extra_metrics: Optional[set] = None,
) -> Tuple[float, Dict[str, float]]:
    """Run the reward jobs for one accumulated clip and return (R, raw_metrics).

    ``frames_dir`` is the decoded accumulated clip (a dir of PNGs, as Stage 1
    writes). ``work_dir`` holds the per-reward job outputs (resumable: existing
    json is reused). Each metric's scorer runs only if the combo needs it.
    """
    spec = COMBOS.get(combo_name)
    if spec is None:
        raise ValueError(f"unknown combo '{combo_name}', expected one of {sorted(COMBOS)}")

    frames_dir = Path(frames_dir)
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if target_end is None:
        target_end = _count_frames(frames_dir)

    # ``needed`` = the combo's metrics ∪ any extra_metrics (validation asks for raw
    # metrics beyond the reward). Everything below is
    # gated on ``needed`` so an extra metric simply switches on its recon + computation;
    # the reward ``R`` is still combine(spec) and ignores the extras.
    needed = {t.metric for t in spec} | set(extra_metrics or ())
    metrics: Dict[str, float] = {}

    # A camera-only combo still needs the recon, because rpe comes from VGGT's estimated w2c.
    # Run the same reproj scorer, keep only the camera terms, and let combine() ignore the rest.
    if needed & {"rpe_rot", "rpe_trans"} and not (needed & {"vggt_mse", "vggt_depth_mae"}):
        from lyra_2._src.rl.scoring import nft_voxel
        if nft_voxel.voxel_enabled():
            rd = nft_voxel.score_reproj_voxel(frames_dir, work, gpu_id, spec=spec)
            _est = (rd.get("voxel_payload") or {}).pop("est_w2c", None) if rd else None
        else:
            rd = _score_reproj_rgbd(frames_dir, work, gpu_id, spec=spec, want_perframe=False)
            _est = rd.pop("est_w2c", None) if isinstance(rd, dict) else None
        if rd is None:
            return float("-inf"), metrics
        metrics.update({k: v for k, v in rd.items() if k != "voxel_payload"})
        metrics.update(_score_camera(_est, scene, int(os.environ.get("NUM_FRAMES", "81"))))
        if not (needed & set(metrics)):
            return float("-inf"), metrics
        return combine(metrics, spec), metrics

    # reproj_rgbd: RGB + depth reprojection over the fused global VGGT cloud. Self-contained
    # (own VGGT recon), so it is self-contained and returns as soon as it is done.
    if needed & {"vggt_mse", "vggt_depth_mae"}:
        # Voxel reward (NFT_REWARD_VOXEL): per-voxel error tables instead of the
        # per-frame decode; disabled during validation (NFT_EVAL_GLOBAL_DEPTH), where the
        # plain path below keeps val metrics comparable. Scalars are reproj_rgbd-identical.
        from lyra_2._src.rl.scoring import nft_voxel
        if nft_voxel.voxel_enabled():
            rd = nft_voxel.score_reproj_voxel(frames_dir, work, gpu_id, spec=spec)
            if rd is None:
                return float("-inf"), metrics
            metrics.update(rd)  # incl. voxel_payload (score_epoch pops it to the record top level)
            if (needed & {"rpe_rot", "rpe_trans"}) or _camera_logging():
                _est = (rd.get("voxel_payload") or {}).pop("est_w2c", None)
                metrics.update(_score_camera(_est, scene, int(os.environ.get("NUM_FRAMES", "81"))))
            return combine(metrics, spec), metrics
        _win = os.environ.get("NFT_REWARD_WINDOW", "full").strip().lower() not in ("", "full", "0")
        # Validation forces the global 16-frame decode (NFT_EVAL_GLOBAL_DEPTH=1) so vggt_depth_mae /
        # vggt_mse are comparable across windowed and non-windowed runs (and faster: 16 vs ~60 frames).
        # Training still uses the windowed per-frame decode for the localized reward.
        if os.environ.get("NFT_EVAL_GLOBAL_DEPTH", "").strip() in ("1", "true", "True"):
            _win = False
        rd = _score_reproj_rgbd(frames_dir, work, gpu_id, spec=spec, want_perframe=_win)
        # always pop est_w2c, even when the camera term is off: rd's leftovers become the metrics
        # dict and every value is float()d, so leaving an array in it kills every rank.
        _est = rd.pop("est_w2c", None) if isinstance(rd, dict) else None
        if rd is not None and ((needed & {"rpe_rot", "rpe_trans"}) or _camera_logging()):
            metrics.update(_score_camera(_est, scene, int(os.environ.get("NUM_FRAMES", "81"))))
        if rd is None:
            return float("-inf"), metrics
        _rpf = rd.pop("R_perframe", None)   # per-latent combo -> drives windowed advantage
        metrics.update(rd)
        if _rpf is not None:
            metrics["R_perframe"] = _rpf
        # hpsv3+reproj blend: the raw-video hpsv3 branch below is unreachable after this
        # return, so score hpsv3_vid here too when the combo needs it.
        if "hpsv3_vid" in needed:
            hv = _score_hpsv3_video(frames_dir, work, gpu_id, target_end)
            if hv is None:
                return float("-inf"), metrics
            metrics["hpsv3_vid"] = hv
        return combine(metrics, spec), metrics

    # hpsv3 on the RAW rollout video: the hpsv3_only term of the alternating schedule.
    if "hpsv3_vid" in needed:
        hv = _score_hpsv3_video(frames_dir, work, gpu_id, target_end)
        if hv is None:
            return float("-inf"), metrics
        metrics["hpsv3_vid"] = hv

    return combine(metrics, spec), metrics

def score_epoch(
    rollout_root: str,
    out_jsonl: str,
    combo_name: str = "reproj_rgbd",
    gpu_id: int = 0,
    extra_metrics: Optional[set] = None,
) -> List[dict]:
    """Score every sample in a rollout store; write rewards_epoch jsonl.

    Each output line carries the sample identity (scene/rollout/chunk/group_id),
    paths Stage 3 needs (x0_path, cond_path), the combo reward ``R`` and the raw
    per-metric values. ``group_id`` is what Stage 3 normalizes within.

    There is no epoch-level resume: ``out_jsonl`` is rewritten on every call and every
    sample is scored again. Resumability is per reward job, inside ``score_clip``'s
    ``work_dir`` -- existing job json is reused, so a re-run skips the expensive
    scorers rather than the record-keeping.

    """
    from collections import defaultdict

    rollout_root = Path(rollout_root)
    manifest = rollout_root / "manifest.jsonl"
    assert manifest.exists(), f"no manifest at {manifest}"
    samples = [json.loads(l) for l in open(manifest)]

    # Group the manifest by (scene, rollout): one autoregressive sequence. We score
    # the full sequence once (the highest-chunk clip is the accumulated [0..end]
    # video) and assign that reward to every chunk of the rollout. The advantage
    # group is the SCENE -- the K rollouts of one scene share identical conditioning,
    # so their full-sequence rewards are what gets z-scored against each other.
    by_rollout: Dict[tuple, List[dict]] = defaultdict(list)
    for m in samples:
        by_rollout[(m["scene"], m.get("rollout", m.get("branch", 0)))].append(m)

    import time as _t
    spec = COMBOS[combo_name]

    def _full(chunks):  # (highest-chunk sample, its clip dir, its reward work dir)
        c = sorted(chunks, key=lambda x: x["chunk"])[-1]
        return c, c.get("clip_path"), Path(os.path.join(os.path.dirname(c["x0_path"]), "reward"))

    # Compute (R, raw) per (scene, rollout). Any single-rollout failure -> -inf so the rollout
    # drops from its group; the loop's min-step all-reduce survives degenerate groups.
    scored: Dict[tuple, tuple] = {}
    n_roll = len(by_rollout)
    # Batch hpsv3_vid (c8hv) across all this rank's rollouts in one subprocess so the
    # hpsv3 model loads once instead of per rollout; score_clip then finds each json.
    if "hpsv3_vid" in ({t.metric for t in spec} | set(extra_metrics or ())):
        items = [(clip, work) for (_s, _r), ch in by_rollout.items()
                 for _, clip, work in [_full(ch)] if clip and os.path.isdir(clip)]
        if items:
            _t0 = _t.time()
            _batch_hpsv3_video(items, gpu_id)
            print(f"[nft_score] gpu{gpu_id} batched hpsv3_vid over {len(items)} rollouts "
                  f"in {_t.time() - _t0:.1f}s", flush=True)
    for ri, ((scene, rollout), chunks) in enumerate(by_rollout.items()):
        _, clip, work = _full(chunks)
        if clip and os.path.isdir(clip):
            print(f"[nft_score] gpu{gpu_id} scoring {scene}/r{rollout} "
                  f"({ri + 1}/{n_roll}) ...", flush=True)
            _t0 = _t.time()
            try:
                R, raw = score_clip(clip, str(work), combo_name, gpu_id,
                                    scene=scene, extra_metrics=extra_metrics)
            except Exception as e:  # noqa: BLE001 -- isolate per-sample scorer faults
                print(f"[nft_score] score_clip failed for {scene}/r{rollout}: "
                      f"{type(e).__name__}: {e}", flush=True)
                R, raw = float("-inf"), {}
            print(f"[nft_score] gpu{gpu_id} scored {scene}/r{rollout} "
                  f"R={R:.4f} in {_t.time() - _t0:.1f}s", flush=True)
        else:
            R, raw = float("-inf"), {}
        scored[(scene, rollout)] = (R, raw)

    # Broadcast each rollout's full-sequence reward to all its chunks.
    results: List[dict] = []
    with open(out_jsonl, "w") as fout:
        for (scene, rollout), chunks in by_rollout.items():
            R, raw = scored.get((scene, rollout), (float("-inf"), {}))
            # Per-latent-frame combo reward for the densified windowed advantage. `rpf` covers
            # the whole rollout's generated latents (scored once on the full clip). Store it
            # whole on every chunk record (not sliced): the loop windows the full rollout on a
            # consistent grid, z-scores each window over the K rollouts, then slices per chunk
            # at assignment so chunk c gets its own windows. (Slicing here + keying local_Rpf by
            # rollout in the loop overwrote all but the last chunk -- the bug this fixes.)
            # Kept as a top-level key (not inside `metrics`, whose values the loop float()s).
            rpf = raw.pop("R_perframe", None) if isinstance(raw, dict) else None
            vpay = raw.pop("voxel_payload", None) if isinstance(raw, dict) else None
            for m in sorted(chunks, key=lambda x: x["chunk"]):
                rec = {
                    "sample_key": f"{scene}/r{rollout}/c{m['chunk']}",
                    "scene": scene,
                    "chunk": m["chunk"],
                    "rollout": rollout,
                    "group_id": scene,  # one group per scene (the K full sequences)
                    "combo": combo_name,
                    "R": R,
                    "metrics": raw,
                    "R_perframe": rpf,  # whole rollout; loop windows full seq, slices per chunk
                    "voxel_payload": vpay,  # whole rollout too (voxel reward; None when off)
                    "x0_path": m["x0_path"],
                    "cond_path": m["cond_path"],
                }
                results.append(rec)
                fout.write(json.dumps(rec) + "\n")
            fout.flush()
    return results

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DiffusionNFT Stage 2: offline reward scoring")
    ap.add_argument("rollout_root", help="rollout_epoch_{E} directory from Stage 1")
    ap.add_argument("out_jsonl", help="output rewards jsonl path")
    ap.add_argument("--combo", default="reproj_rgbd", choices=sorted(COMBOS))
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    res = score_epoch(args.rollout_root, args.out_jsonl, args.combo, args.gpu)
    finite = [r for r in res if r["R"] not in (float("-inf"), float("inf")) and r["R"] == r["R"]]
    print(f"scored {len(res)} samples ({len(finite)} finite) -> {args.out_jsonl}")

if __name__ == "__main__":
    _main()
