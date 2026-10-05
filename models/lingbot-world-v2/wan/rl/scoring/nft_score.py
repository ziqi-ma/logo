# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 of the DiffusionNFT pipeline: offline reward scoring (lingbot port).

Reads the rollout-store manifest produced by Stage 1, runs the reward jobs over
each rollout's decoded clip, combines them into the calibrated z-score combo, and
writes ``rewards.jsonl`` keyed by rollout.

Ported from lyra's ``lyra_2/_src/rl/nft_score.py``. The combo table (``COMBOS``),
per-metric normalizers (``NORM``) and the ``combine`` / per-frame combine math are
kept verbatim so rewards stay comparable across repos. The HPSv3 subprocess wrapper
lives in :mod:`wan.rl.scoring.reward_runners`.

Record granularity (differs from lyra): in lingbot one ROLLOUT = one RECORD.
Manifest rows are::

    {"scene", "rollout", "seed", "group_id", "x0_path", "cond_path",
     "clip_path", "frames_dir"}

(no "chunk" key). ``score_epoch`` scores each rollout's ``frames_dir`` once and
writes one rewards.jsonl line per rollout.

The reward is the fused-cloud RGB+depth reprojection (``vggt_mse`` / ``vggt_depth_mae``,
combo ``reproj_rgbd``). Raw-video HPSv3 (``hpsv3_vid``) is also scored here, but only as a
validation metric (``NFT_VAL_HPSV3``) -- it is not part of the reward.

Each rollout is independent, so scoring is embarrassingly parallel (one job per
rollout / GPU / task).
"""

from __future__ import annotations

import json
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from wan.rl.scoring import reward_runners as gs

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

def _apply_norm_override() -> None:
    """Overlay ``NFT_NORM_OVERRIDE`` (JSON ``{metric: [mu, sigma]}``) onto NORM.

    The frozen constants above are calibrated for one regime (480x832, 81-frame
    rollouts, 16 scored frames). Rollout length changes both terms' within-scene
    spread, and sigma is what sets each term's weight in a combo, so a different
    frame count needs different constants. Applied before COMBOS is built, so the
    Terms are frozen with the overridden values.

    Measured for the 241-frame / 60-scored-frame regime (256 rollouts over 16
    scene-groups of K=16):
        {"vggt_mse": [0.05130, 0.01212], "vggt_depth_mae": [0.17824, 0.05067]}
    With the 81-frame constants that regime splits the reward's within-scene
    spread 55.7/44.3 between mse and depth instead of the nominal 50/50.
    """
    raw = os.environ.get("NFT_NORM_OVERRIDE", "").strip()
    if not raw:
        return
    for metric, pair in json.loads(raw).items():
        if metric not in NORM:
            raise KeyError(f"NFT_NORM_OVERRIDE: unknown metric {metric!r}")
        mu, sd = float(pair[0]), float(pair[1])
        if not (sd > 0):
            raise ValueError(f"NFT_NORM_OVERRIDE: {metric} sigma must be > 0, got {sd}")
        NORM[metric] = (mu, sd, NORM[metric][2])   # sign is a property of the metric
        print(f"[nft_score] NORM override {metric}: mu={mu} sigma={sd}", flush=True)

_apply_norm_override()

def _term(metric: str, weight: float) -> Term:
    mu, sd, sign = NORM[metric]
    return Term(metric, weight, mu, sd, sign)

# The LoGo reward, as the published run uses it: R = -0.5 z(vggt_mse) - 0.5 z(vggt_depth_mae),
# the RGB reprojection MSE and depth reprojection MAE of the fused global VGGT cloud
# (scorers/reproj_rgbd.py). nft_voxel decides whether it is read out globally, per voxel or
# mixed; the weights and the calibrated mu/sigma are the same either way.
COMBOS: Dict[str, List[Term]] = {
    "reproj_rgbd": [_term("vggt_mse", 0.5), _term("vggt_depth_mae", 0.5)],
}

def combine(metrics: Dict[str, float], spec: List[Term]) -> float:
    """Weighted z-score combination. Returns -inf if any required metric is missing
    or non-finite: a rollout with a missing metric drops out of its advantage group."""
    import math

    total = 0.0
    for t in spec:
        v = metrics.get(t.metric, None)
        if v is None or not math.isfinite(float(v)):
            return float("-inf")
        total += t.sign * t.weight * (float(v) - t.mean) / t.std
    return total

def _count_frames(frames_dir: Path) -> int:
    return len(list(Path(frames_dir).glob("*.png")))

# --------------------------------------------------------------------------- #
# MAE from the rewards scorer (in-process), incl. per-frame for densified reward
# --------------------------------------------------------------------------- #
_REPROJ_RGBD_MOD = None
_VAL_FULL_CTX = None    # dl3dv_videogpa ctx for validation-only full scoring (lazy)    # scorers.reproj_rgbd module (lazy)
_REPROJ_RGBD_CTX = None    # loaded VGGT-Omega ctx (once per process)

def _nanmean(xs) -> float:
    import math as _m
    v = [float(x) for x in xs if x is not None and _m.isfinite(float(x))]
    return sum(v) / len(v) if v else float("nan")

def _combine_perframe(vals: Dict[str, float], spec: List[Term]) -> float:
    """Per-frame combine that SKIPS NaN/missing terms (e.g. a frame with no valid depth
    instead of returning -inf; sums available signed z-scores, NaN if none available."""
    import math as _m
    total, used = 0.0, 0
    for t in spec:
        v = vals.get(t.metric)
        if v is None or not _m.isfinite(float(v)):
            continue
        total += t.sign * t.weight * (float(v) - t.mean) / t.std
        used += 1
    return total if used else float("nan")

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
# HPSv3 directly on the raw rollout video (c8hv & hpsv3_* blends). No GS/DA3 recon:
# encode the generated frames to mp4, then score ~N evenly-spaced keyframes with
# HPSv3 in its own env (HPSV3_PY). The number of keyframes is
# NFT_HPSV3_KEYFRAMES (default 15).
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
        cmd = [gs.HPSV3_PY, str(gs.HPSV3_RUNNER), str(frames_dir), str(out_json),
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
        rc = subprocess.run([gs.HPSV3_PY, str(gs.HPSV3_RUNNER), "--manifest", str(mfile)],
                            env=env, stdout=f, stderr=f).returncode
    if rc != 0:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-25:]) if log.exists() else ""
        print(f"[nft_score] hpsv3_vid batch rc={rc} (per-rollout will reload)\n{tail}", flush=True)

def _import_reproj_rgbd():
    """Load ``scorers.reproj_rgbd`` under a synthetic ``scorers`` namespace, without
    running ``scorers/__init__`` (which eagerly imports hpsv3 and fails in the training
    env). ``scorers`` is registered as a namespace-only package over the shared
    ``rewards/scorers`` tree, so the module's relative imports resolve."""
    global _REPROJ_RGBD_MOD
    if _REPROJ_RGBD_MOD is None:
        import importlib
        import types

        # REWARDS_DIR if set, else the repo's shared rewards/ tree (nft_voxel is
        # stdlib-only at import time, so this is safe in every env).
        from wan.rl.scoring.nft_voxel import rewards_dir
        sdir = os.path.join(rewards_dir(), "scorers")
        if "scorers" not in sys.modules:
            pkg = types.ModuleType("scorers")
            pkg.__path__ = [sdir]  # namespace only -- do not execute scorers/__init__
            sys.modules["scorers"] = pkg
        _REPROJ_RGBD_MOD = importlib.import_module("scorers.reproj_rgbd")
    return _REPROJ_RGBD_MOD

def _score_reproj_rgbd(frames_dir: Path, work: Path, gpu_id: int,
                       spec: Optional[List[Term]] = None,
                       want_perframe: bool = False,
                       voxel_frames: int = 0) -> Optional[Dict[str, float]]:
    """{vggt_mse, vggt_depth_mae} from the fused global VGGT cloud (reproj_rgbd combo).
    Encodes the clip to mp4, decodes uniform frames, runs the recon + RGB/depth
    reprojection. VGGT-Omega loads once per process (cached). None on failure.

    When ``want_perframe`` (windowed reward), decodes one frame per generated latent
    (``ceil((NUM_FRAMES-1)/frames_per_latent)`` uniform frames ~ one per latent) so the
    scorer's per-frame errors map directly onto the latent grid, and returns
    ``R_perframe`` = the per-latent combo reward built from those arrays via
    ``_combine_perframe``.

    ``voxel_frames`` > 0 (voxel reward; the value is the rollout's LATENT count, passed
    by nft_loop's training path only) additionally runs the ``scorers.reproj_voxel``
    scorer on ``voxel_frames`` full-frame decodes and returns ``voxel_payload`` -- the
    per-(voxel, frame) error tables that ``nft_voxel.global_pervoxel_r`` consumes.
    A voxel failure degrades the rollout to its scalar r (payload absent), never -inf."""
    global _REPROJ_RGBD_CTX
    import importlib
    import subprocess

    mp4 = Path(work) / "reproj_clip.mp4"
    if not _frames_to_mp4(Path(frames_dir), mp4):
        print(f"[nft_score] reproj_rgbd: no frames in {frames_dir}", flush=True)
        return None
    n = int(os.environ.get("NFT_REPROJ_FRAMES", "16"))
    if want_perframe:
        fpl = 4  # frames_per_latent (framepack)
        nf = int(os.environ.get("NUM_FRAMES", "81"))
        n = max(1, -(-(nf - 1) // fpl))          # ceil((nf-1)/fpl) generated latents

    # SUBPROCESS BY DEFAULT. reproj_rgbd faulted with a CUDA illegal memory access on ~1 in 1e4
    # rollouts; catching the exception in-process is not enough because
    # the fault poisons the CUDA context and the run later died in pipe.model.to(dev), killing
    # all 32 GPUs. Running it in a child process makes the fault cost exactly one rollout.
    # Set NFT_REPROJ_INPROC=1 to restore the old in-process path (faster, unsafe).
    if os.environ.get("NFT_REPROJ_INPROC") != "1":
        wd = Path(work)
        mfile, ofile = wd / "reproj_manifest.json", wd / "reproj_out.json"
        # Serialize the full Term: _combine_perframe z-scores each metric with that term's
        # own mean/std/sign, so weight alone is not enough to reproduce the combo. Term is a
        # frozen dataclass (not iterable), hence the explicit field list rather than unpacking.
        item = {"mp4": str(mp4), "n": n, "want_perframe": bool(want_perframe),
                "spec": ([[t.metric, t.weight, t.mean, t.std, t.sign] for t in spec]
                         if want_perframe and spec else None)}
        if voxel_frames > 0:
            from wan.rl.scoring import nft_voxel
            _pg = nft_voxel.patch_grid()          # None == latent resolution
            item["voxel"] = {"n": int(voxel_frames), "alpha": nft_voxel.voxel_alpha(),
                             "patch": (None if _pg is None else list(_pg)),
                             "depth_cap": nft_voxel.depth_cap()}
        mfile.write_text(json.dumps([item]))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Synchronous launches in the CHILD only. CUDA errors are asynchronous, so the
        # illegal access is *reported* inside reproj_rgbd but may have originated
        # earlier (e.g. the 241-frame KV-cache eviction path during generation) -- the scorer is
        # simply the first thing that synchronizes. Blocking here costs the child some speed and
        # nothing in the training process, and makes the next occurrence name the real kernel.
        # If a captured failure shows a clean scorer stack, the origin is upstream in generation.
        # Set NFT_REPROJ_NOBLOCK=1 to drop this.
        if os.environ.get("NFT_REPROJ_NOBLOCK") != "1":
            env["CUDA_LAUNCH_BLOCKING"] = "1"
            env["TORCH_USE_CUDA_DSA"] = "1"
        log = wd / "reproj_runner.log"
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, "-m", "wan.rl.scoring.reproj_runner",
                                 "--manifest", str(mfile), "--out", str(ofile)],
                                env=env, stdout=f, stderr=f).returncode
        res = None
        if ofile.exists():
            try:
                got = json.loads(ofile.read_text())
                if got and got[0].get("ok"):
                    res = got[0]["metrics"]
                elif got:
                    print(f"[nft_score] reproj_rgbd failed for {frames_dir}: "
                          f"{got[0].get('error')}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[nft_score] reproj_rgbd: unreadable output ({e})", flush=True)
        if res is None:
            tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:]) if log.exists() else ""
            print(f"[nft_score] reproj_rgbd subprocess rc={rc} for {frames_dir}; "
                  f"rollout gets -inf\n{tail}", flush=True)
            _preserve_failed_frames(Path(frames_dir), rc, tail)
        return res

    try:
        rr = _import_reproj_rgbd()
        vg = importlib.import_module("scorers.dl3dv_videogpa")
        if _REPROJ_RGBD_CTX is None:
            _REPROJ_RGBD_CTX = rr.load(vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"),
                                       device="cuda")
        frames = importlib.import_module("scorers.decode").decode_uniform(str(mp4), n)
        res = rr.score(_REPROJ_RGBD_CTX, frames, per_frame=want_perframe)
        if want_perframe and spec is not None:
            mse_pf = res.pop("vggt_mse_pf", []) or []
            dm_pf = res.pop("vggt_depth_mae_pf", []) or []
            res["R_perframe"] = [_combine_perframe({"vggt_mse": m, "vggt_depth_mae": d}, spec)
                                 for m, d in zip(mse_pf, dm_pf)]
        if voxel_frames > 0:
            # Voxel tables from the same loaded ctx. A voxel failure must not void the
            # scalar R -- the rollout just degrades to its scalar advantage.
            from wan.rl.scoring import nft_voxel
            try:
                _pg = nft_voxel.patch_grid()      # None == latent resolution
                vres, payload = nft_voxel.score_voxel_clip(
                    _REPROJ_RGBD_CTX, str(mp4), int(voxel_frames),
                    nft_voxel.voxel_alpha(), (None if _pg is None else list(_pg)),
                    nft_voxel.depth_cap())
                res["voxel_vggt_mse"] = vres["vggt_mse"]
                res["voxel_vggt_dmae"] = vres["vggt_depth_mae"]
                res["voxel_payload"] = payload
            except Exception as e:  # noqa: BLE001 -- voxel must not void R
                print(f"[nft_score] voxel FAILED for {frames_dir} (scalar-r fallback): "
                      f"{type(e).__name__}: {e}", flush=True)
        return res
    except Exception as e:  # noqa: BLE001 -- isolate scorer faults -> -inf for the rollout
        print(f"[nft_score] reproj_rgbd failed for {frames_dir}: {type(e).__name__}: {e}", flush=True)
        return None

def _preserve_failed_frames(frames_dir: Path, rc: int, tail: str) -> None:
    """Upload the frames that killed the scorer so the fault is finally reproducible.

    A CUDA fault cannot be root-caused if the work dir is ephemeral: the
    offending rollout's frames died with the pod, and at ~1e-4 per rollout it is impractical to
    reproduce blind. Capture is best-effort and must never take the run down.
    """
    dest = os.environ.get("NFT_REPROJ_FAIL_URI")
    if not dest:
        return
    try:
        import tarfile
        import tempfile
        from wan.rl.data import gcs_util
        stamp = "_".join(frames_dir.parts[-4:]).replace("/", "_")
        with tempfile.TemporaryDirectory() as td:
            tar = Path(td) / f"{stamp}.tar.gz"
            with tarfile.open(tar, "w:gz") as tf:
                tf.add(frames_dir, arcname="frames")
                note = Path(td) / "failure.txt"
                note.write_text(f"rc={rc}\nframes_dir={frames_dir}\n\n{tail}\n")
                tf.add(note, arcname="failure.txt")
            gcs_util.upload_file(str(tar), f"{dest.rstrip('/')}/{stamp}.tar.gz")
        print(f"[nft_score] preserved failing frames -> {dest.rstrip('/')}/{stamp}.tar.gz",
              flush=True)
    except Exception as e:  # noqa: BLE001 -- capture is diagnostics, never fatal
        print(f"[nft_score] could not preserve frames ({type(e).__name__}: {e})", flush=True)

def score_clip(
    frames_dir: str,
    work_dir: str,
    combo_name: str,
    gpu_id: int = 0,
    target_end: Optional[int] = None,
    extra_metrics: Optional[set] = None,
    voxel_frames: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Run the reward jobs for one rollout clip and return (R, raw_metrics).

    ``frames_dir`` is the decoded rollout clip (a dir of PNGs, as Stage 1 writes).
    ``work_dir`` holds the per-reward job outputs (resumable: existing json is
    reused).

    ``voxel_frames`` > 0 (= the rollout's latent count) additionally computes the
    per-voxel error tables on the reproj path; they ride ``raw_metrics`` under
    the non-float key ``"voxel_payload"`` (callers must pop it before treating the
    metrics as scalars -- ``score_epoch`` promotes it to a top-level jsonl key).
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
    # metrics beyond the reward). Everything below is gated on ``needed`` so an extra
    # metric simply switches on its computation; the reward ``R`` is still
    # combine(spec) and ignores the extras.
    needed = {t.metric for t in spec} | set(extra_metrics or ())
    metrics: Dict[str, float] = {}

    # reproj_rgbd: RGB + depth reprojection over the fused global VGGT cloud
    # (self-contained -- it runs its own VGGT recon).
    if needed & {"vggt_mse", "vggt_depth_mae"}:
        _win = os.environ.get("NFT_REWARD_WINDOW", "full").strip().lower() not in ("", "full", "0")
        rd = _score_reproj_rgbd(frames_dir, work, gpu_id, spec=spec, want_perframe=_win,
                                voxel_frames=voxel_frames)
        if rd is None:
            return float("-inf"), metrics
        _rpf = rd.pop("R_perframe", None)   # per-latent combo -> drives windowed advantage
        _vp = rd.pop("voxel_payload", None)  # per-(voxel, frame) tables -> voxel advantage
        metrics.update(rd)
        if _rpf is not None:
            metrics["R_perframe"] = _rpf
        if _vp is not None:
            metrics["voxel_payload"] = _vp
        # This branch returns, so an extra_metrics request for hpsv3_vid is served here.
        if "hpsv3_vid" in needed:
            hv = _score_hpsv3_video(frames_dir, work, gpu_id, target_end)
            if hv is None:
                return float("-inf"), metrics
            metrics["hpsv3_vid"] = hv
        return combine(metrics, spec), metrics

    # hpsv3 on the RAW rollout video: a validation metric, requested via extra_metrics.
    if "hpsv3_vid" in needed:
        hv = _score_hpsv3_video(frames_dir, work, gpu_id, target_end)
        if hv is None:
            return float("-inf"), metrics
        metrics["hpsv3_vid"] = hv

    return combine(metrics, spec), metrics

# --------------------------------------------------------------------------- #
# Epoch-level scoring over a rollout store (one rollout = one record)
# --------------------------------------------------------------------------- #
def score_epoch(
    manifest_path: str,
    combo_name: str = "reproj_rgbd",
    work_dir: Optional[str] = None,
    gpu_id: int = 0,
    out_jsonl: Optional[str] = None,
    extra_metrics: Optional[set] = None,
    voxel_frames: int = 0,
) -> List[dict]:
    """Score every rollout in a Stage-1 manifest; write one rewards.jsonl line each.

    ``manifest_path`` is the Stage-1 ``manifest.jsonl`` (or its directory). Rows are
    ``{"scene", "rollout", "seed", "group_id", "x0_path", "cond_path", "clip_path",
    "frames_dir"}`` — one row per rollout (no chunks). Each rollout's ``frames_dir``
    is scored once; the output line is::

        {"sample_key", "scene", "rollout", "group_id", "combo", "R", "metrics",
         "R_perframe", "x0_path", "cond_path"}

    ``group_id`` is what Stage 3 normalizes within (defaults to the scene when the
    row omits it). Per-rollout failures are isolated: any scorer fault yields
    ``R = -inf`` so the rollout drops from its advantage group without killing the
    epoch. ``work_dir`` overrides where reward-job intermediates go (default:
    ``<dirname(x0_path)>/reward`` per rollout, so scoring is resumable in place).
    ``out_jsonl`` defaults to ``rewards.jsonl`` next to the manifest.

    ``voxel_frames`` > 0 (voxel reward; = the rollouts' latent count) adds a
    ``voxel_payload`` TOP-LEVEL key to each line that has one (never inside
    ``metrics``, whose values the loop ``float()``s). Only nft_loop's training call
    passes it, so validation scoring stays plain-global by construction.
    """
    import time as _t

    manifest = Path(manifest_path)
    if manifest.is_dir():
        manifest = manifest / "manifest.jsonl"
    assert manifest.exists(), f"no manifest at {manifest}"
    rows = [json.loads(l) for l in open(manifest)]

    spec = COMBOS.get(combo_name)
    if spec is None:
        raise ValueError(f"unknown combo '{combo_name}', expected one of {sorted(COMBOS)}")

    out_path = Path(out_jsonl) if out_jsonl else manifest.parent / "rewards.jsonl"

    def _frames(m) -> Optional[str]:
        return m.get("frames_dir") or m.get("clip_path")

    def _work(m) -> Path:
        if work_dir is not None:
            return Path(work_dir) / str(m["scene"]) / f"r{m['rollout']}"
        return Path(os.path.dirname(m["x0_path"])) / "reward"

    # Batch hpsv3_vid across all this rank's rollouts in one subprocess so the hpsv3
    # model loads once instead of per rollout; score_clip then finds each json.
    if "hpsv3_vid" in ({t.metric for t in spec} | set(extra_metrics or ())):
        items = [(fd, _work(m)) for m in rows
                 for fd in [_frames(m)] if fd and os.path.isdir(fd)]
        if items:
            _t0 = _t.time()
            _batch_hpsv3_video(items, gpu_id)
            print(f"[nft_score] gpu{gpu_id} batched hpsv3_vid over {len(items)} rollouts "
                  f"in {_t.time() - _t0:.1f}s", flush=True)

    results: List[dict] = []
    n_roll = len(rows)
    with open(out_path, "w") as fout:
        for ri, m in enumerate(rows):
            scene, rollout = m["scene"], m["rollout"]
            fd = _frames(m)
            if fd and os.path.isdir(fd):
                print(f"[nft_score] gpu{gpu_id} scoring {scene}/r{rollout} "
                      f"({ri + 1}/{n_roll}) ...", flush=True)
                _t0 = _t.time()
                try:
                    R, raw = score_clip(fd, str(_work(m)), combo_name, gpu_id,
                                        extra_metrics=extra_metrics,
                                        voxel_frames=voxel_frames)
                except Exception as e:  # noqa: BLE001 -- isolate per-rollout scorer faults
                    print(f"[nft_score] score_clip failed for {scene}/r{rollout}: "
                          f"{type(e).__name__}: {e}", flush=True)
                    R, raw = float("-inf"), {}
                print(f"[nft_score] gpu{gpu_id} scored {scene}/r{rollout} "
                      f"R={R:.4f} in {_t.time() - _t0:.1f}s", flush=True)
            else:
                print(f"[nft_score] gpu{gpu_id} {scene}/r{rollout}: missing frames dir "
                      f"({fd!r}) -> R=-inf", flush=True)
                R, raw = float("-inf"), {}
            # Per-latent-frame combo reward for the densified windowed advantage. Kept as
            # a top-level key (not inside `metrics`, whose values the loop float()s).
            # Same for the voxel tables (voxel_payload).
            rpf = raw.pop("R_perframe", None) if isinstance(raw, dict) else None
            vp = raw.pop("voxel_payload", None) if isinstance(raw, dict) else None
            rec = {
                "sample_key": f"{scene}/r{rollout}",
                "scene": scene,
                "rollout": rollout,
                "group_id": m.get("group_id", scene),  # Stage 3 z-scores within this
                "combo": combo_name,
                "R": R,
                "metrics": raw,
                "R_perframe": rpf,
                "x0_path": m["x0_path"],
                "cond_path": m["cond_path"],
            }
            if vp is not None:
                rec["voxel_payload"] = vp
            results.append(rec)
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
    return results

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DiffusionNFT Stage 2: offline reward scoring")
    ap.add_argument("manifest", help="Stage-1 manifest.jsonl (or the rollout_epoch dir holding it)")
    ap.add_argument("out_jsonl", nargs="?", default=None,
                    help="output rewards jsonl path (default: rewards.jsonl next to the manifest)")
    ap.add_argument("--combo", default="reproj_rgbd", choices=sorted(COMBOS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--work-dir", default=None,
                    help="override reward-job work dir (default: <dirname(x0_path)>/reward)")
    ap.add_argument("--voxel-frames", type=int, default=0,
                    help="also emit voxel_payload per rollout: N frames (one per latent, "
                         "so N = the rollouts' latent count) decoded FULL-FRAME and pooled "
                         "into per-(voxel, frame) error tables (wan.rl.scoring.nft_voxel)")
    args = ap.parse_args()
    res = score_epoch(args.manifest, args.combo, work_dir=args.work_dir, gpu_id=args.gpu,
                      out_jsonl=args.out_jsonl, voxel_frames=args.voxel_frames)
    finite = [r for r in res if r["R"] not in (float("-inf"), float("inf")) and r["R"] == r["R"]]
    out = args.out_jsonl or "rewards.jsonl (next to manifest)"
    print(f"scored {len(res)} rollouts ({len(finite)} finite) -> {out}")

if __name__ == "__main__":
    _main()

def score_val_full(frames_dir: str, work: str, gpu_id: int) -> Dict[str, float]:
    """VALIDATION-ONLY: the full VideoGPA suite + raw-video HPSv3 for one clip.

    Training rewards come from ``score_clip``/``score_epoch`` (which compute only the
    metrics the active combo needs). Validation wants everything, so this calls the
    same entry point the offline dl3dv scorer uses
    (``scorers.dl3dv_videogpa._init`` + ``score_video``) and merges HPSv3 on top.

    Every key returned is logged by nft_loop as ``val/<key>``. never raises: any
    sub-scorer failure yields a partial dict so validation can't abort training.
    Metrics needing deps absent from the image (``epipolar`` -> LightGlue,
    ``da3_*`` -> DepthAnything3 weights) simply won't appear if unavailable.
    """
    import gc
    import importlib
    import types as _types

    out: Dict[str, float] = {}
    global _VAL_FULL_CTX
    mp4 = Path(work) / "val_full.mp4"
    try:
        if _frames_to_mp4(Path(frames_dir), mp4):
            _import_reproj_rgbd()          # sets up the synthetic 'scorers' namespace
            vg = importlib.import_module("scorers.dl3dv_videogpa")
            if _VAL_FULL_CTX is None:
                args = _types.SimpleNamespace(
                    device="cuda",
                    vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"),
                    setup_da3=False,
                )
                _VAL_FULL_CTX = vg._init(args)
            n = int(os.environ.get("NFT_VAL_GPA_FRAMES", "16"))
            res = vg.score_video(_VAL_FULL_CTX, mp4, n) or {}
            out.update({k: float(v) for k, v in res.items()
                        if isinstance(v, (int, float))})
    except Exception as e:  # noqa: BLE001 -- val must never abort training
        print(f"[nft_score] val full videogpa failed for {frames_dir}: "
              f"{type(e).__name__}: {e}", flush=True)

    # HPSv3 is OFF by default here. _score_hpsv3_video spawns a per-clip subprocess
    # that loads the HPSv3 VLM; called from validation it runs on every rank at once
    # (32 ranks on a 4-node job) and deadlocks the job -- all val clips reach
    # this point, logged nothing further, and the job sat silent on 32 GPUs until
    # cancelled. lyra only ever drives HPSv3 through score_epoch's BATCHED
    # --manifest path (one subprocess per rank), never per-clip like this.
    # Set NFT_VAL_HPSV3=1 to re-enable; otherwise score HPSv3 offline from the
    # uploaded val videos.
    if os.environ.get("NFT_VAL_HPSV3", "0") == "1":
        try:
            # Reads <work>/hpsv3_vid/hpsv3.json when batch_val_hpsv3() has already
            # written it, so the HPSv3 VLM loads once per rank per val pass. Calling
            # this per clip without the batch pre-pass spawns one VLM subprocess per
            # clip on every rank at once, which deadlocked a 32-GPU job (all val
            # clips reached it, no output, silent until cancelled).
            hv = _score_hpsv3_video(Path(frames_dir), Path(work), gpu_id)
            if hv is not None:
                out["hpsv3_vid"] = float(hv)
        except Exception as e:  # noqa: BLE001
            print(f"[nft_score] val hpsv3 failed for {frames_dir}: "
                  f"{type(e).__name__}: {e}", flush=True)

    return out

def batch_val_hpsv3(items: List[tuple], gpu_id: int) -> None:
    """Pre-score HPSv3 for a whole validation pass in one subprocess per rank.

    ``items`` = [(frames_dir, work_dir)], one per val clip this rank owns. Must run
    before the per-clip score_val_full() calls; each clip then reads the cached json
    instead of reloading the model. Mirrors what score_epoch does for training
    rollouts -- the path lyra has always used, and the absence of which is why the
    naive per-clip version hung a 32-GPU run.
    """
    if os.environ.get("NFT_VAL_HPSV3", "0") != "1" or not items:
        return
    try:
        _batch_hpsv3_video(items, gpu_id)
    except Exception as e:  # noqa: BLE001 -- val must never abort training
        print(f"[nft_score] val hpsv3 batch failed: {type(e).__name__}: {e}", flush=True)

def release_val_full_ctx() -> None:
    """Drop the validation scorer's VGGT + DA3 context and free its GPU memory.

    Call once after a validation pass, before the DiT goes back on the GPU.
    The context is deliberately cached ACROSS the clips of one pass (with
    VAL_K=5/10 a rank scores several clips, and reloading both backbones per
    clip would dominate val time) but must not outlive the pass: holding it
    into training OOM'd a 32-GPU run, where the next rollout's VAE decode died
    at 78.35/79.19 GiB (wan/modules/vae2_1.py -> F.pad, asking for 854 MiB).
    """
    global _VAL_FULL_CTX
    if _VAL_FULL_CTX is None:
        return
    _VAL_FULL_CTX = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 -- cleanup must not mask a real failure
        pass
