#!/usr/bin/env python3
"""Score generated clips with the shared reward scorers, one JSON per clip.

Model-independent: it takes generated videos, not a training manifest, so the same
command scores rollouts from any of the three models (and the base model, for the
paired comparison). Every metric comes from ``rewards/scorers/`` -- the same code the
reward uses during training, so eval numbers and reward are never two implementations.

    python eval/score_videos.py --videos <dir|s3://prefix> --out results/ \
        [--traj-root <dir|s3://prefix>] [--videogpa] [--hpsv3] [--n-frames 16]

Input layout: ``<videos>/<scene>/*.mp4`` (one or more clips per scene, as the
generators write them). A flat directory of ``*.mp4`` also works; the stem is the
scene. ``s3://`` inputs are downloaded to a temp dir, one scene at a time.

Metrics, per clip:

  vggt_mse, vggt_depth_mae   the LoGo geometry terms: RGB squared error and depth
                             absolute error between each frame and the reprojection
                             of the fused VGGT-Omega point cloud (scorers/reproj_rgbd)
  rpe_rot, rpe_trans         camera adherence vs the reference trajectory, with
                             --traj-root (scorers/camera_rpe); degrees and
                             path-length-normalized translation
  hpsv3_vid                  mean HPSv3 over keyframes, with --hpsv3 (runs in
                             HPSV3_PY, a separate interpreter)
  vr_vq                      VideoReward visual quality, with --vq (runs in
                             VIDEOREWARD_PY)
  epipolar, vggt_psnr,       the full VideoGPA suite, with --videogpa (~10x the cost
  vggt_ssim, vggt_lpips,     of the geometry terms alone)
  vggt_mvcs, ...

Writes ``<out>/<scene>__<clip>.json`` per clip and appends to ``<out>/clips.jsonl``.
Resumable: a clip whose JSON already exists is skipped.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

def _rewards_dir() -> Path:
    """The shared rewards/ tree: REWARDS_DIR if set, else this repo's."""
    env = os.environ.get("REWARDS_DIR", "").strip()
    return Path(env) if env else Path(__file__).resolve().parent.parent / "rewards"

def _import_scorers():
    """Register a namespace-only ``scorers`` package over the shared tree and import the
    modules by path, without running ``scorers/__init__`` (which eagerly imports every
    scorer, including ones whose deps are absent here)."""
    rewards = _rewards_dir()
    sdir = rewards / "scorers"
    if not sdir.is_dir():
        raise SystemExit(f"no scorers tree at {sdir} (set REWARDS_DIR)")
    if "scorers" not in sys.modules:
        pkg = types.ModuleType("scorers")
        pkg.__path__ = [str(sdir)]
        sys.modules["scorers"] = pkg
    sys.path.insert(0, str(rewards))       # dl3dv_videogpa needs rewards/ importable
    return (importlib.import_module("scorers.dl3dv_videogpa"),
            importlib.import_module("scorers.reproj_rgbd"))

def _is_uri(p: str) -> bool:
    return "://" in p

def _s3():
    import boto3
    return boto3.client("s3", endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None)

def _split(uri: str):
    scheme, _, rest = uri.partition("://")
    if scheme != "s3":
        raise SystemExit(f"only s3:// URIs are supported, got {uri!r}")
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix

def _list_clips(videos: str):
    """[(scene, clip_name, fetch)] where fetch(dest) puts the mp4 at dest."""
    out = []
    if _is_uri(videos):
        bucket, prefix = _split(videos.rstrip("/") + "/")
        client, token = _s3(), None
        keys = []
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            resp = client.list_objects_v2(**kw)
            keys += [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".mp4")]
            if not resp.get("IsTruncated"):
                break
            token = resp["NextContinuationToken"]
        for k in sorted(keys):
            rel = k[len(prefix):]
            scene = rel.split("/", 1)[0] if "/" in rel else Path(rel).stem
            out.append((scene, Path(rel).stem,
                        (lambda key: lambda dest: client.download_file(bucket, key, dest))(k)))
        return out
    root = Path(videos)
    for mp4 in sorted(root.rglob("*.mp4")):
        rel = mp4.relative_to(root)
        scene = rel.parts[0] if len(rel.parts) > 1 else mp4.stem
        out.append((scene, mp4.stem, (lambda src: lambda dest: __import__("shutil").copyfile(src, dest))(mp4)))
    return out

def _ref_w2c(traj_root: str, scene: str, tmp: Path):
    """Reference w2c [F,4,4] for a scene, from <traj_root>/<scene>/*.npz."""
    import numpy as np
    names = ("lyra2_traj.npz", "trajectory.npz")
    local = None
    if _is_uri(traj_root):
        bucket, prefix = _split(traj_root.rstrip("/") + "/")
        client = _s3()
        for name in names:
            dest = tmp / f"{scene}_{name}"
            try:
                client.download_file(bucket, f"{prefix}{scene}/{name}", str(dest))
                local = dest
                break
            except Exception:  # noqa: BLE001 -- try the next schema, then give up
                continue
    else:
        for name in names:
            cand = Path(traj_root) / scene / name
            if cand.exists():
                local = cand
                break
    if local is None:
        return None
    z = np.load(local)
    key = "w2c" if "w2c" in z else None
    if key is None:
        print(f"[eval] {scene}: {local.name} has no w2c array; skipping camera terms", flush=True)
        return None
    return np.asarray(z[key], dtype=np.float64)

def _score_camera(rpe, est_w2c, ref_w2c):
    """{rpe_rot, rpe_trans} from the recon's estimated poses vs the reference path."""
    import numpy as np
    import torch
    if hasattr(est_w2c, "detach"):
        est_w2c = est_w2c.detach().cpu().numpy()
    est_w2c = np.asarray(est_w2c, dtype=np.float64)
    if len(est_w2c) < 2 or len(ref_w2c) < 2:
        return {}
    # the recon scores a uniform subsample; align the reference to its frame count
    idx = np.linspace(0, len(ref_w2c) - 1, len(est_w2c)).round().astype(int)
    ref = torch.tensor(np.linalg.inv(ref_w2c[idx]), dtype=torch.float32)   # -> c2w
    est = torch.tensor(np.linalg.inv(est_w2c), dtype=torch.float32)        # -> c2w
    rot, trans = rpe.adherence(pred_c2w=est, target_c2w=ref)
    return {"rpe_rot": float(rot), "rpe_trans": float(trans)}

def _score_hpsv3(clip: Path, stride: int) -> dict:
    """Mean HPSv3 over every stride-th frame, in its own interpreter (transformers pin)."""
    runner = _rewards_dir() / "scorers/hpsv3.py"
    py = os.environ.get("HPSV3_PY") or sys.executable
    with tempfile.TemporaryDirectory() as td:
        out_json = Path(td) / "hpsv3.json"
        rc = subprocess.run([py, str(runner), str(clip), str(out_json), "--stride", str(stride)],
                            capture_output=True, text=True)
        if not out_json.exists():
            print(f"[eval] hpsv3 failed for {clip.name} (rc={rc.returncode}): "
                  f"{rc.stdout[-300:]}{rc.stderr[-300:]}", flush=True)
            return {}
        return {"hpsv3_vid": float(json.loads(out_json.read_text())["hpsv3_score"])}

def _score_vq(clip: Path, prompt: str) -> dict:
    """VideoReward VQ for one clip, in its own interpreter (VideoAlign pins its own stack).
    Needs VIDEOREWARD_CKPT and VIDEOALIGN_SRC; see rewards/scorers/videoreward.py."""
    runner = _rewards_dir() / "scorers/videoreward.py"
    py = os.environ.get("VIDEOREWARD_PY") or sys.executable
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "vids").mkdir()
        (td / "vids" / "clip.mp4").write_bytes(clip.read_bytes())
        (td / "prompts.json").write_text(json.dumps({"clip": prompt}))
        rc = subprocess.run([py, str(runner), "--videos", str(td / "vids"),
                             "--prompts", str(td / "prompts.json"), "--out", str(td / "out")],
                            capture_output=True, text=True)
        res = td / "out" / "clip.json"
        if not res.exists():
            print(f"[eval] videoreward failed for {clip.name} (rc={rc.returncode}): "
                  f"{rc.stdout[-300:]}{rc.stderr[-300:]}", flush=True)
            return {}
        r = json.loads(res.read_text())
        return {"vr_vq": float(r["VQ"])}

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--videos", required=True, help="dir or s3:// prefix of <scene>/*.mp4")
    ap.add_argument("--out", required=True, help="output dir for the per-clip JSON")
    ap.add_argument("--traj-root", default="",
                    help="dir or s3:// prefix of <scene>/lyra2_traj.npz (adds rpe_rot/rpe_trans)")
    ap.add_argument("--n-frames", type=int, default=int(os.environ.get("NFT_REPROJ_FRAMES", "16")),
                    help="uniform frames decoded per clip for the geometry terms")
    ap.add_argument("--videogpa", action="store_true", help="also run the full VideoGPA suite")
    ap.add_argument("--hpsv3", action="store_true", help="also score HPSv3 (needs HPSV3_PY)")
    ap.add_argument("--vq", action="store_true",
                    help="also score VideoReward VQ (needs VIDEOREWARD_CKPT, VIDEOALIGN_SRC)")
    ap.add_argument("--vq-prompt", default="",
                    help="text prompt fed to VideoReward; VQ was calibrated with one, so keep it "
                         "consistent across the runs being compared")
    ap.add_argument("--hpsv3-stride", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="score at most N clips (smoke runs)")
    args = ap.parse_args()

    vg, rr = _import_scorers()
    dec = importlib.import_module("scorers.decode")
    rpe = importlib.import_module("scorers.camera_rpe") if args.traj_root else None

    clips = _list_clips(args.videos)
    if args.limit:
        clips = clips[:args.limit]
    if not clips:
        raise SystemExit(f"no *.mp4 under {args.videos}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval] {len(clips)} clips -> {out_dir}", flush=True)

    ctx = rr.load(vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"), device=args.device)
    gpa_ctx = None
    if args.videogpa:
        gpa_ctx = vg._init(types.SimpleNamespace(
            device=args.device, vggt_checkpoint=os.environ.get("VGGT_CHECKPOINT"),
            setup_da3=False))

    refs: dict = {}
    n_ok = 0
    with tempfile.TemporaryDirectory() as td, open(out_dir / "clips.jsonl", "a") as jl:
        tmp = Path(td)
        for scene, name, fetch in clips:
            dest = out_dir / f"{scene}__{name}.json"
            if dest.exists():
                try:
                    done = "error" not in json.loads(dest.read_text())
                except Exception:  # noqa: BLE001 - truncated by an interrupted write
                    done = False
                if done:
                    n_ok += 1
                    continue
            clip = tmp / f"{scene}__{name}.mp4"
            row = {"scene": scene, "clip": name}
            try:
                fetch(str(clip))
                frames = dec.decode_uniform(str(clip), args.n_frames)
                metrics = {k: float(v) for k, v in rr.score(ctx, frames).items()}
                if rpe is not None:
                    if scene not in refs:
                        refs[scene] = _ref_w2c(args.traj_root, scene, tmp)
                    if refs[scene] is not None:
                        est = vg.reproject_vggt(ctx, frames)[2]   # (imgs, reproj, w2c, ...)
                        metrics.update(_score_camera(rpe, est, refs[scene]))
                if args.hpsv3:
                    metrics.update(_score_hpsv3(clip, args.hpsv3_stride))
                if args.vq:
                    metrics.update(_score_vq(clip, args.vq_prompt))
                if gpa_ctx is not None:
                    for k, v in vg.score_video(gpa_ctx, str(clip)).items():
                        metrics.setdefault(k, float(v))
                row["metrics"] = metrics
                n_ok += 1
            except Exception as e:  # noqa: BLE001 -- one bad clip must not abort the pass
                row["error"] = f"{type(e).__name__}: {e}"
                print(f"[eval] {scene}/{name} FAILED: {row['error']}", flush=True)
            finally:
                clip.unlink(missing_ok=True)
            dest.write_text(json.dumps(row, indent=1))
            jl.write(json.dumps(row) + "\n")
            jl.flush()
            print(f"[eval] {scene}/{name}: {row.get('metrics', row.get('error'))}", flush=True)
    print(f"[eval] {n_ok}/{len(clips)} clips scored -> {out_dir}", flush=True)

if __name__ == "__main__":
    main()
