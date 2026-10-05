"""Run scorers.reproj_rgbd in a separate PROCESS, one JSON manifest in, one JSON out.

A CUDA illegal memory access inside the scorer poisons the process's CUDA context, so
catching the exception is not enough -- the next rollouts fail instantly and the run dies
somewhere unrelated. Running the scorer in a child process makes such a fault cost exactly
one rollout. It is rare (order 1e-4 per rollout), which is why it is isolated rather than
blocked on. NFT_REPROJ_INPROC=1 restores the in-process path.

Contract:
  manifest = [{mp4, n, want_perframe, spec}]  (spec = [[metric, weight, mean, std, sign], ...]
             -- all five Term fields; _combine_perframe z-scores with mean/std/sign, so weight
             alone cannot reproduce the combo)
  out      = [{ok, metrics|null, error}]  -- written INCREMENTALLY after each item, so if the
             process dies mid-manifest the caller still gets every result that completed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

def _write(out_path, results):
    tmp = f"{out_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, out_path)   # atomic: the caller never reads a half-written file

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vggt_checkpoint", default=os.environ.get("VGGT_CHECKPOINT"))
    a = ap.parse_args()

    items = json.load(open(a.manifest))
    results = [{"ok": False, "metrics": None, "error": "not reached"} for _ in items]
    _write(a.out, results)

    # Imported here (not at module import) so a bad env fails per-item with a message rather
    # than at process start with no output file for the caller to read.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from wan.rl.scoring.nft_score import Term, _combine_perframe, _import_reproj_rgbd
    import importlib
    rr = _import_reproj_rgbd()          # also registers the synthetic 'scorers' namespace
    dec = importlib.import_module("scorers.decode")
    vg = importlib.import_module("scorers.dl3dv_videogpa")
    ctx = rr.load(vggt_checkpoint=a.vggt_checkpoint, device="cuda")

    for i, it in enumerate(items):
        try:
            frames = dec.decode_uniform(it["mp4"], int(it["n"]))
            res = rr.score(ctx, frames, per_frame=bool(it.get("want_perframe")))
            spec = it.get("spec")
            if it.get("want_perframe") and spec:
                mse_pf = res.pop("vggt_mse_pf", []) or []
                dm_pf = res.pop("vggt_depth_mae_pf", []) or []
                # Rebuild real Term objects: _combine_perframe reads .metric/.weight/.mean/
                # .std/.sign off each one, so bare (metric, weight) tuples would AttributeError.
                terms = [Term(metric=str(m), weight=float(w), mean=float(mu),
                              std=float(sd), sign=float(sg)) for m, w, mu, sd, sg in spec]
                res["R_perframe"] = [
                    _combine_perframe({"vggt_mse": m, "vggt_depth_mae": d}, terms)
                    for m, d in zip(mse_pf, dm_pf)]
            vox = it.get("voxel")
            if vox:
                # Per-voxel error tables (NFT_REWARD_VOXEL), from the same loaded
                # ctx. Own try/except: a voxel failure degrades the rollout to its
                # scalar r (payload absent); it must never void the scalar R.
                try:
                    from wan.rl.scoring.nft_voxel import score_voxel_clip
                    vres, payload = score_voxel_clip(
                        ctx, it["mp4"], int(vox["n"]), float(vox["alpha"]),
                        vox["patch"], float(vox["depth_cap"]))
                    res["voxel_vggt_mse"] = vres["vggt_mse"]
                    res["voxel_vggt_dmae"] = vres["vggt_depth_mae"]
                    res["voxel_payload"] = payload
                except Exception as e:  # noqa: BLE001 -- voxel must not void R
                    print(f"[reproj_runner] item {i} voxel failed (scalar-r fallback): "
                          f"{type(e).__name__}: {e}", flush=True)
            results[i] = {"ok": True, "metrics": res, "error": None}
        except Exception as e:  # noqa: BLE001 -- report and keep going; a CUDA fault will
            # instead kill this process, which is the entire point of running here.
            results[i] = {"ok": False, "metrics": None,
                          "error": f"{type(e).__name__}: {e}"}
            print(f"[reproj_runner] item {i} failed: {type(e).__name__}: {e}", flush=True)
        _write(a.out, results)     # after every item, so a hard crash preserves the rest
    return 0

if __name__ == "__main__":
    sys.exit(main())
