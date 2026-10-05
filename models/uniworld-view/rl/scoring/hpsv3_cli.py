"""HPSv3 perceptual score for one mp4, as a standalone CLI.

Runs in ITS own interpreter (HPSV3_PY, default /opt/hpsv3-venv/bin/python): the hpsv3
package pins transformers==4.45.2, while the training env needs 4.48.3 for the UniView
transformer + peft adapters. Installing hpsv3 into the training env would silently
downgrade transformers and break adapter loading, so the two never share a site-packages.

    <hpsv3-python> -m rl.scoring.hpsv3_cli --clip a.mp4 [--stride 10] [--prompt "..."]
    -> {"hpsv3_vid": 4.72, "n_frames": 9}   (higher is better)

Mirrors the `scorers/hpsv3.py` protocol: score every --stride-th frame with the 7B
Qwen2-VL reward model, one frame per call (the backbone nearly fills the GPU), and mean.
"""
from __future__ import annotations

import argparse
import json
import tempfile

# Matches the shared rewards/scorers/hpsv3.py prompt. HPSv3 is text-conditioned, so the prompt sets
# the scale: this one scores 5-6.5 on the mixed indoor/outdoor/stylized val set, whereas
# scorers/hpsv3.py's "a photorealistic indoor scene" scores ~0 on the same clips.
# Cross-source HPSv3 numbers are only comparable when the prompt matches.
DEFAULT_PROMPT = "a scene from a potentially unconventional view"

def _score_one(clip, model, args, cv2, torch, tempfile):
    """Score one clip with an ALREADY-LOADED model. Same protocol as the single-clip path."""
    cap = cv2.VideoCapture(clip)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {clip}")
    idxs = list(range(0, len(frames), max(1, args.stride))) or [0]
    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in idxs:
            p = f"{tmp}/f{i:05d}.png"
            cv2.imwrite(p, frames[i])
            r = model.reward(prompts=[args.prompt], image_paths=[p])
            scores.append(float(r[0][0].item()))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"hpsv3_vid": sum(scores) / len(scores), "n_frames": len(scores),
            "hpsv3_pf": scores, "hpsv3_pf_idx": idxs[:len(scores)], "stride": max(1, args.stride)}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="",
                    help="single clip; omit when using --clips")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--clips", default="",
                    help="comma-separated clips scored in ONE process (model loads once). "
                         "Training needs this: --clip spawns a fresh 7B load per rollout, which "
                         "at 128 rollouts/step would cost more than the rest of the step.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import cv2
    import torch
    from hpsv3 import HPSv3RewardInferencer

    clip_list = [c for c in args.clips.split(",") if c.strip()] or ([args.clip] if args.clip else [])
    if not clip_list:
        print(json.dumps({"error": "give --clip or --clips"}))
        raise SystemExit(2)
    if len(clip_list) > 1 or args.clips:
        model = HPSv3RewardInferencer(device=args.device)
        results = {}
        for cp in clip_list:
            try:
                results[cp] = _score_one(cp, model, args, cv2, torch, tempfile)
            except Exception as e:  # noqa: BLE001 - one bad clip must not void the batch
                results[cp] = {"error": f"{type(e).__name__}: {e}"}
        print(json.dumps({"batch": results}))
        return
    cap = cv2.VideoCapture(args.clip)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        print(json.dumps({"error": f"no frames decoded from {args.clip}"}))
        raise SystemExit(1)

    model = HPSv3RewardInferencer(device=args.device)
    idxs = list(range(0, len(frames), max(1, args.stride))) or [0]
    scores = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in idxs:
            p = f"{tmp}/f{i:05d}.png"
            cv2.imwrite(p, frames[i])
            r = model.reward(prompts=[args.prompt], image_paths=[p])
            scores.append(float(r[0][0].item()))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    # hpsv3_pf: the per-sampled-frame series with the PIXEL frame index each score came from.
    # The clip mean alone cannot drive a windowed or voxel reward -- those z-score per window, so
    # they need the series. Consumers map pixel index -> latent/window; emitting the indices keeps
    # that mapping out of this script (stride is a CLI knob and must not be re-derived).
    out = {"hpsv3_vid": sum(scores) / len(scores), "n_frames": len(scores),
           "hpsv3_pf": scores, "hpsv3_pf_idx": idxs[:len(scores)], "stride": max(1, args.stride)}
    print(json.dumps(out))

if __name__ == "__main__":
    main()
