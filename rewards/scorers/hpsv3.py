#!/usr/bin/env python3
"""Per-frame HPSv3 score for the beam-search pipeline (higher is better).

Mirrors eval_disc.py's `hpsv3` reward: run HPSv3RewardInferencer on the GS-recon render
(gs_trajectory.mp4) every `stride` frames and average the scores. One frame per call —
the Qwen2-VL backbone nearly fills the GPU, so batching OOMs. hpsv3 is pre-installed into
this (worker base) env once by run_beam_search_worker; it is not installed here.

Usage:
    python scorers/hpsv3.py <video_or_frames_dir> <output_json> \
        [--target-end N] [--stride 10] [--prompt "..."]

Writes:
    <output_json> – {"hpsv3_score": float, "num_frames": int}
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

DEFAULT_PROMPT = "a scene from a potentially unconventional view"

def _load_frames(input_path: Path, max_frames: int | None) -> list[np.ndarray]:
    """Load BGR frames (uint8, HWC) from a video file or a *.png directory."""
    if input_path.is_dir():
        pngs = sorted(input_path.glob("*.png"), key=lambda p: int(p.stem))
        if max_frames is not None:
            pngs = pngs[:max_frames]
        return [img for p in pngs if (img := cv2.imread(str(p))) is not None]

    cap = cv2.VideoCapture(str(input_path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(bgr)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def _score_one(model, torch, device, input_path: Path, output_json: Path,
               stride: int, target_end, prompt: str) -> None:
    """Score one clip (video or PNG dir) and write {hpsv3_score, num_frames} to output_json."""
    max_frames = (target_end + 1) if target_end is not None else None
    frames = _load_frames(input_path, max_frames)
    if not frames:
        raise RuntimeError(f"No frames found in {input_path}")
    idxs = list(range(0, len(frames), stride)) or [0]
    print(f"[HPSV3] scoring {len(idxs)} of {len(frames)} frames (stride={stride}) {input_path}")
    scores: list[float] = []
    with tempfile.TemporaryDirectory() as td:
        for i in idxs:
            p = f"{td}/f{i:05d}.png"
            cv2.imwrite(p, frames[i])
            r = model.reward(prompts=[prompt], image_paths=[p])  # kw order differs pip vs github
            scores.append(float(r[0][0].item()))
            if device == "cuda":
                torch.cuda.empty_cache()
    mean_score = float(np.mean(scores)) if scores else 0.0
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"hpsv3_score": mean_score, "num_frames": len(scores),
                                        "hpsv3_scores": scores, "frame_idxs": idxs}, indent=2))
    print(f"[HPSV3] hpsv3_score={mean_score:.6f} over {len(scores)} frames -> {output_json}")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", type=Path, nargs="?", help="Rendered video file or PNG frames dir.")
    ap.add_argument("output_json", type=Path, nargs="?")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="JSON list of {input, output, stride?, target_end?}: load the model ONCE "
                         "and score every entry (amortizes the model load across a rank's rollouts).")
    ap.add_argument("--target-end", type=int, default=None,
                    help="Consider frames [0, target_end] inclusive. Default: all frames.")
    ap.add_argument("--stride", type=int, default=10, help="Sample every Nth frame.")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = ap.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HPSV3] device={device} manifest={args.manifest} input={args.input}")

    # hpsv3 lives in its own env (transformers==4.45.2 conflicts with lyra2); load it once.
    from hpsv3 import HPSv3RewardInferencer

    model = HPSv3RewardInferencer(device=device)

    if args.manifest is not None:
        entries = json.loads(Path(args.manifest).read_text())
        print(f"[HPSV3] batch: {len(entries)} clips, model loaded once")
        ok = 0
        for e in entries:
            try:
                _score_one(model, torch, device, Path(e["input"]), Path(e["output"]),
                           int(e.get("stride", args.stride)), e.get("target_end", args.target_end),
                           e.get("prompt", args.prompt))
                ok += 1
            except Exception as exc:  # noqa: BLE001 -- one bad clip shouldn't drop the whole batch
                print(f"[HPSV3] FAILED {e.get('input')}: {type(exc).__name__}: {exc}", flush=True)
        print(f"[HPSV3] batch done: {ok}/{len(entries)} scored")
        return

    if args.input is None or args.output_json is None:
        ap.error("input and output_json are required unless --manifest is given")
    _score_one(model, torch, device, args.input, args.output_json,
               args.stride, args.target_end, args.prompt)

if __name__ == "__main__":
    main()
