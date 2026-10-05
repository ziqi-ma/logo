#!/usr/bin/env python3
"""VideoReward (KwaiVGI/VideoAlign) visual-quality score (VQ) per clip.

A VLM reward model (Qwen2-VL-2B + rm_head, human-preference Bradley-Terry). VQ is visual
quality: clarity, aesthetics, frame reasonableness, on a 1-5 scale, optionally z-normalized.

The model also emits motion quality and text alignment from the same forward pass; they are
not kept. Two settings exist for those heads and are still load-bearing for VQ's
comparability across runs, so leave them alone: the dense ``fps=2`` sampling (changing it
changes VQ) and the text prompt (VQ was calibrated with one fed in).

Runs in its own interpreter, like HPSv3: VideoAlign pins its own stack.

  git clone https://github.com/KwaiVGI/VideoAlign            # provides inference.py
  huggingface-cli download KwaiVGI/VideoReward --local-dir <ckpt>

  <videoreward-python> rewards/scorers/videoreward.py \
      --videos <dir of <sid>.mp4> --prompts <json {sid: prompt}> --out <dir> \
      --ckpt <ckpt> --videoalign <VideoAlign checkout> [--fps 2] [--batch 8] [--no_norm]

Operates on local files and is resumable: a sid whose result already exists is skipped.
"""
import argparse
import glob
import json
import os
import sys
import traceback

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of <sid>.mp4")
    ap.add_argument("--prompts", required=True, help="json {sid: prompt}")
    ap.add_argument("--out", required=True, help="output dir for <sid>.json")
    ap.add_argument("--ckpt", default=os.environ.get("VIDEOREWARD_CKPT", ""),
                    help="VideoReward checkpoint dir (or set VIDEOREWARD_CKPT)")
    ap.add_argument("--videoalign", default=os.environ.get("VIDEOALIGN_SRC", ""),
                    help="VideoAlign checkout, for `import inference` (or set VIDEOALIGN_SRC)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--num_frames", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--no_norm", action="store_true")
    a = ap.parse_args()
    if not a.ckpt:
        raise SystemExit("--ckpt (or VIDEOREWARD_CKPT): huggingface-cli download KwaiVGI/VideoReward")
    if not a.videoalign:
        raise SystemExit("--videoalign (or VIDEOALIGN_SRC): git clone "
                         "https://github.com/KwaiVGI/VideoAlign")

    sys.path.insert(0, a.videoalign)
    import torch
    from inference import VideoVLMRewardInference

    prompts = json.load(open(a.prompts))
    os.makedirs(a.out, exist_ok=True)
    sids = sorted(s for s in prompts
                  if os.path.exists(f"{a.videos}/{s}.mp4")
                  and not os.path.exists(f"{a.out}/{s}.json"))
    print(f"VideoReward: {len(prompts)} prompts, {len(sids)} to score  ckpt={a.ckpt}", flush=True)
    if not sids:
        return

    inf = VideoVLMRewardInference(a.ckpt, device="cuda", dtype=torch.bfloat16)
    done = err = 0
    for i in range(0, len(sids), a.batch):
        chunk = sids[i:i + a.batch]
        vps = [f"{a.videos}/{s}.mp4" for s in chunk]
        prs = [prompts[s] for s in chunk]
        try:
            # fps and num_frames are exclusive: if num_frames is given, do not also pass fps
            kw = {"num_frames": a.num_frames} if a.num_frames else {"fps": a.fps}
            if a.max_pixels:
                kw["max_pixels"] = a.max_pixels
            rewards = inf.reward(vps, prs, use_norm=not a.no_norm, **kw)
            for s, r in zip(chunk, rewards):
                json.dump({"VQ": r["VQ"], "prompt": prompts[s]},
                          open(f"{a.out}/{s}.json", "w"))
                done += 1
        except Exception:
            err += len(chunk)
            print(f"ERR chunk {chunk[0]}..: {traceback.format_exc()[-300:]}", flush=True)
        if (done + err) % 40 < a.batch:
            print(f"  {done + err}/{len(sids)} ok={done} err={err}", flush=True)
    print(f"DONE ok={done} err={err}", flush=True)

if __name__ == "__main__":
    main()
