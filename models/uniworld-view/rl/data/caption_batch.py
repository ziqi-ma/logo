"""Precompute BLIP2 captions for staged benchmark first-frames.

Input: a root of <cell>/<sid>.png (stage_uniworldsub.py's CAP_DIR).
Output: <root>/<cell>_uniworld_captions.json  (sid -> raw BLIP2 caption, no
refine_prompt suffix — eval_gen/vibe_gen appends opts.refine_prompt).

    python -m rl.data.caption_batch --root <CAP_DIR> --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--blip_path", default="./checkpoints/blip2-opt-2.7b")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from transformers import AutoProcessor, Blip2ForConditionalGeneration
    proc = AutoProcessor.from_pretrained(args.blip_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        args.blip_path, torch_dtype=torch.float16).to(args.device).eval()

    for cell_dir in sorted(d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)):
        cell = os.path.basename(cell_dir)
        out_path = os.path.join(args.root, f"{cell}_uniworld_captions.json")
        caps = json.load(open(out_path)) if os.path.exists(out_path) else {}
        pngs = sorted(glob.glob(os.path.join(cell_dir, "*.png")))
        for p in pngs:
            sid = os.path.splitext(os.path.basename(p))[0]
            if sid in caps:
                continue
            image = Image.open(p).convert("RGB")
            inputs = proc(images=image, return_tensors="pt").to(args.device, torch.float16)
            with torch.no_grad():
                ids = model.generate(**inputs)
            caps[sid] = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            json.dump(caps, open(out_path, "w"), indent=1)
        print(f"[caption_batch] {cell}: {len(caps)}/{len(pngs)}", flush=True)
    print("CAPTIONS_DONE", flush=True)

if __name__ == "__main__":
    main()
