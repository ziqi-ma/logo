"""On-disk rollout store for one training step (one-shot model: no chunk axis).

Layout:
    root/scene_<sc>/rollout_<kk>/{x0.pt, clip.mp4, meta.json}
    root/manifest.jsonl

Conditioning is not copied per rollout — meta carries the scene's cond.pt path
(policy-independent). rewards.jsonl rows: {scene, rollout, group_id, R, metrics,
x0_path, cond_path, clip_path}.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

class RolloutStoreWriter:
    def __init__(self, root: str, append: bool = False):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest = os.path.join(root, "manifest.jsonl")
        if not append:
            open(self.manifest, "w").close()

    def write(self, scene: str, rollout: int, x0: torch.Tensor, frames: np.ndarray,
              cond_path: str, seed: int, fps: int = 16) -> dict:
        from rl.loop.sampling import write_mp4
        d = os.path.join(self.root, f"scene_{scene}", f"rollout_{rollout:02d}")
        os.makedirs(d, exist_ok=True)
        x0_path = os.path.join(d, "x0.pt")
        clip_path = os.path.join(d, "clip.mp4")
        torch.save(x0.detach().cpu(), x0_path)
        write_mp4(frames, clip_path, fps=fps)
        rec = {"scene": scene, "rollout": rollout, "group_id": scene, "seed": seed,
               "x0_path": x0_path, "cond_path": cond_path, "clip_path": clip_path}
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(rec, f)
        with open(self.manifest, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

def read_manifest(root: str) -> list[dict]:
    path = os.path.join(root, "manifest.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def read_rewards(rewards_jsonl: str) -> list[dict]:
    with open(rewards_jsonl) as f:
        return [json.loads(line) for line in f if line.strip()]
