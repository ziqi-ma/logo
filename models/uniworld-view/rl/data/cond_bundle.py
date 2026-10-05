"""Per-scene conditioning bundle: everything a denoise step needs, precomputed.

UniWorld's conditioning is policy-independent (unlike Lyra's AR history), so the
expensive geometry stage (MoGe lift + render + BLIP2 + UMT5 + VACE VAE encode)
runs once per scene and the training loop never loads those models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import torch

@dataclass
class CondBundle:
    prompt_embeds: torch.Tensor           # [1,512,4096] bf16
    negative_prompt_embeds: torch.Tensor  # [1,512,4096] bf16
    conditioning_latents: torch.Tensor    # [1,96,Tl,60,104] bf16 (VACE control+mask)
    ref_latents: Optional[torch.Tensor]   # [1,16,3,60,104] bf16 or None
    conditioning_scale: torch.Tensor      # [num_vace_layers] float
    height: int
    width: int
    num_frames: int
    meta: dict

    def to(self, device, dtype=None) -> "CondBundle":
        def mv(t):
            if t is None:
                return None
            return t.to(device=device, dtype=dtype) if dtype else t.to(device)
        return CondBundle(mv(self.prompt_embeds), mv(self.negative_prompt_embeds),
                          mv(self.conditioning_latents), mv(self.ref_latents),
                          self.conditioning_scale.to(device),
                          self.height, self.width, self.num_frames, self.meta)

def save_cond_bundle(b: CondBundle, path: str) -> None:
    torch.save({
        "prompt_embeds": b.prompt_embeds.cpu(),
        "negative_prompt_embeds": b.negative_prompt_embeds.cpu(),
        "conditioning_latents": b.conditioning_latents.cpu(),
        "ref_latents": None if b.ref_latents is None else b.ref_latents.cpu(),
        "conditioning_scale": b.conditioning_scale.cpu(),
        "height": b.height, "width": b.width, "num_frames": b.num_frames,
        "meta": b.meta,
    }, path)

def _ensure_local(path: str) -> str:
    """Fetch one cond bundle on first use if it is not already on local disk.

    Every training pod used to pre-stage all cond bundles before rank 0 could start: 1,683
    objects / 57.3 GB per pod, of which a step consumes only `scenes-parallel` scenes (8).
    Staging hundreds of GB keeps the whole gang idle for tens of minutes, and on an
    opportunistic preemption every pod pays it again before it can rejoin the rendezvous.

    A bundle is ~34 MB against a ~27.5 min step, so fetching on demand is free in comparison.
    Set NFT_COND_URI to the remote prefix to enable; unset keeps the old pre-staged behaviour.
    Download goes to a temp file then renames, so the 8 ranks sharing a pod`s filesystem can
    race without any of them reading a half-written bundle.
    """
    if os.path.exists(path):
        return path
    remote = os.environ.get("NFT_COND_URI", "").rstrip("/")
    if not remote:
        return path                      # pre-staged mode: let torch.load raise as before
    scene = os.path.basename(os.path.dirname(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.part.{os.getpid()}"
    from rl.data.gcs_util import download_file
    # Retry with backoff: a transient the object store blip here is otherwise FATAL to the rank
    # (kills the gang mid-step): SSLEOFError -> RetriesExceededError
    # took down rank10 of a 32-GPU run.
    import time
    for attempt in range(5):
        try:
            download_file(f"{remote}/{scene}/cond.pt", tmp)
            break
        except Exception as e:  # noqa: BLE001 - retry any transport error
            if attempt == 4:
                raise
            wait = 2 ** attempt * 3
            print(f"[cond_bundle] fetch {scene} failed ({type(e).__name__}: {e}); "
                  f"retry {attempt + 1}/4 in {wait}s", flush=True)
            time.sleep(wait)
    os.replace(tmp, path)                # atomic; last writer wins, content identical
    return path

def load_cond_bundle(path: str, device="cpu") -> CondBundle:
    d = torch.load(_ensure_local(path), map_location=device, weights_only=False)
    return CondBundle(d["prompt_embeds"], d["negative_prompt_embeds"],
                      d["conditioning_latents"], d["ref_latents"],
                      d["conditioning_scale"], d["height"], d["width"],
                      d["num_frames"], d["meta"])
