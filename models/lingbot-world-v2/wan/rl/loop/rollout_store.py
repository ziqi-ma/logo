"""On-policy rollout generation + on-disk store for lingbot DiffusionNFT.

One record per rollout (no per-chunk records — lingbot's 4-latent-frame
chunks are internal training mechanics, not samples)::

    root/
      scene_{sc}/rollout_{kk}/
        x0_gen.pt        # [C, lat_f, H, W] clean latents of the whole rollout
        cond.pt          # replay conditioning payload (see cond_payload)
        frames/00000.png ...  # decoded clip for reward scoring
        meta.json
      manifest.jsonl     # one line per rollout

The KV cache is deliberately not stored (23+ GB); training rebuilds it by
replaying commits from x0_gen (wan.rl.loop.nft_trainer).
"""
import glob
import hashlib
import json
import os
import subprocess
from typing import Optional

import torch

from wan.image2video import FastGenState, WanI2VCausal

def _stable_hash(s: str) -> int:
    """Process-independent hash (Python's builtin hash() is salted per-process
    via PYTHONHASHSEED, which made sampling seeds differ across
    otherwise-identical jobs)."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

def rollout_seed(scene: str, base_seed: int, k: int, k_total: int) -> int:
    return base_seed + (_stable_hash(scene) % 1_000_000) * k_total + k

def cond_payload(state: FastGenState) -> dict:
    """Everything prepare_replay needs to rebuild the replay state. ~110 MB
    at lat_f=20; dominated by the Plücker embeddings."""
    return {
        "context": state.context[0].detach().to("cpu", torch.bfloat16),
        "y": state.y.detach().cpu(),
        "c2ws_plucker_emb": state.c2ws_plucker_emb.detach().to("cpu", torch.bfloat16),
        "chunk_size": int(state.chunk_size),
        "timesteps_index": list(state.timesteps_index),
        "shift": float(state.shift),
        "seed": int(state.seed),
        "max_attention_size": int(state.max_attention_size),
    }

def sample_rollout(pipe: WanI2VCausal, prompt: str, img, action_path: str,
                   seed: int, frame_num: int = 81, chunk_size: int = 4,
                   timesteps_index=(0, 250, 500, 750), shift: float = 5.0,
                   max_attention_size: Optional[int] = None,
                   decode: bool = True):
    """One on-policy rollout on this rank (no rank-0 gating, no offload).

    Caller is responsible for the adapter context (rollouts run under "old").
    Returns (x0_latents [C, lat_f, H, W], video [3, F, H, W] or None, state).
    """
    state = pipe._prepare_causal_fast(
        prompt, img, action_path,
        chunk_size=chunk_size, frame_num=frame_num,
        timesteps_index=list(timesteps_index), shift=shift, seed=seed,
        offload_model=False, max_attention_size=max_attention_size)
    with torch.amp.autocast('cuda', dtype=pipe.param_dtype), torch.no_grad():
        for chunk_id in range(state.num_chunks):
            x0 = pipe._denoise_chunk_fast(state, chunk_id)
            state.pred_latent_chunks.append(x0)
            pipe._commit_chunk_fast(state, chunk_id, x0)
        x0_latents = torch.cat(state.pred_latent_chunks, dim=1)
        video = pipe.vae.decode([x0_latents])[0] if decode else None
    # Free the caches before the next rollout allocates its own.
    state.self_kv_cache = None
    state.cross_kv_cache = None
    torch.cuda.empty_cache()
    return x0_latents, video, state

class RolloutStoreWriter:

    def __init__(self, root: str, append: bool = False):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        if not append:
            open(self.manifest_path, "w").close()

    def write_rollout(self, scene: str, rollout: int, x0_latents: torch.Tensor,
                      state: FastGenState, video: Optional[torch.Tensor]) -> dict:
        rdir = os.path.join(self.root, f"scene_{scene}", f"rollout_{rollout:02d}")
        os.makedirs(rdir, exist_ok=True)

        x0_path = os.path.join(rdir, "x0_gen.pt")
        torch.save(x0_latents.detach().float().cpu(), x0_path)
        cond_path = os.path.join(rdir, "cond.pt")
        torch.save(cond_payload(state), cond_path)
        frames_dir = save_frames(video, rdir) if video is not None else None

        meta = {
            "scene": scene,
            "rollout": int(rollout),
            "seed": int(state.seed),
            "group_id": scene,
            "x0_path": x0_path,
            "cond_path": cond_path,
            "frames_dir": frames_dir,
            "clip_path": frames_dir,
        }
        with open(os.path.join(rdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return meta

def save_frames(pixels: torch.Tensor, out_dir: str) -> str:
    """Save [3,T,H,W] (or [B,3,T,H,W]) in [-1,1] as a frames/ dir of PNGs.

    A directory of PNGs is accepted by every reward runner, and avoids a hard
    video-codec dependency."""
    from PIL import Image

    if pixels.dim() == 5:
        pixels = pixels[0]
    pixels = pixels.detach().float().clamp(-1, 1)
    pixels = ((pixels + 1.0) * 127.5).round().to(torch.uint8)
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    for t in range(pixels.shape[1]):
        arr = pixels[:, t].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(os.path.join(frames_dir, f"{t:05d}.png"))
    return frames_dir

def run_scene_rollouts(pipe: WanI2VCausal, writer: RolloutStoreWriter,
                       scene: str, prompt: str, img, action_path: str,
                       k_indices, k_total: int, base_seed: int,
                       frame_num: int = 81, chunk_size: int = 4,
                       timesteps_index=(0, 250, 500, 750), shift: float = 5.0,
                       max_attention_size: Optional[int] = None) -> list:
    """Sample this rank's share of a scene's K rollouts and write them.
    Caller wraps this in adapters.adapter_ctx(model, "old")."""
    metas = []
    for k in k_indices:
        seed = rollout_seed(scene, base_seed, k, k_total)
        x0_latents, video, state = sample_rollout(
            pipe, prompt, img, action_path, seed,
            frame_num=frame_num, chunk_size=chunk_size,
            timesteps_index=timesteps_index, shift=shift,
            max_attention_size=max_attention_size, decode=True)
        metas.append(writer.write_rollout(scene, k, x0_latents, state, video))
        del x0_latents, video, state
    return metas

def encode_frames_mp4(frames_dir: str, mp4: str, fps: int = 16) -> bool:
    """Encode a frames/ dir of PNGs to mp4 (ffmpeg libx264 — cv2's mp4v writer
    silently fails for many clips). Returns True on success."""
    if not glob.glob(os.path.join(frames_dir, "*.png")):
        print(f"[encode_frames_mp4] {frames_dir}: no frames; skipping", flush=True)
        return False
    rc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
         "-pattern_type", "glob", "-i", os.path.join(frames_dir, "*.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )
    if rc.returncode != 0 or not os.path.exists(mp4):
        print(f"[encode_frames_mp4] {frames_dir}: ffmpeg rc={rc.returncode} "
              f"({rc.stderr.decode('utf-8', 'replace')[-200:]})", flush=True)
        return False
    return True

def save_rollout_videos(rollout_root: str, scene: str, videos_uri: str,
                        fps: int = 16) -> int:
    """Encode each rollout's frames to mp4 and upload to
    ``videos_uri/scene_<scene>/rollout_NN.mp4``. Best-effort. Returns count."""
    from wan.rl.data.gcs_util import upload_file

    n = 0
    for roll in sorted(glob.glob(os.path.join(rollout_root, f"scene_{scene}", "rollout_*"))):
        try:
            mp4 = roll + ".mp4"
            if not encode_frames_mp4(os.path.join(roll, "frames"), mp4, fps=fps):
                continue
            upload_file(mp4, f"{videos_uri.rstrip('/')}/scene_{scene}/{os.path.basename(roll)}.mp4")
            n += 1
        except Exception as e:  # noqa: BLE001 -- video saving must never abort the run
            print(f"[save_rollout_videos] {roll} failed: {type(e).__name__}: {e}", flush=True)
    return n
