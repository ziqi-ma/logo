"""Lean denoise loop over a CondBundle (mirror of WanVACEPipeline.__call__ steps
6+, pipeline_uniview.py:1011-1078) — no @no_grad decorator, no stateful
scheduler, no text encoding. Used for on-policy rollouts and validation."""
from __future__ import annotations

import numpy as np
import torch

from rl.data.cond_bundle import CondBundle
from rl.loop.schedule import RLSchedule

def denoise_forward(transformer, xt: torch.Tensor, t: torch.Tensor, bundle: CondBundle,
                    use_negative: bool = False) -> torch.Tensor:
    """One conditional transformer forward (velocity prediction)."""
    embeds = bundle.negative_prompt_embeds if use_negative else bundle.prompt_embeds
    return transformer(
        hidden_states=xt.to(transformer.dtype),
        timestep=t,
        encoder_hidden_states=embeds,
        ref_latents=bundle.ref_latents,
        control_hidden_states=bundle.conditioning_latents,
        control_hidden_states_scale=bundle.conditioning_scale.to(transformer.dtype),
        return_dict=False,
    )[0]

@torch.no_grad()
def sample_rollout(transformer, bundle: CondBundle, sched: RLSchedule, seed: int,
                   guidance_scale: float = 4.0, device="cuda") -> torch.Tensor:
    """K=1 rollout: returns clean x0 latents [1,16,Tl,60,104] fp32."""
    Tl = bundle.conditioning_latents.shape[2]
    shape = (1, 16, Tl, bundle.height // 8, bundle.width // 8)
    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=gen, device=device, dtype=torch.float32)
    sig = sched.sigmas.to(device)
    for i in range(sched.num_steps):
        t = sched.timesteps[i].to(device).expand(1)
        v = denoise_forward(transformer, latents, t, bundle)
        if guidance_scale > 1.0:
            v_u = denoise_forward(transformer, latents, t, bundle, use_negative=True)
            v = v_u + guidance_scale * (v - v_u)
        latents = latents + (sig[i + 1] - sig[i]) * v.float()
    return latents

@torch.no_grad()
def decode_latents(vae, latents: torch.Tensor) -> np.ndarray:
    """Denormalize + decode -> [T,H,W,3] uint8."""
    lm = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    ls = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    z = latents.to(vae.device, vae.dtype)
    z = z / ls.to(z.device, z.dtype) + lm.to(z.device, z.dtype)
    video = vae.decode(z, return_dict=False)[0]  # [1,3,T,H,W] in [-1,1]
    video = (video[0].permute(1, 2, 3, 0).float().clamp(-1, 1) + 1.0) * 127.5
    return video.round().to(torch.uint8).cpu().numpy()

def write_mp4(frames: np.ndarray, path: str, fps: int = 16) -> None:
    import subprocess
    import tempfile
    from PIL import Image
    # The conda-base ffmpeg lacks libx264; prefer imageio's bundled binary.
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        ffmpeg = "ffmpeg"
    with tempfile.TemporaryDirectory() as td:
        for i, fr in enumerate(frames):
            Image.fromarray(fr, "RGB").save(f"{td}/{i:05d}.png")
        last_err = ""
        for codec in ("libx264", "libopenh264", "mpeg4"):
            rc = subprocess.run(
                [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-i", f"{td}/%05d.png", "-c:v", codec, "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", path], capture_output=True)
            if rc.returncode == 0:
                return
            last_err = rc.stderr.decode()[-300:]
        raise RuntimeError(f"ffmpeg failed for all codecs: {last_err}")
