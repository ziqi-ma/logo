"""DiffusionNFT training step for the lingbot causal_fast model.

Replays a stored rollout through the model's native AR structure: for each
4-latent-frame chunk, re-noise the stored clean x0 at each sampler-grid
timestep, run the three NFT forwards (old / base-as-ref / new) against the
committed clean-prefix KV cache in ``cache_mode="read_only"``, backward, and
only then commit the chunk (one no-grad forward under "old") to advance the
cache — the exact operation sampling performed.

Ordering law: activation checkpointing re-reads the KV cache at backward, so
every graded microstep of a chunk must complete (forward + backward) before
that chunk's commit mutates the cache. Enforced with a local_end_index guard.
"""
import logging
import os

import torch

from wan.image2video import FastGenState, WanI2VCausal
from wan.rl.loop import adapters
from wan.rl.loop.nft_loss import compute_nft_loss

class LingbotNFTTrainer:

    def __init__(self, pipe: WanI2VCausal, beta: float = 1.0,
                 beta_kl: float | None = None, save_on_cpu: bool = True):
        self.pipe = pipe
        self.model = pipe.model
        self.beta = beta
        self.beta_kl = (float(os.environ.get("NFT_BETA_KL", 1e-4))
                        if beta_kl is None else beta_kl)
        self.save_on_cpu = save_on_cpu
        self.model.gradient_checkpointing = True

    def prepare_replay(self, cond: dict) -> FastGenState:
        """Rebuild a FastGenState (fresh zeroed caches) from a rollout's
        stored conditioning payload (see rollout_store.cond_payload)."""
        pipe = self.pipe
        device = pipe.device
        context = [cond["context"].to(device=device, dtype=pipe.param_dtype)]
        y = cond["y"].to(device=device)
        plucker = cond["c2ws_plucker_emb"].to(device=device, dtype=pipe.param_dtype)
        chunk_size = int(cond["chunk_size"])
        lat_f, lat_h, lat_w = y.shape[-3], y.shape[-2], y.shape[-1]

        pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=cond["shift"])
        timesteps = pipe.scheduler.timesteps[list(cond["timesteps_index"])].to(device)

        frame_seqlen = lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
        if pipe.local_attn_size > -1:
            kv_size = frame_seqlen * pipe.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        model_args = self.model.config
        head_dim = model_args.dim // model_args.num_heads
        self_kv_cache = pipe._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=[1, kv_size, model_args.num_heads, head_dim],
            dtype=pipe.pipe_dtype,
            device=device)
        max_seq_len = chunk_size * frame_seqlen

        noise_placeholder = torch.empty(0)
        state = FastGenState(
            seed=int(cond.get("seed", -1)),
            seed_g=torch.Generator(device=device),
            timesteps=timesteps,
            timesteps_index=list(cond["timesteps_index"]),
            shift=float(cond["shift"]),
            context=context,
            y=y,
            c2ws_plucker_emb=plucker,
            latents_chunk=(noise_placeholder,),
            condition_chunk=y.split(chunk_size, dim=1),
            plucker_chunk=plucker.split(chunk_size, dim=2),
            self_kv_cache=self_kv_cache,
            # Replay never uses the cross-attn cache: all forwards recompute
            # context K/V in-graph (adapter-correct, and sidesteps the
            # is_init state that sampling's first forward would have set).
            cross_kv_cache=[None] * model_args.num_layers,
            chunk_size=chunk_size,
            frame_seqlen=frame_seqlen,
            kv_size=kv_size,
            max_seq_len=max_seq_len,
            max_attention_size=int(cond.get("max_attention_size", kv_size)),
            lat_f=lat_f,
            lat_h=lat_h,
            lat_w=lat_w,
            h=lat_h * pipe.vae_stride[1],
            w=lat_w * pipe.vae_stride[2],
            F=(lat_f - 1) * 4 + 1,
            offload_model=False)
        self._sigma_cache = {}
        return state

    def _sigma_of(self, timestep: torch.Tensor) -> torch.Tensor:
        """Nearest-timestep sigma lookup, same convention as
        WanI2VCausal._convert_flow_pred_to_x0."""
        key = float(timestep)
        if key not in self._sigma_cache:
            sched = self.pipe.scheduler
            tid = torch.argmin((sched.timesteps.to(timestep.device) - timestep).abs())
            self._sigma_cache[key] = sched.sigmas.to(timestep.device)[tid].float()
        return self._sigma_cache[key]

    def _forward(self, state, chunk_id, xt, timestep, grad: bool):
        kwargs = self.pipe._fast_chunk_kwargs(state, chunk_id)
        kwargs["cache_mode"] = "read_only"
        t = torch.stack([timestep]).to(self.pipe.device)
        with torch.amp.autocast('cuda', dtype=self.pipe.param_dtype):
            if grad:
                if self.save_on_cpu:
                    with torch.autograd.graph.save_on_cpu(pin_memory=True):
                        return self.model(x=[xt], t=t, **kwargs)[0]
                return self.model(x=[xt], t=t, **kwargs)[0]
            with torch.no_grad():
                return self.model(x=[xt], t=t, **kwargs)[0]

    def _commit(self, state, chunk_id, x0_c):
        """Advance the cache past this chunk under the rollout policy, with
        the stored clean x0 — bit-equivalent to sampling's commit forward."""
        kwargs = self.pipe._fast_chunk_kwargs(state, chunk_id)
        t = torch.stack([state.timesteps[-1] * 0.0]).to(self.pipe.device)
        with adapters.adapter_ctx(self.model, "old"), torch.no_grad(), \
                torch.amp.autocast('cuda', dtype=self.pipe.param_dtype):
            self.model(x=[x0_c], t=t, **kwargs)

    def _committed_end(self, state) -> int:
        return int(state.self_kv_cache[0]["local_end_index"].item())

    def train_rollout(self, state: FastGenState, x0_full: torch.Tensor,
                      r: torch.Tensor, scale: float, t_indices=None,
                      global_step: int = 0):
        """Accumulate gradients for one rollout: for each native chunk x
        timestep, three forwards + one backward; commit between chunks.

        Args:
            state: fresh replay state from prepare_replay (caches at zero).
            x0_full: [C, lat_f, H, W] stored clean latents of the rollout.
            r: advantage-derived weight in [0, 1]. Scalar (shared by every
                microstep of the rollout), per-latent-frame ``[lat_f]``
                (windowed reward), or per-patch ``[lat_f, gh, gw]`` (voxel
                reward) -- each chunk then trains against its own T-slice,
                so credit lands on the frames/regions that earned it.
            scale: gradient scale, typically 1/(n_rollouts * C * n_t).
            t_indices: which timestep indices of the sampler grid to train
                (default: all).
        Returns (metrics, n_ok, n_oom): metrics are means over microsteps.
        """
        device = self.pipe.device
        x0_full = x0_full.to(device=device, dtype=torch.float32)
        x0_chunks = x0_full.split(state.chunk_size, dim=1)
        if t_indices is None:
            t_indices = range(len(state.timesteps))
        r = r.to(device=device, dtype=torch.float32)
        if r.dim() == 3:
            # Voxel r [lat_f, gh, gw]: keep the patch axes; each chunk slices its
            # own latent frames below. compute_nft_loss takes [B, T, gh, gw] and a
            # spatially-constant map reproduces the per-frame loss exactly.
            if r.shape[0] != x0_full.shape[1]:
                raise ValueError(
                    f"voxel r has {r.shape[0]} frames; expected {x0_full.shape[1]} "
                    "(one per latent frame)")
        else:
            r = r.reshape(-1)
            # Scalar r -> one weight for the whole rollout. Per-latent r -> slice it
            # per chunk below; compute_nft_loss takes [B] or [B, T] and a constant-over-
            # frames [B, T] reproduces the scalar loss exactly.
            if r.numel() not in (1, x0_full.shape[1]):
                raise ValueError(
                    f"r has {r.numel()} entries; expected 1 or {x0_full.shape[1]} "
                    "(one per latent frame)")

        agg, n_ok, n_oom = {}, 0, 0
        lat0 = 0
        for chunk_id, x0_c in enumerate(x0_chunks):
            committed = self._committed_end(state)
            n_lat = x0_c.shape[1]
            if r.dim() == 3:
                r_chunk = r[lat0:lat0 + n_lat][None]       # [1, n_lat, gh, gw]
            else:
                r_chunk = (r.reshape(1) if r.numel() == 1
                           else r[lat0:lat0 + n_lat].reshape(1, n_lat))
            lat0 += n_lat
            for ti in t_indices:
                timestep = state.timesteps[ti]
                sigma = self._sigma_of(timestep)
                try:
                    noise = torch.randn_like(x0_c)
                    xt = ((1.0 - sigma) * x0_c + sigma * noise).to(x0_c.dtype)
                    with adapters.adapter_ctx(self.model, "old"):
                        v_old = self._forward(state, chunk_id, xt, timestep, grad=False)
                    with adapters.adapter_ctx(self.model, "base"):
                        v_ref = self._forward(state, chunk_id, xt, timestep, grad=False)
                    v_new = self._forward(state, chunk_id, xt, timestep, grad=True)
                    loss, metrics = compute_nft_loss(
                        v_new=v_new[None].float(),
                        v_old=v_old[None].float(),
                        v_ref=v_ref[None].float(),
                        xt_gen=xt[None].float(),
                        x0_gen=x0_c[None].float(),
                        sigma=sigma.reshape(1),
                        r=r_chunk,
                        beta=self.beta,
                        beta_kl=self.beta_kl,
                    )
                    assert self._committed_end(state) == committed, (
                        "KV cache advanced between a graded forward and its "
                        "backward — commits must come after all microsteps "
                        "of a chunk")
                    (loss * scale).backward()
                    metrics = {"loss": float(loss.detach()), **{
                        k: float(v) for k, v in metrics.items()}}
                    for k, val in metrics.items():
                        agg[k] = agg.get(k, 0.0) + val
                    n_ok += 1
                except torch.cuda.OutOfMemoryError:
                    n_oom += 1
                    logging.warning(
                        "OOM at rollout chunk %d t-idx %d (step %d); skipping "
                        "this microstep's backward", chunk_id, ti, global_step)
                    torch.cuda.empty_cache()
            self._commit(state, chunk_id, x0_c)

        if n_ok:
            agg = {k: v / n_ok for k, v in agg.items()}
        return agg, n_ok, n_oom
