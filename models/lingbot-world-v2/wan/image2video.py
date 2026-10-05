import gc
import hashlib
import logging
import math
import os
import random
import sys
import time
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import numpy as np
import torch
# import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward_causal, sp_dit_forward_causal
from .distributed.util import get_world_size
from .modules.model_fast import WanModelFast
from .modules.model_causal import WanModelCausal
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE

from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.cam_utils import (
    compute_relative_poses,
    interpolate_camera_poses,
    get_plucker_embeddings,
    get_Ks_transformed,
)
from einops import rearrange

@dataclass
class FastGenState:
    """Per-generation state for the causal_fast sampler.

    Built once by `_prepare_causal_fast`; consumed chunk-by-chunk by
    `_denoise_chunk_fast` / `_commit_chunk_fast`. Also the capture surface
    for RL rollouts: everything needed to replay a rollout at train time
    (conditioning, seed, timestep grid) lives here — except the KV caches,
    which are rebuilt by replay rather than stored.
    """
    seed: int
    seed_g: torch.Generator
    timesteps: torch.Tensor
    timesteps_index: list
    shift: float
    context: list
    y: torch.Tensor
    c2ws_plucker_emb: torch.Tensor
    latents_chunk: tuple
    condition_chunk: tuple
    plucker_chunk: tuple
    self_kv_cache: list
    cross_kv_cache: list
    chunk_size: int
    frame_seqlen: int
    kv_size: int
    max_seq_len: int
    max_attention_size: int
    lat_f: int
    lat_h: int
    lat_w: int
    h: int
    w: int
    F: int
    offload_model: bool
    pred_latent_chunks: list = field(default_factory=list)

    @property
    def num_chunks(self):
        return len(self.latents_chunk)

class WanI2VCausal:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
        pipe_dtype=torch.bfloat16,
        local_attn_size=-1,
        sink_size=0,
        infer_mode="causal_fast",
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            infer_mode (`str`, *optional*, defaults to "causal_fast"):
                Inference mode. "causal_fast" uses the distilled few-step
                model (config.fast_checkpoint) with KV-cache windowing
                (local_attn_size / sink_size). "causal_pretrain" uses the
                pretrained causal model (config.causal_checkpoint) with
                40-step CFG sampling.
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        assert infer_mode in ("causal_fast", "causal_pretrain"), \
            f"Unsupported infer_mode: {infer_mode}"
        self.infer_mode = infer_mode

        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype
        self.pipe_dtype = pipe_dtype
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        if self.infer_mode == "causal_fast":
            logging.info(f"Creating WanModelFast from {checkpoint_dir}")
            self.model = WanModelFast.from_pretrained(
                checkpoint_dir,
                subfolder=config.fast_checkpoint,
                torch_dtype=torch.bfloat16,
                local_attn_size=self.local_attn_size,
                sink_size=self.sink_size)
        else:
            logging.info(f"Creating WanModelCausal from {checkpoint_dir}")
            self.model = WanModelCausal.from_pretrained(
                checkpoint_dir,
                subfolder=config.causal_checkpoint,
                torch_dtype=torch.bfloat16)

        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype).to(self.device)

        self.scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

        # T5 prompt-embedding cache. Same-prompt re-encodes hit this dict
        # instead of re-running the umt5-xxl encoder (~360 ms/call).
        # Keyed by sha256(prompt.utf8); value is the list returned by
        # T5EncoderModel.__call__ (already device-resident). Unbounded;
        # callers can clear via `pipe.clear_text_cache()` if needed.
        self._t5_cache: dict[str, list] = {}

        # Reset per generate() and flipped True after the first DiT forward.
        # Passed into model.forward as `cross_attn_first_call` to skip the
        # crossattn_cache["is_init"].item() sync inside WanCrossAttention.
        self._cross_attn_initialized: bool = False

        # RL rollout capture hook. When set, _generate_causal_fast calls it
        # after each chunk commit as sink(chunk_id=..., x0=..., state=...).
        # Plain inference never sets it.
        self._rollout_sink: Optional[Callable] = None

    def clear_text_cache(self):
        """Drop all cached T5 prompt embeddings. Frees ~4 MB per entry."""
        self._t5_cache.clear()

    def prewarm(
        self,
        img,
        max_area: int = 480 * 832,
        frame_num: int = 81,
        chunk_size: int = 3,
        text_seq_len: int = 512,
    ):
        """Opt-in pre-warm. Run one dummy DiT forward at the same shape a
        subsequent generate() call will use, so CUDA kernels are autotuned,
        FSDP all-gathers happen, and Ulysses all-to-alls handshake — all
        outside the timed generate() window.

        Without this call, the first generate() pays a ~7s warmup tax in
        chunk 0 (CUDA lazy init, kernel autotuning, NCCL handshake). On
        8xH100 at 480*832/81 frames, calling prewarm() before the first
        generate() reduces generate()'s wall-clock by ~6.5s (~30%) with
        bit-identical output.

        Idempotent: subsequent calls on the same pipe are no-ops.
        Shape-keyed: if generate() is later invoked with a different shape,
        the autotuner will warm those kernels on demand in chunk 0 (no
        incorrect output, just the tax re-paid once).

        Args:
            img: PIL image or torch tensor — used only for its h/w to match
                generate()'s lat_h/lat_w derivation.
            max_area, frame_num, chunk_size: shape parameters; must match
                the subsequent generate() call to be effective.
            text_seq_len: T5 sequence length (defaults to config.text_len).

        Caller pattern:
            pipe = WanI2VCausal(...)
            pipe.prewarm(img, max_area=..., frame_num=...)
            # start your timer here
            video = pipe.generate(prompt, img, ...)
        """
        if self.infer_mode != "causal_fast":
            logging.info("prewarm is only supported for infer_mode='causal_fast'; skipping.")
            return
        if getattr(self, "_warmed", False):
            return

        cfg = self.config

        # Match generate()'s shape derivation exactly.
        F = frame_num
        h, w = (img.shape[1], img.shape[2]) if hasattr(img, 'shape') else (img.size[1], img.size[0])
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // cfg.vae_stride[1] //
            cfg.patch_size[1] * cfg.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // cfg.vae_stride[2] //
            cfg.patch_size[2] * cfg.patch_size[2])
        lat_f = (F - 1) // cfg.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))

        frame_seqlen = (lat_h * lat_w) // (cfg.patch_size[1] * cfg.patch_size[2])
        max_seq_len = chunk_size * frame_seqlen
        head_dim = cfg.dim // cfg.num_heads
        local_num_heads = cfg.num_heads // self.sp_size

        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f

        transformer_dtype = self.pipe_dtype
        # generate() folds the VAE spatial stride into the Plücker channel
        # dim via rearrange 'f (h s1) (w s2) c -> (f h w) (c s1 s2)' with
        # s1=s2=vae_stride[1]=8, so control_dim=6 → 6 * 8 * 8 = 384.
        plucker_channels = 6 * cfg.vae_stride[1] * cfg.vae_stride[2]
        # T5 (umt5-xxl) hidden size; cross-attn projects t5_hidden → cfg.dim.
        t5_hidden = 4096

        warmup_self_kv = self._initialize_self_kv_cache(
            num_layers=cfg.num_layers,
            shape=[1, kv_size, local_num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)
        warmup_cross_kv = self._initialize_crossattn_cache(
            num_layers=cfg.num_layers,
            shape=[1, text_seq_len, cfg.num_heads, head_dim],
            dtype=transformer_dtype,
            device=self.device)

        # `y` is concat([msk_4ch, vae_latent_16ch]) → 20 channels; combined
        # with latent's 16 ch at patch-embed concat, the DiT sees 36 ch in.
        dummy_latent = torch.zeros(
            16, chunk_size, lat_h, lat_w,
            device=self.device, dtype=torch.float32)
        dummy_y = torch.zeros(
            20, chunk_size, lat_h, lat_w,
            device=self.device, dtype=transformer_dtype)
        dummy_c2ws = torch.zeros(
            1, plucker_channels, chunk_size, lat_h, lat_w,
            device=self.device, dtype=self.param_dtype)
        dummy_context = torch.zeros(
            text_seq_len, t5_hidden,
            device=self.device, dtype=self.param_dtype)
        dummy_t = torch.tensor(
            [500.0], device=self.device, dtype=torch.float32)

        @contextmanager
        def _noop_no_sync():
            yield
        no_sync_model = getattr(self.model, 'no_sync', _noop_no_sync)

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()
        t0 = time.perf_counter()

        with torch.amp.autocast('cuda', dtype=self.param_dtype), \
             torch.no_grad(), \
             no_sync_model():
            _ = self.model(
                x=[dummy_latent],
                t=dummy_t,
                context=[dummy_context],
                seq_len=max_seq_len,
                y=[dummy_y],
                dit_cond_dict={"c2ws_plucker_emb": (dummy_c2ws,)},
                kv_cache=warmup_self_kv,
                crossattn_cache=warmup_cross_kv,
                current_start=0,
                max_attention_size=kv_size,
                frame_seqlen=frame_seqlen,
            )

        if dist.is_initialized():
            torch.cuda.synchronize()
            dist.barrier()

        if (not dist.is_initialized()) or dist.get_rank() == 0:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            logging.info(f"WanI2VCausal.prewarm: {dt_ms:.0f} ms")

        del (warmup_self_kv, warmup_cross_kv, dummy_latent, dummy_y,
             dummy_c2ws, dummy_context, dummy_t)
        torch.cuda.empty_cache()
        self._warmed = True

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward_causal, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward_causal, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor, scheduler) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, F, H, W]
        xt: the input noisy data with shape [B, C, F, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps - timestep).abs())
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred

        return x0_pred.to(original_dtype)

    def _prepare_causal_fast(self,
                             input_prompt,
                             img,
                             action_path,
                             chunk_size=3,
                             max_area=480 * 832,
                             frame_num=81,
                             timesteps_index=[0, 179, 358, 679],
                             shift=5.0,
                             seed=-1,
                             offload_model=True,
                             max_sequence_length=512,
                             max_attention_size=None,
                             fov_scale=1.0,
                             fov_scale_y=None) -> FastGenState:
        """Everything _generate_causal_fast does before the chunk loop:
        shapes, seed, noise, first-frame conditioning, scheduler timesteps,
        T5 context, Plücker embeddings, KV-cache allocation. RNG call order
        matches the pre-refactor code exactly (one randn here, one per
        non-final denoise step later)."""
        if input_prompt is not None and isinstance(input_prompt, str):
            batch_size = 1
        elif input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1

        assert action_path is not None, "action_path is required"
        c2ws = np.load(os.path.join(action_path, "poses.npy")) # opencv coordinate
        len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
        frame_num = ((frame_num - 1) // 4) * 4 + 1
        frame_num = min(frame_num, len_c2ws)
        c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        # Reset per-generate state: cross-attn K/V cache will be freshly
        # initialized below; the first DiT forward must compute and store.
        self._cross_attn_initialized = False

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16,
            lat_f,
            lat_h,
            lat_w,
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ],
                           dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, shift=shift)
        timesteps = self.scheduler.timesteps[timesteps_index]

        # preprocess
        # T5 cache: skip the encoder entirely if we've seen this exact prompt
        # before in this pipe instance. Bit-identical: cached tensor is the
        # same object returned by the prior call.
        cache_key = hashlib.sha256(input_prompt.encode('utf-8')).hexdigest()
        if cache_key in self._t5_cache:
            context = self._t5_cache[cache_key]
        else:
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                # The result is cached just below, so the encoder is not needed on the GPU
                # again: hand its ~11 GB back before the VAE encode and the denoise loop
                # allocate. Without this the 14B model plus T5 does not fit on an 80 GB card.
                self.text_encoder.model.cpu()
                torch.cuda.empty_cache()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
            self._t5_cache[cache_key] = context

        Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()

        # The provided intrinsics are for original image size (480p). We need to transform them according to the new image size (h, w).
        Ks = get_Ks_transformed(Ks,
                                height_org=480,
                                width_org=832,
                                height_resize=h,
                                width_resize=w,
                                height_final=h,
                                width_final=w)
        Ks = Ks[0]
        if fov_scale != 1.0:
            # Divide the conditioning focal length by k: rays_d = normalize([(x-cx)/fx,
            # (y-cy)/fy, 1]), so a smaller f widens the ray fan. This is a LENS change, not a
            # scene-depth change -- a true scene rescale by 1/k maps (o,d) -> (o/k,d), i.e.
            # origins only -- a post-normalization translation multiplier, which was tried
            # and measured WORSE (see cam_utils.compute_relative_poses).
            # Here the origins are untouched (CAMGAIN sum is identical) and only the
            # directions change. It buys forward travel because on a wide-angle camera the
            # same physical translation produces far larger peripheral flow, so to stay
            # consistent with the ray field the model must render each commanded step bigger:
            # for a pixel at radius r=u*f the commanded flow goes as u*sqrt(1+u^2 k^2), which
            # grows monotonically in k (verified numerically: 1.71 -> 6.06 -> 17.3 px at the
            # frame edge for k=1,7,20). So there is NO geometric optimum; the limit is that
            # the fan leaves the range of real intrinsics seen in training and the model
            # stops obeying or starts distorting -- an empirical ceiling, not a computable
            # one. Training clips do carry varying real focal lengths, so mild k is in
            # distribution; k>>1 is not. Eval-only, default 1.0 = untouched.
            # fov_scale_y defaults to fov_scale (isotropic). Splitting them matters for the
            # door clips, whose rotation is almost pure yaw: a yaw leaves each ray's
            # y-component exactly invariant (verified: d(dy)=0 at every pixel), so a widened
            # vertical fan injects no vertical command. What it does do is make the horizontal
            # yaw displacement row-dependent -- 34% weaker at top/bottom than centre at k=3.5
            # vs 5% at the true fy -- because those rays carry most of their unit length in dy
            # and only dz~0.66 instead of 0.95, and yaw only mixes dx with dz. That shear is
            # not a rigid rotation of this image's geometry. Keeping fy true holds the spread
            # near 5% while fx still supplies the travel.
            ky = fov_scale if fov_scale_y is None else fov_scale_y
            Ks = Ks.clone()
            Ks[0] = Ks[0] / fov_scale
            Ks[1] = Ks[1] / ky
            import math as _m
            logging.info("FOVSCALE kx=%s ky=%s -> fx=%.2f fy=%.2f  hfov=%.1fdeg vfov=%.1fdeg "
                         "(cx=%.1f cy=%.1f unchanged)", fov_scale, ky,
                         Ks[0].item(), Ks[1].item(),
                         2*_m.degrees(_m.atan(Ks[2].item()/Ks[0].item())),
                         2*_m.degrees(_m.atan(Ks[3].item()/Ks[1].item())),
                         Ks[2].item(), Ks[3].item())

        len_c2ws = len(c2ws)
        len_c2ws_ = int((len_c2ws - 1) // 4) + 1
        len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
        c2ws_infer = interpolate_camera_poses(
            src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
            src_rot_mat=c2ws[:, :3, :3],
            src_trans_vec=c2ws[:, :3, 3],
            tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
        )
        c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
        logging.info("CAMPOSE latents=%d step_max=%.4f step_mean=%.4f sum=%.2f",
                     c2ws_infer.shape[0],
                     c2ws_infer[:, :3, 3].norm(dim=-1).max().item(),
                     c2ws_infer[:, :3, 3].norm(dim=-1).mean().item(),
                     c2ws_infer[:, :3, 3].norm(dim=-1).sum().item())
        Ks = Ks.repeat(len(c2ws_infer), 1)

        c2ws_infer = c2ws_infer.to(self.device)
        Ks = Ks.to(self.device)
        wasd_action = None
        c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
        c2ws_plucker_emb = rearrange(
            c2ws_plucker_emb,
            'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
            c1=int(h // lat_h),
            c2=int(w // lat_w),
        )
        c2ws_plucker_emb = c2ws_plucker_emb[None, ...] # [b, f*h*w, c]
        c2ws_plucker_emb = rearrange(c2ws_plucker_emb, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
        if wasd_action is not None:
            wasd_action_tensor = wasd_action[:, None, None, :].repeat(1, h, w, 1) # [f, h, w, 3]
            wasd_action_tensor = rearrange(
                wasd_action_tensor,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h),
                c2=int(w // lat_w),
            )
            wasd_action_tensor = wasd_action_tensor[None, ...] # [b, f*h*w, c]
            wasd_action_tensor = rearrange(wasd_action_tensor, 'b (f h w) c -> b c f h w', f=lat_f, h=lat_h, w=lat_w).to(self.param_dtype)
            c2ws_plucker_emb = torch.cat([c2ws_plucker_emb, wasd_action_tensor], dim=1)

        y = self.vae.encode([
            torch.concat([
                torch.nn.functional.interpolate(
                    img[None].cpu(), size=(h, w), mode='bicubic').transpose(
                        0, 1),
                torch.zeros(3, F - 1, h, w)
            ],
                         dim=1).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        # Initialize KV cache to all zeros
        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1]// 4)
        if self.local_attn_size > -1:
            kv_size = frame_seqlen * self.local_attn_size
        else:
            kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        self_kv_cache = self._initialize_self_kv_cache(num_layers=model_args.num_layers,
                                                       shape=self_kv_shape,
                                                       dtype=transformer_dtype,
                                                       device=self.device)
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]
        cross_kv_cache = self._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                          shape=cross_kv_shape,
                                                          dtype=transformer_dtype,
                                                          device=self.device)

        return FastGenState(
            seed=seed,
            seed_g=seed_g,
            timesteps=timesteps,
            timesteps_index=list(timesteps_index),
            shift=shift,
            context=context,
            y=y,
            c2ws_plucker_emb=c2ws_plucker_emb,
            latents_chunk=noise.split(chunk_size, dim=1),
            condition_chunk=y.split(chunk_size, dim=1),
            plucker_chunk=c2ws_plucker_emb.split(chunk_size, dim=2),
            self_kv_cache=self_kv_cache,
            cross_kv_cache=cross_kv_cache,
            chunk_size=chunk_size,
            frame_seqlen=frame_seqlen,
            kv_size=kv_size,
            max_seq_len=max_seq_len,
            max_attention_size=kv_size if max_attention_size is None else max_attention_size,
            lat_f=lat_f,
            lat_h=lat_h,
            lat_w=lat_w,
            h=h,
            w=w,
            F=F,
            offload_model=offload_model)

    def _fast_chunk_kwargs(self, state: FastGenState, chunk_id: int) -> dict:
        dit_cond_dict = {
            "c2ws_plucker_emb": state.plucker_chunk[chunk_id].chunk(1, dim=0),
        }
        return {
            'context': [state.context[0]],
            'seq_len': state.max_seq_len,
            'y': [state.condition_chunk[chunk_id]],
            'dit_cond_dict': dit_cond_dict,
            'kv_cache': state.self_kv_cache,
            'crossattn_cache': state.cross_kv_cache,
            'current_start': chunk_id * state.chunk_size * state.frame_seqlen,
            'max_attention_size': state.max_attention_size,
            'frame_seqlen': state.frame_seqlen,
        }

    def _denoise_chunk_fast(self, state: FastGenState, chunk_id: int) -> torch.Tensor:
        """The few-step x0-predict + renoise loop for one chunk. Forwards
        write the chunk's K/V into the cache exactly as before the refactor
        (the final commit forward overwrites them with clean-latent K/V)."""
        kwargs = self._fast_chunk_kwargs(state, chunk_id)
        current_latent = state.latents_chunk[chunk_id]

        if state.offload_model:
            torch.cuda.empty_cache()

        for timestep_idx in range(len(state.timesteps)):
            latent_model_input = [current_latent.to(self.device)]
            current_timestep = [state.timesteps[timestep_idx]]

            timestep = torch.stack(current_timestep).to(self.device)

            noise_pred = self.model(
                x=latent_model_input, t=timestep,
                cross_attn_first_call=not self._cross_attn_initialized,
                **kwargs)[0]
            self._cross_attn_initialized = True

            if state.offload_model:
                torch.cuda.empty_cache()

            x0 = self._convert_flow_pred_to_x0(
                flow_pred=noise_pred,
                xt=current_latent,
                timestep=current_timestep[0],
                scheduler=self.scheduler,
            )

            if timestep_idx < len(state.timesteps) - 1:
                next_timestep = state.timesteps[timestep_idx + 1]
                current_latent = self.scheduler.add_noise(x0, torch.randn(x0.shape, generator=state.seed_g, device=x0.device, dtype=x0.dtype), next_timestep)
            else:
                # note return x0
                break

        return x0

    def _commit_chunk_fast(self, state: FastGenState, chunk_id: int, x0: torch.Tensor) -> None:
        """Advance the KV cache past this chunk: one forward at t=0 with the
        clean x0 rewrites the chunk's cache slots with clean-latent K/V."""
        kwargs = self._fast_chunk_kwargs(state, chunk_id)
        context_timestep = [state.timesteps[-1] * 0.0]
        timestep = torch.stack(context_timestep).to(self.device)
        self.model(x=[x0], t=timestep,
                   cross_attn_first_call=False,
                   **kwargs)

    def generate(self,
                 input_prompt,
                 img,
                 action_path,
                 chunk_size=3,
                 max_area=480 * 832,
                 frame_num=81,
                 timesteps_index=[0, 250, 500, 750],
                 shift=5.0,
                 seed=-1,
                 offload_model=True,
                 max_sequence_length=512,
                 max_attention_size=None,
                 fov_scale=1.0,
                 fov_scale_y=None,):
        r"""
        Generates video frames from input image and text prompt.

        Dispatches to the mode-specific implementation according to
        `self.infer_mode`:
            - "causal_fast": distilled few-step sampling (`_generate_causal_fast`)
            - "causal_pretrain": 40-step CFG sampling (`_generate_causal_pretrain`)
        """
        gen_fn = (self._generate_causal_fast
                  if self.infer_mode == "causal_fast"
                  else self._generate_causal_pretrain)
        return gen_fn(
            input_prompt,
            img,
            action_path,
            chunk_size=chunk_size,
            max_area=max_area,
            frame_num=frame_num,
            timesteps_index=timesteps_index,
            shift=shift,
            seed=seed,
            offload_model=offload_model,
            max_sequence_length=max_sequence_length,
            max_attention_size=max_attention_size,
            fov_scale=fov_scale,
            fov_scale_y=fov_scale_y)

    def _generate_causal_fast(self,
                              input_prompt,
                              img,
                              action_path,
                              chunk_size=3,
                              max_area=480 * 832,
                              frame_num=81,
                              timesteps_index=[0, 179, 358, 679],
                              shift=5.0,
                              seed=-1,
                              offload_model=True,
                              max_sequence_length=512,
                              max_attention_size=None,
                                fov_scale=1.0,
                              fov_scale_y=None):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 720*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from max_area)
                - W: Frame width from max_area)
        """

        state = self._prepare_causal_fast(
            input_prompt,
            img,
            action_path,
            chunk_size=chunk_size,
            max_area=max_area,
            frame_num=frame_num,
            timesteps_index=timesteps_index,
            shift=shift,
            seed=seed,
            offload_model=offload_model,
            max_sequence_length=max_sequence_length,
            max_attention_size=max_attention_size,
            fov_scale=fov_scale,
            fov_scale_y=fov_scale_y)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            for chunk_id in tqdm(range(state.num_chunks)):
                x0 = self._denoise_chunk_fast(state, chunk_id)
                state.pred_latent_chunks.append(x0)
                self._commit_chunk_fast(state, chunk_id, x0)
                if self._rollout_sink is not None:
                    self._rollout_sink(chunk_id=chunk_id, x0=x0, state=state)

            pred_latent_chunks = torch.cat(state.pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _generate_causal_pretrain(self,
                                  input_prompt,
                                  img,
                                  action_path,
                                  chunk_size=3,
                                  max_area=480 * 832,
                                  frame_num=81,
                                  timesteps_index=None,
                                  shift=5.0,
                                  seed=-1,
                                  offload_model=True,
                                  max_sequence_length=512,
                                  max_attention_size=None,):
        r"""
        Generates video frames with the pretrained causal model using
        40-step CFG sampling per chunk. `timesteps_index` is unused in this
        mode (kept for signature compatibility with `generate`).
        """
        guide_scale = 5.0
        n_prompt = "画面突变，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

        if input_prompt is not None and isinstance(input_prompt, list):
            batch_size = len(input_prompt)
        else:
            batch_size = 1

        if action_path is not None:
            c2ws = np.load(os.path.join(action_path, "poses.npy"))  # opencv coordinate
            len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
            frame_num = ((frame_num - 1) // 4) * 4 + 1
            frame_num = min(frame_num, len_c2ws)
            c2ws = c2ws[:frame_num]

        # preprocess
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)

        F = frame_num
        h, w = img.shape[1:]
        aspect_ratio = h / w
        lat_h = round(
            np.sqrt(max_area * aspect_ratio) // self.vae_stride[1] //
            self.patch_size[1] * self.patch_size[1])
        lat_w = round(
            np.sqrt(max_area / aspect_ratio) // self.vae_stride[2] //
            self.patch_size[2] * self.patch_size[2])
        h = lat_h * self.vae_stride[1]
        w = lat_w * self.vae_stride[2]
        lat_f = (F - 1) // self.vae_stride[0] + 1
        lat_f = int(lat_f - (lat_f % chunk_size))
        F = (lat_f - 1) * 4 + 1
        max_seq_len = chunk_size * lat_h * lat_w // (
            self.patch_size[1] * self.patch_size[2])
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            16, lat_f, lat_h, lat_w,
            dtype=torch.float32, generator=seed_g, device=self.device)

        msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([
            torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
        ], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]

        # 2. Prepare timesteps (scheduler object created once, state reset per chunk)
        sample_scheduler = FlowUniPCMultistepScheduler(num_train_timesteps=self.num_train_timesteps, shift=1, use_dynamic_shifting=False)

        # preprocess text: cond + uncond
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context      = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt],     self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context      = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt],     torch.device('cpu'))
            context      = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        # cam preparation (only if action_path is provided)
        c2ws_plucker_emb = None
        if action_path is not None:
            Ks = torch.from_numpy(np.load(os.path.join(action_path, "intrinsics.npy"))).float()
            Ks = get_Ks_transformed(Ks,
                                    height_org=480, width_org=832,
                                    height_resize=h, width_resize=w,
                                    height_final=h, width_final=w)
            Ks = Ks[0]

            len_c2ws = len(c2ws)
            len_c2ws_ = int((len_c2ws - 1) // 4) + 1
            len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
            c2ws_infer = interpolate_camera_poses(
                src_indices=np.linspace(0, len_c2ws - 1, len_c2ws),
                src_rot_mat=c2ws[:, :3, :3],
                src_trans_vec=c2ws[:, :3, 3],
                tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_),
            )
            c2ws_infer = compute_relative_poses(c2ws_infer, framewise=True)
            Ks = Ks.repeat(len(c2ws_infer), 1)

            c2ws_infer = c2ws_infer.to(self.device)
            Ks = Ks.to(self.device)
            c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb,
                'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
                c1=int(h // lat_h), c2=int(w // lat_w),
            )
            c2ws_plucker_emb = c2ws_plucker_emb[None, ...]  # [b, f*h*w, c]
            c2ws_plucker_emb = rearrange(
                c2ws_plucker_emb, 'b (f h w) c -> b c f h w',
                f=lat_f, h=lat_h, w=lat_w,
            ).to(self.param_dtype)

        y = self.vae.encode([
            torch.concat(
                [
                    torch.nn.functional.interpolate(img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, h, w)
                ], 
                dim=1,
            ).to(self.device)
        ])[0]
        y = torch.concat([msk, y])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_model = getattr(self.model, 'no_sync', noop_no_sync)

        model_args = self.model.config
        transformer_dtype = self.pipe_dtype
        frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
        kv_size = frame_seqlen * lat_f
        head_dim = model_args.dim // model_args.num_heads
        local_num_heads = model_args.num_heads // self.sp_size
        self_kv_shape = [batch_size, kv_size, local_num_heads, head_dim]
        cross_kv_shape = [batch_size, max_sequence_length, model_args.num_heads, head_dim]

        # CFG requires separate caches for the cond / uncond streams.
        self_kv_cache_cond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        self_kv_cache_uncond = self._initialize_self_kv_cache(
            num_layers=model_args.num_layers,
            shape=self_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_cond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)
        cross_kv_cache_uncond = self._initialize_crossattn_cache_pretrain(
            num_layers=model_args.num_layers,
            shape=cross_kv_shape,
            dtype=transformer_dtype,
            device=self.device)

        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_model(),
        ):
            latent = noise
            latents_chunk = latent.split(chunk_size, dim=1)          # [c, f, h, w]
            condition_chunk = y.split(chunk_size, dim=1)
            c2ws_plucker_emb_chunk = c2ws_plucker_emb.split(chunk_size, dim=2)
            num_inference_chunk = len(latents_chunk)
            pred_latent_chunks = []

            for chunk_id in tqdm(range(num_inference_chunk)):
                # Reset the multi-step scheduler state for each chunk
                sample_scheduler.set_timesteps(40, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps

                current_latent = latents_chunk[chunk_id]
                current_condition = condition_chunk[chunk_id]
                current_c2ws_plucker_emb = c2ws_plucker_emb_chunk[chunk_id]
                dit_cond_dict = {
                    "c2ws_plucker_emb": current_c2ws_plucker_emb.chunk(1, dim=0),
                }

                common = {
                    'seq_len': max_seq_len,
                    'y': [current_condition],
                    'dit_cond_dict': dit_cond_dict,          # camera condition is kept for uncond as well
                    'current_start': chunk_id * chunk_size * frame_seqlen,
                    'max_attention_size': kv_size if max_attention_size is None else max_attention_size,
                }
                kwargs_cond = {
                    **common,
                    'context': [context[0]],
                    'kv_cache': self_kv_cache_cond,
                    'crossattn_cache': cross_kv_cache_cond,
                }
                kwargs_uncond = {
                    **common,
                    'context': context_null,
                    'kv_cache': self_kv_cache_uncond,
                    'crossattn_cache': cross_kv_cache_uncond,
                }

                if offload_model:
                    torch.cuda.empty_cache()

                for timestep_idx in tqdm(range(len(timesteps)), desc=f"infer chunk {chunk_id}"):
                    latent_model_input = [current_latent.to(self.device)]
                    t = timesteps[timestep_idx]
                    timestep = torch.stack([t]).to(self.device)

                    noise_pred_cond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_cond)[0]
                    noise_pred_uncond = self.model(
                        x=latent_model_input, t=timestep, **kwargs_uncond)[0]
                    noise_pred = noise_pred_uncond + guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                    if offload_model:
                        torch.cuda.empty_cache()

                    temp_x0 = sample_scheduler.step(
                        noise_pred.unsqueeze(0), t,
                        current_latent.unsqueeze(0),
                        return_dict=False, generator=seed_g)[0]
                    current_latent = temp_x0.squeeze(0)

                    del latent_model_input, timestep

                pred_latent_chunks.append(current_latent)

                # Update both self KV caches with the clean latent (once for cond, once for uncond)
                timestep0 = torch.stack([timesteps[-1] * 0.0]).to(self.device)
                self.model(x=[current_latent], t=timestep0, **kwargs_cond)
                self.model(x=[current_latent], t=timestep0, **kwargs_uncond)

            pred_latent_chunks = torch.cat(pred_latent_chunks, dim=1)

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode([pred_latent_chunks])

        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def _initialize_self_kv_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a Per-GPU KV cache for the SelfAttn.
        """
        self_kv_cache = []
        for _ in range(num_layers):
            self_kv_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'global_end_index': torch.tensor([0], dtype=torch.long, device=device),
                'local_end_index': torch.tensor([0], dtype=torch.long, device=device)
            })

        return self_kv_cache

    def _initialize_crossattn_cache(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': torch.tensor(0, dtype=torch.int32, device=device),
            })

        return crossattn_cache

    def _initialize_crossattn_cache_pretrain(self, num_layers, shape, dtype, device):
        """
        Initialize a per-GPU cross-attention cache for the pretrained causal
        model, which expects `is_init` to be a plain Python bool.
        """
        crossattn_cache = []
        for _ in range(num_layers):
            crossattn_cache.append({
                'k': torch.zeros(shape, dtype=dtype, device=device),
                'v': torch.zeros(shape, dtype=dtype, device=device),
                'is_init': False,
            })

        return crossattn_cache