# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DiffusionNFT RL fine-tuning model for the Lyra2 DMD policy.

We RL-tune the 14B DMD LoRA itself. Three LoRA adapters live over the same frozen
base net:
  * ``new`` -- the trainable policy (the only adapter with gradients).
  * ``old`` -- the EMA sampling policy (frozen during a training epoch; updated
    ``old <- decay*old + (1-decay)*new`` between NFT epochs).
  * ``ref`` -- a frozen copy of the original DMD LoRA, used for the KL term.

All three are initialized from the same DMD checkpoint
(``config.lora_config.pretrained_lora_path``). Training re-noises stored clean
samples at the DMD timesteps and applies the forward-process NFT loss on the
generated region (see :mod:`lyra_2._src.rl.loop.nft_loss`).
"""

from __future__ import annotations

import contextlib
import os
from typing import Optional

import attrs
import torch
from einops import rearrange

from lyra_2._ext.imaginaire.utils import log
from lyra_2._src.models.lyra2_model import Lyra2Model, Lyra2T2VConfig, WAN2PT1_I2V_COND_LATENT_KEY
from lyra_2._src.modules.conditioner import DataType
from lyra_2._src.rl.loop.nft_loss import compute_nft_loss
from lyra_2._src.utils.context_parallel import broadcast

try:
    from peft.tuners.tuners_utils import BaseTunerLayer
except Exception:  # pragma: no cover - peft always present in the lyra2 env
    BaseTunerLayer = None

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover
    DTensor = None

def _local(t: torch.Tensor) -> torch.Tensor:
    """Return the local shard for a DTensor, else the tensor itself."""
    if DTensor is not None and isinstance(t, DTensor):
        return t.to_local()
    return t

def _mem(stage: str, iteration: int) -> None:
    """Stage-by-stage GPU memory probe (rank 0, first few iters) to localize the
    training peak: where the ~46 GiB of non-weight memory accumulates."""
    if iteration >= 3:
        return
    import torch.distributed as _d
    if _d.is_initialized() and _d.get_rank() != 0:
        return
    print(f"[nft-mem] iter{iteration} {stage}: "
          f"alloc={torch.cuda.memory_allocated() / 1e9:.2f}G "
          f"reserved={torch.cuda.memory_reserved() / 1e9:.2f}G "
          f"peak={torch.cuda.max_memory_allocated() / 1e9:.2f}G", flush=True)

def _trace_nonfinite(stage: str, t, *, iteration, tag: str = "") -> bool:
    """Log (stdout) the first non-finite tensor in the NFT loss path, with stats, to
    root-cause nan/inf. Returns True if non-finite. Cheap vs the 14B forward."""
    if t is None:
        return False
    tl = _local(t) if hasattr(t, "shape") else t
    if not torch.is_tensor(tl) or tl.numel() == 0:
        return False
    fin = torch.isfinite(tl)
    if bool(fin.all()):
        return False
    n_nan = int(torch.isnan(tl).sum())
    n_inf = int(torch.isinf(tl).sum())
    good = tl[fin]
    rng = (float(good.min()), float(good.max())) if good.numel() else ("NA", "NA")
    print(f"[nft-nan]{tag} iter {iteration} NON-FINITE @ {stage}: shape={tuple(tl.shape)} "
          f"dtype={tl.dtype} nan={n_nan} inf={n_inf} finite_min/max={rng}", flush=True)
    return True

@attrs.define(slots=False)
class Lyra2NFTConfig(Lyra2T2VConfig):
    """Lyra2 config extended with DiffusionNFT hyperparameters."""

    nft_enabled: bool = True
    # Mixing coefficient between new/old in the positive/negative predictions.
    nft_beta: float = 1.0
    # Weight of the KL-to-reference regularizer.
    nft_beta_kl: float = 1e-4
    # Advantage clipping used when mapping advantage -> r in [0, 1].
    nft_adv_clip_max: float = 5.0
    # EMA decay for the old (sampling) policy update between epochs.
    nft_ema_decay: float = 0.5
    # Restrict re-noising to the DMD denoising timesteps (keeps the few-step
    # distilled structure intact). When False, falls back to continuous time.
    nft_train_on_dmd_timesteps: bool = True
    # DMD denoising timesteps (in the 0..num_train_timesteps space).
    nft_dmd_denoising_steps: tuple = (1000, 750, 500, 250)
    # Per-adapter init: new/old initialize from the current policy, ref from the
    # frozen original DMD LoRA (the KL anchor). Empty -> fall back to
    # lora_config.pretrained_lora_path (correct for epoch 0, where all coincide).
    nft_policy_lora_path: str = ""   # -> new, old  (the promoted policy each epoch)
    nft_ref_lora_path: str = ""      # -> ref        (pinned to original DMD)

class Lyra2NFTModel(Lyra2Model):
    """Lyra2 model with three RL LoRA adapters and the NFT training step."""

    RL_ADAPTER_NAMES = ("ref", "old", "new")
    TRAINABLE_ADAPTER = "new"

    def __init__(self, config: Lyra2NFTConfig):
        super().__init__(config)
        self._dmd_train_scheduler = None
        if getattr(self.config, "nft_enabled", False):
            if os.environ.get("NFT_DEFER_INJECTION") == "1":
                # Non-FSDP path: defer adapter injection until after the base 14B
                # checkpoint loads (done in build_nft_model). Injecting here wraps the
                # target modules (e.g. head.head -> head.head.base_layer) before the
                # base load, so the checkpoint's keys (head.head.weight) don't map and
                # the base is left at init (head.head norm = 0) -> the net outputs ~0
                # velocity -> noise rollouts + v~0 in training. Deferring reuses the
                # normal load order (base first, LoRA after), like build_inference_model.
                log.info("Lyra2NFTModel: deferring RL adapter injection until after base load",
                         rank0_only=True)
            else:
                # FSDP path: adapters must be injected before fully_shard, so inject now.
                self._finalize_rl_adapters()

    def _finalize_rl_adapters(self) -> None:
        """Inject the 3 RL adapters on clean modules, enable selective-activation
        checkpointing, freeze all but ``new``, and activate ``new``. Order matters:
        checkpointing renames block submodules, which would break LoRA target matching
        if done first. Call this after the base checkpoint is loaded (non-FSDP) or
        before ``fully_shard`` (FSDP)."""
        self.setup_rl_adapters()
        if hasattr(self.net, "enable_selective_checkpoint"):
            self.net.enable_selective_checkpoint(self.net.sac_config, self.net.blocks)
        self.enforce_rl_grad()
        self.activate_rl_adapter(self.TRAINABLE_ADAPTER)
        log.info(
            f"Lyra2NFTModel ready: adapters={self.RL_ADAPTER_NAMES}, "
            f"trainable='{self.TRAINABLE_ADAPTER}', beta={self.config.nft_beta}, "
            f"beta_kl={self.config.nft_beta_kl}",
            rank0_only=True,
        )

    # ------------------------------------------------------------------ #
    # Adapter setup (post-construction, mirrors Lyra2 inference LoRA loading)
    # ------------------------------------------------------------------ #
    def setup_rl_adapters(self) -> None:
        """Inject + load the new/old/ref adapters from the DMD (or promoted) LoRA.

        Idempotent. new/old initialize from the current policy
        (``nft_policy_lora_path``, default DMD), ref from ``nft_ref_lora_path``
        (default DMD). NOTE: this injects after the net is built; for multi-GPU
        FSDP the adapters must instead be injected before ``fully_shard`` (see
        Lyra2Model.build_net) — single-GPU / DDP training is unaffected.
        """
        if any(isinstance(m, BaseTunerLayer) for m in self.net.modules()):
            return  # already injected
        lc = self.config.lora_config
        assert lc.pretrained_lora_path, "NFT requires lora_config.pretrained_lora_path (DMD LoRA)"
        policy_path = getattr(self.config, "nft_policy_lora_path", "") or lc.pretrained_lora_path
        ref_path = getattr(self.config, "nft_ref_lora_path", "") or lc.pretrained_lora_path
        init_paths = {"new": policy_path, "old": policy_path, "ref": ref_path}
        for name in self.RL_ADAPTER_NAMES:
            self.load_lora_weights(
                lora_path=init_paths[name],
                adapter_name=name,
                training_mode=(name == self.TRAINABLE_ADAPTER),
            )
        # PROBE: confirm the 3 adapters actually coexist + which is active on a layer.
        for m in self.net.modules():
            if BaseTunerLayer is not None and isinstance(m, BaseTunerLayer):
                la = getattr(m, "lora_A", None)
                names = list(la.keys()) if la is not None else "<no lora_A>"
                act = getattr(m, "active_adapters", getattr(m, "active_adapter", None))
                log.info(f"[nft-probe] sample LoRA layer adapters={names} active={act}", rank0_only=True)
                break

    def _iter_tuner_layers(self):
        for module in self.net.modules():
            if BaseTunerLayer is not None and isinstance(module, BaseTunerLayer):
                yield module

    def activate_rl_adapter(self, name: str) -> None:
        """Make ``name`` the single active adapter across all LoRA layers."""
        assert name in self.RL_ADAPTER_NAMES, f"unknown adapter {name}"
        for module in self._iter_tuner_layers():
            if hasattr(module, "set_adapter"):
                module.set_adapter(name)
            else:  # pragma: no cover - very old peft
                module.active_adapter = name

    def enforce_rl_grad(self) -> None:
        """Trainable = only the ``new`` adapter; base + old + ref are frozen.

        ``BaseTunerLayer.set_adapter`` flips ``requires_grad`` as a side effect, so
        this is re-applied after every adapter switch to keep the optimizer's view
        of trainable params stable.
        """
        for pname, p in self.net.named_parameters():
            if "lora_" in pname:
                # e.g. blocks.0.self_attn.q.lora_A.new.weight
                p.requires_grad_(f".{self.TRAINABLE_ADAPTER}." in pname)
            else:
                p.requires_grad_(False)

    @contextlib.contextmanager
    def adapter_ctx(self, name: str):
        """Temporarily activate ``name``; restore the trainable adapter on exit.

        The trainable adapter is re-activated (and grads re-enforced) on exit so
        the autograd graph built for ``new`` stays valid through backward (incl.
        activation-checkpoint recomputation).
        """
        try:
            self.activate_rl_adapter(name)
            yield
        finally:
            self.activate_rl_adapter(self.TRAINABLE_ADAPTER)
            self.enforce_rl_grad()

    @torch.no_grad()
    def ema_update_old(self, decay: Optional[float] = None) -> None:
        """old <- decay*old + (1-decay)*new, over matched LoRA params."""
        decay = self.config.nft_ema_decay if decay is None else decay
        params = dict(self.net.named_parameters())
        old_list, new_list = [], []
        for pname, p in params.items():
            if f".{self.TRAINABLE_ADAPTER}." not in pname:
                continue
            old_name = pname.replace(f".{self.TRAINABLE_ADAPTER}.", ".old.")
            if old_name not in params:
                continue
            new_list.append(_local(p).data)
            old_list.append(_local(params[old_name]).data)
        if not old_list:
            log.warning("ema_update_old: no matched new/old LoRA params found")
            return
        torch._foreach_mul_(old_list, decay)
        torch._foreach_add_(old_list, new_list, alpha=1.0 - decay)
        log.info(f"ema_update_old: updated {len(old_list)} LoRA tensors (decay={decay})", rank0_only=True)

    # ------------------------------------------------------------------ #
    # DMD-timestep re-noising
    # ------------------------------------------------------------------ #
    def _ensure_dmd_train_scheduler(self, device):
        """Build the FlowMatchScheduler + mapped DMD timesteps (mirrors inference_dmd)."""
        if self._dmd_train_scheduler is not None:
            return
        from lyra_2._src.schedulers.self_forcing_scheduler import FlowMatchScheduler

        num_train_timestep = 1000
        sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
        sched.set_timesteps(num_train_timestep, training=True)
        sched.timesteps = sched.timesteps.to(device)
        sched.sigmas = sched.sigmas.to(device)
        steps = torch.LongTensor(list(self.config.nft_dmd_denoising_steps))
        timesteps_aug = torch.cat((sched.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        # Map the integer DMD steps to the scheduler's float timestep grid.
        self._dmd_denoising_timesteps = timesteps_aug[num_train_timestep - steps].to(device)
        self._dmd_train_scheduler = sched

    def num_dmd_train_timesteps(self, device=None) -> int:
        """Number of DMD timesteps in the training set (e.g. 4 for {1000,750,500,250})."""
        self._ensure_dmd_train_scheduler(device or self.tensor_kwargs["device"])
        return int(self._dmd_denoising_timesteps.numel())

    def _sample_dmd_train_timesteps(self, batch_size: int, device, force_idx=None):
        """Return (timesteps[B], sigma[B]) for one DMD timestep per item.

        ``force_idx=None`` (default) samples a random DMD-step index per item -- the
        original behavior. Passing an integer ``force_idx`` pins all items to that DMD
        step, so a training loop can sweep the full DMD set (one forward per timestep)
        instead of one random draw per sample-visit.
        """
        self._ensure_dmd_train_scheduler(device)
        choices = self._dmd_denoising_timesteps  # [num_dmd_steps]
        if force_idx is None:
            idx = torch.randint(0, choices.numel(), (batch_size,), device=device)
        else:
            idx = torch.full((batch_size,), int(force_idx), device=device, dtype=torch.long)
        timesteps = choices[idx]  # [B], float timestep grid values
        sched = self._dmd_train_scheduler
        # sigma lookup by nearest scheduler timestep (same as scheduler.add_noise).
        tstep_id = torch.argmin(
            (sched.timesteps.to(device).unsqueeze(0) - timesteps.unsqueeze(1)).abs(), dim=1
        )
        sigma = sched.sigmas.to(device)[tstep_id]  # [B]
        return timesteps, sigma

    # ------------------------------------------------------------------ #
    # Trainer entrypoint
    # ------------------------------------------------------------------ #
    def training_step(self, data_batch, iteration):
        """Dispatch: NFT loss on rollout batches, else the base flow-matching loss.

        Lets the existing imaginaire trainer (FSDP grad-accum / sync) drive NFT
        unchanged -- the rollout dataloader yields batches with ``x0_latents``.
        """
        if isinstance(data_batch, dict) and "x0_latents" in data_batch:
            return self.training_step_from_rollout(data_batch, iteration)
        return super().training_step(data_batch, iteration)

    # ------------------------------------------------------------------ #
    # Condition replay
    # ------------------------------------------------------------------ #
    def _build_condition_from_rollout(self, data_batch):
        """Rebuild the exact sampling-time T2VCondition from stored tensors.

        Mirrors how ``inference_dmd`` assembles the conditioner input, so the net
        sees byte-identical conditioning at train time (text + I2V cond latents +
        buffer). Uses the positive condition only (NFT trains the conditional
        prediction; no CFG in the loss).
        """
        dev = self.tensor_kwargs["device"]

        def _to_dev(x):
            # Stored cond tensors load on CPU; the conditioner/CLIP weights are on
            # GPU. Move to device, preserving dtype (CLIP casts dtype internally).
            return x.to(dev) if isinstance(x, torch.Tensor) else x

        cond_batch = {
            "t5_text_embeddings": _to_dev(data_batch["t5_text_embeddings"]),
            "neg_t5_text_embeddings": _to_dev(data_batch.get(
                "neg_t5_text_embeddings", data_batch["t5_text_embeddings"]
            )),
            "last_hist_frame": _to_dev(data_batch["last_hist_frame"]),
            "cond_latent_mask": _to_dev(data_batch.get("cond_latent_mask")),
            WAN2PT1_I2V_COND_LATENT_KEY: _to_dev(data_batch["cond_latent"]),
            "cond_latent_buffer": _to_dev(data_batch.get("cond_latent_buffer")),
            # fps + padding_mask are required by the conditioner's ReMapkey embedders.
            "fps": _to_dev(data_batch.get("fps")) if data_batch.get("fps") is not None
                   else torch.tensor([16], device=dev),
            "padding_mask": _to_dev(data_batch.get("padding_mask")),
        }
        condition, _uncond = self.conditioner.get_condition_with_negative_prompt(cond_batch)
        return condition.edit_data_type(DataType.VIDEO)

    # ------------------------------------------------------------------ #
    # NFT training step from a stored rollout sample
    # ------------------------------------------------------------------ #
    def training_step_from_rollout(self, data_batch, iteration, force_dmd_idx=None):
        """One DiffusionNFT optimization step on a stored on-policy sample.

        ``force_dmd_idx`` (int) pins the re-noising timestep to that DMD-step index so a
        caller can sweep the full DMD timestep set; ``None`` keeps the original random
        single-timestep draw. The reward-derived weight ``r`` is timestep-independent
        (same ``r`` at every timestep), so this does not touch advantage normalization.

        Expected ``data_batch`` keys (assembled by :class:`NFTRolloutDataset`):
          * ``x0_latents``: ``[B, C, T_total, H, W]`` clean full window (history
            clean + generated region = the on-policy sample).
          * the stored conditioning tensors (``t5_text_embeddings``,
            ``neg_t5_text_embeddings``, ``last_hist_frame``, ``cond_latent``,
            ``cond_latent_mask``, ``cond_latent_buffer``).
          * ``r``: ``[B]`` reward-derived weight in ``[0, 1]``.
        """
        x0_latents = data_batch["x0_latents"].to(**self.tensor_kwargs)
        tag = ""
        try:
            tag = f" grp={data_batch.get('group_id')}" if isinstance(data_batch, dict) else ""
        except Exception:  # noqa: BLE001
            pass
        _trace_nonfinite("x0_latents(input)", x0_latents, iteration=iteration, tag=tag)
        _mem("entry", iteration)
        condition = self._build_condition_from_rollout(data_batch)
        _mem("after_condition(CLIP)", iteration)
        # r is [B] (one weight per sample) or [B, T_gen] (per-frame). Keep the shape;
        # compute_nft_loss broadcasts a [B] over the frame axis and uses a [B, T] as-is.
        r = data_batch["r"].to(device=self.tensor_kwargs["device"], dtype=torch.float32)

        B = x0_latents.shape[0]
        T_hist = self.framepack_total_max_num_latent_frames - self.framepack_num_new_latent_frames
        device = self.tensor_kwargs["device"]

        if getattr(self.config, "nft_train_on_dmd_timesteps", True):
            timesteps_B, sigma_B = self._sample_dmd_train_timesteps(B, device, force_idx=force_dmd_idx)
        else:
            t_B = self.rectified_flow.sample_train_time(B).to(**self.flow_matching_kwargs)
            timesteps_B = self.rectified_flow.get_discrete_timestamp(t_B, self.flow_matching_kwargs).reshape(-1)
            sigma_B = self.rectified_flow.get_sigmas(timesteps_B, self.flow_matching_kwargs).reshape(-1)

        sig = sigma_B.to(x0_latents.dtype).reshape(B, *([1] * (x0_latents.dim() - 1)))

        # Build xt: keep history clean, noise only the generated tail at sigma.
        noise_tail = torch.randn_like(x0_latents[:, :, T_hist:])
        xt = x0_latents.clone()
        xt[:, :, T_hist:] = (1.0 - sig) * x0_latents[:, :, T_hist:] + sig * noise_tail
        _trace_nonfinite("xt", xt, iteration=iteration, tag=tag)

        timesteps = rearrange(timesteps_B, "b -> b 1")

        # Context parallel: broadcast identical inputs to all CP ranks (mirrors base training_step).
        cp_group = self.get_context_parallel_group()
        if cp_group is not None:
            xt = broadcast(xt.contiguous(), cp_group)
            x0_latents = broadcast(x0_latents.contiguous(), cp_group)
            timesteps = broadcast(timesteps.contiguous(), cp_group)
            sigma_B = broadcast(sigma_B.contiguous(), cp_group)
            r = broadcast(r.contiguous(), cp_group)
            condition = condition.broadcast(cp_group)
            self.net.enable_context_parallel(cp_group)
        else:
            self.net.disable_context_parallel()

        # Three predictions on the generated region. Only `new` carries gradients.
        _mem("before_forwards", iteration)
        with self.adapter_ctx("old"), torch.no_grad():
            v_old = self.denoise(xt, timesteps, condition)
        _mem("after_v_old(nograd)", iteration)
        with self.adapter_ctx("ref"), torch.no_grad():
            v_ref = self.denoise(xt, timesteps, condition)
        _mem("after_v_ref(nograd)", iteration)
        # Offload v_new's saved-for-backward activations to host RAM (768Gi available):
        # this is the +17GB term in the train peak. They stream back during backward,
        # so the GPU never holds the full activation graph at once -> lower peak.
        with self.adapter_ctx("new"), torch.autograd.graph.save_on_cpu(pin_memory=True):
            v_new = self.denoise(xt, timesteps, condition)
        _mem("after_v_new(grad)", iteration)
        # PROBE: are the three adapter forwards actually distinct? (kl≈0 / pos≈neg says no)
        if iteration < 2:
            import torch.distributed as _d
            if not (_d.is_initialized() and _d.get_rank() != 0):
                def _rd(a, b):
                    return (a.float() - b.float()).norm().item() / (b.float().norm().item() + 1e-9)
                print(f"[nft-probe] iter{iteration} relΔ new-old={_rd(v_new, v_old):.4e} "
                      f"new-ref={_rd(v_new, v_ref):.4e} old-ref={_rd(v_old, v_ref):.4e}", flush=True)
        _trace_nonfinite("v_old(forward)", v_old, iteration=iteration, tag=tag)
        _trace_nonfinite("v_ref(forward)", v_ref, iteration=iteration, tag=tag)
        _trace_nonfinite("v_new(forward)", v_new, iteration=iteration, tag=tag)

        xt_gen = xt[:, :, T_hist:]
        x0_gen = x0_latents[:, :, T_hist:]

        loss, metrics = compute_nft_loss(
            v_new=v_new,
            v_old=v_old,
            v_ref=v_ref,
            xt_gen=xt_gen.to(v_new.dtype),
            x0_gen=x0_gen.to(v_new.dtype),
            sigma=sigma_B.to(v_new.dtype),
            r=r,
            beta=float(self.config.nft_beta),
            # KL-to-reference weight: env override (regularization against off-manifold
            # drift / pixelation) falling back to the config default (1e-4).
            beta_kl=float(os.environ.get("NFT_BETA_KL", self.config.nft_beta_kl)),
        )

        if _trace_nonfinite("loss", loss, iteration=iteration, tag=tag):
            # loss is bad but the forwards were finite -> dump the loss inputs to localize.
            _trace_nonfinite("x0_gen", x0_gen, iteration=iteration, tag=tag)
            _trace_nonfinite("xt_gen", xt_gen, iteration=iteration, tag=tag)
            _trace_nonfinite("sigma", sigma_B, iteration=iteration, tag=tag)
            _trace_nonfinite("r", r, iteration=iteration, tag=tag)
            print(f"[nft-nan]{tag} iter {iteration} metrics={ {k: float(v) for k, v in metrics.items()} }",
                  flush=True)

        _mem("after_loss(pre-backward)", iteration)
        output_batch = {"nft_loss": loss, **metrics}
        return output_batch, loss
