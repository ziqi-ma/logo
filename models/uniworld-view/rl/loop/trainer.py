"""NFT training step for UniWorld-View: re-noise stored x0 at a schedule step,
run old/ref (no-grad) and new (grad) forwards on the same xt, apply the NFT loss.

train_pass ports Lyra's nft_loop._train_pass: sweep all schedule timesteps per
sample, gradient accumulation with scale=1/(samples*timesteps), one manual
all_reduce(AVG)+clip+opt.step per group. DDP-safe: min-local-count sync, OOM
skips a microstep locally (no mid-loop collective), None grads zero-filled.
"""
from __future__ import annotations

import contextlib
import gc
import os
from collections import defaultdict

import torch

from rl.loop.adapters import adapter_ctx
from rl.data.cond_bundle import CondBundle
from rl.loop.nft_loss import compute_nft_loss
from rl.loop.sampling import denoise_forward
from rl.loop.schedule import RLSchedule

_NFT_KEYS = ("nft/pos_loss", "nft/neg_loss", "nft/kl_loss", "nft/old_deviate",
             "nft/old_kl_div", "nft/amb_rel", "nft/apb_rel", "nft/recon_err_old",
             "nft/v_new_sq", "nft/v_ref_sq", "nft/x0_norm", "nft/r_mean")

def training_microstep(transformer, bundle: CondBundle, x0: torch.Tensor,
                       r: torch.Tensor, sched: RLSchedule, t_idx: int, *,
                       beta: float = 1.0, beta_kl: float = 1e-4,
                       save_on_cpu: bool = True, device="cuda"):
    """One (sample, timestep) forward/loss. x0: [1,C,T,H,W] fp32 on device."""
    sigma = sched.sigmas[t_idx].to(device)
    t = sched.timesteps[t_idx].to(device).expand(x0.shape[0])
    noise = torch.randn_like(x0)
    xt = (1.0 - sigma) * x0 + sigma * noise

    with adapter_ctx(transformer, "old"), torch.no_grad():
        v_old = denoise_forward(transformer, xt, t, bundle)
    with adapter_ctx(transformer, "ref"), torch.no_grad():
        v_ref = denoise_forward(transformer, xt, t, bundle)
    ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
           if save_on_cpu else contextlib.nullcontext())
    with adapter_ctx(transformer, "new"), ctx:
        v_new = denoise_forward(transformer, xt, t, bundle)

    return compute_nft_loss(
        v_new=v_new.float(), v_old=v_old.float(), v_ref=v_ref.float(),
        xt=xt.float(), x0=x0.float(),
        sigma=sigma.expand(x0.shape[0]).float(), r=r,
        beta=beta, beta_kl=float(os.environ.get("NFT_BETA_KL", beta_kl)))

def train_pass(transformer, samples: list, sched: RLSchedule, params, opt, *,
               rank: int, world_size: int, global_step: int,
               beta: float = 1.0, beta_kl: float = 1e-4,
               grad_steps: int = 1, inner_epochs: int = 1,
               max_grad_norm: float = 1.0, save_on_cpu: bool = True,
               lora_l2_lambda: float = 0.0, lora_l2_target: float = 1.3,
               device="cuda"):
    """samples: [{bundle: CondBundle(on device), x0: tensor, r: float}].
    Returns (global_step, metrics)."""
    import torch.distributed as dist
    dist_on = world_size > 1 and dist.is_initialized()

    n_local = torch.tensor([len(samples)], device=device)
    if dist_on:
        dist.all_reduce(n_local, op=dist.ReduceOp.MIN)
    n_steps = int(n_local.item())
    if n_steps == 0:
        print(f"[train_pass] rank{rank}: 0 common samples, skipping", flush=True)
        return global_step, {"n_steps": 0}

    num_t = sched.num_steps
    n_t_train = int(os.environ.get("NFT_TRAIN_TIMESTEPS", "0") or 0) or num_t
    n_t_train = max(1, min(n_t_train, num_t))

    grad_steps = max(1, min(grad_steps, n_steps))
    group_sz = -(-n_steps // grad_steps)

    acc = defaultdict(list)
    n_opt = n_micro = n_oom = n_bad = 0
    gnorms = []
    transformer.train()

    for _inner in range(inner_epochs):
        order = torch.randperm(len(samples))[:n_steps].tolist()
        groups = [order[i:i + group_sz] for i in range(0, n_steps, group_sz)]
        for group in groups:
            opt.zero_grad(set_to_none=True)
            scale = 1.0 / (len(group) * n_t_train)
            for si in group:
                s = samples[si]
                # r is a scalar (global reward), a per-latent list (windowed reward),
                # or a nested [T][gh][gw] list (voxel reward); compute_nft_loss takes
                # [B], [B, T] (broadcast over the frame axis) or [B, T, gh, gw].
                _r = s["r"]
                if isinstance(_r, (list, tuple)) and _r and isinstance(_r[0], (list, tuple)):
                    r = torch.tensor(_r, device=device, dtype=torch.float32)[None]
                elif isinstance(_r, (list, tuple)):
                    r = torch.tensor([[float(x) for x in _r]], device=device, dtype=torch.float32)
                else:
                    r = torch.tensor([float(_r)], device=device, dtype=torch.float32)
                if n_t_train == num_t:
                    t_idxs = range(num_t)
                else:
                    t_idxs = torch.randperm(num_t)[:n_t_train].tolist()
                for ti in t_idxs:
                    try:
                        loss, m = training_microstep(
                            transformer, s["bundle"], s["x0"], r, sched, ti,
                            beta=beta, beta_kl=beta_kl, save_on_cpu=save_on_cpu,
                            device=device)
                        (loss * scale).backward()
                        n_micro += 1
                        acc["loss"].append(float(loss.detach()))
                        for k in _NFT_KEYS:
                            if k in m:
                                acc[k].append(float(m[k]))
                    except torch.cuda.OutOfMemoryError:
                        n_oom += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                if not torch.isfinite(p.grad).all():
                    p.grad.nan_to_num_()
                    n_bad += 1
            if dist_on:
                for p in params:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            # Weight-travel barrier, added once per optimizer step (not per microstep, which
            # would scale it by grad_steps). Deterministic from replicated weights, so it needs
            # no all_reduce and is added after it; included in the clip so the total update stays
            # bounded and the reward/barrier balance is preserved.
            if lora_l2_lambda > 0:
                from rl.loop import adapters as _ad
                pen, r_now = _ad.lora_barrier_penalty(transformer, lora_l2_lambda, lora_l2_target)
                if pen is not None:
                    pen.backward()
                    acc.setdefault("lora_pen", []).append(float(pen.detach()))
                else:
                    acc.setdefault("lora_pen", []).append(0.0)
                acc.setdefault("lora_ratio_pre", []).append(r_now)
            gnorm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            gnorms.append(float(gnorm))
            opt.step()
            n_opt += 1
            global_step += 1

    metrics = {k: sum(v) / len(v) for k, v in acc.items() if v}
    metrics.update({"n_steps": n_opt, "n_microsteps": n_micro, "n_oom": n_oom,
                    "n_bad": n_bad,
                    "grad_norm": sum(gnorms) / len(gnorms) if gnorms else 0.0})
    return global_step, metrics
