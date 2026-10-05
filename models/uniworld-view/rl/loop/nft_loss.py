"""DiffusionNFT loss for UniWorld-View (rectified-flow / sigma parameterization).

Port of Lyra-2's `lyra_2/_src/rl/nft_loss.py` (itself derived from NVIDIA's
DiffusionNFT SD3 reference, arXiv 2509.16117). UniWorld generates one-shot, so
the whole latent is the generated region — no history masking.

Convention (matches rl.loop.schedule / diffusers flow matching):
    x_t = (1 - sigma) * x0 + sigma * noise
    v   = noise - x0
    x0  = x_t - sigma * v
"""
from __future__ import annotations

from typing import Optional

import torch

def reconstruct_x0(xt: torch.Tensor, v: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """x0 = x_t - sigma * v, with sigma broadcast over the non-batch dims."""
    sig = sigma.reshape(sigma.shape[0], *([1] * (xt.dim() - 1)))
    return xt - sig * v

def compute_nft_loss(
    v_new: torch.Tensor,
    v_old: torch.Tensor,
    v_ref: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    beta: float = 1.0,
    beta_kl: float = 1e-4,
    time_weights: Optional[torch.Tensor] = None,
    weight_eps: float = 1e-5,
):
    """Per-batch DiffusionNFT loss.

    All velocity/latent tensors share shape [B, C, T, H, W]; ``sigma`` is [B];
    ``r`` is [B] (per-sample reward weight in [0,1]), [B, T] (per-latent-frame),
    or [B, T, gh, gw] (voxel reward, nearest-upsampled to the latent grid).
    ``v_old``/``v_ref`` are detached internally. Returns (loss, metrics-dict of
    detached scalars).
    """
    assert beta > 0, "beta must be > 0 (it scales the policy loss denominator)"
    v_old = v_old.detach()
    v_ref = v_ref.detach()

    reduce_dims = list(range(1, x0.dim()))            # per-sample dims (C,T,H,W)
    frame_dim = 2                                     # latent-frame axis
    pf_dims = [d for d in reduce_dims if d != frame_dim]
    sig = sigma.reshape(sigma.shape[0], *([1] * (x0.dim() - 1)))

    v_pos = beta * v_new + (1.0 - beta) * v_old
    v_neg = (1.0 + beta) * v_old - beta * v_new
    x0_pos = xt - sig * v_pos
    x0_neg = xt - sig * v_neg

    # Adaptive normalizers stay PER-SAMPLE (reduce over C,T,H,W): a constant-over-
    # frames r then reproduces the scalar-r loss exactly.
    w_pos = (x0_pos.detach() - x0).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=weight_eps)
    w_neg = (x0_neg.detach() - x0).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=weight_eps)

    pos_pf = ((x0_pos - x0) ** 2 / w_pos).mean(dim=pf_dims)  # [B, T]
    neg_pf = ((x0_neg - x0) ** 2 / w_neg).mean(dim=pf_dims)  # [B, T]
    if r.dim() == 4:
        # voxel reward [B, T, gh, gw]: per-latent-location weight. w_pos/w_neg stay
        # per-sample, so a spatially constant r reproduces the scalar-r loss exactly.
        from rl.scoring.nft_voxel import expand_r_to_latent
        r_up = expand_r_to_latent(r, x0.shape[-2], x0.shape[-1])  # [B, 1, T, H, W]
        pos_el = (x0_pos - x0) ** 2 / w_pos
        neg_el = (x0_neg - x0) ** 2 / w_neg
        policy_loss = (r_up * pos_el + (1.0 - r_up) * neg_el).mean(dim=reduce_dims) / beta
    else:
        r2 = r if r.dim() >= 2 else r.reshape(-1, 1)
        policy_loss = (r2 * pos_pf + (1.0 - r2) * neg_pf).mean(dim=1) / beta  # [B]

    kl_loss = ((v_new - v_ref) ** 2).mean(dim=reduce_dims)  # [B]

    per_sample = policy_loss + beta_kl * kl_loss
    if time_weights is not None:
        per_sample = time_weights.reshape(-1) * per_sample
    loss = per_sample.mean()

    r_flat = r.reshape(-1)
    with torch.no_grad():
        # Self-normalized pos/neg losses hide whether `new` moves away from `old`;
        # log the raw deviations + the residual-geometry collapse diagnostics
        # (a-b ~ 0 <=> v_new==v_old healthy; a+b ~ 0 <=> v_old==v_true, reward cancels).
        old_deviate = ((v_new - v_old) ** 2).mean()
        old_kl_div = ((v_old - v_ref) ** 2).mean()
        a = (x0_pos - x0).detach()
        b = (x0_neg - x0).detach()
        a_norm = a.norm().clamp(min=1e-12)
        amb_rel = (a - b).norm() / a_norm
        apb_rel = (a + b).norm() / a_norm
        x0_old = xt - sig * v_old
        recon_err_old = ((x0_old - x0) ** 2).mean()
        v_new_sq = (v_new ** 2).mean()
        v_ref_sq = (v_ref ** 2).mean()
        x0_norm = (x0 ** 2).mean()

    metrics = {
        "nft/loss": loss.detach(),
        "nft/policy_loss": policy_loss.mean().detach(),
        "nft/pos_loss": pos_pf.mean().detach(),
        "nft/neg_loss": neg_pf.mean().detach(),
        "nft/kl_loss": kl_loss.mean().detach(),
        "nft/old_deviate": old_deviate,
        "nft/old_kl_div": old_kl_div,
        "nft/amb_rel": amb_rel,
        "nft/apb_rel": apb_rel,
        "nft/recon_err_old": recon_err_old,
        "nft/v_new_sq": v_new_sq,
        "nft/v_ref_sq": v_ref_sq,
        "nft/x0_norm": x0_norm,
        "nft/r_mean": r_flat.mean().detach(),
        "nft/r_std": r_flat.std().detach(),
    }
    return loss, metrics
