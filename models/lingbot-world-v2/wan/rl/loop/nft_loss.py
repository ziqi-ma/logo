"""Core DiffusionNFT loss math, on plain tensors.

Rectified-flow conventions:

    x_t  = sigma * noise + (1 - sigma) * x0
    v    = noise - x0                          (the flow-matching target / prediction)
    x0   = x_t - sigma * v                     (reconstruction used here)

The NFT objective (NVIDIA, arXiv 2509.16117), ported from the SD3 reference
``DiffusionNFT/scripts/train_nft_sd3.py`` but expressed with *sigma* (not t) to
match the rectified-flow parameterization, and restricted to the generated region:

    v_pos = beta * v_new + (1 - beta) * v_old
    v_neg = (1 + beta) * v_old - beta * v_new
    x0_pos = x_t - sigma * v_pos ;  x0_neg = x_t - sigma * v_neg
    L = mean_b[ tw_b * ( r_b * ||x0_pos - x0||^2 / w_pos / beta
                       + (1 - r_b) * ||x0_neg - x0||^2 / w_neg / beta ) ]
        + beta_kl * mean ||v_new - v_ref||^2

``r in [0, 1]`` is the reward-derived weight (1 => purely positive, 0 => purely
negative). ``w_pos``/``w_neg`` are detached adaptive per-sample normalizers.
``v_old`` and ``v_ref`` are detached: gradients flow only through ``v_new``.
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
    xt_gen: torch.Tensor,
    x0_gen: torch.Tensor,
    sigma: torch.Tensor,
    r: torch.Tensor,
    beta: float = 1.0,
    beta_kl: float = 1e-4,
    time_weights: Optional[torch.Tensor] = None,
    weight_eps: float = 1e-5,
):
    """Compute the per-batch DiffusionNFT loss on the generated region.

    All velocity / latent tensors share shape ``[B, C, T, H, W]`` (T = latent frame
    axis). ``sigma`` is ``[B]``. ``r`` is ``[B]`` (one weight per sample),
    ``[B, T]`` (per-frame weight, T matching the generated frame axis), or
    ``[B, T, gh, gw]`` (per-patch voxel reward, nearest-upsampled to the latent
    grid). ``v_old`` and ``v_ref`` are detached internally so callers need not.

    Returns ``(loss, metrics)`` where ``loss`` is a scalar tensor and ``metrics``
    is a dict of detached scalars for logging.
    """
    assert beta > 0, "beta must be > 0 (it scales the policy loss denominator)"
    v_old = v_old.detach()
    v_ref = v_ref.detach()

    reduce_dims = list(range(1, x0_gen.dim()))              # per-sample dims (C,T,H,W)
    frame_dim = 2                                            # [B, C, T, H, W] -> dim 2 is the latent-frame axis
    pf_dims = [d for d in reduce_dims if d != frame_dim]     # per-frame dims (C,H,W): keep the frame axis
    sig = sigma.reshape(sigma.shape[0], *([1] * (x0_gen.dim() - 1)))

    v_pos = beta * v_new + (1.0 - beta) * v_old
    v_neg = (1.0 + beta) * v_old - beta * v_new
    x0_pos = xt_gen - sig * v_pos
    x0_neg = xt_gen - sig * v_neg

    # Adaptive normalizers stay PER-SAMPLE (reduce over C,T,H,W). Keeping them per-sample (not
    # per-frame) is what makes a constant-over-frame r reproduce the original scalar-r loss
    # exactly -- mean_T then distributes over the shared w.
    w_pos = (x0_pos.detach() - x0_gen).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=weight_eps)
    w_neg = (x0_neg.detach() - x0_gen).abs().mean(dim=reduce_dims, keepdim=True).clamp(min=weight_eps)

    # Per-frame reconstruction losses: reduce over C,H,W, keep the frame axis -> [B, T].
    pos_pf = ((x0_pos - x0_gen) ** 2 / w_pos).mean(dim=pf_dims)  # [B, T]
    neg_pf = ((x0_neg - x0_gen) ** 2 / w_neg).mean(dim=pf_dims)  # [B, T]
    if r.dim() == 4:
        # r is [B, T, gh, gw] (voxel reward): weight the loss per LATENT LOCATION. The
        # patch grid partitions the same image plane as the latent grid, so nearest
        # upsampling IS the index map (latent (y, x) reads patch (y*gh//H, x*gw//W)) --
        # inlined from nft_voxel.expand_r_to_latent to keep this module dependency-free
        # (it is unit-tested with plain tensors, loaded by file path). w_pos/w_neg stay
        # per-sample, so a spatially constant r reproduces the scalar-r loss exactly.
        import torch.nn.functional as F
        r_up = F.interpolate(r, size=x0_gen.shape[-2:], mode="nearest").unsqueeze(1)  # [B,1,T,H,W]
        pos_el = (x0_pos - x0_gen) ** 2 / w_pos
        neg_el = (x0_neg - x0_gen) ** 2 / w_neg
        policy_loss = (r_up * pos_el + (1.0 - r_up) * neg_el).mean(dim=reduce_dims) / beta  # [B]
    else:
        # r is [B] (one weight per sample) or [B, T] (per-frame). Broadcast over the frame
        # axis, weight each frame's pos/neg, then average over frames -> [B].
        r2 = r if r.dim() >= 2 else r.reshape(-1, 1)
        policy_loss = (r2 * pos_pf + (1.0 - r2) * neg_pf).mean(dim=1) / beta  # [B]

    kl_loss = ((v_new - v_ref) ** 2).mean(dim=reduce_dims)  # [B]  == reference "kl_div"

    per_sample = policy_loss + beta_kl * kl_loss  # [B]
    if time_weights is not None:
        per_sample = time_weights.reshape(-1) * per_sample
    loss = per_sample.mean()

    pos_loss = pos_pf.mean(dim=1)   # [B] mean over frames, for logging
    neg_loss = neg_pf.mean(dim=1)
    r_flat = r.reshape(-1)          # r_mean/r_std over all (sample, frame) weights

    # --- Learning / symmetry diagnostics -------------------------------------
    # The self-normalized pos_loss/neg_loss cannot tell whether training is moving
    # `new` away from `old` (the |.| normalizer erases the sign), so -- exactly as
    # the DiffusionNFT SD3 reference does -- log the raw velocity deviations.
    with torch.no_grad():
        old_deviate = ((v_new - v_old) ** 2).mean()              # reference "old_deviate": v_new vs v_old
        old_kl_div = ((v_old - v_ref) ** 2).mean()               # reference "old_kl_div": v_old vs v_ref

        # Sign-preserving residual geometry. a = x0_pos - x0_gen, b = x0_neg - x0_gen.
        #   a - b = 2*beta*sigma*(v_old - v_new)   -> ~0  iff  v_new == v_old   (case 1, healthy: r lives in grad)
        #   a + b = 2*sigma*(v_true - v_old)       -> ~0  iff  v_old == v_true  (case 2, dead:   r cancels in grad)
        a = (x0_pos - x0_gen).detach()
        b = (x0_neg - x0_gen).detach()
        a_norm = a.norm().clamp(min=1e-12)
        amb_rel = (a - b).norm() / a_norm                        # ~0 => case 1 (v_new==v_old)
        apb_rel = (a + b).norm() / a_norm                        # ~0 => case 2 (v_old==v_true: reward cancels)
        # Direct reconstruction error of the old model on its own re-noised sample:
        #   ||x0_gen - (xt - sigma*v_old)|| == ||a + b|| / 2 .  Large => old is not v_true => symmetry not collapsed.
        x0_old = xt_gen - sig * v_old
        recon_err_old = ((x0_old - x0_gen) ** 2).mean()
        # Raw velocity scale: tells whether a small *absolute* kl means "v_new~v_ref"
        # (velocities O(1)) or just "velocities are tiny-norm" (kl = rho^2 * mean(v_ref^2)).
        v_new_sq = (v_new ** 2).mean()
        v_ref_sq = (v_ref ** 2).mean()
        x0_norm = (x0_gen ** 2).mean()  # latent scale (reference logs this too)

    metrics = {
        "nft/loss": loss.detach(),
        "nft/policy_loss": policy_loss.mean().detach(),
        "nft/pos_loss": pos_loss.mean().detach(),
        "nft/neg_loss": neg_loss.mean().detach(),
        "nft/kl_loss": kl_loss.mean().detach(),  # v_new vs v_ref
        "nft/old_deviate": old_deviate,          # v_new vs v_old  (learning signal)
        "nft/old_kl_div": old_kl_div,            # v_old vs v_ref
        "nft/amb_rel": amb_rel,                  # ||a-b||/||a|| : ~0 => case 1 (healthy)
        "nft/apb_rel": apb_rel,                  # ||a+b||/||a|| : ~0 => case 2 (reward cancels)
        "nft/recon_err_old": recon_err_old,      # old-model self-reconstruction error
        "nft/v_new_sq": v_new_sq,                 # velocity scale (resolves kl reading)
        "nft/v_ref_sq": v_ref_sq,
        "nft/x0_norm": x0_norm,                   # latent scale
        "nft/r_mean": r_flat.mean().detach(),
        "nft/r_std": r_flat.std().detach(),
    }
    return loss, metrics
