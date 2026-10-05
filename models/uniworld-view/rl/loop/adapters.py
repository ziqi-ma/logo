"""Three-LoRA-adapter management for DiffusionNFT on the UniView transformer.

Adapters over the frozen 14B base, all initialized from the CausVid rank-32 LoRA
(applied at scale 0.95, the demo's convention):
    ref  — frozen KL anchor
    old  — sampler (EMA of new)
    new  — trained (only params with grads)

PEFT sharp edges handled here (verified against diffusers 0.35 / peft 0.19):
  * pipe.set_adapters(names, weights) ACTIVATES every listed adapter while setting
    persistent per-adapter scaling -> immediately re-activate a single one.
  * BaseTunerLayer.set_adapter flips requires_grad as a side effect -> re-apply
    enforce_new_grad after every switch (adapter_ctx guarantees it).
  * Never fuse_lora in RL (the demo fuses; we must keep adapters separable).
"""
from __future__ import annotations

import contextlib
import json
import os

import torch

RL_ADAPTERS = ("ref", "old", "new")
TRAINABLE = "new"
CAUSVID_WEIGHT = 0.95

def load_rl_adapters(pipe, causvid_path: str, policy_lora: str | None = None,
                     ref_lora: str | None = None, weight: float = CAUSVID_WEIGHT) -> None:
    """Load ref/old/new (Kijai-format CausVid converts via WanLoraLoaderMixin)."""
    # peft 0.19 + transformers 4.48: set_peft_model_state_dict takes a tensor-parallel
    # shard path whenever torch.distributed is initialized and imports
    # transformers.integrations.tensor_parallel, which only exists in transformers
    # >=4.50. This model has no TP plan, so the hook is a no-op by construction.
    import peft.utils.save_and_load as _psl
    _psl._maybe_shard_state_dict_for_tp = lambda *a, **k: None

    init = {"new": policy_lora or causvid_path,
            "old": policy_lora or causvid_path,
            "ref": ref_lora or causvid_path}
    for name in RL_ADAPTERS:
        pipe.load_lora_weights(init[name], adapter_name=name)
    pipe.set_adapters(list(RL_ADAPTERS), adapter_weights=[weight] * len(RL_ADAPTERS))
    set_active_adapter(pipe.transformer, TRAINABLE)

def set_active_adapter(transformer, name: str) -> None:
    transformer.set_adapter(name)
    enforce_new_grad(transformer)

def enforce_new_grad(transformer) -> None:
    """Exactly the `new` LoRA params require grad; everything else frozen."""
    for pname, p in transformer.named_parameters():
        p.requires_grad_("lora_" in pname and f".{TRAINABLE}." in pname)

@contextlib.contextmanager
def adapter_ctx(transformer, name: str):
    """Activate `name`; restore the trainable adapter (+grads) on exit so the
    autograd graph for `new` survives checkpoint recomputation."""
    transformer.set_adapter(name)
    try:
        yield
    finally:
        transformer.set_adapter(TRAINABLE)
        enforce_new_grad(transformer)

def _lora_params(transformer, adapter: str):
    return {n: p for n, p in transformer.named_parameters()
            if "lora_" in n and f".{adapter}." in n}

def _lora_pairs(transformer, adapter: str):
    """[(A, B), ...] for every module of `adapter`, paired by name."""
    ps = _lora_params(transformer, adapter)
    out = []
    for n, p in ps.items():
        if "lora_A" in n:
            b = ps.get(n.replace("lora_A", "lora_B"))
            if b is not None:
                out.append((p, b))
    return out

@torch.no_grad()
def lora_l2(transformer, adapter: str) -> float:
    """Frobenius norm of the adapter's effective weight delta, sum over modules of ||B@A||_F^2.

    Computed via the rank-r grams (B^T B and A A^T are both r x r) so this costs microseconds
    instead of materializing 312 d_out x d_in products -- cheap enough to call every step.
    """
    tot = 0.0
    for A, B in _lora_pairs(transformer, adapter):
        Af, Bf = A.float(), B.float()
        tot += float(((Bf.T @ Bf) * (Af @ Af.T).T).sum())
    return max(tot, 0.0) ** 0.5

@torch.no_grad()
def lora_l2_ratio(transformer, adapter: str = TRAINABLE, ref: str = "ref") -> float:
    """||dW_adapter|| / ||dW_ref||. `ref` is frozen CausVid, so this is drift from the init
    with no external file needed. Empirically tracks visible degradation better than reward:
    ~1.2x still clean, ~1.35x visible texture/detail loss, >=1.6x collapse toward flat colour."""
    den = lora_l2(transformer, ref)
    return lora_l2(transformer, adapter) / den if den > 0 else float("nan")

def lora_l2_ratio_diff(transformer, adapter: str = TRAINABLE, ref: str = "ref"):
    """Differentiable ||dW_adapter|| / ||dW_ref||; `ref` (frozen CausVid) is a constant.

    Same rank-r gram identity as lora_l2, kept in the autograd graph so a penalty on the ratio
    backprops into lora_A/lora_B of the trainable adapter. All intermediates are r x r.
    """
    num_sq = None
    for A, B in _lora_pairs(transformer, adapter):
        Af, Bf = A.float(), B.float()
        t = ((Bf.T @ Bf) * (Af @ Af.T).T).sum()
        num_sq = t if num_sq is None else num_sq + t
    if num_sq is None:
        raise AssertionError(f"lora_l2_ratio_diff: no {adapter} lora pairs found")
    den = lora_l2(transformer, ref)  # no_grad
    return num_sq.clamp_min(1e-12).sqrt() / max(den, 1e-12)

def lora_barrier_penalty(transformer, lam: float, target: float,
                         adapter: str = TRAINABLE, ref: str = "ref"):
    """One-sided weight-travel barrier: ``lam * relu(ratio - target)^2``.

    Chosen over a plain lam*||dW||^2 penalty because the evidence is threshold-shaped, not
    linear: runs look fine up to ~1.3x and degrade past ~1.35x, so a two-sided penalty would
    tax the healthy early phase for nothing. Below `target` this is exactly zero and returns no
    gradient; above it the push-back grows linearly in the overshoot.

    Returns (penalty_tensor_or_None, current_ratio_float).
    """
    if lam <= 0:
        return None, float("nan")
    r = lora_l2_ratio_diff(transformer, adapter, ref)
    over = (r - target).clamp_min(0.0)
    if float(over) <= 0.0:
        return None, float(r.detach())
    return lam * over.pow(2), float(r.detach())

@torch.no_grad()
def ema_update_old(transformer, decay: float) -> int:
    """old <- decay*old + (1-decay)*new, name-paired. Returns pairs updated."""
    new_params = _lora_params(transformer, "new")
    old_list, new_list = [], []
    for n, p in new_params.items():
        on = n.replace(f".{TRAINABLE}.", ".old.")
        op = dict(transformer.named_parameters()).get(on)
        if op is not None:
            old_list.append(op.data)
            new_list.append(p.data)
    assert old_list, "ema_update_old: no old/new pairs matched"
    torch._foreach_mul_(old_list, decay)
    torch._foreach_add_(old_list, new_list, alpha=1.0 - decay)
    return len(old_list)

def save_new_adapter(pipe, out_dir: str, step: int, apply_weight: float = CAUSVID_WEIGHT) -> str:
    """Save `new` as a standard diffusers Wan LoRA, drop-in reloadable by
    pipe.load_lora_weights(...). The runtime scale is not baked in — recorded
    in a sidecar meta json instead."""
    from peft.utils import get_peft_model_state_dict
    os.makedirs(out_dir, exist_ok=True)
    sd = get_peft_model_state_dict(pipe.transformer, adapter_name=TRAINABLE)
    sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    weight_name = f"nft_new_step{step:04d}.safetensors"
    type(pipe).save_lora_weights(save_directory=out_dir, transformer_lora_layers=sd,
                                 weight_name=weight_name)
    with open(os.path.join(out_dir, f"nft_new_step{step:04d}.meta.json"), "w") as f:
        json.dump({"step": step, "apply_weight": apply_weight}, f)
    return os.path.join(out_dir, weight_name)

def load_new_adapter_inplace(pipe, ckpt_path: str) -> int:
    """Resume: copy a saved `new` checkpoint into the live `new` adapter params.
    Leaves old/ref untouched. Returns count loaded."""
    from safetensors.torch import load_file
    sd = load_file(ckpt_path)
    sd = {k[len("transformer."):] if k.startswith("transformer.") else k: v
          for k, v in sd.items()}
    live = _lora_params(pipe.transformer, TRAINABLE)
    n = 0
    with torch.no_grad():
        for name, p in live.items():
            canon = name.replace(f".{TRAINABLE}.", ".")
            if canon in sd:
                p.data.copy_(sd[canon].to(p.device, p.dtype))
                n += 1
    missing = len(live) - n
    if missing:
        print(f"[adapters] load_new_adapter_inplace: {missing} live params not in ckpt", flush=True)
    return n

def save_trainstate(transformer, opt, path: str, step: int, global_step: int) -> None:
    old_sd = {n: p.detach().cpu() for n, p in _lora_params(transformer, "old").items()}
    torch.save({"old": old_sd, "opt": opt.state_dict(),
                "step": step, "global_step": global_step}, path)

def load_trainstate(transformer, opt, path: str):
    st = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(transformer.named_parameters())
    with torch.no_grad():
        for n, t in st["old"].items():
            if n in params:
                params[n].data.copy_(t.to(params[n].device, params[n].dtype))
    opt.load_state_dict(st["opt"])
    return int(st["step"]), int(st["global_step"])
