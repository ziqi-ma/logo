"""LoRA adapter management for DiffusionNFT on WanModelFast.

Two adapters over the frozen distilled base (the upstream DiffusionNFT
scheme — the released causal_fast checkpoint IS the sampling policy):
  * "new" — the trainable policy
  * "old" — the rollout policy, EMA-tracked toward "new" once per loop step
The KL reference is the base model itself, reached by disabling all adapters
(`adapter_ctx(model, "base")`).

Both adapters start as the identity (lora_B = 0), and "old" is hard-copied
from "new" at injection so their lora_A match — the per-param EMA
`old <- d*old + (1-d)*new` is only meaningful when the two adapters share a
common parameterization.
"""
import contextlib
import logging
import os

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer

RL_ADAPTER_NAMES = ("new", "old")
TRAINABLE_ADAPTER = "new"

_ATTN_TARGETS = ["q", "k", "v", "o"]
_FFN_TARGETS = ["ffn.0", "ffn.2"]
_CAM_TARGETS = [
    "cam_injector_layer1",
    "cam_injector_layer2",
    "cam_scale_layer",
    "cam_shift_layer",
]

LORA_SCOPES = {
    "attn": _ATTN_TARGETS,
    "attn+cam": _ATTN_TARGETS + _CAM_TARGETS,
    "attn+ffn+cam": _ATTN_TARGETS + _FFN_TARGETS + _CAM_TARGETS,
}

def iter_tuner_layers(model):
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            yield module

def inject_rl_adapters(model, scope="attn+ffn+cam", rank=32, alpha=64):
    """Inject the new/old adapters into a loaded WanModelFast. Idempotent.

    Must run after from_pretrained (injection wraps target modules, so
    injecting first would break base-checkpoint key mapping). LoRA params are
    upcast to fp32 for optimizer stability; under autocast the forward still
    computes in the model dtype.
    """
    if any(isinstance(m, BaseTunerLayer) for m in model.modules()):
        return
    targets = LORA_SCOPES[scope]
    for name in RL_ADAPTER_NAMES:
        cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=targets,
            init_lora_weights="gaussian",
            bias="none",
        )
        inject_adapter_in_model(cfg, model, adapter_name=name)

    params = dict(model.named_parameters())
    n_copied = 0
    with torch.no_grad():
        for pname, p in params.items():
            if "lora_" not in pname:
                continue
            p.data = p.data.float()
            if f".{TRAINABLE_ADAPTER}." in pname:
                old_name = pname.replace(f".{TRAINABLE_ADAPTER}.", ".old.")
                if old_name in params:
                    params[old_name].data = p.data.clone()
                    n_copied += 1

    activate_adapter(model, TRAINABLE_ADAPTER)
    enforce_grad(model)
    n_layers = sum(1 for _ in iter_tuner_layers(model))
    logging.info(
        "inject_rl_adapters: scope=%s r=%d alpha=%d -> %d LoRA layers, "
        "%d old params copied from new", scope, rank, alpha, n_layers, n_copied)

def activate_adapter(model, name):
    assert name in RL_ADAPTER_NAMES, f"unknown adapter {name}"
    for module in iter_tuner_layers(model):
        module.set_adapter(name)

def enforce_grad(model):
    """Trainable = only the "new" adapter; base + old are frozen.

    ``BaseTunerLayer.set_adapter`` flips ``requires_grad`` as a side effect,
    so this is re-applied after every adapter switch to keep the optimizer's
    view of trainable params stable.
    """
    for pname, p in model.named_parameters():
        if "lora_" in pname:
            p.requires_grad_(f".{TRAINABLE_ADAPTER}." in pname)
        else:
            p.requires_grad_(False)

@contextlib.contextmanager
def adapter_ctx(model, name):
    """Temporarily run under adapter ``name`` ("new", "old", or "base" =
    adapters disabled); restore the trainable adapter + its grads on exit."""
    layers = None
    try:
        if name == "base":
            layers = list(iter_tuner_layers(model))
            for m in layers:
                m.enable_adapters(False)
        else:
            activate_adapter(model, name)
        yield
    finally:
        if layers is not None:
            for m in layers:
                m.enable_adapters(True)
        activate_adapter(model, TRAINABLE_ADAPTER)
        enforce_grad(model)

@torch.no_grad()
def ema_update_old(model, decay):
    """old <- decay*old + (1-decay)*new, over matched LoRA params."""
    params = dict(model.named_parameters())
    old_list, new_list = [], []
    for pname, p in params.items():
        if f".{TRAINABLE_ADAPTER}." not in pname:
            continue
        old_name = pname.replace(f".{TRAINABLE_ADAPTER}.", ".old.")
        if old_name not in params:
            continue
        new_list.append(p.data)
        old_list.append(params[old_name].data)
    if not old_list:
        logging.warning("ema_update_old: no matched new/old LoRA params found")
        return
    torch._foreach_mul_(old_list, decay)
    torch._foreach_add_(old_list, new_list, alpha=1.0 - decay)

def _strip_adapter_name(state_dict, name):
    """Map live PEFT keys ``...lora_A.{name}.weight`` -> ``...lora_A.weight``
    so a saved adapter is adapter-slot-agnostic on reload."""
    out = {}
    for k, v in state_dict.items():
        for suffix in (".weight", ".bias"):
            tag = f".{name}{suffix}"
            if k.endswith(tag):
                k = k[: -len(tag)] + suffix
                break
        out[k] = v
    return out

def adapter_state_dict(model, name):
    live = {}
    for pname, p in model.named_parameters():
        if "lora_" in pname and f".{name}." in pname:
            live[pname] = p.detach().cpu().contiguous()
    return _strip_adapter_name(live, name)

def load_adapter_state_dict(model, name, sd):
    """Inverse of :func:`adapter_state_dict`; loads in place into adapter
    slot ``name``, leaving the other adapter untouched. Returns params loaded."""
    loaded, missing = 0, 0
    for pname, p in model.named_parameters():
        if "lora_" not in pname or f".{name}." not in pname:
            continue
        canon = next(iter(_strip_adapter_name({pname: None}, name)))
        if canon in sd:
            p.data.copy_(sd[canon].to(device=p.device, dtype=p.dtype))
            loaded += 1
        else:
            missing += 1
    if missing:
        logging.warning("load_adapter_state_dict(%s): %d params missing (loaded %d)",
                        name, missing, loaded)
    return loaded

def save_new_adapter(model, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(adapter_state_dict(model, TRAINABLE_ADAPTER), path)
    return path

def load_new_adapter(model, path):
    sd = torch.load(path, map_location="cpu")
    return load_adapter_state_dict(model, TRAINABLE_ADAPTER, sd)
