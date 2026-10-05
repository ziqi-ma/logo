# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 3 of the DiffusionNFT pipeline: train the DMD-RL ``new`` LoRA.

Two ways to run:

* Drive the existing imaginaire FSDP trainer with a :class:`Lyra2NFTModel` and an
  NFT dataloader. ``Lyra2NFTModel.training_step`` dispatches rollout batches to the
  NFT loss, so the trainer's grad-accum / FSDP sync / checkpointing are reused
  unchanged. (Set ``config.model`` target to ``Lyra2NFTModel`` / ``Lyra2NFTConfig``,
  point ``lora_config.pretrained_lora_path`` at the DMD checkpoint, and disable the
  base ``ema`` -- the policy EMA here is the ``old`` adapter, updated per epoch.)

* :func:`train_nft_simple` -- a self-contained loop (single process / debug / 1.3B
  scale-down style) that optimizes only the ``new`` adapter, then EMA-updates
  ``old <- new`` at each epoch boundary and saves the new adapter.

* :func:`train_nft_ddp` -- the 8-GPU the scheduler path. Launched via
  ``torchrun --nproc_per_node=8 -m lyra_2._src.rl.loop.nft_train --ddp ...``. Each rank
  builds the full 14B + 3 adapters on its GPU (DDP, not FSDP -- sidesteps the
  adapter-injection-before-shard gap) from the same checkpoint+LoRA (so ranks start
  identical), runs a DistributedSampler shard, and the ``new`` LoRA grads are
  averaged across ranks with all-reduce before each optimizer step -- keeping ranks
  in lockstep. Only rank 0 saves.

Either way the per-epoch boundary does ``model.ema_update_old(decay)`` and persists
the ``new`` adapter, which becomes the next NFT epoch's sampling policy (Stage 1).
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from lyra_2._src.rl.loop.rollout_dataset import NFTRolloutDataset, nft_collate

# --------------------------------------------------------------------------- #
# Dataloader
# --------------------------------------------------------------------------- #
def make_nft_dataloader(
    rewards_jsonl: str,
    batch_size: int = 1,
    adv_clip_max: float = 5.0,
    shuffle: bool = True,
    num_workers: int = 2,
    drop_nonfinite: bool = True,
):
    """Build a DataLoader over a scored rollout store. Returns (loader, dataset)."""
    ds = NFTRolloutDataset(rewards_jsonl, adv_clip_max=adv_clip_max, drop_nonfinite=drop_nonfinite)
    from lyra_2._ext.imaginaire.utils import log

    log.info(
        f"NFT dataset: {len(ds)} samples, {ds.num_groups} groups, "
        f"zero_std_ratio={ds.zero_std_ratio:.3f}, dropped(nonfinite)={ds.num_dropped}",
        rank0_only=True,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=nft_collate, drop_last=False,
    )
    return loader, ds

# --------------------------------------------------------------------------- #
# Model construction (FSDP, 3 RL adapters)
# --------------------------------------------------------------------------- #
def build_nft_model(
    checkpoint_dir: str,
    experiment: str = "lyra2_nft",
    policy_lora: Optional[str] = None,
    ref_lora: Optional[str] = None,
    enable_fsdp: bool = True,
    config_file: str = "lyra_2/_src/configs/config.py",
):
    """Build the FSDP `Lyra2NFTModel` and load the 14B base checkpoint.

    `policy_lora` initializes new/old (the promoted policy); `ref_lora` initializes
    the frozen KL anchor. Both default to the DMD path baked into the model node
    (correct for epoch 0). Adapter injection/grad-freeze happens inside the model
    constructor; this just selects the init paths.
    """
    from lyra_2._src.utils.model_loader import load_model_from_checkpoint

    # postpone_checkpoint defers block selective-checkpoint wrapping until after the
    # 3 RL adapters are injected (in Lyra2NFTModel.__init__); otherwise the wrapped
    # block submodule names don't match the DMD LoRA targets and only ~14/615
    # modules get adapters. Mirrors how inference loads the DMD LoRA.
    opts = ["model.config.net.postpone_checkpoint=True"]
    if policy_lora:
        opts.append(f"model.config.nft_policy_lora_path={policy_lora}")
    if ref_lora:
        opts.append(f"model.config.nft_ref_lora_path={ref_lora}")

    # Non-FSDP: defer adapter injection so the base 14B checkpoint loads into a clean
    # (un-wrapped) net first, then inject the LoRA -- the normal load order used by
    # build_inference_model. Without this, __init__ injects before the checkpoint load,
    # wrapping modules so the base keys don't map -> base left at init (head.head=0) ->
    # the net outputs ~0 -> noise rollouts + v~0. FSDP must inject pre-shard, so it does
    # not defer (adapters injected in __init__, before fully_shard).
    defer = not enable_fsdp
    if defer:
        os.environ["NFT_DEFER_INJECTION"] = "1"
    try:
        model, config = load_model_from_checkpoint(
            config_file=config_file,
            experiment_name=experiment,
            checkpoint_path=checkpoint_dir,
            enable_fsdp=enable_fsdp,
            instantiate_ema=False,
            load_ema_to_reg=False,
            experiment_opts=opts,
            strict=False,
        )
    finally:
        if defer:
            os.environ.pop("NFT_DEFER_INJECTION", None)
    if defer:
        # Base is loaded; now inject + load the LoRA on the populated net (like eval).
        model._finalize_rl_adapters()
    return model, config

# --------------------------------------------------------------------------- #
# Adapter checkpoint I/O
# --------------------------------------------------------------------------- #
def _strip_adapter_name(state_dict: dict, name: str) -> dict:
    """Map live PEFT keys ``...lora_A.{name}.weight`` -> ``...lora_A.weight``.

    This is the canonical "non-diffusers wan" LoRA layout that
    ``WANDiffusionModel.load_lora_weights`` consumes, so a saved adapter becomes a
    drop-in sampling LoRA (loadable like the original DMD checkpoint).
    """
    out = {}
    for k, v in state_dict.items():
        # Selective-activation checkpointing wraps each block, inserting
        # "._checkpoint_wrapped_module" into the param path. The canonical WAN-LoRA
        # layout has no such segment, so strip it or load_lora_weights can't map the
        # keys ("state_dict should be empty but has ...").
        k = k.replace("._checkpoint_wrapped_module", "")
        for suffix in (".weight", ".bias"):
            tag = f".{name}{suffix}"
            if k.endswith(tag):
                k = k[: -len(tag)] + suffix
                break
        # The loader reads LoRA bias from "diff_b", not "lora_B.bias" (see
        # _convert_non_diffusers_wan_lora_to_diffusers); rename or it's left unconsumed.
        if k.endswith(".lora_B.bias"):
            k = k[: -len(".lora_B.bias")] + ".diff_b"
        out[k] = v
    return out

def save_new_adapter(model, path: str) -> str:
    """Save the trainable ``new`` adapter as a reloadable LoRA checkpoint.

    Keys are stripped of the adapter-name segment so the next NFT epoch can load
    it via ``model.load_lora_weights(path)`` exactly like the DMD LoRA.
    """
    from lyra_2._src.models.lyra2_nft_model import _local

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    name = model.TRAINABLE_ADAPTER
    live = {}
    for pname, p in model.net.named_parameters():
        if "lora_" in pname and f".{name}." in pname:
            live[pname] = _local(p).detach().cpu().contiguous()
    torch.save(_strip_adapter_name(live, name), path)
    return path

def load_new_adapter(model, path: str) -> int:
    """Inverse of :func:`save_new_adapter`: load a canonical LoRA checkpoint into the
    trainable ``new`` adapter in place, leaving ``old``/``ref`` untouched.

    Used to RESUME a preempted loop: rebuild with the DMD policy (so old=ref=DMD,
    the frozen sampler under EMA-off) then overwrite ``new`` with the latest trained
    checkpoint. Returns the number of params loaded."""
    from lyra_2._src.models.lyra2_nft_model import _local

    sd = torch.load(path, map_location="cpu")
    name = model.TRAINABLE_ADAPTER
    loaded, missing = 0, 0
    for pname, p in model.net.named_parameters():
        if "lora_" not in pname or f".{name}." not in pname:
            continue
        canon = next(iter(_strip_adapter_name({pname: None}, name)))  # live key -> canonical
        if canon in sd:
            t = _local(p)
            t.data.copy_(sd[canon].to(device=t.device, dtype=t.dtype))
            loaded += 1
        else:
            missing += 1
    if missing:
        print(f"[load_new_adapter] WARNING: {missing} 'new' params missing from {path} "
              f"(loaded {loaded})", flush=True)
    return loaded

# --------------------------------------------------------------------------- #
# Self-contained training loop
# --------------------------------------------------------------------------- #
def train_nft_simple(
    model,
    dataloader,
    *,
    num_epochs: int = 1,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    ema_decay: Optional[float] = None,
    ckpt_dir: Optional[str] = None,
    log_every: int = 10,
):
    """Minimal NFT training loop over a single process.

    Optimizes only the ``new`` adapter (``enforce_rl_grad`` guarantees that's the
    only trainable set), EMA-updates ``old <- new`` at each epoch boundary, and
    saves the ``new`` adapter. For multi-node / sharded 14B runs prefer the
    imaginaire trainer path (see module docstring).
    """
    from lyra_2._ext.imaginaire.utils import log

    model.enforce_rl_grad()
    model.activate_rl_adapter(model.TRAINABLE_ADAPTER)
    params = [p for p in model.net.parameters() if p.requires_grad]
    assert params, "no trainable params; expected the 'new' LoRA adapter"
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    step = 0
    for epoch in range(num_epochs):
        for it, batch in enumerate(dataloader):
            out, loss = model.training_step_from_rollout(batch, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            if step % log_every == 0:
                log.info(
                    f"[nft] epoch {epoch} step {step} loss={float(loss):.4f} "
                    f"pos={float(out['nft/pos_loss']):.4f} neg={float(out['nft/neg_loss']):.4f} "
                    f"kl={float(out['nft/kl_loss']):.4f} r_mean={float(out['nft/r_mean']):.3f}",
                    rank0_only=True,
                )
            step += 1
        # Epoch boundary: EMA the sampling policy and checkpoint the new adapter.
        model.ema_update_old(ema_decay)
        if ckpt_dir:
            save_new_adapter(model, os.path.join(ckpt_dir, f"nft_new_adapter_epoch{epoch}.pt"))
    return model

# --------------------------------------------------------------------------- #
# DDP training loop (8-GPU the scheduler path)
# --------------------------------------------------------------------------- #
def train_nft_ddp(
    model,
    dataset,
    *,
    rank: int,
    world_size: int,
    num_epochs: int = 1,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
    ema_decay: Optional[float] = None,
    new_lora_out: Optional[str] = None,
    log_every: int = 10,
):
    """Data-parallel NFT training. Ranks start identical (same checkpoint+LoRA),
    each processes a DistributedSampler shard, and the ``new`` LoRA grads are
    all-reduced (averaged) before every optimizer step so ranks stay in lockstep.
    Only rank 0 writes the adapter.
    """
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler

    from lyra_2._ext.imaginaire.utils import log

    model.enforce_rl_grad()
    model.activate_rl_adapter(model.TRAINABLE_ADAPTER)
    params = [p for p in model.net.parameters() if p.requires_grad]
    assert params, "no trainable params; expected the 'new' LoRA adapter"
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=nft_collate, num_workers=2)

    step = 0
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        for batch in loader:
            out, loss = model.training_step_from_rollout(batch, step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # Average the new-adapter grads across ranks (data-parallel).
            for p in params:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            if rank == 0 and step % log_every == 0:
                log.info(
                    f"[nft-ddp] epoch {epoch} step {step} loss={float(loss):.4f} "
                    f"r_mean={float(out['nft/r_mean']):.3f}",
                    rank0_only=False,
                )
            step += 1
        model.ema_update_old(ema_decay)  # identical on every rank (grads were synced)
    if rank == 0 and new_lora_out:
        save_new_adapter(model, new_lora_out)
    dist.barrier()
    return model

def _run_ddp(args):
    import torch.distributed as dist

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)

    ds = NFTRolloutDataset(args.rewards_jsonl, adv_clip_max=args.adv_clip_max)
    model, _ = build_nft_model(
        args.checkpoint_dir, experiment=args.experiment,
        policy_lora=args.policy_lora, ref_lora=args.ref_lora, enable_fsdp=False,
    )
    out = args.new_lora_out or os.path.join(args.ckpt_dir, "nft_new_adapter_final.pt")
    train_nft_ddp(model, ds, rank=rank, world_size=world_size, num_epochs=args.epochs,
                  lr=args.lr, ema_decay=args.ema_decay, new_lora_out=out)
    if rank == 0:
        print(f"saved trained new adapter -> {out}")
    dist.destroy_process_group()

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DiffusionNFT Stage 3: train the DMD-RL new adapter")
    ap.add_argument("rewards_jsonl", help="rewards_epoch_{E}.jsonl from Stage 2")
    ap.add_argument("--ckpt-dir", default="checkpoints/nft")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--adv-clip-max", type=float, default=5.0)
    ap.add_argument("--checkpoint_dir", default="checkpoints/model")
    ap.add_argument("--experiment", default="lyra2_nft")
    ap.add_argument("--policy-lora", default=None, help="init new/old (promoted policy); default DMD")
    ap.add_argument("--ref-lora", default=None, help="init ref (KL anchor); default DMD")
    ap.add_argument("--new-lora-out", default=None, help="path to save the trained new adapter")
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--ddp", action="store_true",
                    help="8-GPU data-parallel (launch via torchrun --nproc_per_node=N)")
    ap.add_argument("--no-fsdp", action="store_true", help="single-GPU debug (no sharding)")
    args = ap.parse_args()

    if args.ddp:
        _run_ddp(args)
        return

    loader, ds = make_nft_dataloader(args.rewards_jsonl, batch_size=args.batch_size,
                                     adv_clip_max=args.adv_clip_max)
    model, _ = build_nft_model(
        args.checkpoint_dir, experiment=args.experiment,
        policy_lora=args.policy_lora, ref_lora=args.ref_lora, enable_fsdp=not args.no_fsdp,
    )
    train_nft_simple(
        model, loader, num_epochs=args.epochs, lr=args.lr, ema_decay=args.ema_decay,
        ckpt_dir=args.ckpt_dir,
    )
    out = args.new_lora_out or os.path.join(args.ckpt_dir, "nft_new_adapter_final.pt")
    save_new_adapter(model, out)
    print(f"saved trained new adapter -> {out}")

if __name__ == "__main__":
    _main()
