# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resident single-process DiffusionNFT loop: sample -> score -> train, for many steps.

One :class:`Lyra2NFTModel` stays resident on each rank, so the weights stage once and a
train step affects the next sample step directly.

The same object serves both phases. Sampling drives a ``Lyra2InferencePipeline(model=model)``
under ``adapter_ctx("old")`` and ``no_grad``. Training runs ``training_step_from_rollout``
under ``adapter_ctx("new")``, with the ``new``-adapter grads all-reduced across ranks (DDP;
``enable_fsdp=False``, which is what gets adapters into all 615 modules -- see
``build_nft_model``). ``ema_update_old`` promotes ``old <- new`` each step.

Data is scene-parallel: rank r owns ``scenes[r::world_size]`` and samples, scores and
normalizes its own scenes, so an advantage group -- the K rollouts of one scene-chunk --
never spans ranks. Train-step counts are synced to the per-rank minimum to keep the grad
all-reduce in lockstep.

Launch (8-GPU gang, one node)::

    torchrun --standalone --nproc_per_node=8 -m lyra_2._src.rl.loop.nft_loop \
        --checkpoint_dir checkpoints/model --experiment lyra2_nft \
        --scenes-root /inputs/scenes --scenes "0000 0001 ... 0031" \
        --combo reproj_rgbd --k-rollouts 4 --num-steps 200 \
        --new-lora-out /outputs/adapters --ckpt-every 10
"""

from __future__ import annotations

import fcntl
import gc
import math
import os
import shutil
import subprocess
import sys
from typing import List, Optional

import torch

def _load_full_videogpa():
    """Import ``scorers.dl3dv_videogpa`` without running ``scorers/__init__`` (which
    eagerly imports hpsv3/worldscore/... and fails in the training env). Injects a
    namespace-only ``scorers`` package rooted at the baked rewards dir so the module's
    relative imports (``from . import videogpa`` etc.) resolve, then imports it."""
    import importlib
    import types

    from lyra_2._src.rl.scoring.nft_voxel import rewards_dir
    sdir = os.path.join(rewards_dir(), "scorers")
    pkg = sys.modules.get("scorers")
    if not (isinstance(pkg, types.ModuleType) and getattr(pkg, "__path__", None)):
        pkg = types.ModuleType("scorers")
        pkg.__path__ = [sdir]  # namespace only -- do not execute scorers/__init__
        sys.modules["scorers"] = pkg
    return importlib.import_module("scorers.dl3dv_videogpa")

def _dist():
    """Return (rank, world_size, local_rank); initialize the NCCL group if launched
    under torchrun. Falls back to a single-process (1-GPU) run otherwise."""
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            # Collectives (ADV gather, grad all-reduce) sit idle while every rank runs
            # its scoring; the fastest rank blocks in the gather until the
            # slowest finishes. The 10-min NCCL default trips on that straggler spread
            # (worse across nodes at 24 ranks), so allow a generous timeout.
            from datetime import timedelta
            _to = timedelta(minutes=int(os.environ.get("NFT_NCCL_TIMEOUT_MIN", "120")))
            dist.init_process_group(backend="nccl", timeout=_to)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def _train_pass(model, dataset, params, opt, *, rank, world_size, global_step,
                max_grad_norm=1.0, log_every=10, grad_steps=1, inner_epochs=1):
    """Train ``new`` over the (pooled) dataset with gradient accumulation.

    Mirrors the DiffusionNFT reference's collect->one-big-step shape (vs the old
    one-opt-step-per-chunk): each optimizer step accumulates over a slice of the pool
    (samples x all DMD timesteps), then one grad all-reduce + step. ``r`` is fixed per
    sample (timestep-independent), so the timestep sweep only adds gradient signal.

    DDP-safety: ``n_steps`` (min local count) and ``grad_steps``/``inner_epochs`` are
    uniform across ranks, so every rank issues the same number of all_reduce collectives
    (one per param per optimizer step). OOM on a (sample,timestep) just skips that
    backward locally -- no mid-loop collective -- and a None grad is zero-filled before
    the per-step all_reduce, so ranks never desync.
    """
    import torch.distributed as dist
    from torch.utils.data import DataLoader

    from lyra_2._ext.imaginaire.utils import log
    from lyra_2._src.rl.loop.rollout_dataset import nft_collate

    n_local = len(dataset)
    if world_size > 1:
        t = torch.tensor([n_local], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_steps = int(t.item())
    else:
        n_steps = n_local
    if n_steps == 0:
        log.warning(f"[nft-loop] rank {rank}: 0 trainable samples this step; skipping train pass",
                    rank0_only=False)
        return global_step, {"n_steps": 0}

    num_t = model.num_dmd_train_timesteps()
    # Subsample DMD timesteps per sample to trade per-step gradient signal for speed
    # (NFT_TRAIN_TIMESTEPS unset/0 -> sweep all num_t). r is timestep-independent, so a
    # random subset still gives an unbiased gradient, just noisier.
    n_t_train = int(os.environ.get("NFT_TRAIN_TIMESTEPS", "0")) or num_t
    n_t_train = min(max(1, n_t_train), num_t)
    # NFT_TIMESTEP_DIST picks WHICH steps the subset draws: "uniform" (equal odds),
    # "high" (weighted toward high-noise steps), or "low" (toward low-noise steps).
    # DMD index 0 = first in nft_dmd_denoising_steps (highest noise, e.g. 1000),
    # index num_t-1 = lowest noise (e.g. 250); "high" weights descend, "low" ascends.
    ts_dist = os.environ.get("NFT_TIMESTEP_DIST", "uniform").lower()
    if ts_dist == "high":
        ts_weights = torch.arange(num_t, 0, -1, dtype=torch.float)
    elif ts_dist == "low":
        ts_weights = torch.arange(1, num_t + 1, dtype=torch.float)
    else:
        ts_weights = None
    grad_steps = max(1, min(grad_steps, n_steps))  # no more groups than samples
    group_sz = math.ceil(n_steps / grad_steps)
    _NFT_KEYS = ("nft/pos_loss", "nft/neg_loss", "nft/kl_loss", "nft/old_deviate",
                 "nft/old_kl_div", "nft/amb_rel", "nft/apb_rel", "nft/recon_err_old",
                 "nft/v_new_sq", "nft/v_ref_sq", "nft/x0_norm",
                 "nft/r_mean", "nft/r_std")
    acc = {"loss": 0.0, **{k: 0.0 for k in _NFT_KEYS}}
    vmin, vmax = float("inf"), 0.0
    n_finite = n_bad = n_oom = n_opt = 0
    acc_gnorm = 0.0

    for inner in range(max(1, inner_epochs)):
        # num_workers=0: samples carry ~1 GB cond tensors; forked workers add IPC/RAM
        # pressure. Re-shuffled each inner epoch so reuse passes aren't identical.
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=nft_collate, num_workers=0)
        it = iter(loader)
        samples = [next(it) for _ in range(n_steps)]  # this rank's slice of the pool
        groups = [samples[i:i + group_sz] for i in range(0, n_steps, group_sz)]
        # Pad to a uniform group count across ranks (already uniform: same n_steps/grad_steps).
        for group in groups:
            opt.zero_grad(set_to_none=True)
            n_micro = max(1, len(group) * n_t_train)
            scale = 1.0 / n_micro  # accumulate the mean over (samples x timesteps) in this step
            for batch in group:
                t_idxs = (range(num_t) if n_t_train >= num_t
                          else torch.multinomial(ts_weights, n_t_train).tolist()
                          if ts_weights is not None
                          else torch.randperm(num_t)[:n_t_train].tolist())
                for ti in t_idxs:
                    try:
                        out, loss = model.training_step_from_rollout(
                            batch, global_step, force_dmd_idx=ti)
                        (loss * scale).backward()  # accumulate into .grad
                    except torch.cuda.OutOfMemoryError:
                        n_oom += 1
                        gc.collect()
                        torch.cuda.empty_cache()
                        continue
                    if torch.isfinite(loss):
                        n_finite += 1
                        acc["loss"] += float(loss)
                        for k in _NFT_KEYS:
                            if k in out:
                                acc[k] += float(out[k])
                        _vsq = float(out.get("nft/v_new_sq", float("nan")))
                        if math.isfinite(_vsq):
                            vmin, vmax = min(vmin, _vsq), max(vmax, _vsq)
            # One collective per optimizer step: zero-fill missing grads so the all_reduce
            # runs uniformly on every rank even if a rank OOM'd every microstep in a group.
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
            if world_size > 1:
                for p in params:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)  # avg over ranks
            bad = any(not torch.isfinite(p.grad).all() for p in params if p.grad is not None)
            if bad:
                n_bad += 1
                for p in params:
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
            gnorm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)  # total grad norm PRE-clip
            acc_gnorm += float(gnorm)
            opt.step()
            n_opt += 1
            global_step += 1
            if rank == 0:
                log.info(f"[nft-loop] opt-step {global_step} (inner {inner}, "
                         f"{len(group)} samples x {n_t_train} t) "
                         f"loss={acc['loss'] / max(1, n_finite):.4f} "
                         f"r={acc['nft/r_mean'] / max(1, n_finite):.3f} "
                         f"v_new_sq={acc['nft/v_new_sq'] / max(1, n_finite):.3e} "
                         f"n_oom={n_oom} n_bad={n_bad}", rank0_only=False)

    denom = max(1, n_finite)
    metrics = {k: v / denom for k, v in acc.items()}
    metrics["nft/v_new_sq_min"] = (vmin if vmin != float("inf") else float("nan"))
    metrics["nft/v_new_sq_max"] = vmax
    metrics["n_steps"] = n_opt            # optimizer steps taken this collection
    metrics["n_microsteps"] = n_finite    # forward/backward passes (samples x timesteps)
    metrics["n_bad"] = n_bad
    metrics["n_oom"] = n_oom
    metrics["grad_norm"] = acc_gnorm / max(1, n_opt)  # mean total grad norm (pre-clip) over opt steps
    return global_step, metrics

def _init_wandb(args, rank: int):
    """Init Weights & Biases on rank 0 only; never let it break the run.

    Logs only if importable AND (a key is set OR WANDB_MODE is offline/disabled).
    Config/name/project/entity come from WANDB_* env (set by the jobspec)."""
    if rank != 0:
        return None
    mode = os.environ.get("WANDB_MODE", "")
    if not os.environ.get("WANDB_API_KEY") and mode not in ("offline", "disabled"):
        print("[nft-loop] no WANDB_API_KEY and WANDB_MODE not offline/disabled; wandb off", flush=True)
        return None
    try:
        import hashlib
        import wandb

        # A preempted job is resubmitted as a new the scheduler job. Without a stable id + resume,
        # wandb opens a fresh run whose x-axis restarts at 0, so the dashboard shows the run
        # "going back to step 0" even though training correctly continues at latest_ckpt+1.
        # The id is derived from the OUTPUT PREFIX, which is what actually identifies a run
        # (it is also what the checkpoint resume scans), not from the display name.
        run_key = (os.environ.get("NEW_LORA_URI") or os.environ.get("WANDB_NAME")
                   or f"{args.combo}-{args.experiment}")
        run_id = "nft" + hashlib.sha1(run_key.encode()).hexdigest()[:20]
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "lyra2-nft"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_NAME") or f"{args.combo}-{args.experiment}",
            id=run_id,
            resume="allow",          # reattach to the same run after a preemption
            config={
                "combo": args.combo, "k_rollouts": args.k_rollouts, "num_steps": args.num_steps,
                "scenes": args.scenes, "lr": args.lr, "ema_decay": args.ema_decay,
                "adv_clip_max": args.adv_clip_max, "num_frames": args.num_frames,
            },
        )
        # Plot everything against the training step, not wandb's own monotonic counter, so a
        # resumed run continues the curve at step N instead of overlaying from 0.
        try:
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
        except Exception:  # noqa: BLE001 -- older wandb without define_metric
            pass
        print(f"[nft-loop] wandb run id={run_id} (resume=allow) key={run_key}", flush=True)
        return wandb
    except Exception as e:  # noqa: BLE001 -- telemetry must never abort training
        print(f"[nft-loop] wandb init failed ({type(e).__name__}: {e}); continuing without it", flush=True)
        return None

def _global_group_advantage(local_R: dict, world_size: int, adv_clip_max: float, std_eps: float = 1e-4):
    """Rollout-parallel advantage. The K rollouts of one scene are split across ranks;
    each rank scores its shard. All-gather the per-rollout full-sequence rewards so the
    whole group (the scene's K rollouts) is z-scored together, and return ``r in [0,1]``
    for this rank's local rollouts.

    ``local_R``: {local_rollout_idx: R}. Returns (r_by_idx, stats)."""
    import math

    import torch.distributed as dist

    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, local_R)
    else:
        gathered = [local_R]
    finite = [R for d in gathered for R in d.values() if math.isfinite(R)]
    if not finite:
        return {}, {"reward_mean": float("nan"), "reward_std": float("nan"),
                    "reward_min": float("nan"), "reward_max": float("nan"),
                    "zero_std": True, "count": 0}
    mean = sum(finite) / len(finite)
    std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite))
    r_by_idx = {}
    for idx, R in local_R.items():
        if math.isfinite(R):
            adv = (R - mean) / (std + std_eps)
            adv_c = max(-adv_clip_max, min(adv_clip_max, adv))
            r_by_idx[idx] = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
    return r_by_idx, {"reward_mean": mean, "reward_std": std,
                      "reward_min": min(finite), "reward_max": max(finite),
                      "zero_std": std < std_eps, "count": len(finite)}

def _global_scene_advantage(local_R, local_metrics, scene, rank, world_size,
                            adv_clip_max, std_eps: float = 1e-4):
    """Scene-parallel advantage (DiffNFT-reference style: global gather + group-by-id).

    Different ranks may sample different scenes (a rank holds one scene's k_local
    rollouts). All-gather every rank's ``(scene, {idx: R})`` globally, group the rewards
    by SCENE, z-score within each scene's rollouts (per-scene normalization, independent
    of GPU layout -- exactly like the reference's PerPromptStatTracker over a global
    gather), and return ``r in [0,1]`` for this rank's local rollouts plus per-scene
    aggregate stats (for logging). The gradient pooling across scenes is left to the
    world grad all-reduce in _train_pass.

    ``local_R``: {local_idx: R} for this rank (all belong to ``scene``).
    ``local_metrics``: {local_idx: {name: val}} of the combo's raw metrics -- gathered and
    averaged per scene for logging only (combo-agnostic: whatever the scorers emitted)."""
    import math
    from collections import defaultdict

    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "R": local_R, "metrics": local_metrics}
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene = defaultdict(list)                            # scene -> [(rank, idx, R)]
    met_by_scene = defaultdict(lambda: defaultdict(list))   # scene -> {name: [vals]}
    for p in gathered:
        for idx, R in p["R"].items():
            by_scene[p["scene"]].append((p["rank"], idx, R))
        for idx, md in p.get("metrics", {}).items():
            for name, v in (md or {}).items():
                if v is not None and math.isfinite(v):
                    met_by_scene[p["scene"]][name].append(v)

    def _means(sc):   # per-scene mean of each raw metric present
        return {name: sum(vs) / len(vs) for name, vs in met_by_scene.get(sc, {}).items() if vs}

    r_global = {}   # (rank, idx) -> r, z-scored within each scene
    agg = {}        # scene -> stats
    for sc, entries in by_scene.items():
        finite = [R for (_, _, R) in entries if math.isfinite(R)]
        if not finite:
            agg[sc] = {"reward_mean": float("nan"), "reward_std": float("nan"),
                       "count": 0, "zero_std": True, "metrics_mean": _means(sc)}
            continue
        mean = sum(finite) / len(finite)
        std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite))
        for (rk, idx, R) in entries:
            if math.isfinite(R):
                adv = (R - mean) / (std + std_eps)
                adv_c = max(-adv_clip_max, min(adv_clip_max, adv))
                r_global[(rk, idx)] = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
        agg[sc] = {"reward_mean": mean, "reward_std": std, "count": len(finite),
                   "zero_std": std < std_eps, "metrics_mean": _means(sc)}

    r_by_idx = {idx: r_global[(rank, idx)] for idx in local_R if (rank, idx) in r_global}

    # Full per-seed records for every rollout this step (all ranks/scenes/seeds), so rank 0
    # can persist per-seed raw metrics + R + r each step (not just per-scene means). The
    # gathered payload already has each rollout's raw metrics; join with the z-scored r.
    per_seed = []
    for p in gathered:
        md_all = p.get("metrics") or {}
        for idx, R in p["R"].items():
            md = md_all.get(idx) or {}
            rec = {"scene": p["scene"], "rank": p["rank"], "rollout": idx,
                   "R": R, "r": r_global.get((p["rank"], idx))}
            rec.update({k: v for k, v in md.items()})   # the combo's raw metrics
            per_seed.append(rec)
    return r_by_idx, agg, per_seed

_VIDEO_METRICS = ("hpsv3_vid",)                     # perceptual (hpsv3) reward term
_REPROJ_METRICS = ("vggt_mse", "vggt_depth_mae")    # geometry (reproj) reward terms

def _reward_decomposition(per_seed, combo):
    """Live diagnostics for the reward's effective component split + per-component variance.

    Each metric's contribution to the reward is its signed z-score term
    ``t.sign*t.weight*(v-t.mean)/t.std``. What the per-scene advantage actually keys on is the
    WITHIN-SCENE std of that term (t.mean cancels in the advantage), so this is the *effective*
    weight -- distinct from the nominal weight. We report:
      * ``eff_wsstd_<metric>`` -- within-scene std of each metric's term (its variance-decrease
        diagnostic over training: a term whose within-scene std ->0 has gone inert).
      * ``eff_wsstd_{hpsv3,reproj}`` -- grouped (reproj = std of the summed geometry terms).
      * ``eff_share_{hpsv3,reproj}`` -- each group's fraction of the effective reward (the real
        blend ratio; e.g. reveals a nominal 0.1/0.9 that is actually ~0.27/0.73).
    Returns a wandb-loggable dict (empty if the combo has no recognized terms)."""
    import math
    import statistics as st
    from collections import defaultdict
    from lyra_2._src.rl.scoring.nft_score import COMBOS

    spec = COMBOS.get(combo, [])
    if not spec:
        return {}
    terms = {t.metric: t for t in spec}
    by_scene = defaultdict(list)
    for rec in per_seed:
        by_scene[rec["scene"]].append(rec.get("metrics") or rec)  # per_seed stores metrics inline

    def _wsstd(names):
        """Mean over scenes of the within-scene std of the summed terms in ``names``."""
        ts = [terms[m] for m in names if m in terms]
        if not ts:
            return float("nan")
        stds = []
        for recs in by_scene.values():
            vals = []
            for md in recs:
                if all(t.metric in md and md[t.metric] is not None
                       and math.isfinite(md[t.metric]) for t in ts):
                    vals.append(sum(t.sign * t.weight * (float(md[t.metric]) - t.mean) / t.std
                                    for t in ts))
            if len(vals) > 1:
                stds.append(st.pstdev(vals))
        return (sum(stds) / len(stds)) if stds else float("nan")

    out = {f"reward/eff_wsstd_{t.metric}": _wsstd((t.metric,)) for t in spec}
    vid, rep = _wsstd(_VIDEO_METRICS), _wsstd(_REPROJ_METRICS)
    if vid == vid:
        out["reward/eff_wsstd_hpsv3"] = vid
    if rep == rep:
        out["reward/eff_wsstd_reproj"] = rep
    if vid == vid and rep == rep and (vid + rep) > 0:
        out["reward/eff_share_hpsv3"] = vid / (vid + rep)
        out["reward/eff_share_reproj"] = rep / (vid + rep)
    return out

def _global_perframe_r(local_Rpf, scene, rank, world_size, adv_clip_max, w_latent, std_eps=1e-4):
    """Windowed per-latent-frame advantage via a global gather (densified reward).

    ``local_Rpf``: {rollout_idx: [per-latent combo reward]} for this rank's rollouts.
    Partitions each scene's generated latent frames into windows of ``w_latent``; for each
    window position, averages each rollout's per-latent reward over the window (skipping
    NaN frames -- e.g. early frames without a k30 partner), z-scores those aggregates over
    the K rollouts of the scene (across ranks), and maps to r in [0,1]. Every frame in a
    window gets that rollout's window r. Degenerate windows (<2 finite rollouts or ~0 std)
    -> r=0.5. Returns ({rollout_idx: [per-latent r]} for this rank, mean_window_std)."""
    import math
    from collections import defaultdict

    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "Rpf": local_Rpf}
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene = defaultdict(list)  # scene -> [(rank, idx, rpf_list)]
    for p in gathered:
        for idx, rpf in (p.get("Rpf") or {}).items():
            by_scene[p["scene"]].append((p["rank"], int(idx), rpf))

    r_global = {}   # (rank, idx) -> [per-latent r]
    win_stds = []
    for sc, entries in by_scene.items():
        lens = [len(rpf) for (_, _, rpf) in entries if rpf]
        if not lens:
            continue
        L = min(lens)
        for (rk, idx, _rpf) in entries:
            r_global[(rk, idx)] = [0.5] * L        # neutral default
        for ws in range(0, L, w_latent):
            we = min(ws + w_latent, L)
            aggs = []
            for (_rk, _idx, rpf) in entries:
                seg = [rpf[f] for f in range(ws, we)
                       if rpf and rpf[f] is not None and math.isfinite(rpf[f])] if rpf else []
                aggs.append(sum(seg) / len(seg) if seg else float("nan"))
            finite = [a for a in aggs if math.isfinite(a)]
            if len(finite) < 2:
                continue                            # can't normalize -> leave neutral
            mean = sum(finite) / len(finite)
            std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite))
            win_stds.append(std)
            if std < std_eps:
                continue                            # degenerate window -> leave neutral
            for (rk, idx, _rpf), a in zip(entries, aggs):
                if not math.isfinite(a):
                    continue
                adv_c = max(-adv_clip_max, min(adv_clip_max, (a - mean) / (std + std_eps)))
                rval = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
                for f in range(ws, we):
                    r_global[(rk, idx)][f] = rval

    r_by_idx = {idx: r_global[(rank, int(idx))]
                for idx in (local_Rpf or {}) if (rank, int(idx)) in r_global}
    mean_win_std = (sum(win_stds) / len(win_stds)) if win_stds else float("nan")
    return r_by_idx, mean_win_std

def _global_meanstd(local_d: dict, world_size: int):
    """(mean, std) of finite values in ``local_d`` gathered across ranks. The gathered
    values are one group's rollouts (one scene/step), so std is the within-group spread."""
    import math
    import statistics

    import torch.distributed as dist

    if world_size > 1:
        g = [None] * world_size
        dist.all_gather_object(g, local_d)
    else:
        g = [local_d]
    vals = [v for d in g for v in d.values() if v is not None and math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    return sum(vals) / len(vals), (statistics.pstdev(vals) if len(vals) > 1 else 0.0)

def _stage_vggt_checkpoint() -> None:
    """Put the VGGT-Omega checkpoint on local disk before anything scores.

    The scorers take ``VGGT_CHECKPOINT`` as a local path and, when the file is missing,
    fall back to ``download_vggt_checkpoint``, which shells out to the AWS CLI -- absent from
    this image, so every rollout scores -inf. The entrypoint stages the file, but a preemption
    restart wipes /tmp and the only re-stage lived in the val branch, so a restart between vals
    leaves every step until the next val scoring nothing.

    Downloads via gcs_util (boto3). Ranks share a host's /tmp, so the
    flock keeps it to one download per pod and the rename keeps a torn file from being loaded.
    """
    from lyra_2._ext.imaginaire.utils import log

    dst = os.environ.get("VGGT_CHECKPOINT")
    src = os.environ.get("VGGT_CKPT_URI")
    if not dst or not src:
        return
    cached = os.path.join("/tmp/vggt_omega", os.path.basename(dst))
    if os.path.exists(dst) and os.path.exists(cached):
        return
    from lyra_2._src.rl.data.gcs_util import download_file

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    os.makedirs("/tmp/vggt_omega", exist_ok=True)
    with open("/tmp/vggt_omega/.stage.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.exists(dst):
            tmp = f"{dst}.tmp"
            download_file(src, tmp)
            os.replace(tmp, dst)
        # download_vggt_checkpoint resolves through this cache dir, so seed it too -- training
        # must not depend on val having run. Same copy the val branch does.
        if not os.path.exists(cached):
            shutil.copy(dst, f"{cached}.tmp")
            os.replace(f"{cached}.tmp", cached)
    log.info(f"[nft-loop] staged VGGT checkpoint {src} -> {dst} (+/tmp/vggt_omega)",
             rank0_only=False)

def run_resident_loop(args) -> None:
    """Build the resident model once and loop sample->score->train for ``num_steps``."""
    import json

    import torch.distributed as dist

    from lyra_2._ext.imaginaire.utils import log
    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._src.rl.loop.sampler import RolloutStoreWriter, run_nft_sampling
    from lyra_2._src.rl.scoring.nft_score import score_epoch
    from lyra_2._src.rl.inference.sample import _build_args, build_scene_data_batch
    from lyra_2._src.rl.loop.nft_train import build_nft_model, load_new_adapter, save_new_adapter
    from lyra_2._src.rl.loop.rollout_dataset import NFTRolloutDataset

    rank, world_size, local_rank = _dist()
    dev = f"cuda:{local_rank}"
    _stage_vggt_checkpoint()

    # Resident model: enable_fsdp=False so adapter injection covers all 615 modules
    # and the per-rank replica matches the DDP train path.
    model, _ = build_nft_model(
        args.checkpoint_dir, experiment=args.experiment,
        policy_lora=args.policy_lora, ref_lora=args.ref_lora, enable_fsdp=False,
    )
    model.enforce_rl_grad()

    # --- RESUME: continue from the latest trained checkpoint in NEW_LORA_URI so an
    # opportunistic preemption (which restarts the job) doesn't reset to step 0. The
    # build above keeps old/ref = DMD (the frozen sampler under EMA-off); here we
    # overwrite only `new` with the latest checkpoint and skip the steps already done.
    start_step = 0
    resume_opt_sd = None      # optimizer state to restore after opt is built (below)
    resume_global_step = 0
    if args.new_lora_uri:
        import re

        # -1 = no checkpoints found (a genuine fresh start); -2 = the scan itself FAILED.
        # These must not be conflated: a transient list_keys error used to fall through to
        # "start fresh", which silently restarts a 50-step run at step 0 AND overwrites its
        # checkpoints/seed_metrics -- indistinguishable from a real fresh start in the logs.
        latest_n = -1
        if rank == 0:
            try:
                from lyra_2._src.rl.data.gcs_util import list_keys

                for key in list_keys(args.new_lora_uri):
                    m = re.search(r"nft_new_step(\d+)\.pt$", key)
                    if m:
                        latest_n = max(latest_n, int(m.group(1)))
            except Exception as e:  # noqa: BLE001 -- signalled to all ranks below, not swallowed
                log.error(f"[nft-loop] resume scan FAILED ({type(e).__name__}: {e})", rank0_only=False)
                latest_n = -2
        if world_size > 1:
            # broadcast the sentinel too, so every rank raises together instead of rank 0
            # dying alone and the rest hanging on the next collective
            t = torch.tensor([latest_n], device=dev)
            dist.broadcast(t, src=0)
            latest_n = int(t.item())
        if latest_n == -2:
            if os.environ.get("NFT_ALLOW_FRESH_START", "").strip() in ("1", "true", "True"):
                log.warning("[nft-loop] resume scan failed but NFT_ALLOW_FRESH_START=1; starting at step 0",
                            rank0_only=False)
                latest_n = -1
            else:
                raise RuntimeError(
                    f"resume scan of {args.new_lora_uri} failed. Refusing to start at step 0: if this "
                    "run already has checkpoints, a fresh start would overwrite them. Fix the object-store "
                    "access, or set NFT_ALLOW_FRESH_START=1 to start over deliberately.")
        if latest_n >= 0:
            from lyra_2._src.rl.data.gcs_util import download_file

            ckpt_uri = f"{args.new_lora_uri.rstrip('/')}/nft_new_step{latest_n:04d}.pt"
            local_ckpt = f"/tmp/resume_step{latest_n:04d}.pt"
            download_file(ckpt_uri, local_ckpt)
            n_loaded = load_new_adapter(model, local_ckpt)
            start_step = latest_n + 1
            # Restore the EMA `old` adapter + optimizer state for a clean resume. TIER 1:
            # a full trainstate file (saved alongside newer checkpoints) -> restore `old`
            # and Adam exactly. TIER 2 (no trainstate, e.g. an older new-only checkpoint):
            # set `old <- new` (ema_update_old(0.0)) -- collapses the new/old gap but avoids
            # the `old = DMD` discontinuity of a naive reload; optimizer restarts fresh.
            ts_uri = f"{args.new_lora_uri.rstrip('/')}/nft_trainstate_step{latest_n:04d}.pt"
            restored = "none"
            try:
                from lyra_2._src.models.lyra2_nft_model import _local
                local_ts = f"/tmp/resume_trainstate_{latest_n:04d}.pt"
                download_file(ts_uri, local_ts)
                ts = torch.load(local_ts, map_location="cpu", weights_only=False)
                old_sd = ts.get("old") or {}
                with torch.no_grad():
                    for pn, p in model.net.named_parameters():
                        if pn in old_sd:
                            _local(p).copy_(old_sd[pn].to(device=_local(p).device, dtype=_local(p).dtype))
                resume_opt_sd = ts.get("opt")             # applied after opt is built
                resume_global_step = int(ts.get("global_step", 0))
                restored = f"EXACT (old={len(old_sd)} params, opt={'yes' if resume_opt_sd else 'no'})"
            except Exception as e:  # noqa: BLE001 -- no trainstate -> old<-new fallback
                model.ema_update_old(0.0)                 # old <- new (avoid old=DMD)
                restored = f"old<-new fallback (no trainstate: {type(e).__name__})"
            log.info(f"[nft-loop] RESUME from {ckpt_uri}: loaded {n_loaded} 'new' params; "
                     f"start_step={start_step}; {restored}", rank0_only=False)

    params = [p for p in model.net.parameters() if p.requires_grad]
    assert params, "no trainable params; expected the 'new' LoRA adapter"
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    if resume_opt_sd is not None:  # exact optimizer resume (Adam moments) from trainstate
        try:
            opt.load_state_dict(resume_opt_sd)
            log.info("[nft-loop] RESUME: restored optimizer state", rank0_only=False)
        except Exception as e:  # noqa: BLE001 -- shape/param mismatch -> keep fresh opt
            log.warning(f"[nft-loop] optimizer state restore failed ({e}); fresh opt", rank0_only=False)

    inf_args = _build_args(args.checkpoint_dir, args.num_frames, args.resolution,
                           args.pose_scale, args.guidance, args.shift, args.base_seed)
    # The model is resident on GPU for training; don't let the inference pipeline
    # offload net<->CPU (it would leave the model off-GPU for the train phase).
    # no_grad sampling has no autograd graph, so it fits alongside the resident model.
    inf_args.offload = False
    da3_model = load_da3_model(da3_model_name=inf_args.da3_model_name,
                               da3_model_path_custom=inf_args.da3_model_path_custom, device=dev)
    da3_model.eval()
    neg_t5 = torch.load("checkpoints/text_encoder/negative_prompt.pt", map_location="cpu",
                        weights_only=False)["t5_text_embeddings"]

    wb = _init_wandb(args, rank)

    all_scenes = args.scenes.split()
    # Scene-parallel (DiffNFT-reference style): partition the world into P scene-groups
    # (--scenes-parallel); each group of R = world_size//P ranks samples one scene's K
    # rollouts (k_local = K//R per rank). All P scenes sample CONCURRENTLY. The advantage
    # is z-scored per scene via a GLOBAL gather + group-by-scene-id
    # (_global_scene_advantage); the gradient pools across the P scenes via the world
    # grad all-reduce in _train_pass. P=1 -> all ranks one scene (legacy).
    P = max(1, args.scenes_parallel)
    assert world_size % P == 0, f"world_size={world_size} not divisible by --scenes-parallel={P}"
    ranks_per_scene = world_size // P
    my_group = rank // ranks_per_scene
    assert args.k_rollouts % ranks_per_scene == 0 and args.k_rollouts // ranks_per_scene >= 1, \
        f"K={args.k_rollouts} must be a positive multiple of ranks_per_scene={ranks_per_scene}"
    k_local = args.k_rollouts // ranks_per_scene
    log.info(f"[nft-loop] rank {rank}/{world_size}: P={P} scene-groups x {ranks_per_scene} ranks "
             f"(group {my_group}); K={args.k_rollouts} -> {k_local} rollouts/rank; "
             f"{len(all_scenes)} scenes, {P}/step", rank0_only=False)

    from lyra_2._src.rl.scoring.nft_score import COMBOS as _COMBOS
    from lyra_2._src.rl.scoring.nft_score import combo_for_step as _combo_for_step
    from lyra_2._src.rl.scoring.nft_score import parse_combo_schedule as _parse_sched
    from lyra_2._src.rl.scoring.nft_score import describe_combo_schedule as _describe_sched
    # Validate NFT_COMBO_SCHEDULE up front: a typo'd combo name must fail at startup, not
    # 7 steps in.
    _sched_combos = [args.combo]
    if os.environ.get("NFT_COMBO_SCHEDULE", "").strip():
        _sched_combos += [n for n, _ in _parse_sched(os.environ["NFT_COMBO_SCHEDULE"])]
        log.info(f"[nft-loop] reward schedule: {_describe_sched()}  (val stays on {args.combo})")
    for _c in _sched_combos:
        if _c not in _COMBOS:
            raise SystemExit(f"unknown combo {_c!r}; known: {sorted(_COMBOS)}")
    # the camera RPE term needs the conditioning trajectory; the scorer only receives the
    # scene id, so hand it the staging root through the environment.
    os.environ["NFT_SCENES_ROOT"] = str(args.scenes_root)
    # Scene data_batches are built LAZILY behind a small LRU cache. Each batch carries
    # ~300 MB of (repeated) depth, so precomputing all scenes OOMs once the scene set is
    # large (e.g. 1773 scenes -> ~500 GB). The caption/T5 is identical across scenes (one
    # prompt), so compute it once, free the umt5 encoder (~5-11 GB), and reuse it for every
    # lazily-built scene (build then needs only DA3 + image + trajectory, both fast).
    from collections import OrderedDict

    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding
    target_hw = tuple(int(x) for x in args.resolution.split(","))
    _t5_cached = get_umt5_embedding(args.prompt, device=dev).detach()
    import lyra_2._src.inference.get_t5_emb as _t5mod
    if _t5mod.t5_encoder is not None:
        _t5mod.t5_encoder.model.to("cpu")
        _t5mod.t5_encoder = None
    torch.cuda.empty_cache()

    _scene_cache: "OrderedDict[str, dict]" = OrderedDict()
    _SCENE_CACHE_CAP = max(4, int(getattr(args, "scene_cache_cap", 8)))

    def scene_batch(sc):
        """Lazily build (or LRU-fetch) scene ``sc``'s data_batch; reuses the cached T5."""
        b = _scene_cache.get(sc)
        if b is not None:
            _scene_cache.move_to_end(sc)
            return b
        b = build_scene_data_batch(
            model, da3_model,
            image_path=os.path.join(args.scenes_root, sc, "image.png"),
            traj_file=os.path.join(args.scenes_root, sc, "lyra2_traj.npz"),
            caption=args.prompt, neg_t5=neg_t5, num_frames=int(args.num_frames),
            target_hw=target_hw, pose_scale=float(args.pose_scale), t5_override=_t5_cached)
        _scene_cache[sc] = b
        _scene_cache.move_to_end(sc)
        while len(_scene_cache) > _SCENE_CACHE_CAP:
            _scene_cache.popitem(last=False)
        torch.cuda.empty_cache()
        return b

    # ---- TEST-SET VALIDATION (held-out scenes; no training) ----------------------------
    # Every --val-every steps, sample the fixed val set under the trainable ("new") policy
    # with FIXED seeds (independent of step, so comparable across steps), score with the same
    # same combo, log val/* means. CRITICAL: distribute (scene, seed) tasks FLAT across all
    # ranks (not leader-only) so validation load is BALANCED -- otherwise idle ranks block at
    # the all_gather_object while a few busy ranks sample for minutes, tripping the NCCL
    # watchdog. No gradient, no EMA.
    import torch.distributed as dist
    val_scenes = args.val_scenes.split() if getattr(args, "val_scenes", "") else []
    val_k = max(1, int(getattr(args, "val_k", 0)) or k_local)
    val_tasks = [(sc, j) for sc in val_scenes for j in range(val_k)]  # scene x seed
    my_tasks = val_tasks[rank::world_size]                             # ~equal per rank
    # Extra raw metrics to compute during validation, beyond the ones the combo needs.
    # Computed on the same accumulated full clip score_epoch reconstructs for the combo.
    val_extra = {x for x in (getattr(args, "val_metrics", "") or "").replace(" ", "").split(",") if x}

    def run_validation(step, wb):
        import json as _json
        import math as _m
        from collections import defaultdict as _dd
        # Val scores with the global 16-frame reproj decode (comparable across windowed/non-windowed
        # runs, and faster) -- see nft_score.py NFT_EVAL_GLOBAL_DEPTH. Training reward is unaffected.
        _prev_eval_global = os.environ.get("NFT_EVAL_GLOBAL_DEPTH")
        os.environ["NFT_EVAL_GLOBAL_DEPTH"] = "1"
        recs = []
        if my_tasks:
            model.eval()
            da3_model.to(dev)
            _vae2 = getattr(getattr(model.tokenizer, "model", None), "model", None)
            if _vae2 is not None:
                _vae2.to(dev)
            roots = []
            with torch.no_grad(), model.adapter_ctx(model.TRAINABLE_ADAPTER):
                for (vsc, j) in my_tasks:
                    rr = os.path.join(args.work_dir, f"val_{step:04d}", f"rank_{rank:03d}",
                                      f"{vsc}_s{j}", "rollout")
                    os.makedirs(rr, exist_ok=True)
                    writer = RolloutStoreWriter(rr, append=False)
                    vseed = args.val_base_seed + j * 7919  # fixed across steps; distinct per seed idx
                    db = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in scene_batch(vsc).items()}
                    run_nft_sampling(model, db, inf_args, writer, vsc, k_rollouts=1,
                                     base_seed=vseed, da3_model=da3_model)
                    roots.append((vsc, j, rr))
            da3_model.to("cpu")
            if _vae2 is not None:
                _vae2.to("cpu")
            model.net.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            # Optional full VideoGPA suite (DA3 + VGGT-Omega + LightGlue) on each val
            # rollout, run here while model.net is on CPU (GPU free). Init once per val
            # step and free after: the models cache on disk so re-init is cheap, and it
            # keeps no GPU memory resident across training steps.
            vgpa = vgpa_ctx = None
            if getattr(args, "val_full_videogpa", False):
                try:
                    import shutil as _shutil
                    import types as _types
                    vgpa = _load_full_videogpa()
                    # The scorer's download_vggt_checkpoint resolves local paths and s3:// URIs,
                    # but the training image may have no AWS CLI. The entrypoint already staged the
                    # checkpoint locally (VGGT_CHECKPOINT), so pre-seed the scorer's cache
                    # dir with it -> download_vggt_checkpoint sees it exists and skips the copy.
                    _bv = os.environ.get("VGGT_CHECKPOINT")
                    _vck = _bv or os.environ.get("VGGT_CKPT_URI")
                    if _bv and os.path.exists(_bv):
                        os.makedirs("/tmp/vggt_omega", exist_ok=True)
                        _cached = os.path.join("/tmp/vggt_omega", os.path.basename(_bv))
                        if not os.path.exists(_cached):
                            _shutil.copy(_bv, _cached)
                    vgpa_ctx = vgpa._init(_types.SimpleNamespace(
                        device=dev, vggt_checkpoint=_vck or None))
                except Exception as e:  # noqa: BLE001 -- full-videogpa must not abort val
                    log.warning(f"[nft-loop] full-videogpa init failed: {e}", rank0_only=False)
                    vgpa = vgpa_ctx = None
            from lyra_2._src.rl.loop.sampler import save_rollout_videos
            for (vsc, j, rr) in roots:
                rj = os.path.join(os.path.dirname(rr), "rewards.jsonl")
                # Unique stage key per (step, rank, seed-idx) -- else the resume-safe
                # workers reuse an earlier round's cached recon (fixed val seeds stage to a
                # step-independent path), freezing val at the step-0 baseline. 900M offset
                # keeps val keys clear of training's (step*100000 + rank*1000).
                score_epoch(rr, rj, combo_name=args.combo, gpu_id=local_rank,
                            extra_metrics=val_extra)
                R, raw = None, {}
                if os.path.exists(rj):
                    for line in open(rj):
                        r = _json.loads(line)
                        R, raw = float(r["R"]), dict(r.get("metrics") or {})
                        break  # k_rollouts=1 -> one full-seq record per store
                rec = {"scene": vsc, "seed": j, "R": R, "metrics": raw}
                recs.append(rec)
                # save the val video (each (scene,seed) task is globally unique -> no collision)
                if args.videos_uri:
                    vbase = f"{args.videos_uri.rstrip('/')}/val_step{step:04d}/scene_{vsc}_seed{j:02d}"
                    try:
                        save_rollout_videos(rr, vsc, vbase)
                    except Exception as e:  # noqa: BLE001 -- video saving must not abort val
                        log.warning(f"[nft-loop] val video save failed {vsc} seed{j}: {e}", rank0_only=False)
                # Full VideoGPA metrics on the local rollout mp4 save_rollout_videos just
                # encoded (da3_/vggt_ psnr/ssim/lpips/mvcs/consistency/mse + shared epipolar).
                # Merged raw so the combo-agnostic aggregation below logs them as val/<metric>.
                if vgpa is not None:
                    import glob as _glob
                    mp4s = sorted(_glob.glob(os.path.join(rr, f"scene_{vsc}", "rollout_*.mp4")))
                    if mp4s:
                        try:
                            rec["metrics"].update(vgpa.score_video(vgpa_ctx, mp4s[0]))
                        except Exception as e:  # noqa: BLE001 -- full-videogpa must not abort val
                            log.warning(f"[nft-loop] full-videogpa score failed {vsc} seed{j}: {e}",
                                        rank0_only=False)
            if vgpa_ctx is not None:
                del vgpa, vgpa_ctx
                gc.collect()
                torch.cuda.empty_cache()
            model.net.to(dev)
            torch.cuda.empty_cache()
            shutil.rmtree(os.path.join(args.work_dir, f"val_{step:04d}", f"rank_{rank:03d}"),
                          ignore_errors=True)
        if world_size > 1:
            g = [None] * world_size
            dist.all_gather_object(g, recs)
            allrecs = [x for sub in g for x in (sub or [])]
        else:
            allrecs = recs
        if rank == 0:
            # Combo-agnostic: aggregate the reward + whatever raw metrics the combo emitted
            # (whatever the active scorers emitted).
            per = _dd(list)                    # scene -> [R]
            permet = _dd(lambda: _dd(list))    # metric -> scene -> [value]
            for r in allrecs:
                if r["R"] is not None and _m.isfinite(r["R"]):
                    per[r["scene"]].append(r["R"])
                for k, v in (r.get("metrics") or {}).items():
                    if v is not None and _m.isfinite(v):
                        permet[k][r["scene"]].append(v)
            flat = lambda d: [v for vs in d.values() for v in vs]
            mean = lambda x: (sum(x) / len(x)) if x else float("nan")
            allR = flat(per)
            met = {k: mean(flat(sc)) for k, sc in permet.items()}
            raw_csv = " ".join(f"{k}={met[k]:.4f}" for k in sorted(met))
            log.info(f"[nft-loop] VAL step {step}: {args.combo}={mean(allR):.4f} {raw_csv} "
                     f"over {len(per)} test scenes ({len(allR)} rollouts)", rank0_only=False)
            if wb is not None:
                wb.log({f"val/{args.combo}_mean": mean(allR), "val/n_scenes": len(per),
                        "val/step": step,
                        **{f"val/{k}": v for k, v in met.items()},
                        **{f"val_reward_scene/{s}": mean(v) for s, v in per.items()}})
            # Persist the full per-scene-per-seed val rewards next to the val videos, so each
            # saved val clip (scene_<sc>_seed<j>) is reward-identifiable.
            if args.new_lora_uri:
                try:
                    from lyra_2._src.rl.data.gcs_util import upload_file
                    vf = os.path.join(args.work_dir, f"val_seed_metrics_step{step:04d}.jsonl")
                    with open(vf, "w") as fh:
                        for r in sorted(allrecs, key=lambda x: (x["scene"], x.get("seed", 0))):
                            fh.write(_json.dumps({"step": step, **r}) + "\n")
                    upload_file(vf, f"{args.new_lora_uri.rstrip('/')}/val_seed_metrics/step{step:04d}.jsonl")
                except Exception as e:  # noqa: BLE001 -- best-effort
                    log.warning(f"[nft-loop] val per-seed metrics upload failed: {e}", rank0_only=False)
        # restore the training decode mode (all ranks) so training scoring stays windowed
        if _prev_eval_global is None:
            os.environ.pop("NFT_EVAL_GLOBAL_DEPTH", None)
        else:
            os.environ["NFT_EVAL_GLOBAL_DEPTH"] = _prev_eval_global

    global_step = resume_global_step  # continue the opt-step counter across a resume
    for step in range(start_step, args.num_steps):
        # This step's reward. NFT_COMBO_SCHEDULE cycles combos across steps; with it unset
        # every step uses --combo. A pure function of the absolute step index, so every rank
        # scores the same reward without communicating.
        step_combo = _combo_for_step(step, args.combo)
        if val_scenes and args.val_every > 0 and step % args.val_every == 0:
            run_validation(step, wb)
            if world_size > 1:
                dist.barrier()
        # ---- COLLECT: this rank samples ITS group's scene (P scenes in parallel). ----
        model.eval()
        step_dir = os.path.join(args.work_dir, f"step_{step:04d}")
        scene = all_scenes[(step * P + my_group) % len(all_scenes)]
        step_root = os.path.join(step_dir, f"rank_{rank:03d}")
        rollout_root = os.path.join(step_root, "rollout")
        rewards_jsonl = os.path.join(step_root, "rewards.jsonl")
        os.makedirs(rollout_root, exist_ok=True)

        # SAMPLE this rank's k_local rollouts of its scene under "old" (frozen if decay=1).
        da3_model.to(dev)
        _vae = getattr(getattr(model.tokenizer, "model", None), "model", None)
        if _vae is not None:
            _vae.to(dev)
        writer = RolloutStoreWriter(rollout_root, append=False)
        base_seed = args.base_seed + step * 100000 + rank * 1000  # rank-distinct rollouts
        with model.adapter_ctx("old"):
            db = {k: (v.clone() if torch.is_tensor(v) else v)
                  for k, v in scene_batch(scene).items()}
            run_nft_sampling(model, db, inf_args, writer, scene,
                             k_rollouts=k_local, base_seed=base_seed, da3_model=da3_model)
        da3_model.to("cpu")
        if _vae is not None:
            _vae.to("cpu")

        # SCORE. Offload the resident net to host RAM so the reward scorers get the GPU.
        # VGGT fits alongside the resident net, but HPSv3 (Qwen2-VL-7B, ~33 GiB)
        # + the ~46 GiB resident process OOMs GPU 0 (rank 0 is the heaviest process). The
        # net isn't used during scoring; reload it before _train_pass. (768Gi host RAM holds
        # all ranks' offloaded nets -- this is what the jobspec's 768Gi is for.)
        model.net.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        score_epoch(rollout_root, rewards_jsonl, combo_name=step_combo, gpu_id=local_rank)
        model.net.to(dev)
        torch.cuda.empty_cache()

        # ADVANTAGE: GLOBAL gather + group-by-scene-id -> per-scene z-scored r for this
        # rank's rollouts (+ per-scene agg stats, available on every rank). The gradient
        # pools across the P scenes via the world all-reduce inside _train_pass.
        local_R, local_metrics, local_Rpf, local_Vp = {}, {}, {}, {}
        if os.path.exists(rewards_jsonl):
            import math as _math
            for line in open(rewards_jsonl):
                rec = json.loads(line)
                local_R[rec["rollout"]] = float(rec["R"])  # identical full-seq R per rollout
                m = rec.get("metrics") or {}
                # combo-agnostic raw metrics, whatever the scorers emitted; logging only.
                local_metrics[rec["rollout"]] = {
                    k: float(v) for k, v in m.items()
                    if v is not None and _math.isfinite(float(v))}
                rpf = rec.get("R_perframe")
                if rpf is not None:
                    local_Rpf[rec["rollout"]] = rpf  # per-latent combo (densified reward)
                vp = rec.get("voxel_payload")
                if vp is not None:
                    local_Vp[rec["rollout"]] = vp    # per-voxel tables (spatial reward)
        r_by_idx, agg, per_seed = _global_scene_advantage(
            local_R, local_metrics, scene, rank, world_size, args.adv_clip_max)

        # Densified windowed per-frame reward: NFT_REWARD_WINDOW is the window size in PIXEL
        # frames (unset/'full'/'0' -> global scalar r = today). w_latent = window / frames_per_latent.
        # Voxel mode (NFT_REWARD_VOXEL) supersedes it: r is [T][gh][gw] per rollout, but slices
        # per chunk along T exactly like the per-frame list, so everything downstream is shared.
        r_perframe_by_idx, r_window_std = {}, float("nan")
        r_voxel_matched = float("nan")
        _win_env = os.environ.get("NFT_REWARD_WINDOW", "full").strip().lower()
        from lyra_2._src.rl.scoring import nft_voxel as _nv
        if _nv.voxel_enabled() and local_Vp:
            from lyra_2._src.rl.scoring.nft_score import COMBOS as _COMBOS
            r_perframe_by_idx, _vdiag = _nv.global_pervoxel_r(
                local_Vp, scene, rank, world_size, args.adv_clip_max, _COMBOS[step_combo])
            r_window_std = _vdiag.get("voxel_std", float("nan"))
            r_voxel_matched = _vdiag.get("matched_frac", float("nan"))
            # NFT_REWARD_MIX: weight on the voxel reward in a convex blend with the global
            # (whole-rollout scalar) one, applied only on steps whose combo carries geometry
            # terms -- the hpsv3_only and camera_only phases have no 3D-consistency signal.
            #   r_cell = w * r_voxel_cell + (1 - w) * r_global_rollout
            # Both sides are rewards in [0,1] off the same clipped-z map, so the combination is
            # well-posed: w=1 is pure voxel, w=0 reproduces the global arm cell-for-cell.
            _mix = float(os.environ.get("NFT_REWARD_MIX", "0.5") or 0.5)
            if _mix < 1 and any(t.metric in _nv._GEOM_METRICS for t in _COMBOS[step_combo]):
                for _idx, _arr in list(r_perframe_by_idx.items()):
                    _g = r_by_idx.get(_idx)
                    if _g is None:
                        continue
                    # Unmatched cells already carry the neutral 0.5 fill from _paint, so they
                    # simply inherit the global reward's pull rather than staying neutral.
                    r_perframe_by_idx[_idx] = _mix * _arr + (1.0 - _mix) * float(_g)
        elif _win_env not in ("", "full", "0") and local_Rpf:
            _fpl = int(getattr(model, "framepack_num_frames_per_latent", 4) or 4)
            _w_lat = max(1, round(int(_win_env) / _fpl))
            r_perframe_by_idx, r_window_std = _global_perframe_r(
                local_Rpf, scene, rank, world_size, args.adv_clip_max, _w_lat)

        # Persist every seed's per-rollout metrics (mae_k1/mae_k30 + R + r) for this step so
        # the full per-seed reward trajectory is recoverable (not just per-scene means /
        # leader-only video metrics). Rank 0 has the global gather; one small jsonl per step.
        if rank == 0 and args.new_lora_uri and per_seed:
            try:
                import json as _json
                from lyra_2._src.rl.data.gcs_util import upload_file
                sf = os.path.join(args.work_dir, f"seed_metrics_step{step:04d}.jsonl")
                with open(sf, "w") as _fh:
                    for rec in sorted(per_seed, key=lambda r: (r["scene"], r["rank"], r["rollout"])):
                        _fh.write(_json.dumps({"step": step, **rec}) + "\n")
                upload_file(sf, f"{args.new_lora_uri.rstrip('/')}/seed_metrics/step{step:04d}.jsonl")
            except Exception as e:  # noqa: BLE001 -- best-effort; must not crash the loop
                log.warning(f"[nft-loop] per-seed metrics upload failed: {e}", rank0_only=False)

        # ---- TRAIN: one accumulated update over this rank's scene's samples x all DMD
        # timesteps; the world grad all-reduce pools across the P scene-groups. ----
        model.train()
        model.activate_rl_adapter(model.TRAINABLE_ADAPTER)
        ds = NFTRolloutDataset(rewards_jsonl, adv_clip_max=args.adv_clip_max)
        ds.records = [rec for rec in ds.records if rec["rollout"] in r_by_idx]
        # Assign r per sample. Windowed reward: r_perframe_by_idx[rollout] is the WHOLE-rollout
        # per-latent r (all 12 windows, z-scored over the K rollouts); slice it per CHUNK so each
        # chunk sample gets its own windows (chunk c -> latents [c*per:(c+1)*per]) instead of the
        # whole-rollout vector. Global reward (window off): scalar per rollout, same for all chunks.
        _by_roll = {}
        for rec in ds.records:
            _by_roll.setdefault(rec["rollout"], []).append(rec)
        for _roll, _recs in _by_roll.items():
            _rpf_full = r_perframe_by_idx.get(_roll)
            if _rpf_full is None or len(_rpf_full) == 0:
                for rec in _recs:
                    rec["r"] = r_by_idx[_roll]
                continue
            _per = max(1, len(_rpf_full) // max(1, len(_recs)))   # per-chunk latent count
            for rec in _recs:
                _c = int(rec.get("chunk", 0))
                _seg = _rpf_full[_c * _per:(_c + 1) * _per]
                rec["r"] = _seg if len(_seg) else r_by_idx[_roll]  # fallback if slice empty
        global_step, tm = _train_pass(
            model, ds, params, opt, rank=rank, world_size=world_size,
            global_step=global_step, max_grad_norm=args.max_grad_norm,
            grad_steps=args.grad_steps_per_collection, inner_epochs=args.inner_epochs,
        )
        model.ema_update_old(args.ema_decay)
        if world_size > 1:
            dist.barrier()

        # Fail fast when a whole round scored nothing: every scene's reward_mean is NaN because
        # the scorer returned no metrics at all (e.g. the VGGT checkpoint missing from local
        # disk sends it down a CLI copy path this image may lack). Everything downstream tolerates
        # it -- scenes without finite rewards are skipped, the loss guard drops the step -- so
        # the loop advances and writes checkpoints while learning nothing, which is how a run
        # burned steps 31-34 unnoticed. Raised on every rank so the collectives stay aligned.
        if agg and not any(math.isfinite(agg[sc]["reward_mean"]) for sc in agg):
            if os.environ.get("NFT_ALLOW_DEAD_STEPS") != "1":
                raise RuntimeError(
                    f"[nft-loop] step {step}: 0/{len(agg)} scenes produced a finite reward; the "
                    f"scorer emitted no metrics. Refusing to train on a dead step "
                    f"(NFT_ALLOW_DEAD_STEPS=1 overrides)."
                )
            log.warning(f"[nft-loop] step {step}: no finite rewards, training through it "
                        f"(NFT_ALLOW_DEAD_STEPS=1)", rank0_only=False)

        if rank == 0:
            # agg (from the global gather) holds per-scene stats for all P scenes.
            scs = sorted(agg.keys())
            rmeans = [agg[sc]["reward_mean"] for sc in scs]
            coll_reward = sum(rmeans) / len(rmeans) if rmeans else float("nan")
            # combo-agnostic raw-metric means over the P scenes.
            metric_names = sorted({n for sc in scs for n in agg[sc]["metrics_mean"]})
            raw_means = {}
            for n in metric_names:
                vals = [agg[sc]["metrics_mean"][n] for sc in scs if n in agg[sc]["metrics_mean"]]
                raw_means[n] = sum(vals) / len(vals) if vals else float("nan")
            scenes_csv = ",".join(scs)
            raw_csv = " ".join(f"{n}={raw_means[n]:.5f}" for n in metric_names)
            # reward + advantage std over all rollouts this step, from the global per-seed
            # gather. (The per-microstep nft/r_std was NaN: the train batch is 1 sample, so
            # std over a single element is undefined -- compute it here over the whole pool.)
            import math as _m2
            import statistics as _stt
            _R = [x["R"] for x in per_seed if x.get("R") is not None and _m2.isfinite(x["R"])]
            _rw = [x["r"] for x in per_seed if x.get("r") is not None]
            reward_std = float(_stt.pstdev(_R)) if len(_R) > 1 else 0.0
            r_mean = (sum(_rw) / len(_rw)) if _rw else float("nan")
            r_std = float(_stt.pstdev(_rw)) if len(_rw) > 1 else 0.0
            # Mean WITHIN-scene reward std -- the advantage signal the z-score divides by.
            # reward_std above is GLOBAL (dominated by cross-scene spread), so it can look
            # healthy while every scene collapses; this isolates the per-scene spread.
            # -> 0 while loss keeps falling = degenerate advantage (loss down, reward flat).
            _ws = [agg[sc]["reward_std"] for sc in scs if _m2.isfinite(agg[sc]["reward_std"])]
            within_scene_std = (sum(_ws) / len(_ws)) if _ws else float("nan")
            log.info(f"[nft-loop] step {step} done ({len(scs)} scenes {scenes_csv}): "
                     f"reward({step_combo})={coll_reward:.4f}+/-{reward_std:.3f} "
                     f"within_scene_std={within_scene_std:.3f} "
                     f"per-scene={[round(x, 2) for x in rmeans]} raw {raw_csv} "
                     f"grad_norm={tm.get('grad_norm', float('nan')):.3f} "
                     f"microsteps={tm.get('n_microsteps', 0)} "
                     f"loss={tm.get('loss', float('nan')):.4f}", rank0_only=False)
            # all per-scene stats -> JSON on the object store (not wandb, to keep it clean):
            # per-scene reward mean/std + the raw metrics (mae_k1/mae_k30) + counts.
            if args.new_lora_uri:
                try:
                    import json as _json
                    from lyra_2._src.rl.data.gcs_util import upload_file
                    scene_stats = {sc: {"reward_mean": agg[sc]["reward_mean"],
                                        "reward_std": agg[sc]["reward_std"],
                                        "count": agg[sc]["count"],
                                        "zero_std": bool(agg[sc]["zero_std"]),
                                        **{n: agg[sc]["metrics_mean"].get(n) for n in metric_names}}
                                   for sc in scs}
                    ssf = os.path.join(args.work_dir, f"scene_stats_step{step:04d}.json")
                    with open(ssf, "w") as _fh:
                        _json.dump({"step": step, "combo": step_combo,
                                    "reward_mean": coll_reward, "reward_std": reward_std,
                                    "scenes": scene_stats}, _fh, indent=2)
                    upload_file(ssf, f"{args.new_lora_uri.rstrip('/')}/scene_stats/step{step:04d}.json")
                except Exception as e:  # noqa: BLE001 -- best-effort
                    log.warning(f"[nft-loop] scene_stats upload failed: {e}", rank0_only=False)
            if wb is not None:
                # Aggregates only -- per-scene series live in scene_stats/*.json. The NFT
                # collapse diagnostics (old_deviate/recon_err_old/a{m,p}b_rel) gate whether
                # the reward tilt still moves `new` off `old`: old_deviate->0 (==v_new==v_old,
                # amb_rel->0, apb_rel->2) is the reward-cancellation collapse.
                wb.log({
                    f"reward/{step_combo}_mean": coll_reward,
                    f"reward/{step_combo}_std": reward_std,
                    "reward/within_scene_std": within_scene_std,
                    **_reward_decomposition(per_seed, step_combo),
                    **{f"raw/{n}": raw_means[n] for n in metric_names},
                    "train/loss": tm.get("loss"),
                    "train/pos_loss": tm.get("nft/pos_loss"),
                    "train/neg_loss": tm.get("nft/neg_loss"),
                    "train/kl_loss": tm.get("nft/kl_loss"),
                    "train/old_deviate": tm.get("nft/old_deviate"),
                    "train/recon_err_old": tm.get("nft/recon_err_old"),
                    "train/amb_rel": tm.get("nft/amb_rel"),
                    "train/apb_rel": tm.get("nft/apb_rel"),
                    "train/grad_norm": tm.get("grad_norm"),
                    "train/r_mean": r_mean,
                    "train/r_std": r_std,
                    "train/n_microsteps": tm.get("n_microsteps"),
                    "train/n_bad": tm.get("n_bad", 0),
                    "train/n_oom": tm.get("n_oom", 0),
                    "data/zero_std_scenes": sum(1 for sc in scs if agg[sc]["zero_std"]),
                    "reward/r_window_std": r_window_std,  # spread of windowed per-frame r (NaN if off)
                    "reward/voxel_matched_frac": r_voxel_matched,  # voxel reward only (NaN if off)
                "reward/mix": float(os.environ.get("NFT_REWARD_MIX", "0.5") or 0.5),
                    "step": step,
                })
            # Rank 0 only: save_new_adapter is collective-free, so a second writer would
            # race on the same path. Step 0 is saved -- the save runs after its training
            # pass, so the adapter has had one update and a restart has something to
            # resume from.
            if args.new_lora_out and rank == 0 and (
                    step % args.ckpt_every == 0 or step + 1 == args.num_steps):
                fname = f"nft_new_step{step:04d}.pt"
                out = os.path.join(args.new_lora_out, fname)
                save_new_adapter(model, out)
                # Training state for an exact resume: EMA `old` adapter + AdamW moments +
                # step counters, in a separate file so nft_new_step*.pt stays a plain
                # reloadable LoRA. new/old + Adam are DDP-replicated, so rank 0's copy is
                # valid for any resume world_size.
                from lyra_2._src.models.lyra2_nft_model import _local as _lc
                ts_name = f"nft_trainstate_step{step:04d}.pt"
                ts_out = os.path.join(args.new_lora_out, ts_name)
                old_sd = {pn: _lc(p).detach().cpu().contiguous()
                          for pn, p in model.net.named_parameters()
                          if "lora_" in pn and ".old." in pn}
                torch.save({"opt": opt.state_dict(), "old": old_sd,
                            "global_step": global_step, "step": step}, ts_out)
                log.info(f"[nft-loop] saved adapter + trainstate (old={len(old_sd)} + opt) -> {out}",
                         rank0_only=False)
                # Push to object store immediately so a mid-run failure keeps progress,
                # then keep only the latest checkpoint (prune older nft_new_step*.pt) so
                # frequent (every-ckpt_every-steps) saves don't blow up storage. all of
                # this is best-effort: a transient GCS error must not crash the run --
                # that would defeat the whole point of frequent checkpointing.
                if args.new_lora_uri:
                    import re as _re
                    from lyra_2._src.rl.data.gcs_util import upload_file, list_keys, delete_file
                    base = args.new_lora_uri.rstrip("/")
                    try:
                        upload_file(out, f"{base}/{fname}")
                        upload_file(ts_out, f"{base}/{ts_name}")
                        log.info(f"[nft-loop] uploaded -> {base}/{fname} (+ trainstate)", rank0_only=False)
                        # --keep-all-checkpoints: retain every nft_new_step*.pt (+trainstate) so
                        # the reward trajectory can be evaluated per step (the prune below is only
                        # for storage-bounded runs that just need resume). Off => keep-latest.
                        if not args.keep_all_checkpoints:
                            for key in list_keys(args.new_lora_uri):
                                m = _re.search(r"nft_(?:new_step|trainstate_step)(\d+)\.pt$", key)
                                if m and int(m.group(1)) != step:
                                    scheme = base.split("://", 1)[0]
                                    bkt = base.split("://", 1)[1].split("/", 1)[0]
                                    delete_file(f"{scheme}://{bkt}/{key}")
                            log.info(f"[nft-loop] pruned old checkpoints (kept step {step})", rank0_only=False)
                    except Exception as e:  # noqa: BLE001 -- upload/prune best-effort
                        log.warning(f"[nft-loop] checkpoint upload/prune failed: {e}", rank0_only=False)
                # Drop the local files too; only the remote copies are retained.
                for _f in (out, ts_out):
                    if os.path.exists(_f):
                        os.remove(_f)

        # Persist a few watchable rollout videos (mp4, ~1 MB) for this rank's scene before
        # dropping the heavy store. Cadence decoupled from checkpointing (--videos-every;
        # ``step % N`` so step 0 / the pre-RL baseline is captured). Each scene-group's
        # rank-0 writes; path keyed by scene so the P parallel scenes don't collide.
        vid_every = args.videos_every if args.videos_every > 0 else args.ckpt_every
        if args.videos_uri and (step % vid_every == 0 or step + 1 == args.num_steps) \
                and (rank % ranks_per_scene == 0):
            from lyra_2._src.rl.loop.sampler import save_rollout_videos
            vbase = f"{args.videos_uri.rstrip('/')}/step{step:04d}/scene_{scene}"
            nv = save_rollout_videos(rollout_root, scene, vbase)
            log.info(f"[nft-loop] uploaded {nv} rollout videos (step {step}, scene {scene})",
                     rank0_only=False)
            # Co-locate per-rollout metrics so each saved video is reward-identifiable
            # (combo R + advantage r + the combo's raw metrics) -- otherwise the videos are
            # unlabeled and best/worst can't be recovered (the heavy store is deleted below).
            try:
                from lyra_2._src.rl.data.gcs_util import upload_file
                mpath = os.path.join(step_root, "video_metrics.jsonl")
                seen = set()
                with open(mpath, "w") as mf:
                    for line in open(rewards_jsonl):
                        rec = json.loads(line)
                        roll = rec.get("rollout")
                        if roll in seen:
                            continue
                        seen.add(roll)
                        m = rec.get("metrics") or {}
                        mf.write(json.dumps({
                            "step": step, "scene": scene, "rank": rank, "rollout": roll,
                            "video": f"scene_{scene}/rollout_{roll:02d}.mp4",
                            "R": rec.get("R"), "r": r_by_idx.get(roll),
                            "metrics": m,
                        }) + "\n")
                upload_file(mpath, f"{vbase}/video_metrics_rank{rank:03d}.jsonl")
            except Exception as e:  # noqa: BLE001 -- best-effort
                log.warning(f"[nft-loop] video metrics save failed: {e}", rank0_only=False)

        # Each rank drops only its own subtree, never the shared step dir: there is no
        # barrier here, so a fast rank would wipe a slower one's frames mid-save.
        if not args.keep_rollouts:
            shutil.rmtree(step_root, ignore_errors=True)

    if wb is not None:
        wb.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DiffusionNFT resident loop (sample/score/train)")
    ap.add_argument("--checkpoint_dir", default="checkpoints/model")
    ap.add_argument("--experiment", default="lyra2_nft")
    ap.add_argument("--scenes-root", required=True, help="dir with <scene>/image.png + lyra2_traj.npz")
    ap.add_argument("--scenes", required=True, help="space-separated scene ids")
    ap.add_argument("--prompt", default="", help="text prompt (caption) for all scenes")
    ap.add_argument("--combo", default="reproj_rgbd")
    ap.add_argument("--k-rollouts", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--work-dir", default="/outputs/loop")
    ap.add_argument("--new-lora-out", default=None, help="dir to checkpoint the trained new adapter")
    ap.add_argument("--new-lora-uri", default=None, help="object-store prefix to upload each adapter to")
    ap.add_argument("--videos-uri", default=None, help="object-store prefix to upload rollout mp4s to (at ckpt steps)")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--videos-every", type=int, default=0,
                    help="save rollout videos every N steps incl step 0 (0 -> use --ckpt-every); "
                         "decoupled from checkpointing")
    ap.add_argument("--keep-rollouts", action="store_true", help="don't delete per-step rollout stores")
    ap.add_argument("--keep-all-checkpoints", action="store_true",
                    help="retain EVERY nft_new_step*.pt in object store (no prune); for per-step eval")
    ap.add_argument("--val-scenes", default="", help="held-out test scene ids (space-sep) to validate on")
    ap.add_argument("--val-every", type=int, default=0, help="run test-set validation every N steps (0=off)")
    ap.add_argument("--val-k", type=int, default=0, help="rollouts per val scene (0 -> k_local)")
    ap.add_argument("--val-metrics", default="hpsv3_vid,rpe_rot,rpe_trans",
                    help="extra raw metrics to compute during validation beyond the combo's own "
                         "(comma-sep), logged as val/<metric>. Empty to compute only the "
                         "combo's own metrics.")
    ap.add_argument("--val-full-videogpa", action="store_true",
                    help="also score each val rollout with the full dl3dv_videogpa suite "
                         "(DA3 + VGGT-Omega + LightGlue): logs val/{da3,vggt}_* + val/epipolar")
    ap.add_argument("--val-base-seed", type=int, default=777000, help="fixed base seed for val sampling")
    ap.add_argument("--scene-cache-cap", type=int, default=8, help="LRU cap for lazily-built scene batches")
    ap.add_argument("--scenes-parallel", type=int, default=1,
                    help="partition the world into P scene-groups (R=world/P ranks each) that sample "
                         "P scenes CONCURRENTLY; advantage z-scored per scene via global gather + "
                         "group-by-scene-id; gradient pools across scenes (DiffNFT-reference style). "
                         "Requires world%%P==0 and K%%(world/P)==0. Default 1 = all ranks one scene.")
    ap.add_argument("--grad-steps-per-collection", type=int, default=1,
                    help="optimizer steps per collection (ref gradient_step_per_epoch); 1 = one "
                         "accumulated step over the whole pool x all DMD timesteps.")
    ap.add_argument("--inner-epochs", type=int, default=1,
                    help="reuse passes over each collection (ref num_inner_epochs).")
    ap.add_argument("--policy-lora", default=None, help="init new/old; default DMD (epoch 0)")
    ap.add_argument("--ref-lora", default=None, help="init ref KL anchor; default DMD")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ema-decay", type=float, default=0.5)
    ap.add_argument("--adv-clip-max", type=float, default=2.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--resolution", default="480,832")
    ap.add_argument("--pose_scale", type=float, default=0.35)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    args = ap.parse_args()
    run_resident_loop(args)

if __name__ == "__main__":
    _main()
