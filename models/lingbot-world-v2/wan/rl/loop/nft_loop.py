"""Resident DiffusionNFT loop for lingbot causal_fast: sample -> score -> train.

Each rank holds one resident WanI2VCausal (18.5B DiT + new/old LoRA adapters)
and loops sample -> score -> train for --num-steps, so the checkpoint stages
once and a train step directly affects the next sample step (true on-policy).
Ported from Lyra-2's lyra_2/_src/rl/nft_loop.py (proven DDP-lockstep shape).

One model serves both phases: sampling runs ``wan.rl.loop.rollout_store.
run_scene_rollouts`` under ``adapters.adapter_ctx(model, "old")`` + no_grad;
training replays each rollout through ``wan.rl.loop.nft_trainer.LingbotNFTTrainer``
under the trainable "new" adapter with grads all-reduced across ranks (DDP).
``adapters.ema_update_old`` promotes ``old <- new`` each loop step.

Scene-parallel data (--scenes-parallel = P): the world is split into P groups
of world//P ranks; each group samples one scene's K rollouts per step (k_local
per rank). Advantages are z-scored per SCENE via a global gather + group-by-
scene-id; the gradient pools across the P scenes via the world grad all-reduce.

Launch (8-GPU gang, one node)::

    torchrun --standalone --nproc_per_node=8 -m wan.rl.loop.nft_loop \
        --checkpoint-dir /ckpts/lingbot --scenes-root /inputs/scenes \
        --scenes "scene_a,scene_b" --combo hpsv3_reproj_r23 \
        --k-rollouts 8 --scenes-parallel 2 --num-steps 200 \
        --work-dir /outputs/loop --new-lora-uri s3://bkt/run1/adapters
"""
from __future__ import annotations

import gc
import json
import logging
import math
import os
import re
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

log = logging.getLogger("nft_loop")

# VAE temporal compression: one latent frame per 4 pixel frames. Fixed by the
# checkpoint, and the same number nft_score.py uses to build R_perframe.
_FRAMES_PER_LATENT = 4

# Both the adapter file and its trainstate sidecar, e.g. nft_new_step0010.pt /
# nft_trainstate_step0010.pt (same pattern as lyra's checkpoint prune/resume).
_CKPT_RE = re.compile(r"nft_(?:new_step|trainstate_step)(\d+)\.pt$")
_NEW_RE = re.compile(r"nft_new_step(\d+)\.pt$")

def _setup_logging(rank: int) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [nft-loop rank {rank}] %(levelname)s %(message)s",
        force=True)

def _dist() -> Tuple[int, int, int]:
    """Return (rank, world_size, local_rank); initialize the NCCL group if launched
    under torchrun. Falls back to a single-process (1-GPU) run otherwise."""
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            # Collectives (ADV gather, grad all-reduce) sit idle while every rank
            # runs its scoring; the fastest rank blocks in the gather until the
            # slowest finishes. The 10-min NCCL default trips on that straggler
            # spread, so allow a generous timeout.
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
        import wandb

        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "lingbot-nft"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            name=os.environ.get("WANDB_NAME") or f"{args.combo}-nft-loop",
            config={
                "combo": args.combo, "k_rollouts": args.k_rollouts, "num_steps": args.num_steps,
                "scenes": args.scenes, "lr": args.lr, "ema_decay": args.ema_decay,
                "adv_clip_max": args.adv_clip_max, "num_frames": args.num_frames,
                "chunk_size": args.chunk_size, "timesteps_index": args.timesteps_index,
                "scenes_parallel": args.scenes_parallel, "lora_scope": args.lora_scope,
                "nft_beta": args.nft_beta,
            },
        )
        return wandb
    except Exception as e:  # noqa: BLE001 -- telemetry must never abort training
        print(f"[nft-loop] wandb init failed ({type(e).__name__}: {e}); continuing without it", flush=True)
        return None

# --------------------------------------------------------------------------------------
# Pure helpers (no torch.distributed) -- unit-tested in tests/test_loop_helpers.py.
# --------------------------------------------------------------------------------------

def _parse_scenes(spec: str) -> List[str]:
    """Comma-separated scene ids, or ``@/path/to/file`` with one scene per line."""
    spec = (spec or "").strip()
    if not spec:
        return []
    if spec.startswith("@"):
        with open(spec[1:]) as f:
            return [ln.strip() for ln in f.read().splitlines() if ln.strip()]
    return [s.strip() for s in spec.split(",") if s.strip()]

def _scene_advantage_records(gathered: List[dict], adv_clip_max: float,
                             std_eps: float = 1e-4):
    """Group the gathered per-rank rollout rewards by scene, z-score within each
    scene, and map to ``r in [0, 1]`` (DiffusionNFT-reference PerPromptStatTracker
    over a global gather).

    ``gathered``: one payload per rank, ``{"rank", "scene", "R": {idx: R},
    "metrics": {idx: {name: val}}}``. Returns ``(r_global, agg, per_seed)``:
      r_global: {(rank, idx): r} for every finite-R rollout,
      agg:      {scene: {reward_mean, reward_std, count, zero_std, metrics_mean}},
      per_seed: full per-rollout records (scene/rank/rollout/R/r + raw metrics).
    """
    by_scene = defaultdict(list)                            # scene -> [(rank, idx, R)]
    met_by_scene = defaultdict(lambda: defaultdict(list))   # scene -> {name: [vals]}
    for p in gathered:
        for idx, R in (p.get("R") or {}).items():
            by_scene[p["scene"]].append((p["rank"], idx, R))
        for idx, md in (p.get("metrics") or {}).items():
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

    per_seed = []
    for p in gathered:
        md_all = p.get("metrics") or {}
        for idx, R in (p.get("R") or {}).items():
            md = md_all.get(idx) or {}
            rec = {"scene": p["scene"], "rank": p["rank"], "rollout": idx,
                   "R": R, "r": r_global.get((p["rank"], idx))}
            rec.update({k: v for k, v in md.items()})
            per_seed.append(rec)
    return r_global, agg, per_seed

_REPROJ_METRICS = ("vggt_mse", "vggt_depth_mae")

def _eff_share_reproj(per_seed: List[dict], terms) -> Optional[float]:
    """Live verification of the NORM sigma calibration: the effective share of the
    reward variance contributed by the reproj terms, from this step's per-rollout
    raw metrics.

    Per scene (>=4 rollouts with all combo metrics finite), compute each rollout's
    reproj z-contribution (sum of sign*w*(v-mu)/sigma over vggt_mse+vggt_depth_mae)
    and hpsv3 z-contribution, take the within-scene variance of each, and return
    the mean over scenes of var_reproj / (var_reproj + var_hpsv3). When NORM's
    sigmas match the true within-scene stds this reads ~= the nominal weight split
    (~0.667 for the equal-thirds hpsv3_reproj_r23). Best-effort: returns None when
    the combo lacks either term group or no scene qualifies."""
    reproj = [t for t in terms if t.metric in _REPROJ_METRICS]
    hps = [t for t in terms if t.metric == "hpsv3_vid"]
    if not reproj or not hps:
        return None
    by_scene = defaultdict(list)   # scene -> [(reproj_contrib, hps_contrib)]
    for rec in per_seed:
        contrib = {"reproj": 0.0, "hps": 0.0}
        ok = True
        for group, ts in (("reproj", reproj), ("hps", hps)):
            for t in ts:
                v = rec.get(t.metric)
                if v is None or not math.isfinite(v):
                    ok = False
                    break
                contrib[group] += t.sign * t.weight * (v - t.mean) / t.std
            if not ok:
                break
        if ok:
            by_scene[rec["scene"]].append((contrib["reproj"], contrib["hps"]))
    shares = []
    for pairs in by_scene.values():
        if len(pairs) < 4:
            continue
        var = lambda xs: sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / len(xs)  # noqa: E731
        vr, vh = var([p[0] for p in pairs]), var([p[1] for p in pairs])
        if math.isfinite(vr) and math.isfinite(vh) and vr + vh > 0:
            shares.append(vr / (vr + vh))
    return (sum(shares) / len(shares)) if shares else None

def _prune_checkpoint_keys(keys: List[str], keep_last: int = 3) -> List[str]:
    """Given the object-store listing under --new-lora-uri, return the keys to
    DELETE so only the newest ``keep_last`` checkpoint steps remain (adapter +
    trainstate pairs are kept/deleted together). Non-checkpoint keys untouched."""
    steps = sorted({int(m.group(1)) for k in keys for m in [_CKPT_RE.search(k)] if m})
    keep = set(steps[-keep_last:]) if keep_last > 0 else set()
    out = []
    for k in keys:
        m = _CKPT_RE.search(k)
        if m and int(m.group(1)) not in keep:
            out.append(k)
    return out

# --------------------------------------------------------------------------------------
# Distributed advantage
# --------------------------------------------------------------------------------------

def _global_scene_advantage(local_R: dict, local_metrics: dict, scene: str,
                            rank: int, world_size: int, adv_clip_max: float,
                            std_eps: float = 1e-4):
    """Scene-parallel advantage (DiffNFT-reference style: global gather + group-by-id).

    Different ranks may sample different scenes (a rank holds one scene's k_local
    rollouts). All-gather every rank's ``(scene, {idx: R})`` globally, group by
    SCENE, z-score within each scene's rollouts (per-scene normalization,
    independent of GPU layout), and return ``r in [0,1]`` for this rank's local
    rollouts plus per-scene aggregate stats. The gradient pooling across scenes
    is left to the world grad all-reduce in _train_pass.

    ``local_R``: {local_idx: R} for this rank (all belong to ``scene``).
    ``local_metrics``: {local_idx: {name: val}} raw combo metrics; logging only."""
    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "R": local_R, "metrics": local_metrics}
    if world_size > 1:
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]
    r_global, agg, per_seed = _scene_advantage_records(gathered, adv_clip_max, std_eps)
    r_by_idx = {idx: r_global[(rank, idx)] for idx in local_R if (rank, idx) in r_global}
    return r_by_idx, agg, per_seed

def _reward_window_latents() -> int:
    """Latent-frame window width from ``NFT_REWARD_WINDOW`` (given in pixel frames),
    or 0 when the reward is scored over the whole rollout ("full"/unset)."""
    win = os.environ.get("NFT_REWARD_WINDOW", "full").strip().lower()
    if win in ("", "full", "0"):
        return 0
    return max(1, round(int(win) / _FRAMES_PER_LATENT))

def _global_perframe_r(local_Rpf: dict, scene: str, rank: int, world_size: int,
                       adv_clip_max: float, w_latent: int, std_eps: float = 1e-4):
    """Windowed per-latent-frame advantage via a global gather (densified reward).

    ``local_Rpf``: {rollout_idx: [per-latent combo reward]} for this rank's rollouts.
    Partitions each scene's generated latent frames into windows of ``w_latent``; for each
    window position, averages each rollout's per-latent reward over the window (skipping
    NaN frames), z-scores those aggregates over the scene's K rollouts (across ranks), and
    maps to r in [0, 1]. Every frame in a window gets that rollout's window r, so credit
    lands on the segment that earned it instead of the whole rollout. Degenerate windows
    (<2 finite rollouts or ~0 std) -> r=0.5. Returns ({rollout_idx: [per-latent r]} for
    this rank, mean_window_std)."""
    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "Rpf": local_Rpf}
    if world_size > 1:
        gathered: List[Optional[dict]] = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene = defaultdict(list)  # scene -> [(rank, idx, rpf_list)]
    for p in gathered:
        for idx, rpf in (p.get("Rpf") or {}).items():
            by_scene[p["scene"]].append((p["rank"], int(idx), rpf))

    r_global, win_stds = {}, []
    for _sc, entries in by_scene.items():
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

# --------------------------------------------------------------------------------------
# Train pass
# --------------------------------------------------------------------------------------

_NFT_KEYS = ("nft/loss", "nft/policy_loss", "nft/pos_loss", "nft/neg_loss",
             "nft/kl_loss", "nft/old_deviate", "nft/old_kl_div", "nft/amb_rel",
             "nft/apb_rel", "nft/recon_err_old", "nft/v_new_sq", "nft/v_ref_sq",
             "nft/x0_norm", "nft/r_mean", "nft/r_std")

def _train_pass(trainer, dataset, params, opt, *, rank: int, world_size: int,
                global_step: int, max_grad_norm: float = 1.0, grad_steps: int = 1,
                inner_epochs: int = 1):
    """Train the "new" adapter over this rank's scored rollouts with gradient
    accumulation, mirroring lyra's collect -> one-big-step shape but iterating
    ROLLOUTS (the shuffle unit here; a rollout's chunks are KV-cache-ordered and
    replayed inside ``trainer.train_rollout``).

    DDP-safety invariants (ported exactly from lyra):
      * ``n_steps`` = all-reduce MIN of local rollout counts, and grad_steps /
        inner_epochs are uniform across ranks, so every rank issues the same
        number of collectives (one all_reduce per param per optimizer step).
      * An OOM inside a rollout just skips microsteps locally (no mid-loop
        collective); a None grad is zero-filled before the per-step all_reduce,
        so ranks never desync.
      * Chunk count C and trained-timestep count n_t are asserted uniform across
        ranks once (they set the accumulation scale 1/(group*C*n_t)).
    """
    import torch.distributed as dist

    n_local = len(dataset)
    if world_size > 1:
        t = torch.tensor([n_local], device="cuda")
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        n_steps = int(t.item())
    else:
        n_steps = n_local
    if n_steps == 0:
        log.warning("rank %d: 0 trainable rollouts this step; skipping train pass", rank)
        return global_step, {"n_steps": 0}

    # REPLICA GUARD. This loop syncs gradients, not weights, so a weight divergence is
    # silent: the reduced gradient becomes an average over different models, which shows
    # up only in grad_norm (the one metric computed after the all-reduce) while loss /
    # reward / old_deviate stay normal because they are rank-0-local. One float64 scalar
    # collective per step is cheap insurance against that class of bug returning.
    if world_size > 1:
        chk = torch.zeros(1, dtype=torch.float64, device="cuda")
        for p in params:
            chk += p.data.double().sum()
        lo, hi = chk.clone(), chk.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        spread = float((hi - lo).abs().item())
        assert spread <= 1e-6 * max(1.0, abs(float(hi.item()))), (
            f"rank {rank}: LoRA replicas DIVERGED (param-sum min={lo.item():.9g} "
            f"max={hi.item():.9g}, spread={spread:.3g}). Gradients would be averaged "
            "over different models; check the DDP-REPLICA SYNC broadcast in main().")

    grad_steps = max(1, min(grad_steps, n_steps))  # no more groups than rollouts
    group_sz = math.ceil(n_steps / grad_steps)
    acc = {"loss": 0.0, **{k: 0.0 for k in _NFT_KEYS}}
    vmin, vmax = float("inf"), 0.0
    n_finite = n_bad = n_oom = n_opt = 0
    acc_gnorm = 0.0
    shape_checked = False

    for inner in range(max(1, inner_epochs)):
        # Re-shuffled each inner epoch so reuse passes aren't identical; only the
        # first n_steps (the synced minimum) are consumed to stay in lockstep.
        order = torch.randperm(n_local).tolist()[:n_steps]
        groups = [order[i:i + group_sz] for i in range(0, n_steps, group_sz)]
        for group in groups:
            opt.zero_grad(set_to_none=True)
            for ds_idx in group:
                item = dataset[ds_idx]
                cond = item["cond"]
                x0 = item["x0_latents"]
                n_chunks = x0.shape[1] // int(cond["chunk_size"])
                n_t_total = len(cond["timesteps_index"])
                # Subsample sampler-grid timesteps per rollout to trade gradient
                # signal for speed (NFT_TRAIN_TIMESTEPS unset/0 -> all). r is
                # timestep-independent, so a random subset is unbiased.
                n_t_train = int(os.environ.get("NFT_TRAIN_TIMESTEPS", "0")) or n_t_total
                n_t_train = min(max(1, n_t_train), n_t_total)
                if not shape_checked:
                    # One collective, same point on every rank (first rollout of
                    # the first group): C and n_t set the accumulation scale, so
                    # they must be uniform for the grad all_reduce AVG to be a
                    # true mean.
                    if world_size > 1:
                        lo = torch.tensor([n_chunks, n_t_train], device="cuda")
                        hi = lo.clone()
                        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
                        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
                        assert torch.equal(lo, hi), (
                            f"non-uniform chunk/timestep counts across ranks: "
                            f"min={lo.tolist()} max={hi.tolist()}")
                    shape_checked = True
                t_indices = (None if n_t_train >= n_t_total
                             else sorted(torch.randperm(n_t_total)[:n_t_train].tolist()))
                scale = 1.0 / max(1, len(group) * n_chunks * n_t_train)
                r_t = torch.tensor(item["r"], dtype=torch.float32)
                if r_t.dim() == 3:
                    # Voxel r [T, gh, gw]: pervoxel_advantage truncates a scene to
                    # its shortest payload, so pad/truncate the FRAME axis with
                    # neutral 0.5 maps (mirrors the windowed pad below).
                    if r_t.shape[0] != x0.shape[1]:
                        pad_n = x0.shape[1] - r_t.shape[0]
                        if pad_n > 0:
                            r_t = torch.cat(
                                [r_t, torch.full((pad_n, *r_t.shape[1:]), 0.5)], dim=0)
                        else:
                            r_t = r_t[:x0.shape[1]]
                elif r_t.numel() > 1 and r_t.numel() != x0.shape[1]:
                    # _global_perframe_r truncates a scene to its shortest rollout,
                    # so a windowed r can be short of the latent count. Pad neutral
                    # (0.5 = no push) rather than drop the rollout.
                    pad = torch.full((x0.shape[1] - r_t.numel(),), 0.5)
                    r_t = torch.cat([r_t[:x0.shape[1]], pad]) if pad.numel() > 0 \
                        else r_t[:x0.shape[1]]
                state = None
                try:
                    state = trainer.prepare_replay(cond)
                    metrics, n_ok, n_oom_r = trainer.train_rollout(
                        state, x0, r_t, scale, t_indices=t_indices,
                        global_step=global_step)
                except torch.cuda.OutOfMemoryError:
                    # Whole-rollout OOM (e.g. the replay KV cache itself): skip
                    # locally; the zero-filled grads keep the collective uniform.
                    n_oom += n_chunks * n_t_train
                    log.warning("OOM preparing/replaying rollout %s (scene %s); skipping",
                                item.get("rollout"), item.get("scene"))
                    metrics, n_ok, n_oom_r = {}, 0, 0
                finally:
                    # Free the replay state (KV cache ~25 GB) before the next
                    # rollout allocates its own; train_rollout does not free it
                    # (the state is caller-owned).
                    del state
                    del item, cond, x0
                    gc.collect()
                    torch.cuda.empty_cache()
                n_oom += n_oom_r
                if n_ok:
                    n_finite += n_ok
                    acc["loss"] += float(metrics.get("loss", 0.0)) * n_ok
                    for k in _NFT_KEYS:
                        if k in metrics:
                            acc[k] += float(metrics[k]) * n_ok
                    _vsq = float(metrics.get("nft/v_new_sq", float("nan")))
                    if math.isfinite(_vsq):
                        vmin, vmax = min(vmin, _vsq), max(vmax, _vsq)
            # One collective per optimizer step: zero-fill missing grads so the
            # all_reduce runs uniformly on every rank even if a rank OOM'd every
            # microstep in a group.
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
            gnorm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm)  # PRE-clip total norm
            acc_gnorm += float(gnorm)
            opt.step()
            opt.zero_grad(set_to_none=True)
            n_opt += 1
            global_step += 1
            if rank == 0:
                log.info("opt-step %d (inner %d, %d rollouts) loss=%.4f r=%.3f "
                         "v_new_sq=%.3e n_oom=%d n_bad=%d",
                         global_step, inner, len(group),
                         acc["loss"] / max(1, n_finite),
                         acc["nft/r_mean"] / max(1, n_finite),
                         acc["nft/v_new_sq"] / max(1, n_finite), n_oom, n_bad)

    denom = max(1, n_finite)
    metrics = {k: v / denom for k, v in acc.items()}
    metrics["nft/v_new_sq_min"] = (vmin if vmin != float("inf") else float("nan"))
    metrics["nft/v_new_sq_max"] = vmax
    metrics["n_steps"] = n_opt            # optimizer steps taken this collection
    metrics["n_microsteps"] = n_finite    # forward/backward passes (chunks x timesteps)
    metrics["n_bad"] = n_bad
    metrics["n_oom"] = n_oom
    metrics["grad_norm"] = acc_gnorm / max(1, n_opt)  # mean pre-clip norm over opt steps
    return global_step, metrics

# --------------------------------------------------------------------------------------
# Scene inputs
# --------------------------------------------------------------------------------------

def _scene_inputs(scenes_root: str, scene: str, default_prompt: str):
    """Load a scene dir's inputs: (PIL image, prompt, action_path). The scene dir
    must hold image.jpg|image.png + poses.npy + intrinsics.npy (the dir itself is
    the ``action_path`` the pipeline loads poses/intrinsics from); an optional
    prompt.txt overrides --prompt."""
    from PIL import Image

    sdir = os.path.join(scenes_root, scene)
    img_path = None
    for name in ("image.jpg", "image.png"):
        p = os.path.join(sdir, name)
        if os.path.exists(p):
            img_path = p
            break
    assert img_path is not None, f"no image.jpg/image.png in {sdir}"
    img = Image.open(img_path).convert("RGB")
    prompt = default_prompt
    pf = os.path.join(sdir, "prompt.txt")
    if os.path.exists(pf):
        txt = open(pf).read().strip()
        if txt:
            prompt = txt
    return img, prompt, sdir

# --------------------------------------------------------------------------------------
# Resident loop
# --------------------------------------------------------------------------------------

def _combo_terms(combo_name):
    from wan.rl.scoring import nft_score
    return nft_score.COMBOS.get(combo_name, ())

def run_resident_loop(args) -> None:
    """Build the resident pipeline once and loop sample->score->train."""
    import torch.distributed as dist

    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    from wan.rl.loop import adapters
    from wan.rl.data.gcs_util import (delete_file, download_file, list_keys, upload_file,
                                 upload_file_verified)
    from wan.rl.scoring.nft_score import score_epoch
    from wan.rl.loop.nft_trainer import LingbotNFTTrainer
    from wan.rl.loop.rollout_dataset import NFTRolloutDataset
    from wan.rl.loop.rollout_store import (RolloutStoreWriter, encode_frames_mp4,
                                      run_scene_rollouts, save_rollout_videos)

    from wan.rl.scoring import nft_voxel

    rank, world_size, local_rank = _dist()
    _setup_logging(rank)
    # The training half of the loop draws from the global RNG -- the graded microstep
    # re-noise (nft_trainer.train_rollout) and the rollout shuffle -- and torch seeds it
    # non-deterministically per process, so training was unreplayable across restarts.
    # Seed it RANK-DEPENDENTLY: replayable, while keeping the re-noise independent
    # across ranks (identical noise on every rank would align the per-rank gradients and
    # inflate grad_norm). Sampling is unaffected: it uses its own generator seeded from
    # rollout_seed(), which is deliberately rank-independent.
    torch.manual_seed(args.base_seed + 1000003 * rank)
    torch.cuda.manual_seed_all(args.base_seed + 1000003 * rank)
    dev = f"cuda:{local_rank}"
    os.makedirs(args.work_dir, exist_ok=True)

    # Resident pipeline: no FSDP/SP (per-rank replica == the DDP train path), T5
    # on CPU (only needed briefly at sampling prep), DiT straight onto the GPU.
    pipe = WanI2VCausal(
        config=WAN_CONFIGS["i2v-A14B"],
        checkpoint_dir=args.checkpoint_dir,
        device_id=local_rank,
        rank=rank,
        use_sp=False,
        dit_fsdp=False,
        t5_fsdp=False,
        # T5 is a bfloat16 model, and bfloat16 matmuls on the CPU are not merely slow but
        # effectively single-threaded, so a CPU encode of one prompt can take a quarter of
        # an hour and look like a hang. Encode on the GPU, as inference does; T5_CPU=1
        # restores the CPU encode for a host that cannot spare the ~11 GB.
        t5_cpu=os.environ.get("T5_CPU", "0") == "1",
        init_on_cpu=False,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode="causal_fast")
    adapters.inject_rl_adapters(pipe.model, scope=args.lora_scope, rank=32, alpha=64)
    # DDP-REPLICA SYNC -- must run before `params`/`opt` are built below.
    # peft's init_lora_weights="gaussian" draws lora_A from the GLOBAL RNG, and torch
    # seeds that NON-deterministically per process; this loop hand-rolls DDP by
    # all-reducing GRADIENTS (see _train_pass) and never broadcasts weights. Without
    # this broadcast a run started from base trains `world_size` different models that
    # share only lora_B -- ||A|| is ~25x ||B|| in practice, so the
    # random per-rank part dominates the learned part. Consequences that were observed
    # before this was found: every checkpoint holds rank 0's member alone (of 128
    # rollouts at a step, exactly rank 0's 2 were bit-reproducible on resume); the
    # reduced gradient is an average over different models, so it partially cancels and
    # `train/grad_norm` reads ~4-8x LOW while every other train metric looks normal
    # (they are rank-0-local); and the first resume silently syncs the ranks, which is
    # what produces the post-resume gradient spike.
    if world_size > 1:
        n_sync = 0
        for pname, p in pipe.model.named_parameters():
            if "lora_" in pname:
                dist.broadcast(p.data, src=0)
                n_sync += 1
        adapters.enforce_grad(pipe.model)   # peft flips requires_grad on adapter ops
        log.info("DDP sync: broadcast %d LoRA params from rank 0", n_sync)
    if args.policy_lora:
        n = adapters.load_new_adapter(pipe.model, args.policy_lora)
        adapters.ema_update_old(pipe.model, 0.0)  # old <- new (matched start)
        log.info("loaded initial policy LoRA %s (%d params); old <- new", args.policy_lora, n)

    # --- RESUME: continue from the latest trained checkpoint in --new-lora-uri so
    # a preemption (which restarts the job) doesn't reset to step 0. Overwrites
    # only `new` (+ `old`/opt if a trainstate exists) and skips completed steps.
    start_step = 0
    resume_opt_sd = None      # optimizer state to restore after opt is built (below)
    resume_global_step = 0
    if args.new_lora_uri:
        latest_n = -1
        if rank == 0:
            try:
                for key in list_keys(args.new_lora_uri):
                    m = _NEW_RE.search(key)
                    if m:
                        latest_n = max(latest_n, int(m.group(1)))
            except Exception as e:  # noqa: BLE001 -- resume is best-effort; fall back to fresh
                log.warning("resume scan failed (%s); starting fresh", e)
        if world_size > 1:
            obj = [latest_n]
            dist.broadcast_object_list(obj, src=0)
            latest_n = int(obj[0])
        if latest_n >= 0:
            base = args.new_lora_uri.rstrip("/")
            ckpt_uri = f"{base}/nft_new_step{latest_n:04d}.pt"
            local_ckpt = os.path.join(args.work_dir, f"_resume_rank{rank:03d}.pt")
            download_file(ckpt_uri, local_ckpt)
            n_loaded = adapters.load_new_adapter(pipe.model, local_ckpt)
            start_step = latest_n + 1
            # TIER 1: full trainstate -> restore `old` + Adam exactly. TIER 2 (no
            # trainstate): old <- new (collapses the new/old gap but avoids an
            # `old = base` discontinuity); optimizer restarts fresh.
            ts_uri = f"{base}/nft_trainstate_step{latest_n:04d}.pt"
            restored = "none"
            try:
                local_ts = os.path.join(args.work_dir, f"_resume_ts_rank{rank:03d}.pt")
                download_file(ts_uri, local_ts)
                ts = torch.load(local_ts, map_location="cpu", weights_only=False)
                n_old = adapters.load_adapter_state_dict(pipe.model, "old", ts.get("old") or {})
                resume_opt_sd = ts.get("opt")             # applied after opt is built
                resume_global_step = int(ts.get("global_step", 0))
                restored = f"EXACT (old={n_old} params, opt={'yes' if resume_opt_sd else 'no'})"
                os.remove(local_ts)
            except Exception as e:  # noqa: BLE001 -- no trainstate -> old<-new fallback
                adapters.ema_update_old(pipe.model, 0.0)  # old <- new
                restored = f"old<-new fallback (no trainstate: {type(e).__name__})"
            if os.path.exists(local_ckpt):
                os.remove(local_ckpt)
            log.info("RESUME from %s: loaded %d 'new' params; start_step=%d; %s",
                     ckpt_uri, n_loaded, start_step, restored)

    params = [p for p in pipe.model.parameters() if p.requires_grad]
    assert params, "no trainable params; expected the 'new' LoRA adapter"
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    if resume_opt_sd is not None:  # exact optimizer resume (Adam moments)
        try:
            opt.load_state_dict(resume_opt_sd)
            log.info("RESUME: restored optimizer state")
        except Exception as e:  # noqa: BLE001 -- shape/param mismatch -> keep fresh opt
            log.warning("optimizer state restore failed (%s); fresh opt", e)

    trainer = LingbotNFTTrainer(pipe, beta=args.nft_beta)  # beta_kl from NFT_BETA_KL

    wb = _init_wandb(args, rank)

    all_scenes = _parse_scenes(args.scenes)
    assert all_scenes, "--scenes resolved to an empty list"
    tsi = tuple(int(x) for x in args.timesteps_index.split(","))

    # Voxel reward (NFT_REWARD_VOXEL): per-(voxel, frame) advantage painted
    # onto a patch grid, r [T][gh][gw] per rollout, superseding the windowed reward.
    # The scorer side is gated by voxel_frames on the training score_epoch call only,
    # so validation always scores the plain global way. latent_T replicates
    # image2video._prepare_causal_fast: (F-1)//4+1 floored to a chunk_size multiple
    # (81 frames -> 20 latents) -- the payload's t axis must equal x0.shape[1].
    latent_T = (args.num_frames - 1) // _FRAMES_PER_LATENT + 1
    latent_T -= latent_T % args.chunk_size
    vox_alpha = nft_voxel.voxel_alpha()
    _lat_credit = nft_voxel.latent_credit()
    # The three spatial rewards are mutually exclusive: voxel paints from
    # world voxels, per-latent keys the same cells in image space, and the window reward is
    # whole-frame. Enabling two would silently let the first branch win.
    assert not (_lat_credit and vox_alpha > 0), (
        "NFT_REWARD_LATENT and NFT_REWARD_VOXEL are mutually exclusive")
    if vox_alpha > 0:
        # The spatial (voxel) r map is a decomposition of the GEOMETRY terms; a reward with
        # no geometry term has no per-voxel reward at all.
        _terms = _combo_terms(args.combo)
        assert any(t.metric in nft_voxel._GEOM_METRICS for t in _terms), (
            f"NFT_REWARD_VOXEL needs a combo with vggt_mse/vggt_depth_mae terms; "
            f"'{args.combo}' has none")
        _nongeom = [t.metric for t in _terms if t.metric not in nft_voxel._GEOM_METRICS]
        if rank == 0 and _nongeom:
            log.warning("voxel reward is GEOMETRY-ONLY: %s stay in the scalar R but do not "
                        "enter the spatial r map", _nongeom)
    if rank == 0:
        _w_lat = _reward_window_latents()
        log.info("reward mode: %s",
                 f"VOXEL alpha={vox_alpha} patch={nft_voxel.patch_grid() or 'latent (per-latent voxel)'} "
                 f"local={nft_voxel.local_lambda()} depth_cap={nft_voxel.depth_cap()} "
                 f"({latent_T} latents)" if vox_alpha > 0 else
                 f"WINDOWED window={os.environ.get('NFT_REWARD_WINDOW')} pixel frames "
                 f"-> {_w_lat} latents (of {latent_T})" if _w_lat else
                 "GLOBAL (one scalar r per rollout)")
        _mix0 = nft_voxel.reward_mix()
        if _mix0 < 1.0:
            log.info("reward MIX: r = %.3f * local_map + %.3f * scalar_advantage "
                     "(per cell; rollouts with no map stay pure scalar)", _mix0, 1.0 - _mix0)

    # Scene-parallel (DiffNFT-reference style): partition the world into P
    # scene-groups (--scenes-parallel); each group of R = world//P ranks samples
    # one scene's K rollouts (k_local = K//R per rank). All P scenes sample
    # CONCURRENTLY; advantages z-score per scene via the global gather; the
    # gradient pools across scenes via the world grad all-reduce in _train_pass.
    P = max(1, args.scenes_parallel)
    assert world_size % P == 0, f"world_size={world_size} not divisible by --scenes-parallel={P}"
    ranks_per_scene = world_size // P
    my_group = rank // ranks_per_scene
    rank_in_group = rank % ranks_per_scene
    assert args.k_rollouts % ranks_per_scene == 0 and args.k_rollouts // ranks_per_scene >= 1, \
        f"K={args.k_rollouts} must be a positive multiple of ranks_per_scene={ranks_per_scene}"
    k_local = args.k_rollouts // ranks_per_scene
    k_indices = list(range(rank_in_group * k_local, (rank_in_group + 1) * k_local))
    log.info("rank %d/%d: P=%d scene-groups x %d ranks (group %d); K=%d -> %d rollouts/rank "
             "(k_indices=%s); %d scenes, %d/step", rank, world_size, P, ranks_per_scene,
             my_group, args.k_rollouts, k_local, k_indices, len(all_scenes), P)

    # ---- TEST-SET VALIDATION (held-out scenes; no training) --------------------
    # Every --val-every steps, sample the fixed val set under the trainable
    # ("new") policy with FIXED seeds (step-independent -> comparable across
    # steps), score with the same combo, log val/* means. Tasks are distributed
    # FLAT across all ranks so validation load is balanced (idle ranks would
    # otherwise block at the all_gather while busy ranks sample for minutes).
    val_scenes = _parse_scenes(args.val_scenes)
    val_k = max(1, int(args.val_k) or k_local)
    val_tasks = [(sc, j) for sc in val_scenes for j in range(val_k)]  # scene x seed
    my_tasks = val_tasks[rank::world_size]                            # ~equal per rank

    def run_validation(step: int) -> None:
        recs = []
        if my_tasks:
            pipe.model.eval()
            roots = []
            with adapters.adapter_ctx(pipe.model, "new"):
                for (vsc, j) in my_tasks:
                    rr = os.path.join(args.work_dir, f"val_{step:04d}",
                                      f"rank_{rank:03d}", f"{vsc}_s{j}", "rollout")
                    os.makedirs(rr, exist_ok=True)
                    writer = RolloutStoreWriter(rr, append=False)
                    vseed = args.val_base_seed + j * 7919  # fixed across steps
                    img, prompt, action_path = _scene_inputs(args.scenes_root, vsc, args.prompt)
                    run_scene_rollouts(pipe, writer, vsc, prompt, img, action_path,
                                       k_indices=[0], k_total=1, base_seed=vseed,
                                       frame_num=args.num_frames,
                                       chunk_size=args.chunk_size,
                                       timesteps_index=tsi, shift=args.shift)
                    roots.append((vsc, j, rr))
            # Score with the DiT off-GPU (reward models need the memory).
            pipe.model.cpu()
            # One HPSv3 subprocess per rank for the whole pass (never per clip --
            # that deadlocked a 32-GPU run). Each clip's score_val_full then reads
            # the cached json. No-op unless NFT_VAL_HPSV3=1.
            if os.environ.get("NFT_VAL_FULL_METRICS", "0") == "1":
                try:
                    from wan.rl.scoring.nft_score import batch_val_hpsv3
                    batch_val_hpsv3(
                        [(os.path.join(rr, f"scene_{vsc}", "rollout_00", "frames"),
                          os.path.dirname(rr)) for (vsc, j, rr) in roots],
                        local_rank)
                except Exception as e:  # noqa: BLE001 -- val must never abort training
                    log.warning("val hpsv3 batch failed: %s", e)
            gc.collect()
            torch.cuda.empty_cache()
            for (vsc, j, rr) in roots:
                rj = os.path.join(os.path.dirname(rr), "rewards.jsonl")
                try:
                    # args.combo, not the step's cycle phase: val must score the same way at
                    # every step or its curve would alternate with the reward schedule and
                    # stop being comparable across steps. NFT_VAL_FULL_METRICS logs the full
                    # suite anyway, so both objectives are visible on the val set regardless
                    # of which phase the step was in.
                    score_epoch(rr, combo_name=args.combo, gpu_id=local_rank, out_jsonl=rj)
                except Exception as e:  # noqa: BLE001 -- val must never abort training
                    log.warning("val scoring failed %s seed%d: %s", vsc, j, e)
                R, raw = None, {}
                if os.path.exists(rj):
                    for line in open(rj):
                        rec = json.loads(line)
                        R, raw = float(rec["R"]), dict(rec.get("metrics") or {})
                        break  # k=1 -> one record per store
                # NFT_VAL_FULL_METRICS=1: also log the full VideoGPA suite + HPSv3 for
                # each val clip (not just the metrics the training combo needs), so a
                # checkpoint can be judged on metrics the reward cannot game. Every key
                # flows through to wandb as val/<key> below. Failures are swallowed by
                # score_val_full, so this can never abort training.
                if os.environ.get("NFT_VAL_FULL_METRICS", "0") == "1":
                    try:
                        from wan.rl.scoring.nft_score import score_val_full
                        raw.update(score_val_full(
                            os.path.join(rr, f"scene_{vsc}", "rollout_00", "frames"),
                            work=os.path.dirname(rr), gpu_id=local_rank))
                    except Exception as e:  # noqa: BLE001 -- val must never abort training
                        log.warning("val full metrics failed %s seed%d: %s", vsc, j, e)
                recs.append({"scene": vsc, "seed": j, "R": R, "metrics": raw})
            # Val seeds are fixed across steps, so the same (scene, seed) mp4
            # accumulates per step -> directly comparable step-0 vs step-N clips.
            if args.new_lora_uri and args.val_videos > 0:
                for (vsc, j, rr) in roots:
                    if j >= args.val_videos:
                        continue
                    try:
                        frames = os.path.join(rr, f"scene_{vsc}", "rollout_00", "frames")
                        mp4 = os.path.join(os.path.dirname(rr), "val.mp4")
                        if encode_frames_mp4(frames, mp4):
                            upload_file(mp4, f"{args.new_lora_uri.rstrip('/')}/val_videos/"
                                             f"step{step:04d}/scene_{vsc}/seed{j}.mp4")
                    except Exception as e:  # noqa: BLE001 -- val must never abort training
                        log.warning("val video upload failed %s seed%d: %s", vsc, j, e)
            # Free the val scorer's VGGT + DA3 before the DiT returns to the GPU.
            # They are cached across the pass's clips; carrying them into training
            # is what OOM'd the 32-GPU run at the next rollout's VAE decode.
            if os.environ.get("NFT_VAL_FULL_METRICS", "0") == "1":
                try:
                    from wan.rl.scoring.nft_score import release_val_full_ctx
                    release_val_full_ctx()
                except Exception as e:  # noqa: BLE001 -- val must never abort training
                    log.warning("val scorer cleanup failed: %s", e)
            pipe.model.to(dev)
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
            per = defaultdict(list)                    # scene -> [R]
            permet = defaultdict(lambda: defaultdict(list))  # metric -> scene -> [v]
            for r in allrecs:
                if r["R"] is not None and math.isfinite(r["R"]):
                    per[r["scene"]].append(r["R"])
                for k, v in (r.get("metrics") or {}).items():
                    if v is not None and math.isfinite(v):
                        permet[k][r["scene"]].append(v)
            flat = lambda d: [v for vs in d.values() for v in vs]  # noqa: E731
            mean = lambda x: (sum(x) / len(x)) if x else float("nan")  # noqa: E731
            allR = flat(per)
            met = {k: mean(flat(sc)) for k, sc in permet.items()}
            raw_csv = " ".join(f"{k}={met[k]:.4f}" for k in sorted(met))
            log.info("VAL step %d: %s=%.4f %s over %d test scenes (%d rollouts)",
                     step, args.combo, mean(allR), raw_csv, len(per), len(allR))
            if wb is not None:
                # Per-scene PER-METRIC, not just the combined reward: a pooled mean can come
                # back bimodal -- {1.021, 1.029} at steps 45/55 vs {1.164, 1.185} at 40/50 --
                # while old_deviate sat at ~1e-4, i.e. the weights were not moving. Two tight
                # clusters is not unimodal noise, and val_reward_scene/* cannot explain it:
                # epipolar is a val-only metric and is not in the training combo, so the
                # combined per-scene R says nothing about it. Logging permet per scene is what
                # distinguishes "one scene intermittently blows up and drags the mean" from
                # "every scene shifts together".
                per_scene_met = {f"val_scene/{k}/{s}": mean(v)
                                 for k, sc in permet.items() for s, v in sc.items()}
                # Spread across scenes, per metric: a heavy-tailed metric shows a large
                # max-min while its mean looks stable.
                spread = {}
                for k, sc in permet.items():
                    sv = [mean(v) for v in sc.values() if v]
                    if len(sv) > 1:
                        spread[f"val_spread/{k}"] = max(sv) - min(sv)
                wb.log({f"val/{args.combo}_mean": mean(allR), "val/n_scenes": len(per),
                        "val/step": step,
                        **{f"val/{k}": v for k, v in met.items()},
                        **{f"val_reward_scene/{s}": mean(v) for s, v in per.items()},
                        **per_scene_met, **spread})

    global_step = resume_global_step  # continue the opt-step counter across a resume
    for step in range(start_step, args.num_steps):
        if val_tasks and args.val_every > 0 and step % args.val_every == 0:
            run_validation(step)
            if world_size > 1:
                dist.barrier()

        step_combo = args.combo
        step_terms = _combo_terms(step_combo)
        # Spatial (voxel/windowed) r needs per-frame geometry. Gate it on that rather than
        # letting the per-frame lookup come back empty -- that path is how a silent all -inf
        # reward happened before.
        step_geom = any(t.metric in nft_voxel._GEOM_METRICS for t in step_terms)
        step_vox = vox_alpha if (vox_alpha > 0 and step_geom) else 0.0

        # ---- COLLECT: this rank samples ITS group's scene (P scenes in parallel).
        pipe.model.eval()
        step_dir = os.path.join(args.work_dir, f"step_{step:04d}")
        scene = all_scenes[(step * P + my_group) % len(all_scenes)]
        step_root = os.path.join(step_dir, f"rank_{rank:03d}")
        rollout_root = os.path.join(step_root, "rollout")
        rewards_jsonl = os.path.join(step_root, "rewards.jsonl")
        os.makedirs(rollout_root, exist_ok=True)

        # SAMPLE this rank's k_local rollouts of its scene under "old". Seeds are
        # step-distinct (base term) and rollout-distinct (rollout_seed's k term;
        # k_indices are disjoint across the ranks of a scene-group).
        img, prompt, action_path = _scene_inputs(args.scenes_root, scene, args.prompt)
        writer = RolloutStoreWriter(rollout_root, append=False)
        base_seed = args.base_seed + step * 100000
        with adapters.adapter_ctx(pipe.model, "old"):
            run_scene_rollouts(pipe, writer, scene, prompt, img, action_path,
                               k_indices, k_total=args.k_rollouts, base_seed=base_seed,
                               frame_num=args.num_frames, chunk_size=args.chunk_size,
                               timesteps_index=tsi, shift=args.shift)

        # SCORE. Offload the resident DiT to host RAM so the reward workers get
        # the GPU (hpsv3's VLM + the 18.5B resident DiT don't fit together); it
        # is not used during scoring and reloads before _train_pass.
        pipe.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        score_epoch(rollout_root, combo_name=step_combo, gpu_id=local_rank,
                    out_jsonl=rewards_jsonl,
                    voxel_frames=latent_T if (step_vox > 0 or _lat_credit) else 0)
        pipe.model.to(dev)
        torch.cuda.empty_cache()

        # ADVANTAGE: GLOBAL gather + group-by-scene-id -> per-scene z-scored r for
        # this rank's rollouts (+ per-scene agg stats, available on every rank).
        local_R, local_metrics, local_Rpf, local_Vp = {}, {}, {}, {}
        if os.path.exists(rewards_jsonl):
            for line in open(rewards_jsonl):
                rec = json.loads(line)
                local_R[rec["rollout"]] = float(rec["R"])
                m = rec.get("metrics") or {}
                local_metrics[rec["rollout"]] = {
                    k: float(v) for k, v in m.items()
                    if v is not None and math.isfinite(float(v))}
                rpf = rec.get("R_perframe")
                if rpf:
                    local_Rpf[rec["rollout"]] = rpf
                vp = rec.get("voxel_payload")
                if vp:
                    local_Vp[rec["rollout"]] = vp
        else:
            open(rewards_jsonl, "w").close()  # empty pool -> 0-rollout dataset below
        r_by_idx, agg, per_seed = _global_scene_advantage(
            local_R, local_metrics, scene, rank, world_size, args.adv_clip_max)

        # WINDOWED advantage (NFT_REWARD_WINDOW = window length in PIXEL frames):
        # z-score per window position instead of per rollout, so each segment of a
        # rollout is graded against the same segment of its siblings. Falls back to
        # the scalar r above for any rollout the scorer gave no per-frame reward.
        # The gather inside is a collective, so the gate must be the env alone --
        # local_Rpf is per-rank (a rank whose scorer failed for every rollout has
        # none) and gating on it would hang the ranks that do enter.
        # VOXEL advantage (NFT_REWARD_VOXEL) supersedes it: r [T][gh][gw] per
        # rollout, same scalar-r fallback and same env-only collective gating.
        r_perframe_by_idx, r_window_std = {}, float("nan")
        r_voxel_matched = float("nan")
        w_latent = _reward_window_latents() if step_geom else 0
        if step_vox > 0:
            r_perframe_by_idx, vdiag = nft_voxel.global_pervoxel_r(
                local_Vp, scene, rank, world_size, args.adv_clip_max, step_terms)
            r_window_std = vdiag["voxel_std"]        # cell std, the voxel analogue
            r_voxel_matched = vdiag["matched_frac"]
        elif _lat_credit and step_geom:
            # PER-LATENT credit (NFT_REWARD_LATENT): r per (latent frame, latent cell), keyed in
            # image space. Same statistics as voxel, different keying -- see nft_voxel.
            # latent_credit() is env-only, so every rank takes this branch together and the
            # collective inside cannot deadlock.
            r_perframe_by_idx, ldiag = nft_voxel.global_perlatent_r(
                local_Vp, scene, rank, world_size, args.adv_clip_max, step_terms)
            r_window_std = ldiag["voxel_std"]
            r_voxel_matched = ldiag["matched_frac"]
        elif w_latent:
            r_perframe_by_idx, r_window_std = _global_perframe_r(
                local_Rpf, scene, rank, world_size, args.adv_clip_max, w_latent)

        # Persist every seed's per-rollout metrics (R + r + raw) for this step so
        # the full per-seed reward trajectory is recoverable. Best-effort.
        if rank == 0 and args.new_lora_uri and per_seed:
            try:
                sf = os.path.join(args.work_dir, f"seed_metrics_step{step:04d}.jsonl")
                with open(sf, "w") as fh:
                    for rec in sorted(per_seed, key=lambda r: (r["scene"], r["rank"], r["rollout"])):
                        fh.write(json.dumps({"step": step, **rec}) + "\n")
                upload_file(sf, f"{args.new_lora_uri.rstrip('/')}/seed_metrics/step{step:04d}.jsonl")
                os.remove(sf)
            except Exception as e:  # noqa: BLE001 -- best-effort; must not crash the loop
                log.warning("per-seed metrics upload failed: %s", e)

        # ---- TRAIN: one accumulated update over this rank's rollouts x chunks x
        # timesteps; the world grad all-reduce pools across the P scene-groups.
        ds = NFTRolloutDataset(rewards_jsonl, adv_clip_max=args.adv_clip_max)
        # Overwrite the locally-computed r with the globally-gathered per-scene one.
        ds.records = [rec for rec in ds.records if rec["rollout"] in r_by_idx]
        # NFT_REWARD_MIX blends the voxel map with the rollout's own scalar advantage
        # (mix=1.0, the default, keeps the historical "map replaces scalar"). Rollouts with
        # no map -- scorer failure, or an hpsv3-only phase -- stay pure scalar either way.
        _mix = nft_voxel.reward_mix()
        for rec in ds.records:
            _loc = r_perframe_by_idx.get(rec["rollout"])
            _glob = r_by_idx[rec["rollout"]]
            rec["r"] = nft_voxel.blend_local_global(_loc, _glob, _mix) if _loc else _glob
        pipe.model.train()
        global_step, tm = _train_pass(
            trainer, ds, params, opt, rank=rank, world_size=world_size,
            global_step=global_step, max_grad_norm=args.max_grad_norm,
            grad_steps=args.grad_steps_per_collection, inner_epochs=args.inner_epochs)
        adapters.ema_update_old(pipe.model, args.ema_decay)
        if world_size > 1:
            dist.barrier()

        if rank == 0:
            # agg (from the global gather) holds per-scene stats for all P scenes.
            import statistics as _stt
            scs = sorted(agg.keys())
            rmeans = [agg[sc]["reward_mean"] for sc in scs]
            coll_reward = sum(rmeans) / len(rmeans) if rmeans else float("nan")
            metric_names = sorted({n for sc in scs for n in agg[sc]["metrics_mean"]})
            raw_means = {}
            for n in metric_names:
                vals = [agg[sc]["metrics_mean"][n] for sc in scs if n in agg[sc]["metrics_mean"]]
                raw_means[n] = sum(vals) / len(vals) if vals else float("nan")
            _R = [x["R"] for x in per_seed if x.get("R") is not None and math.isfinite(x["R"])]
            _rw = [x["r"] for x in per_seed if x.get("r") is not None]
            reward_std = float(_stt.pstdev(_R)) if len(_R) > 1 else 0.0
            r_mean = (sum(_rw) / len(_rw)) if _rw else float("nan")
            r_std = float(_stt.pstdev(_rw)) if len(_rw) > 1 else 0.0
            # Mean WITHIN-scene reward std -- the spread the z-score divides by;
            # -> 0 while loss keeps falling = degenerate advantage.
            _ws = [agg[sc]["reward_std"] for sc in scs if math.isfinite(agg[sc]["reward_std"])]
            within_scene_std = (sum(_ws) / len(_ws)) if _ws else float("nan")
            log.info("step %d done (%d scenes %s): reward(%s)=%.4f+/-%.3f "
                     "within_scene_std=%.3f per-scene=%s grad_norm=%.3f "
                     "microsteps=%d loss=%.4f",
                     step, len(scs), ",".join(scs), step_combo, coll_reward, reward_std,
                     within_scene_std, [round(x, 3) for x in rmeans],
                     tm.get("grad_norm", float("nan")), tm.get("n_microsteps", 0),
                     tm.get("loss", float("nan")))
            # all per-scene stats -> JSON on the object store (not wandb).
            if args.new_lora_uri:
                try:
                    scene_stats = {sc: {"reward_mean": agg[sc]["reward_mean"],
                                        "reward_std": agg[sc]["reward_std"],
                                        "count": agg[sc]["count"],
                                        "zero_std": bool(agg[sc]["zero_std"]),
                                        **{n: agg[sc]["metrics_mean"].get(n) for n in metric_names}}
                                   for sc in scs}
                    ssf = os.path.join(args.work_dir, f"scene_stats_step{step:04d}.json")
                    with open(ssf, "w") as fh:
                        json.dump({"step": step, "combo": step_combo,
                                   "reward_mean": coll_reward, "reward_std": reward_std,
                                   "scenes": scene_stats}, fh, indent=2)
                    upload_file(ssf, f"{args.new_lora_uri.rstrip('/')}/scene_stats/step{step:04d}.json")
                    os.remove(ssf)
                except Exception as e:  # noqa: BLE001 -- best-effort
                    log.warning("scene_stats upload failed: %s", e)
            # Live sigma-calibration check (see _eff_share_reproj). Best-effort:
            # a degenerate scene / missing metrics must never crash the step.
            try:
                eff_share_reproj = _eff_share_reproj(per_seed, step_terms)
            except Exception as e:  # noqa: BLE001
                log.warning("eff_share_reproj failed: %s", e)
                eff_share_reproj = None
            if wb is not None:
                wb.log({
                    f"reward/{step_combo}_mean": coll_reward,
                    f"reward/{step_combo}_std": reward_std,
                    "reward/mean": coll_reward,
                    "reward/std": reward_std,
                    "reward/within_scene_std": within_scene_std,
                    **({"reward/eff_share_reproj": eff_share_reproj}
                       if eff_share_reproj is not None else {}),
                    **{f"raw/{n}": raw_means[n] for n in metric_names},
                    # Per-term reward contribution (sign*w*(v-mu)/sigma with the
                    # frozen NORM constants): makes term balance and per-term
                    # trends readable directly, instead of eyeballing raw/*
                    # series that live on wildly different scales.
                    **{f"reward/contrib_{t.metric}": (
                           t.sign * t.weight * (raw_means[t.metric] - t.mean) / t.std)
                       for t in step_terms
                       if t.metric in raw_means and raw_means[t.metric] == raw_means[t.metric]},
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
                    **({"train/r_window_std": r_window_std}
                       if r_window_std == r_window_std else {}),
                    **({"reward/voxel_matched_frac": r_voxel_matched}
                       if r_voxel_matched == r_voxel_matched else {}),
                    "train/n_microsteps": tm.get("n_microsteps"),
                    "train/n_bad": tm.get("n_bad", 0),
                    "train/n_oom": tm.get("n_oom", 0),
                    "data/zero_std_scenes": sum(1 for sc in scs if agg[sc]["zero_std"]),
                    "step": step,
                })  # no explicit step= -> wandb commits each step immediately

            # CHECKPOINT (rank 0 only; save_new_adapter is collective-free and
            # new/old + Adam are DDP-replicated, so rank 0's copy is valid for
            # any resume world_size). Saved on multiples of ckpt_every + the
            # final step; all of it best-effort (a transient object-store error
            # must not crash the run).
            if (step > 0 and step % args.ckpt_every == 0) or step + 1 == args.num_steps:
                fname = f"nft_new_step{step:04d}.pt"
                ts_name = f"nft_trainstate_step{step:04d}.pt"
                out = os.path.join(args.work_dir, fname)
                ts_out = os.path.join(args.work_dir, ts_name)
                ckpt_ok = False
                try:
                    adapters.save_new_adapter(pipe.model, out)
                    torch.save({"opt": opt.state_dict(),
                                "old": adapters.adapter_state_dict(pipe.model, "old"),
                                "global_step": global_step, "step": step}, ts_out)
                    if args.new_lora_uri:
                        base = args.new_lora_uri.rstrip("/")
                        # Retry + read-back size check. A single-shot upload silently lost
                        # a lost checkpoint costs the steps since the previous one on the
                        # next resume. Upload the TRAINSTATE first: the resume scan keys off
                        # nft_new_step*.pt, so an adapter that lands without its trainstate
                        # advertises a checkpoint that can only resume via the old<-new
                        # fallback. This order means a partial failure leaves the adapter
                        # absent and the scan falls back to the previous COMPLETE checkpoint.
                        upload_file_verified(ts_out, f"{base}/{ts_name}")
                        upload_file_verified(out, f"{base}/{fname}")
                        ckpt_ok = True
                        log.info("saved adapter + trainstate -> %s/%s (+ trainstate)", base, fname)
                        # Keep only the newest 3 checkpoints unless --keep-all-checkpoints
                        # (retain every step for per-step eval of the reward trajectory).
                        if not args.keep_all_checkpoints:
                            scheme = base.split("://", 1)[0]
                            bkt = base.split("://", 1)[1].split("/", 1)[0]
                            for key in _prune_checkpoint_keys(list_keys(args.new_lora_uri), keep_last=3):
                                delete_file(f"{scheme}://{bkt}/{key}")
                    else:
                        log.info("saved adapter + trainstate -> %s (no --new-lora-uri; kept local)", out)
                except Exception as e:  # noqa: BLE001 -- checkpointing must not kill the run
                    # ERROR, not warning: a lost checkpoint costs real training time and
                    # used to leave only a warning nobody read.
                    log.error("CHECKPOINT LOST at step %d: %s", step, e, exc_info=True)
                    if wb is not None:
                        try:
                            wb.log({"ckpt/failed_step": step, "step": step})
                        except Exception:  # noqa: BLE001
                            pass
                # Keep the local copies when the upload did not verify -- deleting them
                # unconditionally (as before) meant a failed upload left the checkpoint
                # nowhere at all.
                if args.new_lora_uri and ckpt_ok:
                    for _f in (out, ts_out):
                        if os.path.exists(_f):
                            os.remove(_f)
                elif args.new_lora_uri:
                    log.error("keeping local %s / %s (upload unverified)", out, ts_out)

        # Persist a few watchable rollout mp4s for this rank's scene before
        # dropping the heavy store. Each scene-group's leader writes; path keyed
        # by scene so the P parallel scenes don't collide.
        vid_every = args.videos_every if args.videos_every > 0 else args.ckpt_every
        if args.videos_uri and (step % vid_every == 0 or step + 1 == args.num_steps) \
                and rank_in_group == 0:
            vbase = f"{args.videos_uri.rstrip('/')}/step{step:04d}/scene_{scene}"
            try:
                nv = save_rollout_videos(rollout_root, scene, vbase)
                log.info("uploaded %d rollout videos (step %d, scene %s)", nv, step, scene)
                # Co-locate per-rollout metrics so each saved video is
                # reward-identifiable (combo R + advantage r + raw metrics).
                mpath = os.path.join(step_root, "video_metrics.jsonl")
                seen = set()
                with open(mpath, "w") as mf:
                    for line in open(rewards_jsonl):
                        rec = json.loads(line)
                        roll = rec.get("rollout")
                        if roll in seen:
                            continue
                        seen.add(roll)
                        mf.write(json.dumps({
                            "step": step, "scene": scene, "rank": rank, "rollout": roll,
                            "video": f"scene_{scene}/rollout_{roll:02d}.mp4",
                            "R": rec.get("R"), "r": r_by_idx.get(roll),
                            "metrics": rec.get("metrics") or {},
                        }) + "\n")
                upload_file(mpath, f"{vbase}/video_metrics_rank{rank:03d}.jsonl")
            except Exception as e:  # noqa: BLE001 -- video saving must never abort the run
                log.warning("video save/metrics failed: %s", e)

        # Each rank drops only its own subtree, so no barrier is needed.
        if not args.keep_rollouts:
            shutil.rmtree(step_root, ignore_errors=True)

    if wb is not None:
        wb.finish()
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()

def _main():
    import argparse

    ap = argparse.ArgumentParser(
        description="Resident DiffusionNFT loop for lingbot (sample/score/train)")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="lingbot checkpoint dir (WAN_CONFIGS['i2v-A14B'] layout)")
    ap.add_argument("--scenes-root", required=True,
                    help="dir with <scene>/{image.jpg|image.png, poses.npy, intrinsics.npy"
                         "[, prompt.txt]} (staged locally by the jobspec)")
    ap.add_argument("--scenes", required=True,
                    help="comma-separated scene ids, or @file with one id per line")
    ap.add_argument("--prompt", default="", help="fallback prompt when a scene has no prompt.txt")
    ap.add_argument("--combo", default="hpsv3_reproj_r23", help="reward combo (wan.rl.scoring.nft_score)")
    ap.add_argument("--k-rollouts", type=int, default=8, help="rollouts per scene per step")
    ap.add_argument("--num-steps", type=int, default=100)
    ap.add_argument("--work-dir", default="/tmp/nft_loop")
    ap.add_argument("--new-lora-uri", default=None,
                    help="object-store prefix for adapter checkpoints (+ resume scan)")
    ap.add_argument("--videos-uri", default=None,
                    help="object-store prefix to upload rollout mp4s to")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--videos-every", type=int, default=0,
                    help="save rollout videos every N steps incl step 0 (0 -> use --ckpt-every)")
    ap.add_argument("--keep-rollouts", action="store_true",
                    help="don't delete per-step rollout stores")
    ap.add_argument("--keep-all-checkpoints", action="store_true",
                    help="retain EVERY nft_new_step*.pt in object store (no prune)")
    ap.add_argument("--val-scenes", default="",
                    help="held-out scene ids (comma-sep or @file) to validate on")
    ap.add_argument("--val-every", type=int, default=0,
                    help="run test-set validation every N steps (0=off)")
    ap.add_argument("--val-k", type=int, default=0, help="rollouts per val scene (0 -> k_local)")
    ap.add_argument("--val-base-seed", type=int, default=777000,
                    help="fixed base seed for val sampling")
    ap.add_argument("--val-videos", type=int, default=1,
                    help="upload the first N val rollouts per scene as mp4s under "
                         "new-lora-uri/val_videos/ (0=off)")
    ap.add_argument("--scenes-parallel", type=int, default=1,
                    help="partition the world into P scene-groups (world/P ranks each) that "
                         "sample P scenes CONCURRENTLY; requires world%%P==0 and K%%(world/P)==0")
    ap.add_argument("--grad-steps-per-collection", type=int, default=1,
                    help="optimizer steps per collection; 1 = one accumulated step over the pool")
    ap.add_argument("--inner-epochs", type=int, default=1,
                    help="reuse passes over each collection (ref num_inner_epochs)")
    ap.add_argument("--policy-lora", default=None,
                    help="initial 'new' adapter .pt (also copied to 'old'); default identity")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--adv-clip-max", type=float, default=1.3)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--num-frames", type=int, default=81)
    ap.add_argument("--chunk-size", type=int, default=4)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--timesteps-index", default="0,250,500,750",
                    help="comma-separated sampler-grid timestep indices")
    ap.add_argument("--local-attn-size", type=int, default=-1)
    ap.add_argument("--sink-size", type=int, default=0)
    ap.add_argument("--lora-scope", default="attn+ffn+cam",
                    choices=("attn", "attn+cam", "attn+ffn+cam"))
    ap.add_argument("--nft-beta", type=float, default=1.0)
    args = ap.parse_args()
    run_resident_loop(args)

if __name__ == "__main__":
    _main()
