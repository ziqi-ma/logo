"""Group advantage -> r in [0,1] (verbatim ports of Lyra rollout_dataset.compute_group_r
and nft_loop._global_scene_advantage: gather-all + group-by-scene, no NCCL subgroups)."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

def compute_group_r(records: List[dict], adv_clip_max: float = 2.0,
                    std_eps: float = 1e-4) -> Dict[str, object]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)
    zero_std_groups = 0
    for _gid, recs in groups.items():
        rs = [float(r["R"]) for r in recs]
        n = len(rs)
        mean = sum(rs) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in rs) / n)
        if std < std_eps:
            zero_std_groups += 1
        for rec in recs:
            adv = (float(rec["R"]) - mean) / (std + std_eps)
            adv_c = max(-adv_clip_max, min(adv_clip_max, adv))
            rec["advantage"] = adv
            rec["r"] = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
    return {"records": records,
            "zero_std_ratio": zero_std_groups / max(1, len(groups)),
            "num_groups": len(groups)}

def global_perframe_r(local_Rpf: Dict[int, list], scene: str, rank: int, world_size: int,
                      w_latent: int, adv_clip_max: float = 2.0, std_eps: float = 1e-4):
    """Windowed per-latent-frame advantage (verbatim port of Lyra's _global_perframe_r).

    A single scalar reward per 81-frame rollout gives the same credit to every latent, so
    a clip that is excellent for 60 frames and collapses at the end is trained as uniformly
    mediocre. Here each scene's T latents are cut into windows of ``w_latent``; within each
    window position the rollouts' window-mean rewards are z-scored ACROSS the K rollouts of
    that scene (so windows compete like-for-like, never against other timestamps), mapped to
    r in [0,1], and every latent in the window inherits its rollout's window r.

    Degenerate windows (<2 finite rollouts, or ~zero spread) stay at the neutral r=0.5 so
    they contribute no gradient direction rather than noise.

    ``local_Rpf``: {rollout_idx: [per-latent reward]} for this rank.
    Returns ({rollout_idx: [per-latent r]} for this rank, mean window std).
    """
    import torch.distributed as dist

    payload = {"rank": rank, "scene": scene, "Rpf": dict(local_Rpf)}
    if world_size > 1 and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene = defaultdict(list)  # scene -> [(rank, idx, rpf)]
    for p in gathered:
        for idx, rpf in (p.get("Rpf") or {}).items():
            by_scene[p["scene"]].append((p["rank"], int(idx), rpf))

    r_global: Dict[tuple, list] = {}
    win_stds = []
    for _sc, entries in by_scene.items():
        lens = [len(rpf) for (_r, _i, rpf) in entries if rpf]
        if not lens:
            continue
        L = min(lens)  # ragged decodes -> use the common prefix
        for (rk, idx, _rpf) in entries:
            r_global[(rk, idx)] = [0.5] * L
        for ws in range(0, L, w_latent):
            we = min(ws + w_latent, L)
            aggs = []
            for (_rk, _idx, rpf) in entries:
                seg = [rpf[f] for f in range(ws, we)
                       if rpf and rpf[f] is not None and math.isfinite(rpf[f])]
                aggs.append(sum(seg) / len(seg) if seg else float("nan"))
            finite = [a for a in aggs if math.isfinite(a)]
            if len(finite) < 2:
                continue
            mean = sum(finite) / len(finite)
            std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite))
            win_stds.append(std)
            if std < std_eps:
                continue
            for (rk, idx, _rpf), a in zip(entries, aggs):
                if not math.isfinite(a):
                    continue
                adv_c = max(-adv_clip_max, min(adv_clip_max, (a - mean) / (std + std_eps)))
                rval = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
                for f in range(ws, we):
                    r_global[(rk, idx)][f] = rval

    r_by_idx = {int(idx): r_global[(rank, int(idx))]
                for idx in (local_Rpf or {}) if (rank, int(idx)) in r_global}
    return r_by_idx, (sum(win_stds) / len(win_stds) if win_stds else float("nan"))

def global_scene_advantage(local_R: Dict[int, float], scene: str, rank: int,
                           world_size: int, adv_clip_max: float = 2.0,
                           std_eps: float = 1e-4, local_metrics: Dict[int, dict] | None = None):
    """All-gather every rank's finite rewards, z-score within scene, return
    (r_by_local_idx, per_scene_stats, per_seed_records)."""
    import torch.distributed as dist
    payload = {"rank": rank, "scene": scene, "R": dict(local_R),
               "metrics": dict(local_metrics or {})}
    if world_size > 1 and dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, payload)
    else:
        gathered = [payload]

    by_scene = defaultdict(list)
    for p in gathered:
        for idx, R in p["R"].items():
            m = (p.get("metrics") or {}).get(idx, {})
            by_scene[p["scene"]].append((p["rank"], int(idx), float(R), m))

    r_by_idx: Dict[int, float] = {}
    agg = {}
    per_seed = []
    for sc, entries in by_scene.items():
        finite = [R for _r, _i, R, _m in entries if math.isfinite(R)]
        if not finite:
            agg[sc] = {"reward_mean": float("nan"), "reward_std": 0.0,
                       "count": 0, "zero_std": True}
            continue
        mean = sum(finite) / len(finite)
        std = math.sqrt(sum((x - mean) ** 2 for x in finite) / len(finite))
        agg[sc] = {"reward_mean": mean, "reward_std": std,
                   "count": len(finite), "zero_std": std < std_eps}
        for rk, idx, R, m in entries:
            if not math.isfinite(R):
                continue
            adv = (R - mean) / (std + std_eps)
            adv_c = max(-adv_clip_max, min(adv_clip_max, adv))
            r = max(0.0, min(1.0, adv_c / adv_clip_max / 2.0 + 0.5))
            per_seed.append({"scene": sc, "rank": rk, "rollout": idx, "R": R, "r": r,
                             **{k: v for k, v in m.items() if isinstance(v, (int, float))}})
            if rk == rank and sc == scene:
                r_by_idx[idx] = r
    return r_by_idx, agg, per_seed
