"""Dataset over the scored rollout store (one item = one rollout).

Joins samples with rewards, computes the per-``group_id`` (= per-scene)
advantage and maps it to ``r in [0, 1]``. Advantage / r follow the
DiffusionNFT SD3 reference: within a group,
    adv = (R - mean) / (std + eps);  adv_clip = clip(adv, ±M);
    r   = clip(adv_clip / M / 2 + 0.5, 0, 1)
so best-in-group -> r≈1 (positive flow-matching target), worst -> r≈0
(negative target), ~0-std groups -> r≈0.5 (neutral; fraction tracked as
``zero_std_ratio``).

The loop normally overwrites ``r`` with the globally-gathered per-scene value
(nft_loop._global_scene_advantage); the locally-computed one is a fallback.
"""
import json
import math
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset

def compute_group_r(
    records: List[dict],
    adv_clip_max: float = 5.0,
    std_eps: float = 1e-4,
) -> Dict[str, object]:
    """Annotate each record with a per-group advantage and ``r``.

    Returns ``{"records": [...], "zero_std_ratio": float, "num_groups": int}``.
    Records with non-finite ``R`` are assumed already filtered out by the caller.
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        groups[rec["group_id"]].append(rec)

    zero_std_groups = 0
    for gid, recs in groups.items():
        rs = [float(r["R"]) for r in recs]
        n = len(rs)
        mean = sum(rs) / n
        var = sum((x - mean) ** 2 for x in rs) / n
        std = math.sqrt(var)
        if std < std_eps:
            zero_std_groups += 1
        for rec in recs:
            adv = (float(rec["R"]) - mean) / (std + std_eps)
            adv_clip = max(-adv_clip_max, min(adv_clip_max, adv))
            rec["advantage"] = adv
            rec["r"] = max(0.0, min(1.0, adv_clip / adv_clip_max / 2.0 + 0.5))
    return {
        "records": records,
        "zero_std_ratio": zero_std_groups / max(1, len(groups)),
        "num_groups": len(groups),
    }

class NFTRolloutDataset(Dataset):
    """Each item loads one rollout's clean latents + replay conditioning.

    The trainer consumes rollouts one at a time (chunks of a rollout are
    cache-ordered, so the shuffle unit is the rollout), so there is no collate.
    """

    def __init__(
        self,
        rewards_jsonl: str,
        adv_clip_max: float = 5.0,
        std_eps: float = 1e-4,
        drop_nonfinite: bool = True,
    ):
        raw = [json.loads(l) for l in open(rewards_jsonl)]
        recs = [r for r in raw if math.isfinite(float(r["R"]))] if drop_nonfinite else raw
        self.num_dropped = len(raw) - len(recs)
        info = compute_group_r(recs, adv_clip_max=adv_clip_max, std_eps=std_eps)
        self.records: List[dict] = info["records"]
        self.zero_std_ratio: float = info["zero_std_ratio"]
        self.num_groups: int = info["num_groups"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        cond = torch.load(rec["cond_path"], map_location="cpu", weights_only=False)
        x0_latents = torch.load(rec["x0_path"], map_location="cpu", weights_only=False)
        return {
            "x0_latents": x0_latents,  # [C, lat_f, H, W]
            "cond": cond,
            # scalar, or a per-latent list when NFT_REWARD_WINDOW is set
            "r": (list(rec["r"]) if isinstance(rec["r"], (list, tuple))
                  else float(rec["r"])),
            "R": float(rec["R"]),
            "group_id": rec["group_id"],
            "scene": rec.get("scene"),
            "rollout": rec.get("rollout"),
        }
