# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset over the scored rollout store.

Joins the sampled rollouts with their rewards, computes the per-``group_id``
advantage and maps it to ``r in [0, 1]`` (the DiffusionNFT reward weight), and
yields a training batch of (x0_latents, conditioning tensors, r) for
:meth:`Lyra2NFTModel.training_step_from_rollout`.

Advantage / r follow the SD3 reference: within a group,
    adv = (R - mean) / (std + eps);  adv_clip = clip(adv, ±M);
    r   = clip(adv_clip / M / 2 + 0.5, 0, 1)
so the best-in-group -> r≈1 (positive flow-matching target) and worst -> r≈0
(negative target). Groups with ~0 std contribute r≈0.5 (neutral); their fraction
is tracked (``zero_std_ratio``) since high-variance video rewards can make many
groups degenerate.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover
    Dataset = object

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
    """Dataset of scored rollout samples for DiffusionNFT training.

    Reads a ``rewards_epoch_{E}.jsonl``. Each item loads the generated
    latent + stored conditioning from disk and returns a model-free batch dict
    (the condition is rebuilt on the model side in ``training_step_from_rollout``).
    """

    # Mapping from cond.pt keys -> training_step_from_rollout keys.
    _COND_REMAP = {"pos_text": "t5_text_embeddings", "neg_text": "neg_t5_text_embeddings"}
    _COND_PASS = ("cond_latent", "cond_latent_mask", "cond_latent_buffer", "last_hist_frame", "fps", "padding_mask")

    def __init__(
        self,
        rewards_jsonl: str,
        adv_clip_max: float = 5.0,
        std_eps: float = 1e-4,
        drop_nonfinite: bool = True,
    ):
        raw = [json.loads(l) for l in open(rewards_jsonl)]
        if drop_nonfinite:
            recs = [r for r in raw if math.isfinite(float(r["R"]))]
        else:
            recs = raw
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
        x0_gen = torch.load(rec["x0_path"], map_location="cpu", weights_only=False)
        if x0_gen.dim() == 4:  # [C,T,H,W] -> [1,C,T,H,W]
            x0_gen = x0_gen.unsqueeze(0)

        history = cond["history_window"]  # [1,C,T_hist,H,W]
        if history.dim() == 4:
            history = history.unsqueeze(0)
        x0_latents = torch.cat([history, x0_gen.to(history.dtype)], dim=2)  # [1,C,T_total,H,W]

        # Chunk index (parsed from the store path chunk_{NNN}) so the loop can log
        # diagnostics per chunk -- chunk 0 conditions on the real image, chunk c>0 on
        # the accumulated generated history, so velocity scale / replay fidelity can
        # differ by chunk and must not be averaged away.
        _m = re.search(r"chunk_(\d+)", str(rec.get("x0_path", "")))
        chunk_id = int(_m.group(1)) if _m else -1
        # r is a scalar (one weight per sample), a per-latent-frame list (windowed reward),
        # or a [T][gh][gw] array (voxel reward -- an ndarray, since the direct paint's grid
        # is far too large to round-trip through nested Python lists).
        _r = rec["r"]
        if isinstance(_r, np.ndarray):
            r_val = torch.as_tensor(_r, dtype=torch.float32)
        elif isinstance(_r, (list, tuple)) and _r and isinstance(_r[0], (list, tuple)):
            r_val = torch.tensor(_r, dtype=torch.float32)          # [T, gh, gw]
        elif isinstance(_r, (list, tuple)):
            r_val = torch.tensor([float(x) for x in _r], dtype=torch.float32)
        else:
            r_val = float(_r)
        item = {
            "x0_latents": x0_latents,
            "r": r_val,
            "group_id": rec["group_id"],
            "chunk_id": chunk_id,
        }
        for src, dst in self._COND_REMAP.items():
            item[dst] = cond.get(src)
        for k in self._COND_PASS:
            item[k] = cond.get(k)
        return item

def nft_collate(batch: List[dict]) -> dict:
    """Collate rollout items: cat tensors over the batch dim (each carries a
    leading singleton batch), stack r, keep None as None, keep group_id as a list."""
    out: dict = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if k == "r":
            if all(isinstance(v, torch.Tensor) and v.dim() > 1 for v in vals):
                out[k] = torch.stack(vals, dim=0)  # [B, T_gen, gh, gw] voxel reward
            elif all(isinstance(v, torch.Tensor) for v in vals):
                out[k] = torch.stack([v.reshape(-1) for v in vals], dim=0)  # [B, T_gen] per-frame
            else:
                out[k] = torch.tensor([float(v) for v in vals], dtype=torch.float32)  # [B] scalar
        elif k == "group_id":
            out[k] = list(vals)
        elif all(v is None for v in vals):
            out[k] = None
        elif all(isinstance(v, torch.Tensor) for v in vals):
            out[k] = torch.cat(vals, dim=0)
        else:
            out[k] = vals
    return out
