# SPDX-License-Identifier: Apache-2.0
"""Effective reward decomposition for the DiffusionNFT loop (wandb diagnostics).

Ported from Lyra-2's ``nft_loop._reward_decomposition``, which the UniWorld port dropped.

Why it matters: a metric's contribution to the reward is its signed z-score term
``sign * weight * (v - mu) / sigma``, and what the per-scene advantage actually keys on is the
WITHIN-SCENE std of that term -- ``mu`` cancels in the advantage, and a term whose within-scene
spread is small contributes almost no gradient no matter what its nominal weight says. So the
*effective* weight is generally not the nominal one.

Two UniWorld runs paid for not having this:
  * uw-nft-reproj-2 used Lyra's NORM (depth sigma 9.4x too large) -> the nominal 50/50 reward was
    really ~3.5:1 toward mse, and depth flatlined at -7.7% while mse kept falling.
  * after recalibration the mirror image appeared: measured on held-out val, depth moved 4.19
    sigma and mse 0.78 sigma over 160 steps -- a 5.4x skew in a nominal 50/50 reward. Caught only
    by hand-computing sigma-units from val, which is exactly what these curves show live.

Logged keys (all under ``reward/``):
  eff_wsstd_<metric>        within-scene std of that metric's term (-> 0 means the term went inert)
  eff_wsstd_<group>         same for a summed group (geometry terms summed before taking the std)
  eff_share_<group>         each group's fraction of the effective reward -- the real blend ratio
"""
from __future__ import annotations

import math
import statistics as st
from collections import defaultdict

# groups reported as shares. hpsv3 is listed now so the panel appears the moment an hpsv3 term
# enters COMBO; with a geometry-only reward only the reproj group is emitted.
GROUPS = {
    "reproj": ("vggt_mse", "vggt_depth_mae"),
    "hpsv3": ("hpsv3_vid",),
}

def reward_decomposition(per_seed, combo, norm) -> dict:
    """``per_seed``: records from advantage.global_scene_advantage (scene + raw metric values).

    ``combo``: [(metric, weight), ...] as in nft_score_cli.COMBO.
    ``norm``:  {metric: (mu, sigma, sign)} as in nft_score_cli.NORM.
    Returns a wandb-loggable dict; empty if no combo metric is present in the records.
    """
    if not per_seed or not combo:
        return {}
    by_scene = defaultdict(list)
    for rec in per_seed:
        by_scene[rec.get("scene")].append(rec)

    def wsstd(metrics):
        """Mean over scenes of the within-scene std of the summed terms for ``metrics``."""
        terms = [(m, w, norm[m]) for m, w in combo if m in metrics and m in norm]
        if not terms:
            return float("nan")
        stds = []
        for recs in by_scene.values():
            vals = []
            for rec in recs:
                if all(isinstance(rec.get(m), (int, float)) and math.isfinite(rec[m])
                       for m, _w, _n in terms):
                    vals.append(sum(sign * w * (float(rec[m]) - mu) / sigma
                                    for m, w, (mu, sigma, sign) in terms))
            if len(vals) > 1:
                stds.append(st.pstdev(vals))
        return (sum(stds) / len(stds)) if stds else float("nan")

    out = {}
    for m, _w in combo:
        v = wsstd((m,))
        if v == v:
            out[f"reward/eff_wsstd_{m}"] = v
    gvals = {}
    for gname, gmetrics in GROUPS.items():
        v = wsstd(gmetrics)
        if v == v:
            out[f"reward/eff_wsstd_{gname}"] = v
            gvals[gname] = v
    tot = sum(gvals.values())
    if tot > 0:
        for gname, v in gvals.items():
            out[f"reward/eff_share_{gname}"] = v / tot
    return out
