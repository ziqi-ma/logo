"""Standalone voxel reprojection reward for RL on Lyra-2 rollouts.

Self-contained facade over the DiffusionNFT voxel reward so any training loop can
call it without touching nft_loop/nft_score internals or env vars. All K rollouts of
a group must share the same conditioning image and camera trajectory (that is what
makes voxel cells comparable across rollouts without alignment).

Usage::

    from lyra_2._src.rl.scoring.voxel_reward import VoxelReward

    vr = VoxelReward()                       # finalized defaults: alpha=0.5, lam=0.5
    # rollouts: {id: frames}, frames = uint8 RGB array [T_px, H, W, 3]
    # (or a video/frames-dir path; T_px frames are subsampled to one per latent)
    r_maps, scalars, diag = vr(rollouts)
    # r_maps[id]  = [T, gh, gw] floats in [0, 1] -- 0.5 neutral, <0.5 suppress,
    #               >0.5 reinforce; T = generated latent count (one scored frame per
    #               latent), grid default 8x8. Nearest-upsample to the latent HxW for
    #               a per-location loss weight (see nft_loss.compute_nft_loss and
    #               nft_voxel.expand_r_to_latent); NFT_VOXEL_PATCH=latent paints the
    #               latent grid directly and makes that upsample a no-op.
    # scalars[id] = {"R", "vggt_mse", "vggt_depth_mae"} -- the whole-clip combo
    #               reward and raw metrics (identical math to the scalar reward).
    # diag        = {"matched_frac", "voxel_std"}

The two stages are also exposed separately for distributed callers: run
``score_rollout`` per rollout wherever its frames live (GPU; VGGT loads once per
instance), gather the payloads however you like, then run ``advantage`` on the
gathered group (CPU, pure).

Scoring runs VGGT-Omega; hpsv3-blended combos need the per-frame hpsv3 scores
injected by the caller (payload key ``hpsv3_pf``) since hpsv3 lives in its own env.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from lyra_2._src.rl.scoring.nft_score import COMBOS, combine
from lyra_2._src.rl.scoring.nft_voxel import _decode_uniform_full, pervoxel_advantage

_PAYLOAD_KEYS = ("voxel_keys", "voxel_stats", "patch_voxels", "voxel_stats_t",
                 "alpha", "grid", "n_frames")

class VoxelReward:
    """Voxel group-relative reward. See module docstring for usage.

    Args mirror the finalized training defaults; ``combo`` names an entry of
    ``nft_score.COMBOS`` (geometry combos: ``reproj_rgbd``, ``reproj_rgbd_g70``).
    ``temporal="rollout"`` (the default) pools a voxel over time; ``"frame"`` compares
    (voxel, frame) cells across rollouts;
    ``"rollout"`` pools each voxel over the whole clip (severity-blind, kept for
    ablation). ``lam`` weights the within-rollout contrast term that localizes
    artifacts; 0 disables it.
    """

    def __init__(self, combo: str = "reproj_rgbd", alpha: float = 0.5,
                 lam: float = 0.5, patch_grid: Tuple[int, int] = (8, 8),
                 adv_clip_max: float = 2.0, depth_cap: float = 4.0,
                 temporal: str = "frame", num_frames: Optional[int] = None,
                 vggt_checkpoint: Optional[str] = None, device: str = "cuda"):
        self.spec = COMBOS[combo]
        self.alpha, self.lam = alpha, lam
        self.patch_grid, self.adv_clip_max = patch_grid, adv_clip_max
        self.depth_cap, self.temporal = depth_cap, temporal
        # scored frames per rollout: one per generated latent (framepack fpl=4)
        nf = num_frames or int(os.environ.get("NUM_FRAMES", "81"))
        self.n_frames = max(1, -(-(nf - 1) // 4))
        self._vggt_checkpoint = vggt_checkpoint
        self._device = device
        self._ctx = None
        self._scorer = None

    def _load(self):
        if self._ctx is None:
            import importlib

            from lyra_2._src.rl.scoring import nft_score as ns
            rr = ns._import_reproj_rgbd()
            self._scorer = importlib.import_module("scorers.reproj_voxel")
            self._ctx = rr.load(vggt_checkpoint=self._vggt_checkpoint
                                or os.environ.get("VGGT_CHECKPOINT"),
                                device=self._device)
        return self._ctx, self._scorer

    def score_rollout(self, frames) -> Dict:
        """Score one rollout (GPU). ``frames`` is a uint8 RGB array [T_px, H, W, 3],
        a video path, or a directory of PNG frames; paths are decoded to
        ``n_frames`` uniform samples. Returns the payload for :meth:`advantage`
        (plus the whole-clip scalars under ``"scalars"``)."""
        import numpy as np

        if isinstance(frames, (str, os.PathLike)):
            path = str(frames)
            if os.path.isdir(path):
                import cv2

                names = sorted(os.listdir(path))
                idxs = np.linspace(0, len(names) - 1,
                                   min(self.n_frames, len(names))).round().astype(int)
                frames = np.stack([cv2.cvtColor(cv2.imread(os.path.join(path, names[i])),
                                                cv2.COLOR_BGR2RGB) for i in idxs])
            else:
                frames = _decode_uniform_full(path, self.n_frames)
        elif frames.shape[0] > self.n_frames:
            idxs = np.linspace(0, frames.shape[0] - 1, self.n_frames).round().astype(int)
            frames = frames[idxs]

        ctx, scorer = self._load()
        res = scorer.score_voxel(ctx, frames, alpha=self.alpha,
                                 patch_grid=self.patch_grid, depth_cap=self.depth_cap)
        payload = {k: res[k] for k in _PAYLOAD_KEYS}
        scal = {"vggt_mse": res["vggt_mse"], "vggt_depth_mae": res["vggt_depth_mae"]}
        payload["scalars"] = {**scal, "R": combine(scal, self.spec)}
        return payload

    def advantage(self, payloads: Dict) -> Tuple[Dict, Dict]:
        """Group advantage over ``{rollout_id: payload}`` (CPU, pure). Returns
        ``({rollout_id: [T][gh][gw] r in [0,1]}, diagnostics)``."""
        return pervoxel_advantage(payloads, self.spec, adv_clip_max=self.adv_clip_max,
                                  lam=self.lam, temporal=self.temporal)

    def __call__(self, rollouts: Dict) -> Tuple[Dict, Dict, Dict]:
        """Score + advantage in one call for a co-located group ``{id: frames}``.
        Returns ``(r_maps, scalars, diag)``."""
        payloads = {rid: self.score_rollout(fr) for rid, fr in rollouts.items()}
        scalars = {rid: p.pop("scalars") for rid, p in payloads.items()}
        r_maps, diag = self.advantage(payloads)
        return r_maps, scalars, diag
