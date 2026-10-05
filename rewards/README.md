# Reward scorers

Reward functions in this directory are used across models.

## What each metric is

| metric | scorer | what it computes | direction |
|---|---|---|---|
| `vggt_mse` | `scorers/reproj_rgbd.py` | squared RGB error between each frame and the reprojection of the fused cloud, over channels and pixels | lower |
| `vggt_depth_mae` | `scorers/reproj_rgbd.py` | absolute error between each frame's predicted depth and the rendered depth, over covered unoccluded pixels | lower |
| per-voxel `vggt_mse` / `vggt_depth_mae` | `scorers/reproj_voxel.py` | the same two errors accumulated per voxel, plus each rollout's patch-to-voxel membership | lower |
| `rpe_rot`, `rpe_trans` | `scorers/camera_rpe.py` | relative pose error over all ordered frame pairs against the trajectory the sampler was given, scale-aligned, translation normalized by path length | lower |
| `hpsv3_vid` | `scorers/hpsv3.py` | mean [HPSv3](https://github.com/MizzenAI/HPSv3) over keyframes of the clip | higher |
| `vr_vq` | `scorers/videoreward.py` | [VideoReward](https://github.com/KwaiVGI/VideoAlign)'s visual-quality score | higher |
| [`epipolar`](https://github.com/KupynOrest/epipolar-dpo), `vggt_psnr/ssim/lpips/mvcs/...` | `scorers/dl3dv_videogpa.py` | the [VideoGPA](https://github.com/Hongyang-Du/VideoGPA) suite, over the same reconstruction | mixed |

Every metric's `mu`, `sigma` and direction live in the model's `scoring/norm.json`, since they
are measured per model.

## The voxel reward

For one scene group -- the K rollouts sharing a conditioning image and camera trajectory --
[VGGT-Omega](https://github.com/facebookresearch/vggt-omega) runs on one decoded frame per generated latent at the clip's own aspect ratio, giving
per-pixel RGB and depth reprojection errors against the fused-cloud render. Those errors are
aggregated into voxels of edge `alpha * s`, with `s` the p90 depth of the first frame, which is
shared across the group, so the same voxel means the same place in every rollout with no
alignment step. Each voxel is then z-scored across the rollouts that observed it; optionally,
for small group sizes, a score against the median of the rollout's own voxels can be added. The
result is clipped and mapped to `r` in [0, 1]. During DiffusionNFT, each latent patch receives
the average score of the voxels that project to it.

## Hyperparameters

| env | default | meaning |
|---|---|---|
| `NFT_REWARD_VOXEL` | unset/0 = off | voxel edge as a fraction of frame 1's p90 depth |
| `NFT_VOXEL_LOCAL` | 0 | weight of the within-rollout median term |
| `NFT_VOXEL_TEMPORAL` | `rollout` | `rollout` = one error per voxel; `frame` = split each voxel per frame |
| `NFT_VOXEL_PATCH` | `latent` | grid the score is applied on; `latent` is the latent dimension, `GHxGW` a coarser one |
| `NFT_VOXEL_DEPTH_CAP` | 4.0 | drop pixels deeper than this multiple of the scale anchor |
| `NFT_REWARD_MIX` | 0.5 | weight on the voxel reward in the blend with the global one; 1.0 is pure voxel |

## Scoring generated clips

Scoring infrastructure is shared across 3 models. `../eval/score_videos.py` runs these same
scorers over a directory or `s3://` prefix of generated clips, writing one JSON per clip. A clip
whose JSON already exists is skipped, so an interrupted pass just runs again.

```bash
python eval/score_videos.py --videos <dir|s3://prefix> --out results/<run> \
    [--traj-root <dir|s3://prefix>] [--videogpa] [--hpsv3] [--vq] [--limit N]
```

| flag | adds |
|---|---|
| (always) | `vggt_mse`, `vggt_depth_mae` |
| `--traj-root` | `rpe_rot`, `rpe_trans` |
| `--hpsv3` | `hpsv3_vid` |
| `--vq` | `vr_vq` |
| `--videogpa` | `epipolar`, `vggt_psnr/ssim/lpips/mvcs/...` |
