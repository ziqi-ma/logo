# LoGo Training for Lyra-2

## Setup

```bash
./setup.sh
conda activate lyra2
./stage.sh
```

`setup.sh` creates the conda environment, installs the CUDA 12.8 toolchain and torch 2.7.1, and
compiles the extensions against them. Persist the `LD_LIBRARY_PATH` it prints in your shell
profile. We tested on Ubuntu 22.04 with CUDA 12.8 on H100. `stage.sh` downloads the base weights
(`nvidia/Lyra-2.0`), VGGT-Omega (`facebook/VGGT-Omega`, gated — accept the terms once and
`huggingface-cli login`), the packages that are not on PyPI and the DL3DV training scenes into
`/inputs/scenes`. The train/val split is
[`data/scene_sets/dl3dv_lyra2.json`](../../data/scene_sets/dl3dv_lyra2.json).

## Train

```bash
./train.sh configs/runs/logo.json
```

Adapters are written to `$OUT_DIR/adapters` (default `/outputs`).

For training, `POSE_SCALE=0.4` scales dl3dv's COLMAP unit to Lyra2 unit, used for training and
evaluation on DL3DV.

The configuration is the run — every value below comes from that file, and the reward dials are
read from the environment by `nft_voxel.py` / `nft_score.py` rather than passed as flags:

| | |
|---|---|
| reward | `NFT_COMBO=reproj_rgbd`, schedule `reproj_rgbd_g70:5,hpsv3_only:6,camera_only:3` |
| voxel reward | `NFT_REWARD_VOXEL=0.1` (voxel edge = 0.1 x frame-1 p90 depth), `NFT_VOXEL_PATCH=latent`, `NFT_VOXEL_LOCAL=0.5` |
| voxel/global mix | `NFT_REWARD_MIX=0.5` |
| optimization | `LR=3.33e-5`, `NFT_BETA_KL=1e-4`, `EMA_DECAY=0.9`, `GRAD_STEPS=1`, `INNER_EPOCHS=1` |
| rollouts | `K_ROLLOUTS=16`, `SCENES_PARALLEL=8`, `NUM_STEPS=300`, `NUM_FRAMES=241`, `RESOLUTION=480,832`, `POSE_SCALE=0.4` |
| scorer frames | `NFT_REPROJ_FRAMES=16`, `NFT_HPSV3_KEYFRAMES=60` |
| validation | `VAL_EVERY=5`, `VAL_K=4` |
| calibration | `Lyra-2/lyra_2/_src/rl/scoring/norm.json`, with `NFT_NORM_OVERRIDE` for the camera terms |

## Evaluate

```bash
torchrun --standalone --nproc_per_node=8 -m lyra_2._src.rl.inference.evaluate \
  --checkpoint_dir checkpoints/model --experiment lyra2 \
  --adapter /outputs/adapters/nft_new_step0150.pt \
  --scenes-root /inputs/scenes --scenes "$VAL_SCENES" \
  --combo reproj_rgbd --k-rollouts 8 --num_frames 241 --pose_scale 0.4 \
  --videos-uri s3://<bucket>/<run>/eval-videos

python ../../eval/score_videos.py --videos s3://<bucket>/<run>/eval-videos \
  --out results/logo --traj-root /inputs/scenes --videogpa --hpsv3
```

For the base-model clips pass the DMD-distillation LoRA as `--adapter`. See
`../../eval/README.md`; the metrics are listed in `../../rewards/README.md`.

The Lyra-2 code here builds on NVIDIA Cosmos; its open-source license attributions are kept
upstream at [ATTRIBUTIONS.md](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/ATTRIBUTIONS.md).
