# LoGo Training for UniWorld-View

## Setup

Python 3.10, CUDA 12.1.

```bash
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install carvekit --no-deps
bash extern/install_pytorch3d.sh
./stage.sh
```

Downloads the base weights (UniView, Wan2.1-VACE, CausVid LoRA), VGGT-Omega
(`facebook/VGGT-Omega`, gated — accept the terms once and `huggingface-cli login`), the packages
that are not on PyPI, and the train/val scene lists into `rl/scenes/`.

Training reads a precomputed `CondBundle` per scene, so the loop never loads BLIP-2/T5/MoGe.
Build them and point the reward at the trajectories it scores camera adherence against:

```bash
python -m rl.data.scene_prep --scenes-root <scene dir> \
    --scenes $(cat rl/scenes/train_scenes_1658.txt)     # -> /inputs/cond

export VGGT_CHECKPOINT=weights/vggt/vggt_omega_1b_512.pt
export NFT_TRAJ_URI=/inputs/traj                         # or an s3:// prefix
export HPSV3_PY=/opt/hpsv3-venv/bin/python               # hpsv3 pins its own transformers
```

## Train

```bash
./train.sh rl/run_configs/logo.json
```

Adapters are written to the config's `ckpt_out` (default `/outputs/adapters`).

The config is the run — `train.sh` emits its `args` as flags and its `voxel` block as the
`NFT_*` environment, so the launch cannot drift from the record:

| | |
|---|---|
| reward | `combo reproj_rgbd` = `[[vggt_mse, 0.5], [vggt_depth_mae, 0.5]]`, `alt_schedule 12:5:3` over reproj_rgbd / hpsv3_only / camera_only |
| voxel reward | `voxel.alpha 0.1` (voxel edge = 0.1 x frame-1 p90 depth), `voxel.patch 60x104` (the full latent grid), `voxel.local 0.5`, `voxel.depth_cap 4.0` |
| local/global mix | `voxel.mix 0.5` |
| optimization | `lr 3e-4`, `beta_kl 0.5`, `beta 1.0`, `ema_decay 0.9`, `adv_clip_max 1.3`, `lora_l2_lambda 10.0`, `lora_l2_target 1.3` |
| rollouts | `k_rollouts 16`, `scenes_parallel 8`, `num_steps 250`, `num_sample_steps 8`, `guidance 4.0` |
| calibration | `rl/scoring/norm.json` |

## Evaluate

```bash
python -m rl.inference.eval_gen --ws_prefix s3://<bucket>/<run> \
    --rl-ckpt /outputs/adapters/nft_new_step0170.safetensors --pose-scale 1.0

python ../../eval/score_videos.py --videos s3://<bucket>/<run>/videos --out results/logo \
    --traj-root /inputs/traj --videogpa --hpsv3
```

Omit `--rl-ckpt` for the base-model clips. Training conditions on a BLIP-2 caption baked into
each bundle, so evaluate with captions too. See `../../eval/README.md`; the metrics are listed
in `../../rewards/README.md`.
