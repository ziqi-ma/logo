# LoGo Training for Lingbot2

## Setup

```bash
pip install -r requirements.txt -r requirements-logo.txt
./stage.sh
```

Downloads the base weights at a pinned revision, VGGT-Omega (`facebook/VGGT-Omega`, gated —
accept the terms once and `huggingface-cli login`), the packages that are not on PyPI, and the
training scenes into `data/scenes`.

## Train

```bash
./train.sh runs/logo.env
```

Adapters are written to `$OUT_DIR/loop` (default `./out/<run>`); set `OUT_DIR` to an `s3://`
prefix to upload them instead.
The train/val split is [`data/scene_sets/dl3dv.json`](../../data/scene_sets/dl3dv.json).

The `.env` is the run — the reward dials are read from the environment by `wan/rl/scoring/nft_voxel.py`
rather than passed as flags:

| | |
|---|---|
| reward | `NFT_COMBO=reproj_rgbd` = `[[vggt_mse, 0.5], [vggt_depth_mae, 0.5]]` |
| voxel reward | `NFT_REWARD_VOXEL=0.1` (voxel edge = 0.1 x frame-1 p90 depth), `NFT_VOXEL_LOCAL=0.5` |
| local/global mix | `NFT_REWARD_MIX=0.5` |
| optimization | `LR=3e-4`, `NFT_BETA_KL=1e-4`, `NFT_BETA=1`, `EMA_DECAY=0.9`, `ADV_CLIP_MAX=1.3`, `GRAD_STEPS=1`, `INNER_EPOCHS=1` |
| rollouts | `K_ROLLOUTS=16`, `SCENES_PARALLEL=8`, `NUM_STEPS=75`, `NUM_FRAMES=241` |
| long horizon | `LOCAL_ATTN_SIZE=18`, `SINK_SIZE=6` (the KV cache costs ~1.28 GB per latent frame) |
| adapters | `LORA_SCOPE=attn+ffn+cam`, rank 32 / alpha 64 |
| calibration | `wan/rl/scoring/norm.json`, with `NFT_NORM_OVERRIDE` for the 241-frame regime |

128 H100. Gradient checkpointing is mandatory: 81-frame training peaks near 71 GB on one card.

## Evaluate

```bash
python -m wan.rl.inference.gen_scenes --input_base <scene root> --videos_base <out> \
    --lora_path <adapter.pt>

python ../../eval/score_videos.py --videos s3://<bucket>/<run>-dl3dv --out results/logo \
    --traj-root s3://videogen-rl/training_data/dl3dv-mirror-train-fhalf241-lingbot \
    --videogpa --hpsv3
```

Omit `--lora_path` for the base-model clips. See `../../eval/README.md`; the metrics are listed
in `../../rewards/README.md`.
