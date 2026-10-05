# LoGo: Local-Global Rewards for Consistent Long-Horizon Video Generation

Ziqi Ma<sup>1*</sup>, Shreya Sharma<sup>2</sup>, Mohamed El Banani<sup>2</sup>, Katja Schwarz<sup>2</sup>, Chongjie Ye<sup>2</sup>, Chao-Yuan Wu<sup>2</sup>, Li Fei-Fei<sup>2</sup>, Ben Mildenhall<sup>2</sup>, Georgia Gkioxari<sup>1</sup>, Justin Johnson<sup>2</sup>, Gowthami Somepalli<sup>2</sup>\
<sup>1</sup>California Institute of Technology &nbsp;&nbsp; <sup>2</sup>World Labs\
<sub><sup>*</sup>Work done during an internship at World Labs</sub>

[[`Project Page`](https://ziqi-ma.github.io/logo-website/)] [[`arXiv`](https://arxiv.org/abs/2610.03636)]

![teaser](media/teaser.png?raw=true)

## Table of Contents:
1. [Overview](#overview)
2. [Environment Setup](#environment)
3. [Reward](#reward)
4. [Training](#training)
5. [Data](#data)
6. [Evaluation](#evaluation)
7. [Checkpoints](#checkpoints)
8. [Citing](#citing)

## Overview <a name="overview"></a>
LoGo is a post-training method for camera-controlled video models that combines a global reward
with a spatially localized one. It leverages depth estimation (VGGT-Omega) and computes per-voxel
reprojection error, both in RGB and in depth. This repository provides code for running LoGo on
three base models: **Lyra-2** under [`models/lyra2/`](models/lyra2), **Lingbot2** under
[`models/lingbot-world-v2/`](models/lingbot-world-v2), and **UniWorld-View** under
[`models/uniworld-view/`](models/uniworld-view). Each model supplies its own sampler and trainer,
and its README gives that model's setup, launch command and configuration.

## Environment Setup <a name="environment"></a>
Environment setup for each model is shown below.

| model | environment | staging |
|---|---|---|
| Lyra-2 | [`models/lyra2/setup.sh`](models/lyra2/setup.sh) | [`models/lyra2/stage.sh`](models/lyra2/stage.sh) |
| Lingbot2 | [`models/lingbot-world-v2/requirements.txt`](models/lingbot-world-v2/requirements.txt) + [`models/lingbot-world-v2/requirements-logo.txt`](models/lingbot-world-v2/requirements-logo.txt) | [`models/lingbot-world-v2/stage.sh`](models/lingbot-world-v2/stage.sh) |
| UniWorld-View | [`models/uniworld-view/requirements.txt`](models/uniworld-view/requirements.txt) | [`models/uniworld-view/stage.sh`](models/uniworld-view/stage.sh) |

The staging script installs the packages that are not on PyPI from their own repositories (vggt-omega, utils3d,
MoGe), downloads the base model weights from their own releases, and pulls the training scenes.

## Reward <a name="reward"></a>
[`rewards/scorers/`](rewards/scorers) is the reward model implementation, shared by all three
models. Each rollout is reconstructed with VGGT-Omega, every frame's depth is fused into one
colored world point cloud, and that cloud is reprojected into every estimated camera.

| scorer | what it computes |
|---|---|
| [`reproj_rgbd.py`](rewards/scorers/reproj_rgbd.py) | the global terms — `vggt_mse` (squared RGB error against the reprojection) and `vggt_depth_mae` (absolute depth error over covered, unoccluded pixels), one scalar each per rollout |
| [`reproj_voxel.py`](rewards/scorers/reproj_voxel.py) | the same two errors pooled into world-space voxels, per (voxel, frame), plus each rollout's patch-to-voxel membership |
| [`camera_rpe.py`](rewards/scorers/camera_rpe.py) | camera adherence — relative pose error over all ordered frame pairs against the trajectory the sampler was given |
| [`scorers/hpsv3.py`](rewards/scorers/hpsv3.py) | the appearance term, HPSv3 over keyframes of the clip |

The voxel edge is a fraction of the first frame's 90th-percentile depth, set by
`NFT_REWARD_VOXEL`. Each model's `nft_voxel.py` contains the reward localization logic.

## Training <a name="training"></a>
Each model's training setup is detailed in the respective README.

| model | instructions |
|---|---|
| Lyra-2 | [`models/lyra2/README.md`](models/lyra2/README.md) |
| Lingbot2 | [`models/lingbot-world-v2/README.md`](models/lingbot-world-v2/README.md) |
| UniWorld-View | [`models/uniworld-view/README.md`](models/uniworld-view/README.md) |

Each one is a single config and a single command, for example:
```
cd models/lingbot-world-v2
./train.sh runs/logo.env
```

## Data <a name="data"></a>
We train on 1700 scenes from [DL3DV-10K](https://github.com/DL3DV-10K/Dataset), excluding the
VideoGPA eval subset, and convert the public COLMAP poses into the various base models'
conventions. [`data/README.md`](data) details how to build the training set from the public
DL3DV dataset.

## Evaluation <a name="evaluation"></a>
Evaluation runs the same scoring functions in [`rewards/scorers/`](rewards/scorers). First
generate clips with the model's own sampler:

| model | generation script |
|---|---|
| Lyra-2 | `python -m lyra_2._src.rl.inference.evaluate --adapter <adapter> --scenes-root <dir> --scenes "..."` |
| Lingbot2 | `python -m wan.rl.inference.gen_scenes --input_base <scenes> --videos_base <out> --lora_path <adapter>` |
| UniWorld-View | `python -m rl.inference.eval_gen --ws_prefix <out> --rl-ckpt <adapter>` |

For the base-model clips, omit the adapter argument -- except Lyra-2, where `--adapter`
is required and the baseline is the DMD-distillation LoRA itself. Then score them:

```
python eval/score_videos.py --videos <out> --out results/logo \
    --traj-root <scene root> --videogpa --hpsv3
```
`--traj-root` is where the reference camera paths live, as `<scene root>/<scene>/trajectory.npz`.
This writes one JSON per clip. [`eval/README.md`](eval) has the generation command for each
model; [`rewards/README.md`](rewards) covers the flags and what each metric comes from.

## Checkpoints <a name="checkpoints"></a>
The LoRA adapter each model was post-trained to is at
[huggingface.co/ziqima/LoGo](https://huggingface.co/ziqima/LoGo), one file per model. The base
weights they load on top of come from each model's own release.

| model | adapter | pass it to |
|---|---|---|
| Lyra-2 | `logo-lyra2-lora.pt` | `evaluate --adapter` |
| Lingbot2 | `logo-lingbot-world-v2-lora.pt` | `gen_scenes --lora_path` |
| UniWorld-View | `logo-uniworld-view-lora.safetensors` | `eval_gen --rl-ckpt` |

The commands to pass them to are in [Evaluation](#evaluation) above, and in
[`eval/README.md`](eval).

## Citing <a name="citing"></a>
Please use the following BibTeX entry if you find our work helpful!

```BibTex
@misc{ma2026logo,
  title={LoGo: Local-Global Rewards for Consistent Long-Horizon Video Generation},
  author={Ziqi Ma and Shreya Sharma and Mohamed El Banani and Katja Schwarz and Chongjie Ye and Chao-Yuan Wu and Li Fei-Fei and Ben Mildenhall and Georgia Gkioxari and Justin Johnson and Gowthami Somepalli},
  year={2026},
  eprint={2610.03636},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2610.03636},
}
```
