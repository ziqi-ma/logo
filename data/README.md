# Data

Training scenes come from the public [DL3DV-10K](https://github.com/DL3DV-10K/Dataset) dataset,
excluding the evaluation split.

## Scene input

Every model reads one directory per scene, named by the DL3DV scene hash:

```
<scene_hash>/image.png          frame 0, the conditioning image
<scene_hash>/lyra2_traj.npz     w2c [F,4,4] f32, world-to-camera, OpenCV, w2c[0] = I
                                intrinsics [F,3,3] f32, pixel-space K at the frame size
                                image_height, image_width  int scalars
```

## Training set

`scene_sets/dl3dv.json` lists the scene ids, as `{"train": [...], "val": [...]}`: 1700 training
scenes plus 25 held out for in-training validation. They are not the first 1700 of DL3DV -- the
pool is DL3DV-10K minus the entire VideoGPA `captions_1K` subset, so training never sees a scene
from the eval distribution, and the run's scenes were drawn from that pool with a fixed seed.
Lingbot2 and UniWorld-View train on exactly this set; Lyra-2 used
`scene_sets/dl3dv_lyra2.json`, the same split less one scene.

## Building the trajectories

```bash
python3 build_trajectories.py --dl3dv-root /path/to/DL3DV-10K --out /inputs/scenes \
    --scenes scene_sets/dl3dv.json
```

Per scene this reads the native COLMAP poses DL3DV ships in `transforms.json` (nerfstudio
export, 287-419 poses per scene), converts them to OpenCV, resamples the first half of the path
to 241 poses by position lerp and rotation slerp, and relativizes `w2c` to frame 0. This length
gives reasonable spread and camera speed in 241 frames.

Two kinds of scene are dropped: fewer than `--num-frames` + 4 posed frames, which means a
truncated COLMAP reconstruction, and a maximum `||t - t0||` above `--max-motion`, which means a
broken one. Normal scenes sit at 8-14, and the script prints the distribution so the cutoff
can be retuned.

Intrinsics are scaled to the size of the conditioning frame that was used, and every loader
rescales them again to its own generation resolution. Each model's `stage.sh` skips this step and
downloads scenes that were already built this way.

## Per-model conversion

Lyra-2 trains on the layout above directly. The other two convert it once:

| model | script | produces |
|---|---|---|
| Lingbot2 | `models/lingbot-world-v2/wan/rl/data/scene_convert.py` | `poses.npy` (c2w) + `intrinsics.npy` (fx, fy, cx, cy) rescaled to 480x832 |
| UniWorld-View | `models/uniworld-view/rl/data/scene_prep.py` | a `CondBundle` per scene: the BLIP-2 caption, its T5 embedding and the MoGe geometry, so the loop never loads those models |

The translation unit is not uniform across models, so for the ones that do not self-normalize we
have to find a translation scale that gives an appropriate camera trajectory -- neither too fast
nor too slow.

The stored trajectory carries COLMAP's own scale, which might not align with each model's
translation scale, so it has to be scaled appropriately. Lyra-2 scales the translation column at
load, `POSE_SCALE=0.4` in its run config. UniWorld-View uses
`pose_scale` 1.0, so `scene_prep.py` bakes the trajectory into the bundle as stored.
Lingbot2 internally normalizes translations by the clip's peak step, and thus setting pose
scale has no effect.
