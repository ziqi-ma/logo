# Evaluation

## 1. Generate clips

Each model samples with its own sampler. For the base-model clips to compare against, run each
command again without the adapter argument; Lyra-2 takes its DMD-distillation LoRA there instead.

**Lyra-2**

```bash
cd models/lyra2
torchrun --standalone --nproc_per_node=8 -m lyra_2._src.rl.inference.evaluate \
  --checkpoint_dir checkpoints/model --experiment lyra2 \
  --adapter /outputs/adapters/nft_new_step0150.pt \
  --scenes-root /inputs/scenes --scenes "$VAL_SCENES" \
  --combo reproj_rgbd --k-rollouts 8 --num_frames 241 \
  --videos-uri s3://<bucket>/<run>/eval-videos
```

**Lingbot2**

```bash
cd models/lingbot-world-v2
python -m wan.rl.inference.gen_scenes --input_base <scene root> --videos_base <out> \
    --lora_path <adapter.pt>
```

**UniWorld-View**

```bash
cd models/uniworld-view
python -m rl.inference.eval_gen --ws_prefix s3://<bucket>/<run> \
    --rl-ckpt /outputs/adapters/nft_new_step0170.safetensors
```

Each writes `<scene>/<clip>.mp4` under its output prefix.

## 2. Score them

```bash
python eval/score_videos.py --videos <dir|s3://prefix> --out results/<run> \
    [--traj-root <dir|s3://prefix>] [--videogpa] [--hpsv3] [--vq] [--limit N]
```

One JSON per clip. A clip already scored is skipped, so an interrupted pass just runs again;
clips whose JSON records an error are retried. `--limit N` scores N clips, for checking a setup
before committing a GPU to the full set. `../rewards/README.md` lists what each flag adds and
which scorer computes it.

## 3. Evaluation on TrajectoryBench

### Get the data

```bash
huggingface-cli download ziqima/TrajectoryBench --repo-type dataset --local-dir trajbench
```

`metadata.jsonl` has one line per clip:

| field | meaning |
|---|---|
| `id` | four-digit clip id, matching `clips/<id>/` |
| `category` | `indoor`, `outdoor`, or `transition` (passing through a door into a new space) |
| `difficulty` | `easy`, `medium` or `hard` camera motion |
| `stylized` | whether the scene is stylized rather than photoreal |
| `num_frames` | how many frames to generate for this clip: 81 or 241 |
| `caption`, `camera_prompt` | text descriptions, also in `clips/<id>/prompt.json` |

Each `clips/<id>/` holds `first_frame.png` (832x480) and `trajectory.npz`, whose `w2c`,
`intrinsics` and `image_wh` arrays are what the generators read.

### Stage a group

Clips are evaluated a group at a time, a group being all the clips that share a `category`, a
`difficulty` and whether they are stylized -- the 16 combinations of those three fields. Generation
settings differ between groups, and results are reported per group. List them with:

```bash
python eval/stage_trajbench.py --list
```

Then stage the one you want to evaluate:

```bash
python eval/stage_trajbench.py --category indoor --difficulty hard --out runs/indoor_hard
```

That writes, per clip, the three files each generator reads:

```
<out>/inputs/<id>/<id>.png             <- that clip's first_frame.png
<out>/inputs/<id>/trajectory.npz       <- that clip's trajectory.npz
<out>/inputs/<id>/captions.json        <- {"0": ""}
```

`eval/stage_trajbench.py` also writes `<out>/settings.json`, holding that group's generation
settings. `eval/run_trajbench.sh` depends on this file and will not run without it.

`<out>` can be any local directory. Keep the group in its name, for example `runs/ext_hard_photo`
-- the transition groups are recognised by `transition_` appearing there.

### Settings

Evaluation settings are in the `settings.json` that `eval/stage_trajbench.py` writes into the group
directory.

For Lingbot2, since it internally normalizes translation scale and we cannot control the
translation amount directly, we set `--transition_fov_scale` 6.0 for transition scenes so that it passes
through the door like the other models, rather than only getting close.

Lyra-2 and Lingbot2 should be evaluated with the empty caption from `captions.json`.
UniWorld-View uses its internal BLIP-2 captioner. Lyra-2 and Lingbot2 generate up to 241
frames. Since UniWorld-View's base model is trained only for 81-frame generation, it generates
81-frame videos using every third pose.

### Generate and score

`eval/run_trajbench.sh` reads the group's `settings.json`, works out the ids to run, and passes each
model what it needs -- the frame count, and the wider lens for Lingbot2, which is the only
model with one:

```bash
eval/run_trajbench.sh lingbot  runs/ext_hard <adapter.pt>
eval/run_trajbench.sh lyra2    runs/ext_hard <adapter.pt>
eval/run_trajbench.sh uniworld runs/ext_hard <adapter.safetensors>
```

`SEED`, `NPROC`, `CKPT_DIR`, `CHECKPOINT_DIR` and `DEVICE` override the defaults, and any further
arguments are passed straight through to the generator. Clips land in
`<group-dir>/videos/<id>/video/<id>.mp4`.

```bash
python -m wan.rl.inference.gen_trajbench --ws_prefix <out> --ids 0003 \
  --num_frames 241 --transition_fov_scale 6.0 --base_seed 1 \
  --lora_path <adapter.pt> --ckpt_dir weights/<base>
```

Then score as in section 2, with `--videos <group-dir>/videos`.

## 4. Exact reproducibility

The published clips were generated on H100s with 132 streaming multiprocessors, and should be
reproduced in the same environment. Generation could look different on other architectures, even
with the same seed. One source of discrepancy is RNG. To mitigate this, we provide `eval/rng_emu`,
invoked by setting `EMU_MPC=132` when running `eval/run_trajbench.sh`, so that RNG is aligned to
the original environment even when on another architecture, e.g. an A100. While this mitigates
discrepancies, bit exactness should not be expected on other architectures due to low-level
execution differences.
