#!/usr/bin/env bash
# Reproduce the DiffusionNFT run with nothing but this repo, plain torchrun and S3.
#
#   ./stage.sh                                   # once: base weights, scenes, VGGT
#   ./train.sh runs/logo.env
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR the way torchrun wants them. The published
# run was 16 nodes x 8 H100 (128 GPUs); fewer nodes changes the rollout batch and therefore
# the trajectory.
set -euo pipefail
CFG="${1:?usage: train.sh runs/<run>.env}"
# shellcheck disable=SC1090
source "$CFG"
: "${RUN_NAME:?config has no RUN_NAME}"

BASE_DIR="${BASE_DIR:-./weights/lingbot-world-v2-14b-causal-fast}"
SCENES_ROOT="${SCENES_ROOT:-./data/scenes}"
OUT_DIR="${OUT_DIR:-./out/$RUN_NAME}"
# Train/val split. A sibling <run>.scenes / <run>.val_scenes wins if present; otherwise
# SCENE_SET names the split JSON, and the default is the one the published run used.
SCENES_FILE="${CFG%.env}.scenes"
VAL_SCENES_FILE="${CFG%.env}.val_scenes"
SCENE_SET="${SCENE_SET:-../../data/scene_sets/dl3dv.json}"
if [ -f "$SCENES_FILE" ]; then
  SCENES="$(paste -sd, "$SCENES_FILE")"
  VAL_SCENES=""
  [ -f "$VAL_SCENES_FILE" ] && VAL_SCENES="$(paste -sd, "$VAL_SCENES_FILE")"
else
  SET_JSON="$SCENE_SET"
  case "$SET_JSON" in
    s3://*) SET_JSON="./data/$(basename "$SCENE_SET")"
            [ -f "$SET_JSON" ] || { mkdir -p ./data; aws s3 cp "$SCENE_SET" "$SET_JSON"; } ;;
  esac
  read -r SCENES VAL_SCENES <<EOF
$("${PYTHON:-python3}" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(','.join(d['train']), ','.join(d.get('val', [])))" "$SET_JSON")
EOF
  [ -n "$SCENES" ] || { echo "no train scenes in $SET_JSON"; exit 1; }
  echo "[train.sh] split from $SCENE_SET" >&2
fi

# The reward's own weights, downloaded by stage.sh.
export VGGT_CHECKPOINT="${VGGT_CHECKPOINT:-./weights/vggt/vggt_omega_1b_512.pt}"
# Reward dials are read from the environment by wan/rl/nft_voxel.py, not passed as flags,
# which is exactly why they had to be captured in runs/*.env.
export NFT_COMBO NFT_REWARD_VOXEL NFT_REWARD_MIX NFT_VOXEL_LOCAL \
       NFT_VOXEL_PATCH NFT_VOXEL_TEMPORAL NFT_VOXEL_DEPTH_CAP NFT_REWARD_LATENT \
       NFT_LATENT_LOCAL NFT_REWARD_WINDOW NFT_BETA_KL NFT_NORM_OVERRIDE NFT_REPROJ_FRAMES \
       NFT_HPSV3_KEYFRAMES NFT_VAL_HPSV3 NFT_VAL_GPA_FRAMES NFT_VAL_FULL_METRICS 2>/dev/null || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export PYTHONPATH="$PWD:$PWD/wan/third_party:${PYTHONPATH:-}"

# DRY_RUN=1 prints the composed command and exits, so a config can be checked without a
# GPU allocation (and so the flags below can be diffed against nft_loop's parser).
# torchrun is not always on PATH (a venv may ship only the module), and `exec torchrun`
# then dies with "torchrun: not found" after all the config work is done. Prefer the binary,
# fall back to the module, which is the same entry point.
if command -v torchrun >/dev/null 2>&1; then LAUNCH=(torchrun)
else LAUNCH=("${PYTHON:-python3}" -m torch.distributed.run); fi
# --new-lora-uri / --videos-uri go through gcs_util, which understands s3:// URIs only. Handing them a local path fails in upload_file_verified ("Invalid bucket name") on
# every checkpoint. With a local OUT_DIR they are omitted: the loop still writes
# nft_new_step*.pt into $OUT_DIR/loop, which is what a local run wants.
URI_FLAGS=()
case "$OUT_DIR" in
  *://*) URI_FLAGS=(--new-lora-uri "$OUT_DIR/adapters" --videos-uri "$OUT_DIR/videos");;
  *) echo "[train.sh] local OUT_DIR: checkpoints stay in $OUT_DIR/loop (no upload)" >&2;;
esac

RUN=("${LAUNCH[@]}" --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" --nproc_per_node="${GPUS_PER_NODE:-8}" \
  --rdzv_backend=c10d --rdzv_id="$RUN_NAME" --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:29500" \
  -m wan.rl.loop.nft_loop --keep-all-checkpoints \
  --checkpoint-dir "$BASE_DIR" \
  --scenes-root "$SCENES_ROOT" --scenes "$SCENES" \
  --combo "${NFT_COMBO}" --k-rollouts "${K_ROLLOUTS:-16}" \
  --num-steps "${NUM_STEPS:-75}" --num-frames "${NUM_FRAMES:-241}" \
  --local-attn-size "${LOCAL_ATTN_SIZE:--1}" --sink-size "${SINK_SIZE:-0}" \
  --work-dir "$OUT_DIR/loop" "${URI_FLAGS[@]}" \
  --ckpt-every "${CKPT_EVERY:-1}" --videos-every "${VIDEOS_EVERY:-0}" \
  --val-scenes "$VAL_SCENES" --val-every "${VAL_EVERY:-0}" --val-k "${VAL_K:-0}" --val-videos 1 \
  --scenes-parallel "${SCENES_PARALLEL:-8}" \
  --grad-steps-per-collection "${GRAD_STEPS:-1}" --inner-epochs "${INNER_EPOCHS:-1}" \
  --lr "${LR:-3e-4}" --ema-decay "${EMA_DECAY:-0.9}" --nft-beta "${NFT_BETA:-1.0}" \
  --adv-clip-max "${ADV_CLIP_MAX:-1.3}")
if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${RUN[@]}"; echo; exit 0; fi
exec "${RUN[@]}"
