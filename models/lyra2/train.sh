#!/usr/bin/env bash
# Reproduce the DiffusionNFT run with nothing but this repo, plain torchrun and S3.
#
#   ./stage.sh                                                  # once
#   ./train.sh configs/runs/logo.json
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR the way torchrun wants them. The published
# run was 8 nodes x 8 H100 (64 GPUs); fewer nodes changes the rollout batch and therefore the
# trajectory. DRY_RUN=1 prints the composed command without needing a GPU.
set -euo pipefail
CFG="${1:?usage: train.sh configs/runs/<run>.json}"
PY="${PYTHON:-python3}"
# The loop resolves lyra_2/_src/configs/config.py and ./checkpoints/... against the working
# directory, so it runs from the Lyra-2 tree, where stage.sh puts the weights. Paths given
# on the command line are made absolute first.
HERE="$(cd "$(dirname "$0")" && pwd)"
CFG="$(cd "$(dirname "$CFG")" && pwd)/$(basename "$CFG")"

# The config's envvars ARE the run: the reward dials are read from the environment by
# nft_voxel.py / nft_score.py, not passed as flags, which is why they live here.
while IFS='=' read -r k v; do
  [ -n "$k" ] && export "$k=$v"
done < <("$PY" -c "
import json,sys
e=json.load(open(sys.argv[1]))['envvars']
print('\n'.join(f'{k}={v}' for k,v in e.items() if v != ''))" "$CFG")

RUN_NAME="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['run'])" "$CFG")"
SCENES_ROOT="${SCENES_ROOT:-/inputs/scenes}"
OUT_DIR="${OUT_DIR:-/outputs}"
CKPT_DIR="${CKPT_DIR:-checkpoints/model}"

# Train/val split from the scene-set definition stage.sh downloaded; SCENE_SET overrides.
SCENE_SET="${SCENE_SET:-$HERE/../../data/scene_sets/dl3dv_lyra2.json}"
[ -f "$SCENE_SET" ] || { echo "missing $SCENE_SET (run ./stage.sh)"; exit 1; }
SCENES="$("$PY" -c "
import json,sys; print(' '.join(json.load(open(sys.argv[1]))['train']))" "$SCENE_SET")"
VAL_SCENES="$("$PY" -c "
import json,sys; print(' '.join(json.load(open(sys.argv[1])).get('val', [])))" "$SCENE_SET")"

# The DMD LoRA is both the policy and the reference, so step 0 is exactly the DMD model.
POLICY_LORA="${POLICY_LORA:-checkpoints/lora/dmd_distillation.safetensors}"
REF_LORA="${REF_LORA:-$POLICY_LORA}"

cd "$HERE/Lyra-2"
if command -v torchrun >/dev/null 2>&1; then LAUNCH=(torchrun)
else LAUNCH=("$PY" -m torch.distributed.run); fi

# Keep --rdzv-conf=join_timeout: plain timeout= is silently ignored, which leaves the
# rendezvous on its 600 s default and loses slow-starting nodes.
RUN=("${LAUNCH[@]}" --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="${GPUS_PER_NODE:-8}" \
  --rdzv_backend=c10d --rdzv_id="$RUN_NAME" \
  --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:29500" \
  --rdzv-conf=join_timeout="${JOIN_TIMEOUT:-2400}" \
  -m lyra_2._src.rl.loop.nft_loop \
  --checkpoint_dir "$CKPT_DIR" --experiment "${EXPERIMENT:-lyra2_nft}" \
  --scenes-root "$SCENES_ROOT" --scenes "$SCENES" \
  --combo "${NFT_COMBO}" --k-rollouts "${K_ROLLOUTS:-16}" \
  --num-steps "${NUM_STEPS:-75}" --num_frames "${NUM_FRAMES:-241}" \
  --resolution "${RESOLUTION:-480,832}" --pose_scale "${POSE_SCALE:-0.4}" \
  --policy-lora "$POLICY_LORA" --ref-lora "$REF_LORA" \
  --work-dir "$OUT_DIR/loop" --new-lora-out "$OUT_DIR/adapters" \
  --ckpt-every "${CKPT_EVERY:-1}" --ema-decay "${EMA_DECAY:-0.9}" --lr "${LR:-3e-5}" \
  --scenes-parallel "${SCENES_PARALLEL:-8}" \
  --grad-steps-per-collection "${GRAD_STEPS:-1}" --inner-epochs "${INNER_EPOCHS:-1}" \
  --val-scenes "$VAL_SCENES" --val-every "${VAL_EVERY:-0}" --val-k "${VAL_K:-0}" \
  --adv-clip-max "${ADV_CLIP_MAX:-1.3}")
[ -n "${VAL_METRICS:-}" ] && RUN+=(--val-metrics "$VAL_METRICS")

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${RUN[@]}"; echo; exit 0; fi
exec "${RUN[@]}"
