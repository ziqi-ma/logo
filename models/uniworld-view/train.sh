#!/usr/bin/env bash
# Reproduce the DiffusionNFT run with nothing but this repo, plain torchrun and S3.
#
#   ./stage.sh                                     # once
#   ./train.sh rl/run_configs/logo.json
#
# Multi-node: set NNODES / NODE_RANK / MASTER_ADDR the way torchrun wants them.
set -euo pipefail
CFG="${1:?usage: train.sh rl/run_configs/<run>.json}"
PY="${PYTHON:-python3}"

# The config's `voxel` block is the reward localization, read from the environment by
# rl/nft_voxel.py rather than passed as flags.
while IFS='=' read -r k v; do
  [ -n "$k" ] && export "$k=$v"
done < <("$PY" -c "
import json, sys
v = json.load(open(sys.argv[1])).get('voxel') or {}
if v:
    print(f\"NFT_REWARD_VOXEL={v['alpha']}\")
    print(f\"NFT_REWARD_MIX={v['mix']}\")
    print(f\"NFT_VOXEL_LOCAL={v['local']}\")
    print(f\"NFT_VOXEL_TEMPORAL={v['temporal']}\")
    print(f\"NFT_VOXEL_DEPTH_CAP={v['depth_cap']}\")
    p = v.get('patch')
    if p: print(f\"NFT_VOXEL_PATCH={p[0]}x{p[1]}\")" "$CFG")

# The scene lists stage.sh downloaded; the config names which it used.
SCENES_FILE="${SCENES_FILE:-rl/scenes/train_scenes_1658.txt}"
VAL_SCENES_FILE="${VAL_SCENES_FILE:-rl/scenes/val_scenes_25.txt}"
[ -f "$SCENES_FILE" ] || { echo "missing $SCENES_FILE (run ./stage.sh)"; exit 1; }

# args map one-to-one onto rl.loop.nft_loop's flags (underscore -> dash). Emitted from the config
# so the launch cannot drift from the record of what ran.
mapfile -t ARGS < <("$PY" -c "
import json, sys
a = json.load(open(sys.argv[1]))['args']
skip = {'scenes', 'val_scenes', 'scenes_file', 'val_scenes_file'}
for k, v in a.items():
    if k in skip or v is None:
        continue
    flag = '--' + k.replace('_', '-')
    if isinstance(v, bool):
        if v: print(flag)
    else:
        print(flag); print(v)" "$CFG")

if command -v torchrun >/dev/null 2>&1; then LAUNCH=(torchrun)
else LAUNCH=("$PY" -m torch.distributed.run); fi

RUN=("${LAUNCH[@]}" --nnodes="${NNODES:-1}" --node_rank="${NODE_RANK:-0}" \
  --nproc_per_node="${GPUS_PER_NODE:-8}" \
  --rdzv_backend=c10d --rdzv_id="$(basename "${CFG%.json}")" \
  --rdzv_endpoint="${MASTER_ADDR:-127.0.0.1}:29500" \
  -m rl.loop.nft_loop "${ARGS[@]}" \
  --scenes "$(cat "$SCENES_FILE")")
[ -f "$VAL_SCENES_FILE" ] && RUN+=(--val-scenes "$(cat "$VAL_SCENES_FILE")")

if [ -n "${DRY_RUN:-}" ]; then printf '%q ' "${RUN[@]}"; echo; exit 0; fi
exec "${RUN[@]}"
