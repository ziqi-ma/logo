#!/usr/bin/env bash
# Evaluate one TrajectoryBench group with one model.
#
#   eval/run_trajbench.sh <model> <group-dir> <adapter> [extra generator args...]
#
#     model       lyra2 | lingbot | uniworld
#     group-dir   a directory staged by eval/stage_trajbench.py
#     adapter     the LoGo adapter for that model
#
#   python eval/stage_trajbench.py --category transition --difficulty hard --out runs/transition_hard
#   eval/run_trajbench.sh lingbot runs/transition_hard checkpoints/logo-lingbot-world-v2-lora.pt
set -euo pipefail

MODEL="${1:?model: lyra2 | lingbot | uniworld}"
GROUP_DIR="${2:?a directory staged by eval/stage_trajbench.py}"
ADAPTER="${3:?the LoGo adapter for this model}"
shift 3

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SETTINGS="$GROUP_DIR/settings.json"
[ -f "$SETTINGS" ] || { echo "no $SETTINGS -- stage the group first with eval/stage_trajbench.py" >&2; exit 1; }

read -r GROUP FRAMES FOV IDS <<EOF
$(python - "$SETTINGS" "$GROUP_DIR" <<'PY'
import json, os, sys
s = json.load(open(sys.argv[1]))
ids = sorted(os.listdir(os.path.join(sys.argv[2], "inputs")))
print(s["group"], s["num_frames"], s["transition_fov_scale"], ",".join(ids))
PY
)
EOF

echo "group $GROUP: ${FRAMES}f, lingbot lens $FOV, $(echo "$IDS" | tr ',' '\n' | wc -l) clip(s)"

# With EMU_MPC set, the generator runs under eval/rng_emu, which reproduces the CUDA random
# stream of a card with that many multiprocessors (132 for the H100s the clips were published
# from). Everything else, including the settings above, is unchanged.
EMU_DIR="$HERE/rng_emu"
launch() {            # launch <emu-driver> <module>  [args...]
  local driver="$1" module="$2"; shift 2
  if [ -n "${EMU_MPC:-}" ]; then
    echo "rng emulation: mpc=$EMU_MPC via $driver"
    exec python "$EMU_DIR/$driver" "$@"
  fi
  exec python -m "$module" "$@"
}

case "$MODEL" in
  lyra2)
    # no lens: Lyra-2 has no such parameter
    cd "$REPO/models/lyra2/Lyra-2"
    if [ -n "${EMU_MPC:-}" ]; then
      echo "rng emulation: mpc=$EMU_MPC via run_lyra2.py"
      set -- --ws_prefix "$GROUP_DIR" --ids "$IDS" \
        --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/model}" --adapter "$ADAPTER" \
        --num_frames "$FRAMES" --base-seed "${SEED:-1}" "$@"
      exec python "$EMU_DIR/run_lyra2.py" "$@"
    fi
    exec torchrun --standalone --nproc_per_node="${NPROC:-1}" \
      -m lyra_2._src.rl.inference.gen_trajbench \
      --ws_prefix "$GROUP_DIR" --ids "$IDS" \
      --checkpoint_dir "${CHECKPOINT_DIR:-checkpoints/model}" --adapter "$ADAPTER" \
      --num_frames "$FRAMES" --base-seed "${SEED:-1}" "$@"
    ;;
  lingbot)
    # the lens is LingBot-World's own parameter
    cd "$REPO/models/lingbot-world-v2"
    launch run_lingbot.py wan.rl.inference.gen_trajbench \
      --ws_prefix "$GROUP_DIR" --ids "$IDS" \
      --num_frames "$FRAMES" --transition_fov_scale "$FOV" --base_seed "${SEED:-1}" \
      --lora_path "$ADAPTER" \
      --ckpt_dir "${CKPT_DIR:-weights/lingbot-world-v2-14b-causal-fast}" "$@"
    ;;
  uniworld)
    # no lens; always 81 frames, subsampling a 241-pose path
    cd "$REPO/models/uniworld-view"
    launch run_uniworld.py rl.inference.eval_gen \
      --ws_prefix "$GROUP_DIR" --sids "$IDS" \
      --num_frames 81 --seed "${SEED:-42}" \
      --rl-ckpt "$ADAPTER" --device "${DEVICE:-cuda:0}" "$@"
    ;;
  *)
    echo "unknown model '$MODEL' (lyra2 | lingbot | uniworld)" >&2; exit 1 ;;
esac
