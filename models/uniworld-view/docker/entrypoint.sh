#!/bin/bash
# UniWorld wrapper entrypoint: `/entrypoint.sh bash -lc '<script>'` runs the
# payload script with the repo importable and checkpoints/ pointed at /weights.
set -u
export PYTHONPATH="/app/uniworld:${PYTHONPATH:-}"
cd /app/uniworld
mkdir -p /weights
# jobs stage the model bundle to /weights/checkpoints; the repo's relative
# ./checkpoints paths resolve through this symlink
if [ ! -e /app/uniworld/checkpoints ] || [ -z "$(ls /app/uniworld/checkpoints 2>/dev/null)" ]; then
  rm -rf /app/uniworld/checkpoints
  ln -sfn /weights/checkpoints /app/uniworld/checkpoints
fi
case "${1:-bash}" in
  bash|sh) shift 2>/dev/null || true; exec bash "$@" ;;
  *) exec "$@" ;;
esac
