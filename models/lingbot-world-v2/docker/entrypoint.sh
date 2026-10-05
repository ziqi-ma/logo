#!/bin/bash
# Container entrypoint for the lingbot DiffusionNFT training image.
#
# Minimal by design -- the job script owns staging and launch.
# Activates the `lingbot` conda env so `python` / `torchrun` / `pip` resolve to
# it (the reward subprocesses pick their own interpreters via HPSV3_PY /
# VGGT_PY), then execs the given command.
set -eo pipefail

# Conda's per-env activation scripts reference vars that are unset on first
# activation, so drop `set -u` around `conda activate` (mirrors lyra2-wrapper).
# shellcheck disable=SC1091
source /opt/conda/etc/profile.d/conda.sh
set +u
conda activate lingbot
set -u

exec "$@"
