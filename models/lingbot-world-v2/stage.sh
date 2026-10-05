#!/usr/bin/env bash
# Fetch everything a run needs: base weights from HuggingFace, scenes + reward weights from S3.
set -euo pipefail
S3="${S3:-s3://videogen-rl}"
AWS="${AWS:-aws}"                      # add --profile if your creds need one
mkdir -p weights data

echo "== base model (public) =="
# Pin the revision: the shipped checkpoint is 18.544 B params in F32 despite the 14B name, and
# a moved default branch would silently change what you fine-tune from.
# Pinned to the revision the runs actually trained from (verified 2026-09-11: 19 files,
# 8 transformer shards, last modified 2026-07-08).
HF_REVISION="${HF_REVISION:-5c33dd40b213598c418fd25bff30fdbd23fd38a7}"
huggingface-cli download robbyant/lingbot-world-v2-14b-causal-fast \
  --revision "$HF_REVISION" \
  --local-dir weights/lingbot-world-v2-14b-causal-fast

# Meta VGGT-Omega -- a public checkpoint, so it is NOT mirrored: fetch it from the source.
# It is gated, so do this once: accept the terms at https://huggingface.co/facebook/VGGT-Omega
# and `huggingface-cli login`.
echo "== reward backbone: VGGT-Omega (public, gated -- accept terms once) =="
huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir weights/vggt

# These are not on PyPI, so nothing pip-installs them for you. Without
# vggt_omega the reward raises ModuleNotFoundError inside the scorer subprocess and EVERY
# rollout scores -inf, which the loop reports as "0 trainable rollouts this step" -- training
# appears to run and learns nothing.
echo "== packages not on PyPI, from their own repos =="
pip install --no-deps \
  "git+https://github.com/facebookresearch/vggt-omega.git" \
  "git+https://github.com/EasternJournalist/utils3d.git" \
  "git+https://github.com/microsoft/MoGe.git" \
  "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

echo "== training scenes (1700) =="
$AWS s3 sync "$S3/training_data/dl3dv-mirror-train-fhalf241-lingbot/" data/scenes/

