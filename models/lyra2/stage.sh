#!/usr/bin/env bash
# Fetch everything a run needs: public model weights from their own releases, the
# training scenes from S3. Nothing here is mirrored that is published elsewhere.
set -euo pipefail
S3="${S3:-s3://videogen-rl}"
AWS="${AWS:-aws}"

echo "== Lyra-2 base checkpoints (HuggingFace: nvidia/Lyra-2.0) =="
# DiT + T5 text encoder + VAE + image encoder + the DMD-distillation LoRA (~101 GB).
huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir Lyra-2

echo "== VGGT-Omega, the reward's geometry backbone (HuggingFace: facebook/VGGT-Omega) =="
# GATED: accept the terms once at https://huggingface.co/facebook/VGGT-Omega and
# `huggingface-cli login`.
huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir Lyra-2/weights/vggt

echo "== packages not on PyPI, from their own repos =="
pip install --no-deps \
  "git+https://github.com/facebookresearch/vggt-omega.git" \
  "git+https://github.com/EasternJournalist/utils3d.git" \
  "git+https://github.com/microsoft/MoGe.git"

echo "== training scenes (1700) =="
$AWS s3 sync "$S3/training_data/dl3dv-mirror-train-fhalf241/" /inputs/scenes/

cat <<'MSG'

Done. The LoRA the loop starts from is
  checkpoints/lora/dmd_distillation.safetensors
used as BOTH --policy-lora and --ref-lora, so step 0 is exactly the DMD model.
Point VGGT_CHECKPOINT at Lyra-2/weights/vggt/vggt_omega_1b_512.pt.
MSG
