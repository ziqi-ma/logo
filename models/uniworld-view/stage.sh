#!/usr/bin/env bash
# Fetch everything a run needs: public model weights from their own releases, the scene
# lists from S3. Nothing here is mirrored that is published elsewhere.
set -euo pipefail
S3="${S3:-s3://videogen-rl}"
AWS="${AWS:-aws}"

export WAN_LORA_REPO="${WAN_LORA_REPO:-Kijai/WanVideo_comfy}"
export WAN_LORA_FILENAME="${WAN_LORA_FILENAME:-Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors}"

echo "== base weights (HuggingFace: UniView, Wan2.1-VACE, CausVid LoRA) =="
# Core weights are always downloaded. The optional bundles (--mosca, --stream3r, --recon)
# are off by default; pass the flag to pull each one.
python checkpoints/download_hf.py "$@"

echo "== VGGT-Omega, the reward's geometry backbone (HuggingFace: facebook/VGGT-Omega) =="
# Accept the terms once at https://huggingface.co/facebook/VGGT-Omega and
# `huggingface-cli login`.
huggingface-cli download facebook/VGGT-Omega vggt_omega_1b_512.pt --local-dir weights/vggt

echo "== packages not on PyPI, from their own repos =="
pip install --no-deps \
  "git+https://github.com/facebookresearch/vggt-omega.git" \
  "git+https://github.com/EasternJournalist/utils3d.git" \
  "git+https://github.com/microsoft/MoGe.git" \
  "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"

echo "== train/val scene lists =="
$AWS s3 cp "$S3/training_data/uniworld80f/train_scenes_1658.txt" rl/scenes/
$AWS s3 cp "$S3/training_data/uniworld80f/val_scenes_25.txt" rl/scenes/

cat <<'MSG'

Done. Point VGGT_CHECKPOINT at weights/vggt/vggt_omega_1b_512.pt, then build the
CondBundles the loop trains on:

  python -m rl.data.scene_prep --scenes-root <scene dir> \
      --scenes $(cat rl/scenes/train_scenes_1658.txt)
MSG
