#!/usr/bin/env bash
# Build the lyra2 environment: conda env, CUDA 12.8 toolchain, torch 2.7.1, and the
# extensions that must be compiled against them. Tested on Ubuntu 22.04, H100.
set -euo pipefail
ENV_NAME="${ENV_NAME:-lyra2}"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n "$ENV_NAME" python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
conda activate "$ENV_NAME"
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y
conda install cuda -c nvidia/label/cuda-12.8.0 -y

export CUDA_HOME=$CONDA_PREFIX
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128

SITE=$CONDA_PREFIX/lib/python3.10/site-packages
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

cd Lyra-2
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"   # transformer_engine expects cudart
MAX_JOBS="${MAX_JOBS:-16}" pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
# ViPE and Depth Anything 3 are upstream projects, not vendored here: install them from
# their own repositories. DA3 is needed by the reward's VGGT path too, which imports its
# geometry helpers.
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation \
    "git+https://github.com/nv-tlabs/vipe.git"
pip install --no-deps \
    "git+https://github.com/ByteDance-Seed/Depth-Anything-3.git"
pip install piq lpips plyfile          # the shared reward path (rewards/requirements.txt)

cat <<MSG

Done. Add this to your shell profile, the compiled extensions need it at run time:

  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

MSG
