# syntax=docker/dockerfile:1.6
# UniWorld-View eval/RL wrapper image.
# Build from the repo root (exclude checkpoints/ and output/ via .dockerignore):
#   docker build -f docker/Dockerfile -t <your-registry>/uniworld:<tag> .
# Weights are NOT baked: jobs stage them from $UNIWORLD_DATA_ROOT/base-train via rl.data.gcs_util (S3).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git g++ libgl1 libglib2.0-0 libsm6 libxext6 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/uniworld
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
    pip install carvekit --no-deps && \
    pip install boto3

# PyTorch3D from source (pinned commit; no GPU at build -> pin arch for H100)
ENV TORCH_CUDA_ARCH_LIST="9.0" FORCE_CUDA=1 MAX_JOBS=16
RUN pip install ninja fvcore iopath && \
    pip install --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@71db7a0ea293f2626f5ea1c70366870a607129d0" && \
    python -c "import pytorch3d; print('pytorch3d', pytorch3d.__version__)"

COPY . /app/uniworld/

# Pre-build the vipe JIT extension so jobs don't compile at startup. Needs nvcc
# only (no GPU); tolerated to fail -> falls back to JIT at first import.
ENV TORCH_EXTENSIONS_DIR=/opt/torch_extensions
RUN python -c "import sys; sys.path.insert(0,'/app/uniworld/extern/vipe'); import vipe.ext" || \
    echo 'vipe ext prebuild skipped (will JIT at job start)'

RUN python - <<'EOF'
import sys
sys.path.insert(0, "/app/uniworld")
sys.path.insert(1, "/app/uniworld/extern")
import diffusers, peft, transformers, pytorch3d, decord, carvekit  # noqa: F401
from moge.model.v2 import MoGeModel  # noqa: F401
from rl.data.gcs_util import download_prefix  # noqa: F401
print("image smoke OK")
EOF

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
