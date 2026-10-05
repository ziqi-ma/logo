"""VGGT-Omega depth and camera estimation.

Installs the vggt-omega package at runtime (from git) in environments where it is not
importable. One shared copy: ``scorers.dl3dv_videogpa`` imports it as a sibling, and each
model's recon and eval scripts import it as ``scorers.vggt``.

Outputs under output_dir:
  depth_maps.npz  — depth (N, H, W) float32, depth_conf (N, H, W), frame_indices (N,)
  cameras.npz     — w2c (N, 4, 4), intrinsics (N, 3, 3), frame_indices (N,)
"""

import gc
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# ────────────────────── setup ──────────────────────

def install_vggt_deps(vggt_wheel: Path | None = None) -> None:
    """Install VGGT-Omega runtime deps; optionally install from a pre-built wheel."""
    import os
    os.environ.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    os.environ.setdefault("PIP_NO_CACHE_DIR", "1")

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "numpy<2", "pillow", "einops", "safetensors", "opencv-python",
    ])

    if vggt_wheel is not None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(vggt_wheel)])
    else:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/facebookresearch/vggt-omega.git",
        ])

def download_vggt_checkpoint(checkpoint_uri: str, local_dir: Path) -> Path:
    """Resolve the VGGT-Omega checkpoint to a local file.

    A local path is used as-is. An ``s3://`` URI is copied into ``local_dir`` once and
    reused. Anything else is a missing local file and fails here, by name."""
    import fcntl
    import os
    import shutil
    import tempfile

    if os.path.isfile(checkpoint_uri):
        return Path(checkpoint_uri)
    if not checkpoint_uri.startswith("s3://"):
        raise FileNotFoundError(
            f"VGGT-Omega checkpoint not found at {checkpoint_uri!r}. Download "
            "vggt_omega_1b_512.pt from https://huggingface.co/facebook/VGGT-Omega (gated) "
            "and set VGGT_CHECKPOINT to its path.")
    local_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = checkpoint_uri.rstrip("/").split("/")[-1]
    ckpt_path = local_dir / ckpt_name
    if ckpt_path.exists():
        return ckpt_path

    def fetch(dst: str) -> None:
        import subprocess
        subprocess.check_call(["aws", "s3", "cp", checkpoint_uri, dst])

    # The scoring wrappers run one worker per local GPU, all calling this at once with
    # the same local_dir. Copying straight into ckpt_path published a half-written file
    # the other workers' exists() check accepted, so they died in torch.load with
    # "PytorchStreamReader failed reading zip archive", leaving holes in the scored set.
    # Serialize on a lock and publish by atomic rename.
    with open(f"{ckpt_path}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if ckpt_path.exists():
            return ckpt_path
        fd, part = tempfile.mkstemp(dir=str(local_dir), suffix=".part")
        os.close(fd)
        try:
            fetch(part)
            os.replace(part, ckpt_path)
        finally:
            if os.path.exists(part):
                os.unlink(part)
        print(f"Downloaded checkpoint → {ckpt_path}")
    return ckpt_path

# ────────────────────── helpers ──────────────────────

def _uniform_subsample(n: int, max_frames: int) -> list[int]:
    if max_frames <= 0 or n <= max_frames:
        return list(range(n))
    return np.floor(np.linspace(0, n - 1, num=max_frames)).astype(np.int64).tolist()

# ────────────────────── inference ──────────────────────

def run_vggt_depth_estimation(
    image_dir: Path,
    output_dir: Path,
    checkpoint_path: Path,
    image_resolution: int = 512,
    max_frames: int = 300,
) -> None:
    """Run VGGT-Omega depth and camera estimation on PNG/JPG frames.

    Imports from vggt_omega are deferred until after install_vggt_deps() runs.

    Args:
        image_dir: directory containing PNG/JPG frames (sorted alphabetically).
        output_dir: where to write depth_maps.npz and cameras.npz.
        checkpoint_path: path to vggt_omega_1b_512.pt (or other checkpoint).
        image_resolution: model input resolution (512 for the 1B-512 checkpoint).
        max_frames: uniformly subsample to at most this many frames; 0 = no limit.
    """
    import glob as _glob

    from vggt_omega.models import VGGTOmega
    from vggt_omega.utils.load_fn import load_and_preprocess_images
    from vggt_omega.utils.pose_enc import encoding_to_camera

    output_dir.mkdir(parents=True, exist_ok=True)

    img_list = sorted(
        _glob.glob(str(image_dir / "*.png")) + _glob.glob(str(image_dir / "*.jpg"))
    )
    if not img_list:
        raise FileNotFoundError(f"No PNG/JPG images in {image_dir}")
    N = len(img_list)

    indices_sub = _uniform_subsample(N, max_frames)
    img_list_sub = [img_list[i] for i in indices_sub]
    print(f"{N} images. Using {len(img_list_sub)} frames (max_frames={max_frames}).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VGGT-Omega from {checkpoint_path}")
    model = VGGTOmega().eval().to(device)
    # mmap=True: share one OS page-cache read of the checkpoint across the node's GPU procs
    # and across steps (see run_da3._load_da3_model). Fall back if the format lacks mmap support.
    try:
        state_dict = torch.load(str(checkpoint_path), map_location="cpu", mmap=True)
    except (RuntimeError, TypeError, ValueError):
        state_dict = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(state_dict)

    images = load_and_preprocess_images(img_list_sub, image_resolution=image_resolution).to(device)
    print(f"Input shape: {tuple(images.shape)}")

    with torch.inference_mode():
        predictions = model(images)

    extrinsic, intrinsic = encoding_to_camera(
        predictions["pose_enc"],
        predictions["images"].shape[-2:],
    )

    # extrinsic: (1, N, 3, 4) → (N, 3, 4); intrinsic: (1, N, 3, 3) → (N, 3, 3)
    extrinsic_np = extrinsic[0].float().cpu().numpy()
    intrinsic_np = intrinsic[0].float().cpu().numpy()
    depth_np = predictions["depth"][0].float().cpu().numpy()[..., 0]   # (N, H, W)
    depth_conf_np = predictions["depth_conf"][0].float().cpu().numpy()  # (N, H, W)

    del model, predictions, images
    gc.collect()
    torch.cuda.empty_cache()

    # Pad (N, 3, 4) extrinsics to (N, 4, 4) for compatibility with DA3 cameras.npz format
    n = extrinsic_np.shape[0]
    w2c = np.zeros((n, 4, 4), dtype=np.float32)
    w2c[:, :3, :4] = extrinsic_np
    w2c[:, 3, 3] = 1.0

    np.savez_compressed(
        str(output_dir / "depth_maps.npz"),
        depth=depth_np,
        depth_conf=depth_conf_np,
        frame_indices=np.asarray(indices_sub, dtype=np.int64),
    )
    print(f"Depth maps → {output_dir / 'depth_maps.npz'}")

    np.savez(
        str(output_dir / "cameras.npz"),
        w2c=w2c,
        intrinsics=intrinsic_np,
        frame_indices=np.asarray(indices_sub, dtype=np.int64),
    )
    print(f"Cameras    → {output_dir / 'cameras.npz'}")

    print(f"Depth estimation complete. Results: {output_dir}")
