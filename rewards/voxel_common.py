"""The voxel-edge dial, in one copy for every model."""

import os

_DEFAULT_ALPHA = 0.5

def voxel_alpha() -> float:
    v = os.environ.get("NFT_REWARD_VOXEL", "").strip().lower()
    if v in ("", "0", "off", "false"):
        return 0.0
    if v in ("on", "true"):
        return _DEFAULT_ALPHA
    try:
        return float(v)
    except ValueError:
        return 0.0
