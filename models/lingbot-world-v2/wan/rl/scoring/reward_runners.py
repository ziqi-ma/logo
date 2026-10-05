# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HPSv3 subprocess wrapper for :mod:`wan.rl.scoring.nft_score`.

HPSv3 (transformers==4.45.2, 7B Qwen2-VL) cannot share the training env, so it runs
in its own interpreter over the shared runner script ``rewards/scorers/hpsv3.py``. This
module only resolves the interpreter + script path; ``nft_score`` shells out.

``HPSV3_PY`` selects the interpreter (default: this one). ``HPSV3_RUNNER`` overrides
the script path, which otherwise resolves to the repo's shared ``rewards/`` tree
(``REWARDS_DIR`` if set).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# HPSv3 interpreter: its own env when set, else this one.
HPSV3_PY = os.environ.get("HPSV3_PY") or sys.executable

# The shared runner script under rewards/ (REWARDS_DIR if set, else the repo tree).
def _default_runner() -> Path:
    from wan.rl.scoring.nft_voxel import rewards_dir
    return Path(rewards_dir()) / "scorers/hpsv3.py"

HPSV3_RUNNER = Path(os.environ["HPSV3_RUNNER"]) if os.environ.get("HPSV3_RUNNER") \
    else _default_runner()
