"""Reward scorers, loaded by path.

The training loops and ``eval/score_videos.py`` register ``scorers`` as a namespace-only
package and import the modules they need directly, so this file is never executed on the
reward path. It exists so the package can also be imported normally, and importing it
requires every scorer's dependencies to be present.
"""

from . import (  # noqa: F401
    camera_rpe,
    dl3dv_videogpa,
    mvcs,
    reproj_rgbd,
    reproj_voxel,
    videogpa,
)
