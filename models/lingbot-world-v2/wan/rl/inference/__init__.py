"""Generation for evaluation (dl3dv VideoGPA + trajbench).

Shared helper: convert a lyra-staging ``trajectory.npz`` into the lingbot
camera format (``poses.npy`` + ``intrinsics.npy``) consumed by
``WanI2VCausal.generate(action_path=...)``.
"""
import numpy as np

REF_HEIGHT = 480
REF_WIDTH = 832

def convert_trajectory(npz_path, out_dir):
    """Write lingbot ``poses.npy`` (c2w [F,4,4]) + ``intrinsics.npy``
    ([F,4] fx,fy,cx,cy at 480x832) from a staged ``trajectory.npz``.

    Handles both staging schemas in the wild:
      * dl3dv ``stage_inputs_s3.py``: ``w2c`` [F,4,4] float32 relativized
        to frame 0, ``intrinsics`` [F,3,3], ``image_wh`` [2] = (width, height)
        at the resolution the intrinsics are expressed in.
      * trajbench ``stage_worldscore_inputs.py``: ``w2c`` [F,4,4],
        ``intrinsics`` [F,3,3], ``image_height``/``image_width`` int64.
    Missing size keys fall back to 720x1280 (the dl3dv first-frame size).

    Same math as ``wan.rl.data.scene_convert.convert_scene`` (which reads the
    training-side ``lyra2_traj.npz`` schema instead). Returns pose count.
    """
    import os

    traj = np.load(npz_path)
    w2c = traj["w2c"].astype(np.float64)          # [F, 4, 4]
    K = traj["intrinsics"].astype(np.float64)     # [F, 3, 3]
    if "image_wh" in traj:
        width, height = (float(v) for v in np.asarray(traj["image_wh"]).ravel())
    elif "image_height" in traj and "image_width" in traj:
        height, width = float(traj["image_height"]), float(traj["image_width"])
    else:
        height, width = 720.0, 1280.0

    poses = np.linalg.inv(w2c).astype(np.float32)  # c2w, stays OpenCV
    sx = REF_WIDTH / width
    sy = REF_HEIGHT / height
    intrinsics = np.stack(
        [K[:, 0, 0] * sx, K[:, 1, 1] * sy, K[:, 0, 2] * sx, K[:, 1, 2] * sy],
        axis=1).astype(np.float32)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "poses.npy"), poses)
    np.save(os.path.join(out_dir, "intrinsics.npy"), intrinsics)
    return len(poses)
