"""Relative pose error (camera adherence), shared by every model's reward.

RPE as defined for the TUM RGB-D benchmark: compare the relative pose between every
ordered pair of frames in the estimated camera path against the same pair in the
reference path, which makes the measure invariant to a global world transform. A
single least-squares scale is fitted over all pairs before the translation residual,
because a monocular reconstruction recovers the path only up to scale, and the
translation error is then normalized by the reference path length.

Self-contained on purpose: the reference implementation lives in an evaluation package
whose import pulls a SLAM stack that the training environments do not have. Both
returned values are CLIP-level scalars -- an RPE belongs jointly to the two frames of
a pair and cannot be attributed to either one, so there is deliberately no per-frame
decomposition.

torch is imported lazily so this module loads in any environment.
"""

def rotation_angle(rot, epsilon: float = 1e-8):
    """Geodesic angle (radians) of a batch of rotation matrices."""
    import torch
    trace = torch.einsum("...ii->...", rot).clamp(-1.0 + epsilon, 3.0 - epsilon)
    return torch.acos((trace - 1) / 2)

def all_pairs(traj_ref, traj_est):
    """(translation_error, rotation_error_deg) over all ordered pairs; scale-aligned."""
    import torch
    n = traj_ref.shape[0]
    ii, jj = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    # relative pose of j as seen from i -- invariant to a global world transform
    d_ref = traj_ref[ii].inverse() @ traj_ref[jj]
    d_est = traj_est[ii].inverse() @ traj_est[jj]
    rot_ref, tr_ref = d_ref[..., :3, :3], d_ref[..., :3, 3]
    rot_est, tr_est = d_est[..., :3, :3], d_est[..., :3, 3]
    s = (tr_ref * tr_est).sum() / (tr_est * tr_est).sum().clamp(min=1e-12)   # align scale
    translation_error = (tr_ref - s * tr_est).norm(dim=-1)
    rotation_error_deg = (180 / torch.pi) * rotation_angle(rot_ref.inverse() @ rot_est)
    return translation_error, rotation_error_deg

def adherence(pred_c2w, target_c2w):
    """(rot_deg, trans_normalized). Scale invariant; translation is ALREADY divided by the
    reference trajectory length -- never divide by path length a second time."""
    te, re_deg = all_pairs(traj_ref=target_c2w, traj_est=pred_c2w)
    adjacent = target_c2w[1:, :3, 3] - target_c2w[:-1, :3, 3]
    traj_len = adjacent.norm(p=2, dim=1).sum()
    return re_deg.mean(), te.mean() / traj_len.clamp(min=1e-12)
