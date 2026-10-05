# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sampling: build the inference model + scene data_batch, run K rollouts.

Reuses the existing single-image custom-trajectory inference setup
(``lyra2_custom_traj_inference`` / ``lyra2_zoomgs_inference`` / ``model_loader``)
so there is one source of truth for model + scene preparation. The sampling
policy is whatever LoRA is passed as ``sampling_lora`` (the DMD LoRA at step 0,
the promoted ``new`` adapter thereafter). Produces a rollout store the scorer
consumes. ``loop/sampler.py`` holds the samplers themselves; this module builds
what they run on, and is also the standalone CLI for sampling a checkpoint.

Run (single scene/GPU):
    python -m lyra_2._src.rl.inference.sample \
        --checkpoint_dir checkpoints/model --experiment <exp> \
        --input_image_path img.png --trajectory_path traj.npz --prompt "..." \
        --sampling-lora checkpoints/lora/dmd_distillation.safetensors \
        --rollout-root rollout_epoch_0 --scene 0008 --k-rollouts 4 --num_frames 81 --use_dmd
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np
import torch

from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._src.inference.lyra2_ar_inference import safe_to
from lyra_2._src.inference.lyra2_custom_traj_inference import load_trajectory
from lyra_2._src.inference.lyra2_zoomgs_inference import _da3_infer_depth_intrinsics_single
from lyra_2._src.utils.model_loader import load_model_from_checkpoint
from lyra_2._src.rl.loop.sampler import RolloutStoreWriter, run_nft_sampling

def build_inference_model(checkpoint_dir: str, experiment: str, sampling_lora: str,
                          lora_weight: float = 1.0):
    """Load the 14B base + activate `sampling_lora` as the (DMD) sampling policy."""
    experiment_opts = [
        "model.config.use_mp_policy_fsdp=False",
        "model.config.keep_original_net_dtype=False",
        "model.config.net.postpone_checkpoint=True",
    ]
    model, config = load_model_from_checkpoint(
        config_file="lyra_2/_src/configs/config.py",
        experiment_name=experiment,
        checkpoint_path=checkpoint_dir,
        enable_fsdp=False,
        instantiate_ema=False,
        load_ema_to_reg=False,
        experiment_opts=experiment_opts,
    )
    if sampling_lora:
        name = model.load_lora_weights(sampling_lora)
        model.set_weights_and_activate_adapters([name], [lora_weight])
    if hasattr(model.net, "enable_selective_checkpoint"):
        model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)
    dt, dev = model.tensor_kwargs.get("dtype"), model.tensor_kwargs.get("device")
    if dt is not None:
        model.net = model.net.to(device=dev, dtype=dt)
    model.eval()
    return model, config

_MOGE_MODEL = None


def _align_depth_to_moge(depth_hw, mask_hw, img_rgb_uint8, target_hw):
    """Divide DA3 depth by the least-squares DA3->MoGe inverse-depth factor.

    The same alignment lyra2_custom_traj_inference performs under --use_moge_scale, for
    trajectories with no ``moge_scale_s`` baked in. Returns the depth unchanged if MoGe is
    unavailable or the overlap is too small to fit, after saying so.
    """
    global _MOGE_MODEL

    try:
        from lyra_2._src.inference.depth_utils import load_moge_model, moge_infer_depth_intrinsics
    except Exception as e:  # noqa: BLE001 - MoGe is an optional install
        log.warning(f"[sample] no MoGe ({type(e).__name__}); depth stays on the raw DA3 scale "
                    f"and the clip's geometry will be off", rank0_only=True)
        return depth_hw

    dev = depth_hw.device
    if _MOGE_MODEL is None:
        _MOGE_MODEL = load_moge_model(dev)
        _MOGE_MODEL.eval()
    _MOGE_MODEL.to(dev)
    # Both sizes, as lyra2_custom_traj_inference passes them: depth_pred_hw alone leaves the
    # MoGe depth at its own resolution and the DA3 mask no longer lines up. MATH sdpa keeps
    # MoGe off the fused kernels, matching that path.
    with torch.no_grad(), torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
            _MOGE_MODEL, img_rgb_uint8,
            depth_pred_hw=tuple(target_hw), target_hw=tuple(target_hw),
        )
    _MOGE_MODEL.cpu()

    moge_depth_hw = moge_depth_hw.to(dev)
    valid = (mask_hw.to(dev) > 0.5) & (moge_mask_hw.to(dev) > 0.5)
    if valid.sum() <= 10:
        log.warning("[sample] MoGe/DA3 overlap too small to fit the depth scale", rank0_only=True)
        return depth_hw
    inv_da3 = 1.0 / (depth_hw[valid] + 1e-6)
    inv_moge = 1.0 / (moge_depth_hw[valid] + 1e-6)
    den = (inv_da3 * inv_da3).sum()
    if den <= 1e-8:
        log.warning("[sample] degenerate depth-scale fit; leaving DA3 depth as is", rank0_only=True)
        return depth_hw
    scale = ((inv_da3 * inv_moge).sum() / den).item()
    if scale <= 1e-6:
        log.warning(f"[sample] depth-scale fit returned {scale:.3g}; leaving DA3 depth as is",
                    rank0_only=True)
        return depth_hw
    log.info(f"[sample] fitted DA3->MoGe depth scale {scale:.4f}", rank0_only=True)
    return depth_hw / scale


def build_scene_data_batch(model, da3_model, *, image_path: str, traj_file: str,
                           caption: str, neg_t5, num_frames: int, target_hw, pose_scale: float,
                           fps: int = 16, t5_override=None) -> dict:
    """Assemble the single-image data_batch (mirrors lyra2_custom_traj_inference).

    ``t5_override``: reuse a precomputed caption embedding instead of calling umt5. The
    caption is the same for every scene, so callers that build many scenes lazily can
    compute the T5 embedding once and free the umt5 encoder (saves ~11 GB + per-scene cost).
    """
    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding

    dev = model.tensor_kwargs.get("device")
    dt = model.tensor_kwargs.get("dtype")
    target_h, target_w = target_hw

    w2cs_T_44, Ks_T_33 = load_trajectory(traj_file, num_frames, target_hw=target_hw, pose_scale=pose_scale)

    bgr = cv2.imread(image_path)
    assert bgr is not None, f"cannot read image {image_path}"
    rgb_t = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    image_chw01, depth_hw, _K_da3, _mask = _da3_infer_depth_intrinsics_single(
        da3_model=da3_model, img_rgb_uint8=rgb_t, target_hw=target_hw,
    )
    # MoGe-normalize the raw DA3 depth to the (MoGe) scale the base model / eval use.
    # `moge_scale_s` is baked per scene into the trajectory npz (s = DA3->MoGe inverse-depth
    # LS factor, identical to the inference use_moge_scale alignment); depth_moge = depth_da3/s.
    # Without it the loop feeds raw DA3NESTED depth, off the eval/base scale by a per-scene factor.
    _tf = np.load(traj_file)
    if "moge_scale_s" in _tf.files:
        depth_hw = depth_hw / float(_tf["moge_scale_s"])
    else:
        # Trajectories staged outside the DL3DV pipeline (the trajbench cells, for one) carry
        # no baked factor. Fitting it here matches the --use_moge_scale alignment the trajbench
        # generation path applies; skipping it leaves raw DA3 depth and desynchronises the
        # geometry against the camera translations.
        depth_hw = _align_depth_to_moge(depth_hw, _mask, rgb_t, target_hw)
    H, W = image_chw01.shape[-2:]
    img_bchw = image_chw01.to(device=dev) * 2.0 - 1.0

    t5 = (t5_override.to(device=dev, dtype=dt) if t5_override is not None
          else get_umt5_embedding(caption, device=dev).to(dtype=dt))
    if t5.dim() == 2:
        t5 = t5.unsqueeze(0)
    elif t5.dim() == 3 and t5.shape[0] != 1:
        t5 = t5[:1]

    N = num_frames
    data_batch = {
        "video": img_bchw.unsqueeze(2),
        "t5_text_embeddings": t5,
        "neg_t5_text_embeddings": misc.to(neg_t5, **model.tensor_kwargs),
        "fps": torch.tensor([fps], dtype=torch.int32, device=dev),
        "padding_mask": torch.zeros((1, 1, H, W), dtype=dt, device=dev),
        "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=dev),
        "camera_w2c": w2cs_T_44.unsqueeze(0).to(dtype=torch.float32, device=dev),
        "intrinsics": Ks_T_33.unsqueeze(0).to(dtype=torch.float32, device=dev),
        "depth": depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, N, 1, 1).to(device=dev),
    }
    skip = {"camera_w2c", "intrinsics", "depth"}
    return safe_to(data_batch, device=dev, dtype=dt, skip_keys=skip)

def stage1_sample_scene(model, da3_model, args, *, scene: str, image_path: str, traj_file: str,
                        caption: str, neg_t5, writer: RolloutStoreWriter,
                        k_rollouts: int, base_seed: int, process_group=None) -> List[dict]:
    """Build the scene data_batch and run K independent rollouts into `writer`."""
    target_hw = tuple(int(x) for x in args.resolution.split(","))
    data_batch = build_scene_data_batch(
        model, da3_model, image_path=image_path, traj_file=traj_file, caption=caption,
        neg_t5=neg_t5, num_frames=int(args.num_frames), target_hw=target_hw,
        pose_scale=float(args.pose_scale), fps=int(args.fps),
    )
    return run_nft_sampling(
        model, data_batch, args, writer, scene,
        k_rollouts=k_rollouts, base_seed=base_seed, da3_model=da3_model, process_group=process_group,
    )

def _build_args(checkpoint_dir, num_frames, resolution, pose_scale, guidance, shift, seed):
    """Full inference args namespace with DMD defaults (reuses the inference CLI defaults)."""
    import sys
    from lyra_2._src.inference.lyra2_custom_traj_inference import parse_arguments, _apply_dmd_defaults

    argv = sys.argv
    sys.argv = [
        "logo-sample",   # argv[0], for the borrowed inference argparser
        "--trajectory_path", "PLACEHOLDER", "--input_image_path", "PLACEHOLDER",
        "--checkpoint_dir", checkpoint_dir, "--num_frames", str(num_frames),
        "--resolution", resolution, "--pose_scale", str(pose_scale),
        "--guidance", str(guidance), "--shift", str(shift), "--seed", str(seed),
        "--use_dmd", "--offload",  # offload swaps diffusion/DA3 CPU<->GPU to fit one GPU
    ]
    try:
        args = parse_arguments()
    finally:
        sys.argv = argv
    _apply_dmd_defaults(args)
    return args

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DiffusionNFT Stage 1: K-rollout sampling for one scene")
    ap.add_argument("--checkpoint_dir", default="checkpoints/model")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--input_image_path", required=True)
    ap.add_argument("--trajectory_path", required=True)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--sampling-lora", required=True)
    ap.add_argument("--rollout-root", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--k-rollouts", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--resolution", default="704,1280")
    ap.add_argument("--pose_scale", type=float, default=0.35)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--append", action="store_true", help="append to an existing rollout store (multi-scene)")
    a = ap.parse_args()

    from lyra_2._src.inference.depth_utils import load_da3_model

    args = _build_args(a.checkpoint_dir, a.num_frames, a.resolution, a.pose_scale,
                       a.guidance, a.shift, a.base_seed)
    model, _ = build_inference_model(a.checkpoint_dir, a.experiment, a.sampling_lora)
    da3_model = load_da3_model(da3_model_name=args.da3_model_name,
                               da3_model_path_custom=args.da3_model_path_custom,
                               device=model.tensor_kwargs.get("device", "cuda"))
    da3_model.eval()
    neg = torch.load("checkpoints/text_encoder/negative_prompt.pt", map_location="cpu",
                     weights_only=False)["t5_text_embeddings"]

    writer = RolloutStoreWriter(a.rollout_root, append=a.append)
    metas = stage1_sample_scene(
        model, da3_model, args, scene=a.scene, image_path=a.input_image_path,
        traj_file=a.trajectory_path, caption=a.prompt, neg_t5=neg, writer=writer,
        k_rollouts=a.k_rollouts, base_seed=a.base_seed,
    )
    log.info(f"Stage 1: wrote {len(metas)} samples to {a.rollout_root}", rank0_only=True)

if __name__ == "__main__":
    _main()
