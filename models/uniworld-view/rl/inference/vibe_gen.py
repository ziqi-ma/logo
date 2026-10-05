"""Vibe-check generation: UniWorld-View single-view NVS along benchmark trajectories.

Mirrors ``UniScene.nvs_single_view`` (demo.py:365-691, hybrid render branch) but with
the camera trajectory injected from a benchmark ``trajectory.npz`` instead of
``build_cameras`` presets, at a configurable frame count.

Usage (one process per GPU):
    python -m rl.inference.vibe_gen --scenes-root <dir> --scenes indoor_000 indoor_001 \
        --settings 241f-native 81f-3x 81f-2x --out-root <dir> --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.infer_config import get_parser
from rl.data.traj_adapter import align_to_world, load_trajectory_npz, rescale_intrinsics, slice_setting
from utils.utils import np_points_padding, points_padding, set_initial_camera
from utils.warp_utils import save_video

SETTINGS = ("241f-native", "81f-3x", "81f-2x")

def build_opts(args) -> argparse.Namespace:
    opts = get_parser().parse_args([])
    opts.device = args.device
    opts.height, opts.width = 480, 832
    opts.video_length = 81
    opts.ddim_steps = 8
    opts.diffusion_guidance_scale = 4.0
    opts.prompt = ""
    opts.mode = "single_view"
    opts.save_dir = getattr(args, "out_root", "/tmp/eval_gen")
    opts.weight_dtype = torch.bfloat16
    for k in ("blip_path", "transformer_path", "model_name", "lora_path", "stream3r_path",
              "moge_path", "segnet_path"):
        v = getattr(args, k, None)
        if v:
            setattr(opts, k, v)
    # run_diffusion reads opts.seed (default 42 = the demo's hardcoded value).
    opts.seed = int(getattr(args, "seed", 42) or 42)
    return opts

def nvs_custom_traj(pvd, image_path: str, w2c_rel: torch.Tensor, K_traj: torch.Tensor,
                    traj_hw, out_dir: str, fps: int = 16) -> dict:
    """One generation along an injected frame-0-relative trajectory.

    ``K_traj`` is at ``traj_hw``; it is rescaled to the working resolution after the
    input-resize policy has (possibly) adjusted ``opts.height/width``.
    Returns meta (prompt, num_frames, timings). Writes render/mask/diffusion mp4s.
    """
    opts = pvd.opts
    T = int(w2c_rel.shape[0])
    opts.video_length = T
    device = opts.device
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    with pvd.active_aux_modules():
        image = Image.open(image_path).convert("RGB")
        image = pvd._apply_input_resize_policy(image)
        K_traj = rescale_intrinsics(K_traj, traj_hw, (opts.height, opts.width))
        validation_image = ToTensor()(image)[None].to(device)
        depth_image = validation_image[0]
        depth, masks, normal, _, _K_moge, _ = pvd.run_moge(depth_image)
        # WorldScore convention: the input is treated as captured by the benchmark
        # camera (npz K), not MoGe's estimate — lifting with the same K the targets
        # are rendered with makes frame 0 an identity warp (no zoom jump).
        K = K_traj[0].to(device)
        K_inv = K.inverse()

        points2d = torch.stack(
            torch.meshgrid(
                torch.arange(opts.width, dtype=torch.float32),
                torch.arange(opts.height, dtype=torch.float32),
                indexing="xy",
            ),
            -1,
        ).to(device)
        points3d = points_padding(points2d).reshape(opts.height * opts.width, 3)
        points3d = (K_inv @ points3d.T * depth.reshape(1, opts.height * opts.width)).T
        colors = (depth_image * 255).to(torch.uint8).permute(1, 2, 0).reshape(opts.height * opts.width, 3)

        points3d_np = points3d.detach().cpu().numpy()
        colors_np = colors.detach().cpu().numpy()

        with torch.no_grad():
            origin_w_, origin_h_ = image.size
            image_pil = image.resize((512, 512))
            fg_mask = pvd.seg_net([image_pil])[0]
            fg_mask = fg_mask.resize((origin_w_, origin_h_))
        fg_mask_np = np.array(fg_mask, dtype=np.float32)
        fg_mask_bool = fg_mask_np > 0.5
        if fg_mask_bool.mean() < 0.05:
            fg_mask_bool[...] = True
        # Anchor depth: median over fg ∩ MoGe-valid pixels (run_moge fills invalid
        # depth with 1000, which poisons a plain fg median on outdoor/sky images).
        valid = depth[0, 0] < 999.0
        fg_valid = torch.from_numpy(fg_mask_bool).to(device, torch.bool) & valid
        if not fg_valid.any():
            fg_valid = valid if valid.any() else torch.ones_like(valid)
        depth_avg = torch.median(depth[0, 0][fg_valid]).item()
        w2c_0, c2w_0 = set_initial_camera(opts.elevation, depth_avg)

        # `if override` was a truthiness test, so an empty override ("" = eval_gen's
        # --dl3dv-setting, i.e. sample with no text as the RL rollouts do) fell through to
        # BLIP2 captioning -- silently reintroducing the text conditioning it was meant to
        # remove, and paying for a BLIP2 forward per clip. Distinguish "" from None.
        override = getattr(pvd, "_eval_prompt_override", None)
        if override is None:
            prompt = pvd.get_caption(image)
        elif override == "":
            prompt = ""                      # no text conditioning at all
        else:
            prompt = override + pvd.opts.refine_prompt

    # Trajectory translations are in WorldScore normalized units (scene scale ~ 1);
    # our warp world is MoGe-metric. Anchor them to the scene (the demo's presets
    # scale offsets by radius = depth_avg); the 0.5 factor calibrates apparent
    # travel against Lyra's videos on the same scenes (full depth_avg overshoots).
    ts = float(getattr(pvd.opts, "trans_scale", -1.0))
    w2c_rel = w2c_rel.clone()
    w2c_rel[:, :3, 3] *= (0.5 * depth_avg if ts < 0 else ts)

    c2w_0_np = c2w_0.detach().cpu().numpy()
    points3d_np = (c2w_0_np[:3] @ np_points_padding(points3d_np).T).T
    points_world_tensor = torch.from_numpy(points3d_np.reshape(opts.height, opts.width, 3)).to(device, torch.float32)
    colors_tensor = torch.from_numpy(((colors_np / 255.0) * 2.0 - 1.0).reshape(opts.height, opts.width, 3)).to(device, torch.float32)
    mask_reliable_tensor = masks.view(opts.height, opts.width).to(device, torch.float32)

    w2cs, c2ws = align_to_world(w2c_rel, w2c_0)
    w2cs = w2cs.to(device)
    c2ws = c2ws.to(device)
    intrinsic = K_traj.to(device)
    w2c_0 = w2c_0.to(device)
    K = K.to(device)
    depth = depth.to(device)

    from utils.meshrender import MeshWarper
    from utils.pointcloud import run_render

    t_geo = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
        meshwarp = MeshWarper(resolution=(opts.height, opts.width), device=str(device))
        pcd = points_world_tensor.unsqueeze(0)
        imgs = colors_tensor.unsqueeze(0)
        masks_rgb = mask_reliable_tensor.view(1, opts.height, opts.width, 1).repeat(1, 1, 1, 3)

        warped_images2, warped_images1, masks1 = [], [], []
        for i in range(T):
            _, _, warped_depth_pcd = run_render(
                pcd=pcd, imgs=imgs, masks=masks_rgb,
                H=opts.height, W=opts.width,
                c2ws=c2ws[i:i + 1], K=intrinsic[i:i + 1],
                num_views=1, return_mask=True, return_depth=True, device=device,
            )
            warped_frame2, _, warped_depth_mesh = meshwarp.forward_warp(
                imgs.permute(0, 3, 1, 2), depth, w2c_0.unsqueeze(0), w2cs[i:i + 1],
                K.unsqueeze(0), intrinsic[i:i + 1],
            )
            warped_images2.append(warped_frame2)
            warped_mask_rgb, _, _ = meshwarp.forward_warp(
                mask_reliable_tensor.view(1, 1, opts.height, opts.width).repeat(1, 3, 1, 1) * 2 - 1,
                depth, w2c_0.unsqueeze(0), w2cs[i:i + 1],
                K.unsqueeze(0), intrinsic[i:i + 1],
            )
            warped_images1.append(warped_mask_rgb)
            masks1.append((warped_depth_pcd.to(warped_depth_mesh.device) <= warped_depth_mesh).to(torch.float32))

        cond_video = (torch.cat(warped_images2) + 1.0) / 2.0
        cond_video1 = (torch.cat(warped_images1) + 1.0) / 2.0
        cond_video1 = (cond_video1 >= 0.5).float()
        cond_masks1 = torch.cat(masks1)

        control_imgs = cond_video * cond_video1 * cond_masks1
        render_masks = cond_video1[:, 0:1] * cond_masks1
        control_imgs = control_imgs * 2.0 - 1.0

        control_imgs[0:1] = validation_image * 2.0 - 1.0
        render_masks[0:1] = 1.0

        import einops
        render_results = einops.rearrange(control_imgs, "f c h w -> f h w c", f=T)
        view_masks = einops.rearrange(render_masks, "f c h w -> f h w c", f=T)
        view_masks = 1.0 - view_masks
        render_results = (render_results + 1.0) / 2.0
        save_video(render_results, os.path.join(out_dir, "render.mp4"), fps=fps)
        save_video(view_masks.repeat(1, 1, 1, 3), os.path.join(out_dir, "mask.mp4"), fps=fps)
        render_results = render_results * 2.0 - 1.0

    del warped_images2, warped_images1, masks1, cond_video, cond_video1, cond_masks1
    del control_imgs, render_masks, pcd, imgs, masks_rgb, points_world_tensor, colors_tensor
    torch.cuda.empty_cache()

    if getattr(pvd.opts, "render_only", False):
        return {"prompt": prompt, "num_frames": T, "depth_avg": depth_avg, "render_only": True,
                "_render_results": render_results.cpu(), "_view_masks": view_masks.cpu()}

    t_render = time.time()
    print(f"[vibe_gen] {out_dir}: prompt={prompt!r}", flush=True)
    diffusion_results = pvd.run_diffusion(render_results, view_masks, prompt, ref_video=None)
    t_diff = time.time()
    save_video(diffusion_results, os.path.join(out_dir, "diffusion.mp4"), fps=fps)

    meta = {
        "prompt": prompt,
        "num_frames": T,
        "depth_avg": depth_avg,
        "sec_geometry": round(t_render - t_geo, 1),
        "sec_prep": round(t_geo - t0, 1),
        "sec_diffusion": round(t_diff - t_render, 1),
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True, help="dir with <scene>/{first_frame.png, trajectory.npz}")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--settings", nargs="+", default=list(SETTINGS))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pose-scale", type=float, default=1.0)
    ap.add_argument("--low-gpu", action="store_true",
                    help="enable_model_cpu_offload (needed for 241f: T5 alone pins ~11GB)")
    ap.add_argument("--render-only", action="store_true", help="stop after render.mp4/mask.mp4")
    ap.add_argument("--trans-scale", type=float, default=-1.0,
                    help="translation multiplier; <0 = auto (depth_avg anchor)")
    ap.add_argument("--blip_path", default="./checkpoints/blip2-opt-2.7b")
    ap.add_argument("--transformer_path", default="./checkpoints/UniView")
    ap.add_argument("--model_name", default="./checkpoints/Wan2.1-VACE-14B-diffusers")
    ap.add_argument("--lora_path", default="./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors")
    ap.add_argument("--stream3r_path", default="./checkpoints/STream3R")
    ap.add_argument("--moge_path", default="./checkpoints/moge/model.pt")
    ap.add_argument("--segnet_path", default="./checkpoints/tracer_b7.pth")
    args = ap.parse_args()

    opts = build_opts(args)
    opts.low_gpu_memory_mode = bool(args.low_gpu)
    opts.render_only = bool(args.render_only)
    opts.trans_scale = float(args.trans_scale)
    from demo import UniScene
    pvd = UniScene(opts)

    for scene in args.scenes:
        w2c_all, K_all, src_hw = load_trajectory_npz(
            os.path.join(args.scenes_root, scene, "trajectory.npz"), pose_scale=args.pose_scale)
        for setting in args.settings:
            out_dir = os.path.join(args.out_root, scene, setting)
            if os.path.exists(os.path.join(out_dir, "diffusion.mp4")):
                print(f"[vibe_gen] skip existing {out_dir}", flush=True)
                continue
            w2c, K_traj = slice_setting(w2c_all, K_all, setting)
            meta = nvs_custom_traj(
                pvd, os.path.join(args.scenes_root, scene, "first_frame.png"),
                w2c, K_traj, src_hw, out_dir)
            print(f"[vibe_gen] {scene}/{setting} done: {meta}", flush=True)

if __name__ == "__main__":
    main()
