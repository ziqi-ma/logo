"""Precompute CondBundles for RL training scenes.

Runs the full demo stack once per scene (MoGe lift + hybrid render along the
benchmark trajectory + BLIP2 caption + UMT5 embeds + VACE VAE encode), mirroring
WanVACEPipeline.__call__ steps 3-5 (pipeline_uniview.py:936-987), and saves a
~40 MB bundle. The training loop then never loads BLIP2/T5/MoGe/tracer.

    python -m rl.data.scene_prep --scenes-root <dir> --scenes s1 s2 ... \
        --out-root <dir> --num_frames 81 --device cuda:0

Scene dirs follow the lyra/trajbench contract: <scene>/{image.png|first_frame.png,
lyra2_traj.npz|trajectory.npz}; 241-pose trajectories are strided to num_frames.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.data.cond_bundle import CondBundle, save_cond_bundle
from rl.data.traj_adapter import load_trajectory_npz, rescale_intrinsics
from rl.inference.vibe_gen import build_opts, nvs_custom_traj

def encode_bundle(pipe, render_results: torch.Tensor, view_masks: torch.Tensor,
                  prompt: str, opts, num_frames: int, meta: dict) -> CondBundle:
    """(T,H,W,C in [-1,1]) render + (T,H,W,1) hole mask -> CondBundle."""
    device = pipe.transformer.device
    tdt = pipe.transformer.dtype
    video = render_results.permute(3, 0, 1, 2).unsqueeze(0).to(device, torch.float32)
    mask = view_masks.permute(3, 0, 1, 2).unsqueeze(0).to(device, torch.float32)

    with torch.no_grad():
        pe, ne = pipe.encode_prompt(
            prompt=prompt, negative_prompt=opts.negative_prompt,
            do_classifier_free_guidance=True, num_videos_per_prompt=1,
            max_sequence_length=512, device=device)
        cond = pipe.prepare_video_latents_noref(video, mask, None, device)
        m = pipe.prepare_masks_noref(mask)
        conditioning_latents = torch.cat([cond, m], dim=1).to(tdt)

    n_vace = len(pipe.transformer.config.vace_layers)
    return CondBundle(
        prompt_embeds=pe.to(tdt), negative_prompt_embeds=ne.to(tdt),
        conditioning_latents=conditioning_latents, ref_latents=None,
        conditioning_scale=torch.ones(n_vace),
        height=opts.height, width=opts.width, num_frames=num_frames, meta=meta)

def find_inputs(scene_dir: str):
    img = next((os.path.join(scene_dir, n) for n in ("image.png", "first_frame.png")
                if os.path.exists(os.path.join(scene_dir, n))), None)
    npz = next((os.path.join(scene_dir, n) for n in ("lyra2_traj.npz", "trajectory.npz")
                if os.path.exists(os.path.join(scene_dir, n))), None)
    assert img and npz, f"missing inputs in {scene_dir}"
    return img, npz

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes-root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--pose-scale", type=float, default=1.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--upload-uri", default=None,
                    help="upload each bundle to <uri>/<scene>/cond.pt and drop the local copy")
    ap.add_argument("--skip-existing", action="store_true",
                    help="also skip scenes whose bundle already exists at --upload-uri")
    for k, v in (("blip_path", "./checkpoints/blip2-opt-2.7b"),
                 ("transformer_path", "./checkpoints/UniView"),
                 ("model_name", "./checkpoints/Wan2.1-VACE-14B-diffusers"),
                 ("lora_path", "./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors"),
                 ("stream3r_path", "./checkpoints/STream3R"),
                 ("moge_path", "./checkpoints/moge/model.pt"),
                 ("segnet_path", "./checkpoints/tracer_b7.pth")):
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()

    opts = build_opts(args)
    opts.render_only = True
    opts.trans_scale = -1.0  # 0.5*depth_avg anchor
    from demo import UniScene
    pvd = UniScene(opts)

    n_done = n_skip = n_fail = 0
    for scene in args.scenes:
        out_dir = os.path.join(args.out_root, scene)
        if os.path.exists(os.path.join(out_dir, "cond.pt")):
            print(f"[scene_prep] {scene}: exists, skip", flush=True)
            n_skip += 1
            continue
        if args.skip_existing and args.upload_uri:
            from rl.data.gcs_util import exists as _remote_exists
            if _remote_exists(f"{args.upload_uri.rstrip('/')}/{scene}/cond.pt"):
                print(f"[scene_prep] {scene}: exists remote, skip", flush=True)
                n_skip += 1
                continue
        os.makedirs(out_dir, exist_ok=True)
        try:
            img, npz = find_inputs(os.path.join(args.scenes_root, scene))
            w2c, K, src_hw = load_trajectory_npz(npz, pose_scale=args.pose_scale)
            n = w2c.shape[0]
            stride = max(1, round((n - 1) / (args.num_frames - 1)))
            idx = torch.arange(0, n, stride)[: args.num_frames]

            meta = nvs_custom_traj(pvd, img, w2c[idx], K[idx], src_hw, out_dir)
            rr = meta.pop("_render_results")
            vm = meta.pop("_view_masks")
            bundle = encode_bundle(pvd.pipeline, rr, vm, meta["prompt"], pvd.opts,
                                   len(idx), {"scene": scene, "stride": int(stride), **meta})
            local = os.path.join(out_dir, "cond.pt")
            save_cond_bundle(bundle, local)
            if args.upload_uri:
                from rl.data.gcs_util import upload_file
                upload_file(local, f"{args.upload_uri.rstrip('/')}/{scene}/cond.pt")
                # a full set is ~64GB; keep the pod's disk flat
                os.remove(local)
            n_done += 1
            print(f"[scene_prep] {scene}: bundle saved "
                  f"(cond {tuple(bundle.conditioning_latents.shape)}) "
                  f"[done={n_done} skip={n_skip} fail={n_fail}]", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad scene must not kill the shard
            n_fail += 1
            print(f"[scene_prep] {scene}: FAILED {type(e).__name__}: {e} "
                  f"[done={n_done} skip={n_skip} fail={n_fail}]", flush=True)
    print(f"[scene_prep] shard complete: done={n_done} skip={n_skip} fail={n_fail}", flush=True)

if __name__ == "__main__":
    main()
