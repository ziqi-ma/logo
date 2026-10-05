"""Resident DiffusionNFT loop for UniWorld-View.

Per step (scene-parallel like Lyra's loop): SAMPLE k_local rollouts with `old`
(no grad) -> decode + store -> SCORE out-of-process (reproj_rgbd reward) ->
per-scene z-score advantage (all-gather) -> TRAIN `new` (all timesteps sweep,
grad accumulation, manual all_reduce) -> EMA old<-new -> checkpoint/wandb.

    torchrun --standalone --nproc_per_node=N -m rl.loop.nft_loop \
        --cond-root <scene_prep output> --scenes s1 s2 ... --work-dir ... \
        --ckpt-out ... [--scenes-parallel P] [--k-rollouts K]

Scenes must be pre-processed by rl.data.scene_prep (cond.pt per scene). The reward
scorer runs via NFT_SCORER_PY (a python with the scorer deps, e.g. the
score-venv) and shares the rollout GPU.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import timedelta

import torch

# The package root, i.e. the directory holding rl/. `rl.scoring.nft_score_cli` is imported
# from here and is also run as a module by the scorer subprocess, whose cwd must be this
# same directory -- one level above rl/, not above rl/loop/.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PKG_ROOT)

from rl.loop import adapters
from rl.scoring import nft_voxel
from rl.loop.advantage import global_perframe_r, global_scene_advantage
from rl.scoring.nft_score_cli import COMBO as _TRAIN_COMBO
from rl.data.cond_bundle import load_cond_bundle
from rl.loop.dist_env import scorer_env
from rl.loop.rollout_store import RolloutStoreWriter, read_rewards
from rl.loop.sampling import decode_latents, sample_rollout
from rl.loop.schedule import build_rl_schedule
from rl.loop.trainer import train_pass

def _dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        timeout = timedelta(minutes=int(os.environ.get("NFT_NCCL_TIMEOUT_MIN", "120")))
        dist.init_process_group(backend="nccl", timeout=timeout)
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world, local_rank

def _init_wandb(args, rank):
    if rank != 0:
        return None
    try:
        import wandb
        if not (os.environ.get("WANDB_API_KEY")
                or os.environ.get("WANDB_MODE") in ("offline", "disabled")):
            # loud: a silent return here means a multi-day run logs no curves at all
            print("[wandb] WARNING: WANDB_API_KEY is empty -> NO METRICS WILL BE LOGGED "
                  "to wandb for this run (step/VAL lines still go to stdout)", flush=True)
            return None
        # Deterministic id keyed on the RUN NAME (== the checkpoint prefix), so a resubmission
        # -- provider move, preemption, priority change -- reattaches to the same wandb run
        # instead of starting a sibling whose x-axis restarts at 0. resume="allow" creates it
        # on first launch and appends afterwards.
        # not basename(ckpt_out): in the container that is always "/outputs/adapters", so every
        # run would hash to the same id and they would all append into one merged wandb run.
        # NEW_LORA_URI is .../uniworld/nft/<run-name>/adapters -- unique per run and stable across
        # resubmissions, which is exactly the key we want.
        _lora_uri = os.environ.get("NEW_LORA_URI", "").rstrip("/")
        run_name = (os.environ.get("WANDB_NAME")
                    or (os.path.basename(os.path.dirname(_lora_uri)) if _lora_uri else "")
                    or f"nft-{args.combo}")
        run_id = os.environ.get("WANDB_RUN_ID") or hashlib.sha1(
            run_name.encode()).hexdigest()[:16]
        wandb.init(project=os.environ.get("WANDB_PROJECT", "uniworld-diffnft"),
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=run_name, id=run_id, resume="allow",
                   config=vars(args))
        print(f"[wandb] run={run_name} id={run_id} resume=allow", flush=True)
        return wandb
    except Exception as e:  # noqa: BLE001
        print(f"[wandb] disabled: {type(e).__name__}: {e}", flush=True)
        return None

def alt_phases(args):
    """[(n_steps, combo), ...] for --alt-schedule, or None when alternation is off.

    'A:B' -> two phases (combo a, combo b); 'A:B:C' -> three (a, b, c). Zero-length phases are
    dropped so '12:5:0' behaves exactly like '12:5'.
    """
    if not args.alt_schedule:
        return None
    parts = args.alt_schedule.split(":")
    combos = [args.alt_combo_a, args.alt_combo_b, args.alt_combo_c]
    if len(parts) > len(combos):
        raise SystemExit(f"--alt-schedule supports at most {len(combos)} phases, "
                         f"got {args.alt_schedule!r}")
    try:
        ns = [int(x) for x in parts]
    except ValueError as e:
        raise SystemExit(f"--alt-schedule must look like '12:5:3', got {args.alt_schedule!r}: {e}")
    if any(n < 0 for n in ns) or sum(ns) == 0:
        raise SystemExit(f"--alt-schedule needs a positive total, got {args.alt_schedule!r}")
    return [(n, c) for n, c in zip(ns, combos) if n > 0]

def alt_combo_for_step(step: int, args):
    """Which combo this step trains on. Pure function of the ABSOLUTE step index, so a resume
    lands mid-cycle without shifting the pattern."""
    phases = alt_phases(args)
    if not phases:
        return None
    cycle = sum(n for n, _c in phases)
    off = step % cycle
    for n, c in phases:
        if off < n:
            return c
        off -= n
    return phases[-1][1]

def build_rl_pipeline(args, device):
    from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
    from model.pipeline_uniview import WanVACEPipeline
    from model.uniview_transformer import WanVACETransformer3DModel
    from rl.loop.schedule import SCHEDULER_CONFIG

    transformer = WanVACETransformer3DModel.from_pretrained(
        args.transformer_path, torch_dtype=torch.bfloat16)
    vae = AutoencoderKLWan.from_pretrained(args.model_name, subfolder="vae",
                                           torch_dtype=torch.float32)
    pipe = WanVACEPipeline(
        tokenizer=None, text_encoder=None, transformer=transformer, vae=vae,
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG))
    transformer.to(device)
    vae.to(device)
    adapters.load_rl_adapters(pipe, args.causvid_lora, args.policy_lora, args.ref_lora)
    transformer.enable_gradient_checkpointing()
    return pipe

def try_resume(args, pipe, opt, rank):
    """Scan ckpt-out for nft_new_step*.safetensors; Tier1 = +trainstate, Tier2 = old<-new."""
    import glob
    ckpts = sorted(glob.glob(os.path.join(args.ckpt_out, "nft_new_step*.safetensors")))
    if not ckpts:
        return 0, 0
    # PREFER THE NEWEST ADAPTER THAT still HAS ITS TRAINSTATE. The pair is written back-to-back
    # locally but mirrored to remote by a periodic, non-atomic upload_prefix, and the adapter
    # (204MB) lands ~45s before the trainstate (614MB) -- so a crash inside that window leaves a
    # TORN pair remotely: adapter N, no trainstate N. Taking ckpts[-1] unconditionally then fell to
    # Tier 2 (Adam moments dropped, `old` reset to `new` via ema_update_old(..,0.0), global_step
    # reset) and the run silently continued from a cold optimizer with no reference policy. That is
    # otherwise a crash in validation after writing an adapter loses the step,
    # resumed Tier 2 from 19, and trained 33 steps on a degraded basis before anyone noticed.
    # Giving up one step of progress is far cheaper, so resume from the newest COMPLETE pair.
    _step_of = lambda p: int(p.rsplit("step", 1)[1].split(".")[0])
    _ts_for = lambda n: os.path.join(args.ckpt_out, f"nft_trainstate_step{n:04d}.pt")
    ranked = sorted(((_step_of(p), p) for p in ckpts), reverse=True)
    paired = [(n, p) for n, p in ranked if os.path.exists(_ts_for(n))]
    if paired and paired[0][0] != ranked[0][0]:
        print(f"[resume] rank{rank}: newest adapter is step{ranked[0][0]:04d} but its trainstate is "
              f"MISSING (torn checkpoint); resuming from step{paired[0][0]:04d} instead, which has "
              f"one -- this keeps the optimizer/EMA state instead of restarting them cold",
              flush=True)
    step, latest = paired[0] if paired else ranked[0]
    n = adapters.load_new_adapter_inplace(pipe, latest)
    print(f"[resume] rank{rank}: loaded {latest} ({n} params), resuming at step {step + 1}",
          flush=True)
    ts = os.path.join(args.ckpt_out, f"nft_trainstate_step{step:04d}.pt")
    if os.path.exists(ts):
        _s, gs = adapters.load_trainstate(pipe.transformer, opt, ts)
        return step + 1, gs
    adapters.ema_update_old(pipe.transformer, 0.0)  # old <- new
    return step + 1, 0

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond-root", required=True, help="rl.data.scene_prep output root")
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--work-dir", default="/tmp/nft_loop")
    ap.add_argument("--ckpt-out", required=True)
    ap.add_argument("--combo", default="reproj_rgbd")
    ap.add_argument("--k-rollouts", type=int, default=8)
    ap.add_argument("--scenes-parallel", type=int, default=1)
    ap.add_argument("--num-steps", type=int, default=200)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--num-sample-steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--beta-kl", type=float, default=0.5)
    ap.add_argument("--ema-decay", type=float, default=0.99)
    ap.add_argument("--lora-l2-lambda", type=float, default=0.0,
                    help="weight of the one-sided barrier lam*relu(lora_l2_ratio-target)^2; "
                         "0 disables (ratio is still logged)")
    ap.add_argument("--lora-l2-target", type=float, default=1.3,
                    help="ratio at which the barrier starts pushing back")
    ap.add_argument("--alt-schedule", default="",
                    help="ALTERNATE objectives instead of a weighted sum: 'A:B' trains A steps on "
                         "--alt-combo-a then B steps on --alt-combo-b, repeating (e.g. '4:1'). "
                         "Empty = weighted sum via --combo (default).")
    ap.add_argument("--alt-combo-a", default="reproj_rgbd")
    ap.add_argument("--alt-combo-b", default="hpsv3_only")
    ap.add_argument("--alt-combo-c", default="camera_only",
                    help="third phase combo, used only when --alt-schedule has 3 parts")
    ap.add_argument("--adv-clip-max", type=float, default=1.3)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--grad-steps", type=int, default=1)
    ap.add_argument("--inner-epochs", type=int, default=1)
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--keep-rollouts", action="store_true")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--transformer_path", default="./checkpoints/UniView")
    ap.add_argument("--model_name", default="./checkpoints/Wan2.1-VACE-14B-diffusers")
    ap.add_argument("--causvid-lora",
                    default="./checkpoints/loras/Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors")
    ap.add_argument("--policy-lora", default=None)
    ap.add_argument("--ref-lora", default=None)
    ap.add_argument("--reward-window", default="full",
                    help="windowed reward: window size in PIXEL frames (e.g. 27 for 81f). "
                         "'full'/0 = one scalar reward per rollout (default). Converted to "
                         "latents via 81f/21latents -> ~4 frames per latent.")
    ap.add_argument("--frames-per-latent", type=int, default=4)
    ap.add_argument("--val-scenes", nargs="*", default=[],
                    help="held-out scene ids (cond bundles must exist under --cond-root)")
    ap.add_argument("--val-every", type=int, default=0, help="0 disables validation")
    ap.add_argument("--val-k", type=int, default=1, help="rollouts (fixed seeds) per val scene")
    ap.add_argument("--val-base-seed", type=int, default=777000)
    ap.add_argument("--val-videogpa", action="store_true",
                    help="score val clips with the full VideoGPA suite, not just the reward")
    ap.add_argument("--val-videos-uri", default="",
                    help="upload each validation's clips to <uri>/stepNNNN/<scene12>_seedJJ.mp4 "
                         "(rollouts are deleted after scoring, so without this a checkpoint's "
                         "output can only be seen by re-sampling it on a GPU)")
    ap.add_argument("--val-hpsv3", action="store_true",
                    help="also log HPSv3 on val clips, so a perceptual metric is reported "
                         "next to the geometry reward")
    args = ap.parse_args()

    rank, world, local_rank = _dist()
    device = f"cuda:{local_rank}"
    torch.backends.cuda.matmul.allow_tf32 = True

    P = max(1, args.scenes_parallel)
    assert world % P == 0, f"world {world} % scenes-parallel {P} != 0"
    ranks_per_scene = world // P
    my_group = rank // ranks_per_scene
    assert args.k_rollouts % ranks_per_scene == 0
    k_local = args.k_rollouts // ranks_per_scene

    pipe = build_rl_pipeline(args, device)
    transformer = pipe.transformer
    sched = build_rl_schedule(args.num_sample_steps, device=device)
    params = [p for p in transformer.parameters() if p.requires_grad]
    assert params, "no trainable params (new adapter missing?)"
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)

    os.makedirs(args.ckpt_out, exist_ok=True)
    start_step, global_step = try_resume(args, pipe, opt, rank)
    wb = _init_wandb(args, rank)

    if rank == 0:
        # Durable settings snapshot next to the checkpoints (uploaded to the object store with
        # them), so a checkpoint is never separated from the recipe that made it --
        # including the reward calibration, which lives in the scorer, not in args.
        from rl.scoring.nft_score_cli import COMBO, NORM
        cfg = {"args": {k: v for k, v in vars(args).items()},
               "reward": {"combo": COMBO, "norm": NORM},
               "topology": {"world": world, "scenes_parallel": P,
                            "ranks_per_scene": ranks_per_scene, "k_local": k_local,
                            "k_rollouts": args.k_rollouts},
               "schedule": {"timesteps": [float(t) for t in sched.timesteps],
                            "sigmas": [float(s) for s in sched.sigmas]}}
        if nft_voxel.voxel_alpha() > 0:
            cfg["voxel"] = {
                "alpha": nft_voxel.voxel_alpha(),
                "mix": nft_voxel.mix_lambda(),
                "local": nft_voxel.local_lambda(),
                "temporal": os.environ.get("NFT_VOXEL_TEMPORAL", "rollout"),
                "patch": list(nft_voxel.patch_grid()),
                "depth_cap": nft_voxel.depth_cap()}
        with open(os.path.join(args.ckpt_out, "run_config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        print(f"[loop] reward calibration: {json.dumps(NORM)}", flush=True)
        if wb:
            wb.config.update({"reward_norm": NORM, "reward_combo": COMBO,
                              "world": world, "k_local": k_local}, allow_val_change=True)

    # windowed reward: window size in PIXEL frames -> latents. 81 frames map to 21
    # latents (1 + 20*4), so ~4 pixel frames per latent.
    _win = str(args.reward_window).strip().lower()
    latent_T = 1 + (81 - 1) // args.frames_per_latent
    w_latent = 0 if _win in ("", "full", "0", "none") else max(
        1, round(int(_win) / args.frames_per_latent))
    # voxel reward (NFT_REWARD_VOXEL, supersedes windowed): per-(voxel,
    # frame) advantage painted onto a patch grid, r [T][gh][gw] per rollout. The
    # scorer is gated by --voxel on the TRAIN command only, so validation below
    # keeps its plain global scoring untouched.
    vox_alpha = nft_voxel.voxel_alpha()
    vox_mix = nft_voxel.mix_lambda()
    if rank == 0:
        print(f"[loop] reward mode: "
              + (f"VOXEL alpha={vox_alpha} mix={vox_mix} patch={nft_voxel.patch_grid()} "
                 f"local={nft_voxel.local_lambda()} ({latent_T} latents)"
                 if vox_alpha > 0 else
                 f"WINDOWED window={_win} pixel frames -> {w_latent} latents "
                 f"(of {latent_T})" if w_latent else "GLOBAL (one scalar r per rollout)"),
              flush=True)

    scorer_py = os.environ.get(
        "NFT_SCORER_PY", sys.executable)

    _cache: OrderedDict = OrderedDict()

    def bundle(scene):
        b = _cache.get(scene)
        if b is None:
            b = load_cond_bundle(os.path.join(args.cond_root, scene, "cond.pt"), device)
            _cache[scene] = b
            if len(_cache) > 16:
                _cache.popitem(last=False)
        _cache.move_to_end(scene)
        return b

    import torch.distributed as dist
    dist_on = world > 1 and dist.is_initialized()

    # ---- held-out validation ------------------------------------------------------
    # (scene, seed) tasks are spread FLAT across all ranks, not leader-only: with a
    # leader-only pass the idle ranks sit in the all_gather while one rank samples for
    # many minutes, which trips the NCCL watchdog (Lyra hit exactly this).
    val_tasks = [(sc, j) for sc in args.val_scenes for j in range(max(1, args.val_k))]
    my_val = val_tasks[rank::world]

    def run_validation(step: int):
        recs = []
        transformer.eval()
        # Sample every val clip first, then offload the 17.76B transformer to CPU before
        # scoring. The scorers are large models in their own right (DA3 + VGGT-Omega, and
        # HPSv3 is a 7B Qwen2-VL needing ~16GB); with 35GB of transformer resident they
        # OOM on an 80GB card. Lyra's loop offloads for the same reason -- step-0
        # validation of uw-nft-full-k16 silently lost hpsv3_vid because this was missing.
        pending = []
        for vsc, j in my_val:
            vdir = os.path.join(args.work_dir, f"val_{step:04d}", f"rank_{rank:03d}", f"{vsc}_s{j}")
            vroot = os.path.join(vdir, "rollouts")
            vwriter = RolloutStoreWriter(vroot)
            vb = bundle(vsc)
            vseed = args.val_base_seed + j * 7919  # fixed across steps -> comparable
            with adapters.adapter_ctx(transformer, "new"), torch.no_grad():
                x0 = sample_rollout(transformer, vb, sched, vseed,
                                    guidance_scale=args.guidance, device=device)
                frames = decode_latents(pipe.vae, x0)
                vwriter.write(vsc, j, x0, frames, os.path.join(args.cond_root, vsc, "cond.pt"), vseed)
            pending.append((vsc, j, vdir, vroot))
            gc.collect()
            torch.cuda.empty_cache()

        offloaded = False
        if pending and (args.val_videogpa or args.val_hpsv3):
            transformer.to("cpu")
            pipe.vae.to("cpu")
            offloaded = True
            gc.collect()
            torch.cuda.empty_cache()
        for vsc, j, vdir, vroot in pending:
            vrew = os.path.join(vdir, "rewards.jsonl")
            cmd = [scorer_py, "-m", "rl.scoring.nft_score_cli", "--root", vroot,
                   "--out", vrew, "--device", "cuda:0"]
            if args.val_videogpa:
                cmd.append("--videogpa")
            if args.val_hpsv3:
                cmd.append("--hpsv3")
            # NFT_LOG_CAMERA: val runs under the container's --combo (geometry), so _need_camera()
            # would be False and val would carry no camera numbers at all. The training-side
            # rpe_rot/rpe_trans are the REWARD, measured on the policy's own rollouts for scenes it
            # just trained on -- they cannot say whether camera adherence generalizes. This is the
            # held-out measurement. Cost is one extra VGGT forward per val clip (16 clips / 10
            # steps vs 128 rollouts per step), and val/* aggregation is generic so the keys appear
            # as val/rpe_rot and val/rpe_trans with no further plumbing.
            vrc = subprocess.run(cmd, cwd=_PKG_ROOT,
                                 # not a raw os.environ: the scorer must not inherit torchrun's
                                 # MASTER_PORT or hpsv3 dies with EADDRINUSE on rank 0 only,
                                 # which ReduceOp.MIN then turns into a skipped step on every
                                 # rank. See rl/dist_env.py.
                                 env=scorer_env(local_rank),
                                 capture_output=True, text=True)
            # print the tail even on success: the scorer exits 0 when an individual
            # metric fails, so a silently-missing hpsv3/videogpa is otherwise invisible
            if vrc.returncode != 0 or "FAILED" in vrc.stdout:
                print(f"[loop] rank{rank} VAL scorer {vsc}/s{j} rc={vrc.returncode}: "
                      f"{vrc.stdout[-800:]}\n{vrc.stderr[-500:]}", flush=True)
            for line in (open(vrew).read().splitlines() if os.path.exists(vrew) else []):
                row = json.loads(line)
                recs.append({"scene": vsc, "seed": j, "R": row["R"],
                             "metrics": row.get("metrics", {})})
            # Keep the clip: without this the only way to SEE a checkpoint's output is to
            # re-sample it locally on a GPU. Uploaded per (step, scene, seed) so any
            # checkpoint's held-out videos are one `aws s3 cp` away.
            if args.val_videos_uri:
                clip = os.path.join(vroot, f"scene_{vsc}", "rollout_00", "clip.mp4")
                if os.path.exists(clip):
                    dest = (f"{args.val_videos_uri.rstrip('/')}/step{step:04d}/"
                            f"{vsc[:12]}_seed{j:02d}.mp4")
                    try:
                        from rl.data.gcs_util import upload_file
                        upload_file(clip, dest)
                    except Exception as e:  # noqa: BLE001 - never abort val over a video
                        print(f"[loop] rank{rank} val video upload failed {vsc}/s{j}: "
                              f"{type(e).__name__}: {e}", flush=True)
            shutil.rmtree(vdir, ignore_errors=True)

        if offloaded:  # back to GPU before training resumes
            transformer.to(device)
            pipe.vae.to(device)
            gc.collect()
            torch.cuda.empty_cache()

        allrecs = recs
        if dist_on:
            gathered = [None] * world
            dist.all_gather_object(gathered, recs)
            allrecs = [x for sub in gathered for x in (sub or [])]
        if rank != 0:
            return
        per_scene = defaultdict(list)
        per_metric = defaultdict(list)
        for r in allrecs:
            if r["R"] is not None and math.isfinite(r["R"]):
                per_scene[r["scene"]].append(r["R"])
            for k, v in (r.get("metrics") or {}).items():
                if isinstance(v, (int, float)) and math.isfinite(v):
                    per_metric[k].append(v)
        # PER-CLIP rows next to the checkpoints (so they upload to the object store with them).
        # The aggregate VAL line hides which scenes moved and which regressed, and the
        # rollouts themselves are deleted -- without this the only per-clip numbers live
        # in pod stdout, and answering "which scene got worse?" means re-sampling on a GPU.
        # Keys match the uploaded videos: stepNNNN / <scene12>_seedJJ.
        try:
            vpath = os.path.join(args.ckpt_out, f"val_metrics_step{step:04d}.jsonl")
            with open(vpath, "w") as f:
                for r in sorted(allrecs, key=lambda x: (x["scene"], x["seed"])):
                    f.write(json.dumps({"step": step, "scene": r["scene"],
                                        "clip": f"{r['scene'][:12]}_seed{r['seed']:02d}",
                                        "seed": r["seed"], "R": r["R"],
                                        **(r.get("metrics") or {})}) + "\n")
            print(f"[loop] wrote {len(allrecs)} per-clip val rows -> {vpath}", flush=True)
        except Exception as e:  # noqa: BLE001 - never lose the aggregate over this
            print(f"[loop] per-clip val write failed: {type(e).__name__}: {e}", flush=True)

        allR = [x for v in per_scene.values() for x in v]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")
        vmsg = {"val/step": step, f"val/{args.combo}_mean": mean(allR),
                "val/n_scenes": len(per_scene), "val/n_rollouts": len(allR)}
        vmsg.update({f"val/{k}": mean(v) for k, v in per_metric.items()})
        print(f"[loop] VAL step {step}: " +
              " ".join(f"{k.split('/', 1)[1]}={vmsg[k]:.4f}" for k in sorted(vmsg)
                       if isinstance(vmsg[k], float)), flush=True)
        if wb:
            wb.log(vmsg, step=step)

    # NO `my_val and` GUARD: run_validation() ends in a COLLECTIVE (all_gather_object at
    # L410) and is followed by dist.barrier(). With world > len(val_tasks) -- 128 ranks vs
    # 25 scenes x val_k 4 = 100 tasks -- the idle ranks skipped both and ran straight into the
    # next train step, whose advantage.py:111 all_gather_object then matched call-order with
    # this one. Rank 0 received dicts where it expected lists of records and died with
    # `TypeError: string indices must be integers`, killing the whole 128-GPU jobset.
    # The body already no-ops on an empty my_val, so every rank can and must enter.
    if args.val_every > 0 and start_step == 0:
        run_validation(0)  # step-0 baseline: `new` == the untrained CausVid policy

    for step in range(start_step, args.num_steps):
        t0 = time.time()
        scene = args.scenes[(step * P + my_group) % len(args.scenes)]
        step_dir = os.path.join(args.work_dir, f"step_{step:04d}", f"rank_{rank:03d}")
        rollout_root = os.path.join(step_dir, "rollouts")
        writer = RolloutStoreWriter(rollout_root)
        b = bundle(scene)
        cond_path = os.path.join(args.cond_root, scene, "cond.pt")

        # SAMPLE with `old`
        transformer.eval()
        with adapters.adapter_ctx(transformer, "old"):
            for j in range(k_local):
                seed = args.base_seed + step * 100000 + rank * 1000 + j
                x0 = sample_rollout(transformer, b, sched, seed,
                                    guidance_scale=args.guidance, device=device)
                frames = decode_latents(pipe.vae, x0)
                writer.write(scene, rank * k_local + j, x0, frames, cond_path, seed)
        t_sample = time.time()

        # SCORE (subprocess in the scorer env; shares this GPU)
        rewards_jsonl = os.path.join(step_dir, "rewards.jsonl")
        gc.collect()
        torch.cuda.empty_cache()
        # hpsv3 at TRAIN time only when it is actually a reward term: driven by the combo, not a
        # flag, so a geometry-only run can never pay for a 7B forward it will not use. The scorer
        # batches all of this step's clips into one interpreter (one model load per step per rank).
        #
        # PER-PHASE scoring. alt_combo_for_step is a pure function of the absolute step index, so
        # this step's objective is known before the scorer launches -- there is no need to emit
        # every phase's metrics on every step. Telling the scorer the phase via NFT_COMBO makes its
        # own _need_hps/_need_camera fall out of that one combo, which skips a 7B HPSv3 forward on
        # the 15-in-20 non-hpsv3 steps and a second full VGGT forward (reproject_vggt, for the
        # camera terms) on the 17-in-20 non-camera steps.
        #
        # Safe now, unlike when this emitted everything unconditionally: a term missing when the
        # phase needs it is no longer silent -- nft_voxel drops unusable terms loudly and returns
        # {} (scalar fallback) rather than a neutral 0.5 grid, and the -inf recompute below drops
        # rollouts with a count instead of quietly keeping the scorer's combo.
        _phase_combo = alt_combo_for_step(step, args) if args.alt_schedule else None
        _phase_terms = _TRAIN_COMBO
        if _phase_combo is not None:
            from rl.scoring.nft_score_cli import COMBOS as _ALL_COMBOS
            _phase_terms = _ALL_COMBOS[_phase_combo]
        _need_hps = any(m == "hpsv3_vid" for m, _w in _phase_terms)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(local_rank)}
        if _phase_combo is not None:
            # subprocess env only -- never os.environ, which persists across steps and would pin
            # every later step to whichever phase happened to run first.
            env["NFT_COMBO"] = _phase_combo
            env.pop("NFT_ALT_COMBOS", None)      # else _need_camera() forces camera every step
        score_cmd = [scorer_py, "-m", "rl.scoring.nft_score_cli", "--root", rollout_root,
                     "--out", rewards_jsonl, "--device", "cuda:0"]
        if w_latent:
            score_cmd += ["--per-frame", str(latent_T)]
        if vox_alpha > 0:
            # latent_T, not a re-derived frame count: the payload's t axis must
            # match x0.shape[2] or the shape guard below downgrades every rollout
            score_cmd += ["--voxel", str(latent_T)]
        if _need_hps:
            score_cmd += ["--hpsv3"]
        rc = subprocess.run(
            score_cmd,
            cwd=_PKG_ROOT,
            env=env, capture_output=True, text=True)
        if rc.returncode != 0:
            print(f"[loop] rank{rank} scorer failed:\n{rc.stdout[-1500:]}\n{rc.stderr[-1500:]}",
                  flush=True)
        else:
            # the scorer exits 0 even when every rollout failed; surface its tail
            # grep the lines that explain a missing reward term; the 400-char tail is filled by
            # the metrics dump, which is why rank0's "hpsv3 batch: 0/2" never reached the log.
            for _l in rc.stdout.splitlines():
                if any(k in _l for k in ("hpsv3 batch", "BATCH FAILED", "hpsv3 unavailable",
                                         "hpsv3 single FAILED", "camera term skipped")):
                    print(f"[loop] rank{rank} SCORER: {_l.strip()}", flush=True)
            print(f"[loop] rank{rank} scorer tail: {rc.stdout[-400:]}", flush=True)
        t_score = time.time()

        # ADVANTAGE (per-scene z-score, global gather)
        local_R = {}
        local_metrics = {}
        local_Rpf = {}
        local_Vp = {}
        rows = read_rewards(rewards_jsonl) if os.path.exists(rewards_jsonl) else []
        _rows_R = {}
        for row in rows:
            local_R[int(row["rollout"])] = float(row["R"])
            _rows_R[int(row["rollout"])] = float(row["R"])   # scorer's R, for the empty-rank fallback
            m = row.get("metrics", {})
            rpf = m.pop("R_perframe", None)  # keep the vector out of the scalar metric means
            local_metrics[int(row["rollout"])] = m
            if rpf:
                local_Rpf[int(row["rollout"])] = rpf
            vp = row.get("voxel_payload")
            if vp:
                local_Vp[int(row["rollout"])] = vp
        # ALTERNATING objective: rebind COMBO for this step, then recompute the scalar R from
        # the raw metrics (row["R"] was computed by the scorer under ITS env combo, which is not
        # this step's objective). combo_spec() in the voxel path re-reads COMBO on call, so it
        # picks the switch up with no further plumbing.
        alt_name = alt_combo_for_step(step, args)
        if alt_name is not None:
            import rl.scoring.nft_score_cli as _sc
            _sc.set_combo(alt_name)
            _n_bad = 0
            for _idx, _m in local_metrics.items():
                _r = _sc.combine(_m)
                if math.isfinite(_r):
                    local_R[_idx] = _r
                else:
                    # combine() returns -inf when a term of this phase's combo is missing. Keeping
                    # the scorer's R here silently trains the phase on the SCORER's combo instead
                    # (camera phases ran on reproj_rgbd for 3 steps before this was caught,
                    # Drop the rollout so the failure shows up as a rollout count.
                    local_R.pop(_idx, None)
                    _n_bad += 1
            if _n_bad:
                print(f"[loop] rank{rank} step {step}: {_n_bad}/{len(local_metrics)} rollouts have "
                      f"no finite R under combo '{alt_name}' (missing term); dropped", flush=True)
            # A rank at ZERO samples is fatal for every rank: train_pass takes ReduceOp.MIN over the
            # per-rank sample counts, so one empty rank makes all N skip the step. rank0's hpsv3 has
            # been missing on every hpsv3 step since the first alternating run; before the drop was
            # added it silently kept the scorer's R and the step still trained. Restore that
            # fallback only when dropping would empty the rank, and say so loudly -- losing 2/128
            # rollouts to the wrong objective beats losing the whole step on all 64 ranks.
            if not local_R and local_metrics:
                for _idx, _m in local_metrics.items():
                    _r0 = _rows_R.get(_idx)
                    if _r0 is not None and math.isfinite(_r0):
                        local_R[_idx] = _r0
                print(f"[loop] rank{rank} step {step}: ALL {len(local_metrics)} rollouts lacked a "
                      f"finite R under '{alt_name}'; restored the scorer's R for {len(local_R)} of "
                      f"them so this rank is not empty (a zero-sample rank skips the step on ALL "
                      f"ranks via ReduceOp.MIN). These rollouts are scored on the SCORER's combo, "
                      f"not '{alt_name}' -- fix the missing term.", flush=True)
        r_by_idx, agg, per_seed = global_scene_advantage(
            local_R, scene, rank, world, adv_clip_max=args.adv_clip_max,
            local_metrics=local_metrics)

        # WINDOWED reward: per-latent r from window-wise z-scores across the scene's K
        # rollouts. Falls back to the scalar r for any rollout without a usable vector.
        # VOXEL reward supersedes it: r [T][gh][gw] per rollout, same fallback.
        r_perframe_by_idx, r_window_std = {}, float("nan")
        r_voxel_matched = float("nan")
        _vox_extra = {}
        if vox_alpha > 0:
            # gate on the env alone, not `and local_Vp`: this contains a collective,
            # and a rank whose scorer produced no payloads must still join the gather
            r_perframe_by_idx, vdiag = nft_voxel.global_pervoxel_r(
                local_Vp, scene, rank, world, args.adv_clip_max, nft_voxel.combo_spec())
            r_window_std = vdiag["voxel_std"]
            r_voxel_matched = vdiag["matched_frac"]
            # pre_contrib_* / pre_share_*: each term's magnitude before the cross-rollout z, i.e.
            # the real 0.45/0.45/0.1. Previously computed and then dropped on the floor here.
            _vox_extra = {f"reward/{k}": v for k, v in vdiag.items()
                          if k.startswith(("pre_contrib_", "pre_share_"))}
        elif w_latent and local_Rpf:
            r_perframe_by_idx, r_window_std = global_perframe_r(
                local_Rpf, scene, rank, world, w_latent, adv_clip_max=args.adv_clip_max)

        # TRAIN `new`
        samples = []
        for row in rows:
            idx = int(row["rollout"])
            if idx not in r_by_idx:
                continue
            x0 = torch.load(row["x0_path"], map_location=device, weights_only=False)
            # per-latent r when the windowed vector matches this sample's latent count,
            # else the scalar (a length mismatch would silently mis-align credit)
            rv = r_perframe_by_idx.get(idx)
            if rv is not None and len(rv) != x0.shape[2]:
                print(f"[loop] rank{rank} windowed r len {len(rv)} != latents {x0.shape[2]}; "
                      f"using scalar r for rollout {idx}", flush=True)
                rv = None
            # NFT_REWARD_MIX<1: blend the per-voxel grid with this scene's GLOBAL scalar
            # instead of letting the grid supersede it outright. Gated on vox_alpha so the
            # windowed path (whose rv is a flat [T] of floats) keeps its own shape, and only
            # reproj phases yield a grid at all -- camera/hpsv3 steps stay on the plain scalar.
            if rv and vox_alpha > 0 and vox_mix < 1.0:
                _s = r_by_idx[idx]
                rv = [[[(1.0 - vox_mix) * _s + vox_mix * c for c in row] for row in frame]
                      for frame in rv]
            samples.append({"bundle": b, "x0": x0.float(), "r": rv if rv else r_by_idx[idx]})
        global_step, tm = train_pass(
            transformer, samples, sched, params, opt, rank=rank, world_size=world,
            global_step=global_step, beta=args.beta, beta_kl=args.beta_kl,
            grad_steps=args.grad_steps, inner_epochs=args.inner_epochs,
            max_grad_norm=args.max_grad_norm,
            lora_l2_lambda=args.lora_l2_lambda, lora_l2_target=args.lora_l2_target,
            device=device)
        # Weight-travel diagnostic: ||dW_new|| / ||dW_ref||, i.e. drift from the CausVid init
        # (`ref` IS CausVid, so no external file). Measured every step because beta_kl only
        # slows drift -- 0.0017/step at 0.5 vs 0.0030 at 1e-2, linear, no plateau in 190 steps --
        # and this number tracks visible degradation where the reward curve does not.
        lora_ratio = adapters.lora_l2_ratio(transformer)
        adapters.ema_update_old(transformer, args.ema_decay)
        if dist_on:
            dist.barrier()
        t_train = time.time()

        # LOG / CKPT / CLEANUP
        if rank == 0:
            rmean = sum(a["reward_mean"] for a in agg.values()) / max(1, len(agg))
            within = sum(a["reward_std"] for a in agg.values()) / max(1, len(agg))
            try:
                from rl.scoring.reward_diag import reward_decomposition
                from rl.scoring.nft_score_cli import COMBO as _COMBO, NORM as _NORM
                _dec = reward_decomposition(per_seed, _COMBO, _NORM)
            except Exception as e:  # noqa: BLE001 - diagnostics must never break a step
                print(f"[loop] reward_decomposition: {type(e).__name__}: {e}", flush=True)
                _dec = {}
            msg = {"step": step, f"reward/{args.combo}_mean": rmean,
                   "reward/within_scene_std": within,
                   **_dec, **_vox_extra,
                   "reward/r_window_std": r_window_std,
                   "nft/lora_l2_ratio": lora_ratio,
                   **({"reward/alt_combo_is_b": float(alt_name == args.alt_combo_b),
                       "reward/alt_combo": alt_name} if alt_name is not None else {}),
                   "train/loss": tm.get("loss", float("nan")),
                   "train/grad_norm": tm.get("grad_norm", 0.0),
                   "train/n_microsteps": tm.get("n_microsteps", 0),
                   "train/n_oom": tm.get("n_oom", 0),
                   # barrier diagnostics: train_pass computes these but nothing copied them out,
                   # so the penalty was only inferable from the ratio trajectory
                   **({"train/lora_pen": tm["lora_pen"]} if "lora_pen" in tm else {}),
                   **({"train/lora_ratio_pre": tm["lora_ratio_pre"]}
                      if "lora_ratio_pre" in tm else {}),
                   "time/sample_s": round(t_sample - t0, 1),
                   "time/score_s": round(t_score - t_sample, 1),
                   "time/train_s": round(t_train - t_score, 1)}
            if vox_alpha > 0:
                msg["reward/voxel_matched_frac"] = r_voxel_matched
            msg.update({f"train/{k.split('/', 1)[1]}": v for k, v in tm.items()
                        if k.startswith("nft/")})
            # raw reward-component means across this step's rollouts
            for mk in ("vggt_mse", "vggt_depth_mae", "hpsv3_vid", "rpe_rot", "rpe_trans"):
                vals = [p[mk] for p in per_seed if mk in p]
                if vals:
                    msg[f"raw/{mk}"] = sum(vals) / len(vals)
            print(f"[loop] {json.dumps(msg)}", flush=True)
            if wb:
                wb.log(msg, step=step)
            with open(os.path.join(args.ckpt_out, f"seed_metrics_step{step:04d}.jsonl"), "w") as f:
                for row in per_seed:
                    f.write(json.dumps(row) + "\n")
            if (step > 0 and step % args.ckpt_every == 0) or step + 1 == args.num_steps:
                path = adapters.save_new_adapter(pipe, args.ckpt_out, step)
                adapters.save_trainstate(
                    transformer, opt,
                    os.path.join(args.ckpt_out, f"nft_trainstate_step{step:04d}.pt"),
                    step, global_step)
                print(f"[loop] checkpoint -> {path}", flush=True)
        if not args.keep_rollouts:
            shutil.rmtree(step_dir, ignore_errors=True)

        # VALIDATE (after the step, so step N's numbers reflect N optimizer steps)
        # every rank enters: run_validation is collective (see note above)
        if args.val_every > 0 and (step + 1) % args.val_every == 0:
            run_validation(step + 1)
            if dist_on:
                dist.barrier()

    if rank == 0 and wb:
        wb.finish()

if __name__ == "__main__":
    main()
