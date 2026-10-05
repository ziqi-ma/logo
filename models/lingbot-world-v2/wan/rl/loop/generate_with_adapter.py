"""Single-GPU generation CLI with an optional DiffusionNFT LoRA adapter.

Upstream's generation flags for the causal_fast single-GPU path (no SP/FSDP)
and adds adapter loading: --lora_path takes a ``nft_new_step*.pt`` saved by
``wan.rl.loop.adapters.save_new_adapter``. The adapter is injected on top of the
frozen base (both "new"/"old" slots exist, as in training; only "new" is
loaded and active at inference). With --lora_merge the adapter is folded into
the base weights via PEFT merge so inference pays no LoRA overhead.

    python -m wan.rl.loop.generate_with_adapter --task i2v-A14B --size 480*832 \
        --ckpt_dir lingbot-world-v2-14b-causal-fast \
        --image scene/image.jpg --action_path scene \
        --frame_num 241 --local_attn_size 18 --sink_size 6 \
        --lora_path adapters/nft_new_step0150.pt --save_file out.mp4
"""
import argparse
import logging
import os
import sys

from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video, str2bool

def apply_lora(pipe, lora_path, scope="attn+ffn+cam", rank=32, alpha=64,
               merge=False):
    """Inject the RL adapter pair into pipe.model, load ``lora_path`` into the
    "new" slot, and activate it. rank/alpha/scope must match how the adapter
    was trained. With merge=True the "new" adapter is merged into the base
    weights (the layer then runs the plain base forward)."""
    from wan.rl.loop import adapters

    adapters.inject_rl_adapters(pipe.model, scope=scope, rank=rank, alpha=alpha)
    n_loaded = adapters.load_new_adapter(pipe.model, lora_path)
    adapters.activate_adapter(pipe.model, adapters.TRAINABLE_ADAPTER)
    if merge:
        for module in adapters.iter_tuner_layers(pipe.model):
            module.merge(adapter_names=[adapters.TRAINABLE_ADAPTER])
    pipe.model.requires_grad_(False)
    logging.info("apply_lora: loaded %d params from %s (scope=%s r=%d a=%d merge=%s)",
                 n_loaded, lora_path, scope, rank, alpha, merge)

def build_pipe(args, cfg):
    """Load a single-GPU WanI2VCausal and (optionally) its LoRA adapter."""
    pipe = wan.WanI2VCausal(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
        local_attn_size=args.local_attn_size,
        sink_size=args.sink_size,
        infer_mode=args.infer_mode,
    )
    if args.lora_path:
        apply_lora(pipe, args.lora_path, scope=args.lora_scope,
                   rank=args.lora_rank, alpha=args.lora_alpha,
                   merge=args.lora_merge)
    return pipe

def add_pipe_args(parser):
    """Flags shared by every caller that builds the generation pipe."""
    parser.add_argument("--task", type=str, default="i2v-A14B",
                        choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--infer_mode", type=str, default="causal_fast",
                        choices=["causal_fast", "causal_pretrain"])
    parser.add_argument("--size", type=str, default="480*832",
                        choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--ckpt_dir", type=str,
                        default=os.environ.get("LINGBOT_CKPT_DIR",
                                               "lingbot-world-v2-14b-causal-fast"),
                        help="checkpoint directory (env LINGBOT_CKPT_DIR overrides)")
    parser.add_argument("--chunk_size", type=int, default=4)
    parser.add_argument("--sample_shift", type=float, default=None,
                        help="flow-matching shift (default: config value)")
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False,
                        help="place T5 on CPU (fits DiT + 241-frame KV on one GPU)")
    # The generators default this on, and it is not free: the encode costs minutes on a
    # busy CPU, and CPU and GPU embeddings differ slightly, which changes the sample. Keep
    # it on to match clips generated that way; turn it off when the card has room for umT5.
    parser.add_argument("--no-t5-cpu", dest="t5_cpu", action="store_false",
                        help="run T5 on the GPU instead (needs ~11 GB beyond the DiT)")
    parser.add_argument("--local_attn_size", type=int, default=-1,
                        help="KV-cache window in latent frames (-1 = full)")
    parser.add_argument("--sink_size", type=int, default=0)
    parser.add_argument("--max_attention_size", type=int, default=None)
    parser.add_argument("--lora_path", type=str, default=None,
                        help="nft_new_step*.pt adapter from wan.rl.loop.adapters.save_new_adapter")
    parser.add_argument("--lora_scope", type=str, default="attn+ffn+cam",
                        help="LoRA target scope; must match training")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_merge", type=str2bool, default=False,
                        help="merge the adapter into base weights (faster inference)")

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU lingbot generation with an optional RL LoRA adapter")
    add_pipe_args(parser)
    parser.add_argument("--frame_num", type=int, default=None,
                        help="frames to generate (4n+1; default: config value)")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--action_path", type=str, required=True,
                        help="dir with poses.npy + intrinsics.npy (lingbot camera format)")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--offload_model", type=str2bool, default=True)
    parser.add_argument("--save_file", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="output")
    args = parser.parse_args()

    cfg = WAN_CONFIGS[args.task]
    assert args.size in SUPPORTED_SIZES[args.task], \
        f"Unsupported size {args.size} for task {args.task}"
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.prompt is None:
        args.prompt = "A camera walkthrough of a realistic 3D scene"
    return args

def main():
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(stream=sys.stdout)])
    args = _parse_args()
    cfg = WAN_CONFIGS[args.task]
    logging.info("Generation args: %s", args)

    pipe = build_pipe(args, cfg)
    img = Image.open(args.image).convert("RGB")
    video = pipe.generate(
        args.prompt,
        img,
        action_path=args.action_path,
        chunk_size=args.chunk_size,
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        seed=args.base_seed,
        offload_model=args.offload_model,
        max_attention_size=args.max_attention_size)

    if args.save_file is None:
        os.makedirs(args.save_dir, exist_ok=True)
        args.save_file = os.path.join(
            args.save_dir, f"lingbot_lora_seed{args.base_seed}.mp4")
    logging.info("Saving generated video to %s", args.save_file)
    save_video(tensor=video[None], save_file=args.save_file, fps=cfg.sample_fps,
               nrow=1, normalize=True, value_range=(-1, 1))
    logging.info("Finished.")

if __name__ == "__main__":
    main()
