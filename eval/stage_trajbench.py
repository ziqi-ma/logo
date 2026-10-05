"""Stage a group of TrajectoryBench clips and report the settings that group needs.

The generation settings depend on the clip's type, so work them out from the dataset's own
metadata rather than by hand:

    fov scale    6.0 on transition clips, 1.0 elsewhere. LingBot-World only.
    frames       the clip's own num_frames (81 or 241).

    # what groups exist
    python eval/stage_trajbench.py --list

    # stage one group and print the commands for it
    python eval/stage_trajbench.py --category indoor --difficulty hard --out runs/indoor_hard
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

REPO_ID = "ziqima/TrajectoryBench"


def load_metadata(local_dir: str | None):
    """The dataset's metadata.jsonl, from a local checkout or from the Hub."""
    if local_dir:
        path = os.path.join(local_dir, "metadata.jsonl")
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(REPO_ID, "metadata.jsonl", repo_type="dataset")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def fov_scale(clip) -> float:
    """LingBot-World's wider lens, used on transition clips only."""
    return 6.0 if clip["category"] == "transition" else 1.0


def group_name(clip) -> str:
    """A directory name LingBot-World can read the clip type back out of."""
    style = "styl" if clip["stylized"] else "photo"
    if clip["category"] == "transition":
        return f"transition_{clip['difficulty']}_{style}"
    return f"{clip['category']}_{clip['difficulty']}_{style}"


def fetch(local_dir: str | None, clip_id: str, name: str) -> str:
    if local_dir:
        return os.path.join(local_dir, "clips", clip_id, name)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(REPO_ID, f"clips/{clip_id}/{name}", repo_type="dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-dir", default=None,
                    help="a downloaded copy of the dataset; otherwise it is pulled from the Hub")
    ap.add_argument("--list", action="store_true", help="list the groups and their settings")
    ap.add_argument("--category", choices=["indoor", "outdoor", "transition"])
    ap.add_argument("--difficulty", choices=["easy", "medium", "hard"])
    ap.add_argument("--stylized", action="store_true", help="the stylized half of the group")
    ap.add_argument("--out", default=None, help="where to write inputs/<id>/...")
    ap.add_argument("--limit", type=int, default=0, help="stage only the first N clips")
    args = ap.parse_args()

    clips = load_metadata(args.local_dir)

    if args.list or not (args.category and args.difficulty):
        seen = {}
        for c in clips:
            seen.setdefault(group_name(c), []).append(c)
        print(f"{'group':26s} {'clips':>5s} {'frames':>6s} {'fov':>4s}")
        for name in sorted(seen):
            g = seen[name]
            print(f"{name:26s} {len(g):5d} {g[0]['num_frames']:6d} {fov_scale(g[0]):4.1f}")
        if not args.list:
            print("\nPick one with --category/--difficulty [--stylized] --out <dir>")
        return

    group = [c for c in clips
             if c["category"] == args.category
             and c["difficulty"] == args.difficulty
             and bool(c["stylized"]) == bool(args.stylized)]
    if not group:
        raise SystemExit("no clips match that group")
    group.sort(key=lambda c: c["id"])
    if args.limit:
        group = group[:args.limit]

    first = group[0]
    fov, frames = fov_scale(first), first["num_frames"]
    name = group_name(first)
    ids = [c["id"] for c in group]

    out = args.out
    if out:
        for c in group:
            d = os.path.join(out, "inputs", c["id"])
            os.makedirs(d, exist_ok=True)
            shutil.copyfile(fetch(args.local_dir, c["id"], "first_frame.png"),
                            os.path.join(d, f"{c['id']}.png"))
            shutil.copyfile(fetch(args.local_dir, c["id"], "trajectory.npz"),
                            os.path.join(d, "trajectory.npz"))
            with open(os.path.join(d, "captions.json"), "w") as fh:
                json.dump({"0": ""}, fh)
        # The generators read this, so the group's settings do not have to be passed by hand.
        with open(os.path.join(out, "settings.json"), "w") as fh:
            json.dump({"group": name, "num_frames": first["num_frames"],
                       "transition_fov_scale": fov_scale(first)}, fh, indent=1)
        print(f"staged {len(group)} clip(s) of {name} under {out}/inputs/")
        print(f"wrote {out}/settings.json (num_frames, transition_fov_scale)")

    print(f"\ngroup {name}: {len(group)} clip(s), num_frames {frames}, LingBot fov {fov}")
    if fov != 1.0:
        print("this is a transition group, so keep the output directory name ending in "
              f"{name!r} for LingBot-World to use the wider lens")
    idlist = ",".join(ids[:8]) + ("..." if len(ids) > 8 else "")
    print(f"ids: {idlist}")
    print(f"""
# Lyra-2
torchrun --standalone --nproc_per_node=1 -m lyra_2._src.rl.inference.gen_trajbench \\
  --ws_prefix {out or '<out>'} --ids {','.join(ids[:2])} \\
  --checkpoint_dir checkpoints/model --adapter <adapter.pt> \\
  --num_frames {frames} --base-seed 1

# LingBot-World
python -m wan.rl.inference.gen_trajbench \\
  --ws_prefix {out or '<out>'} --ids {','.join(ids[:2])} \\
  --num_frames {frames} --base_seed 1 --transition_fov_scale {fov} \\
  --lora_path <adapter.pt> --ckpt_dir weights/<base>

# UniWorld-View
python -m rl.inference.eval_gen \\
  --ws_prefix {out or '<out>'} --sids {','.join(ids[:2])} \\
  --num_frames 81 --seed 42 --rl-ckpt <adapter.safetensors>""")


if __name__ == "__main__":
    main()
