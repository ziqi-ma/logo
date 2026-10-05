# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-policy rollout collection.

Two samplers are provided:

* :class:`IndependentRolloutSampler` (DEFAULT) -- K independent full-sequence
  rollouts per scene, each a fresh AR pass with a distinct seed. Fully on-policy
  and embarrassingly parallel. The K rollouts share scene-level conditioning, so
  per-chunk samples are grouped by ``(scene, chunk)`` for advantage normalization.
  This is the natural fit for DiffusionNFT, whose reward-derived ``r in [0,1]`` is
  a soft reconstruction weight (not a policy-gradient coefficient), so it tolerates
  the divergent-history comparison without the variance blow-up GRPO/PPO would see.

* :class:`BranchingRolloutSampler` (OPTIONAL) -- a branching tree where the K
  samples in a group share a byte-identical prefix (snapshot S_{c-1}), giving the
  cleanest per-chunk credit. Heavier (snapshot/revert state surgery). Kept as the
  shared-prefix comparison for the reward-attribution ablation.

Both persist all K samples as training data (winners and losers; the losers are the
negative-advantage data NFT consumes) and need no rewards inline: sampling is fully
decoupled from scoring. See the on-disk schema in
:class:`RolloutStoreWriter`.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, List, Optional, Sequence, Tuple

import torch

def _stable_hash(s: str) -> int:
    """Process-independent hash (Python's builtin hash() is salted per-process via
    PYTHONHASHSEED, which made sampling seeds differ across otherwise-identical jobs)."""
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16)

def save_rollout_videos(rollout_root: str, scene: str, videos_uri: str, fps: int = 16) -> int:
    """Encode each rollout's full-sequence frames (the highest chunk's accumulated clip)
    to mp4 and upload to ``videos_uri/scene_<scene>/rollout_NN.mp4``.

    Uses **ffmpeg (libx264/yuv420p)** rather than cv2.VideoWriter: cv2's mp4v writer
    silently fails to open for many clips (-> no file -> FileNotFoundError on upload,
    swallowed -> missing videos) AND produces mp4v that most players can't decode.
    ffmpeg encodes reliably and yields directly-playable H.264. Best-effort per rollout.
    Returns the number of videos uploaded."""
    import glob
    import subprocess

    from lyra_2._src.rl.data.gcs_util import upload_file

    n = 0
    for roll in sorted(glob.glob(os.path.join(rollout_root, f"scene_{scene}", "rollout_*"))):
        chunks = sorted(glob.glob(os.path.join(roll, "chunk_*")))
        frames_dir = os.path.join(chunks[-1], "frames") if chunks else None
        pngs = sorted(glob.glob(os.path.join(frames_dir, "*.png"))) if frames_dir else []
        if not pngs:
            print(f"[save_rollout_videos] {roll}: no frames; skipping", flush=True)
            continue
        try:
            mp4 = roll + ".mp4"
            # -pattern_type glob is robust to non-contiguous %05d numbering.
            rc = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                 "-pattern_type", "glob", "-i", os.path.join(frames_dir, "*.png"),
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", mp4],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            )
            if rc.returncode != 0 or not os.path.exists(mp4):
                print(f"[save_rollout_videos] {roll}: ffmpeg rc={rc.returncode} "
                      f"({rc.stderr.decode('utf-8','replace')[-200:]})", flush=True)
                continue
            upload_file(mp4, f"{videos_uri.rstrip('/')}/scene_{scene}/{os.path.basename(roll)}.mp4")
            n += 1
        except Exception as e:  # noqa: BLE001 -- video saving must never abort the run
            print(f"[save_rollout_videos] {roll} failed: {type(e).__name__}: {e}", flush=True)
    return n

def _accum_clip(pipe, strip_prefix: bool = True):
    """The generated video exactly as normal lyra2 inference saves it
    (``Lyra2InferencePipeline.build_outputs``): ``history_frames`` with the
    ``repeat_pixels`` init-history prefix stripped via ``start_index``. Without this
    slice the saved/scored clip carries ~``start_index`` static copies of the init
    frame, which both look like a frozen camera and contaminate the reward.

    ``strip_prefix=False`` returns the full buffer (the pre-fix, untrimmed clip) --
    used to reproduce exactly what an old reward run scored."""
    hf = getattr(pipe, "history_frames", None)
    if hf is None:
        return None
    if not strip_prefix:
        return hf
    return hf[:, :, int(getattr(pipe, "start_index", 0)):]

# --------------------------------------------------------------------------- #
# Rollout store writer
# --------------------------------------------------------------------------- #
class RolloutStoreWriter:
    """Writes the on-disk rollout store for one NFT epoch.

    Layout::

        root/
          scene_{sc}/chunk_{c}/
            cond.pt                  # group-shared conditioning (written once)
            branch_{k}/
              x0_gen.pt              # generated clean latent  [C, T_new, H, W]
              frames/ or accum.mp4   # accumulated decoded clip [0..chunk_end]
              meta.json
          manifest.jsonl             # one line per branch sample
    """

    # Keys copied verbatim from the capture payload into the group cond file.
    COND_KEYS = (
        "history_window",
        "cond_latent",
        "cond_latent_mask",
        "cond_latent_buffer",
        "pos_text",
        "neg_text",
        "last_hist_frame",
        "fps",
        "padding_mask",
    )

    def __init__(self, root: str, frame_saver: Optional[Callable] = None, append: bool = False):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.jsonl")
        self._frame_saver = frame_saver or _default_frame_saver
        # Truncate the manifest unless appending (multi-scene runs append per scene).
        if not append:
            open(self.manifest_path, "w").close()

    def group_dir(self, scene: str, chunk: int) -> str:
        return os.path.join(self.root, f"scene_{scene}", f"chunk_{chunk:03d}")

    def write_sample(
        self,
        scene: str,
        rollout: int,
        chunk: int,
        payload: dict,
        accum_pixels: Optional[torch.Tensor],
        group_id: str,
        committed: bool = False,
    ) -> dict:
        """Write one rollout sample (independent-rollout mode).

        Unlike the branching mode, conditioning differs per rollout, so cond.pt is
        stored per sample. ``group_id`` (default ``scene/chunkNNN``) is the
        advantage-normalization group key shared across the K rollouts.
        """
        sdir = os.path.join(self.root, f"scene_{scene}", f"rollout_{rollout:02d}", f"chunk_{chunk:03d}")
        os.makedirs(sdir, exist_ok=True)

        x0 = payload["gen_chunk"]
        if x0.dim() == 5:
            x0 = x0[0]
        x0_path = os.path.join(sdir, "x0_gen.pt")
        torch.save(x0, x0_path)

        cond_path = os.path.join(sdir, "cond.pt")
        torch.save({k: payload.get(k) for k in self.COND_KEYS}, cond_path)

        clip_path = self._frame_saver(accum_pixels, sdir) if accum_pixels is not None else None

        meta = {
            "scene": scene,
            "rollout": int(rollout),
            "chunk": int(chunk),
            "seed": int(payload.get("seed", -1)),
            "ar_idx": int(payload.get("ar_idx", chunk)),
            "group_id": group_id,
            "committed": bool(committed),
            "x0_path": x0_path,
            "cond_path": cond_path,
            "clip_path": clip_path,
        }
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return meta

    def write_group_cond(self, scene: str, chunk: int, payload: dict) -> str:
        gdir = self.group_dir(scene, chunk)
        os.makedirs(gdir, exist_ok=True)
        cond = {k: payload.get(k) for k in self.COND_KEYS}
        cond_path = os.path.join(gdir, "cond.pt")
        torch.save(cond, cond_path)
        return cond_path

    def write_branch(
        self,
        scene: str,
        chunk: int,
        branch: int,
        payload: dict,
        accum_pixels: Optional[torch.Tensor],
        committed: bool,
    ) -> dict:
        bdir = os.path.join(self.group_dir(scene, chunk), f"branch_{branch:02d}")
        os.makedirs(bdir, exist_ok=True)

        x0 = payload["gen_chunk"]
        if x0.dim() == 5:  # [B,C,T,H,W] -> drop singleton batch
            x0 = x0[0]
        torch.save(x0, os.path.join(bdir, "x0_gen.pt"))

        clip_path = None
        if accum_pixels is not None:
            clip_path = self._frame_saver(accum_pixels, bdir)

        group_id = f"{scene}/chunk{chunk:03d}"
        meta = {
            "scene": scene,
            "chunk": int(chunk),
            "branch": int(branch),
            "seed": int(payload.get("seed", -1)),
            "ar_idx": int(payload.get("ar_idx", chunk)),
            "group_id": group_id,
            "committed": bool(committed),
            "x0_path": os.path.join(bdir, "x0_gen.pt"),
            "cond_path": os.path.join(self.group_dir(scene, chunk), "cond.pt"),
            "clip_path": clip_path,
        }
        with open(os.path.join(bdir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(self.manifest_path, "a") as f:
            f.write(json.dumps(meta) + "\n")
        return meta

def _default_frame_saver(pixels: torch.Tensor, out_dir: str) -> str:
    """Save [B,3,T,H,W] or [3,T,H,W] in [-1,1] as a frames/ dir of PNGs.

    A directory of PNGs is accepted by every run_*_for_lyra.py reward script, and
    avoids a hard video-codec dependency. Returns the frames directory path.
    """
    from PIL import Image

    if pixels.dim() == 5:
        pixels = pixels[0]
    pixels = pixels.detach().float().clamp(-1, 1)
    pixels = ((pixels + 1.0) * 127.5).round().to(torch.uint8)  # [3,T,H,W]
    frames_dir = os.path.join(out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    T = pixels.shape[1]
    for t in range(T):
        arr = pixels[:, t].permute(1, 2, 0).cpu().numpy()  # [H,W,3]
        Image.fromarray(arr).save(os.path.join(frames_dir, f"{t:05d}.png"))
    return frames_dir

# --------------------------------------------------------------------------- #
# Independent rollout sampler (DEFAULT for DiffusionNFT)
# --------------------------------------------------------------------------- #
class IndependentRolloutSampler:
    """K independent full-sequence rollouts per scene (the default for NFT).

    Each rollout is a fresh AR pass with a distinct seed, fully on-policy: a
    chunk's conditioning is the history that rollout actually produced. The K
    rollouts of a scene share scene-level conditioning (init frame + camera path +
    text), so per-chunk samples are grouped by ``(scene, chunk)`` for advantage
    normalization across rollouts.

    No snapshot/revert and no inter-stage reward dependency -- the K rollouts are
    embarrassingly parallel (one per seed / GPU / the scheduler job).
    """

    def __init__(
        self,
        pipeline_factory: Callable[[int], object],
        writer: RolloutStoreWriter,
        k_rollouts: int = 4,
        base_seed: int = 0,
        save_accum_clip: bool = True,
        strip_init_prefix: bool = True,
    ):
        assert k_rollouts >= 1
        self.pipeline_factory = pipeline_factory
        self.writer = writer
        self.K = int(k_rollouts)
        self.base_seed = int(base_seed)
        self.save_accum_clip = save_accum_clip
        self.strip_init_prefix = strip_init_prefix

    def _seed(self, scene: str, k: int) -> int:
        return self.base_seed + (_stable_hash(scene) % 1_000_000) * self.K + k

    def run_scene(
        self,
        scene: str,
        cam_chunks: Sequence[torch.Tensor],
        intr_chunks: Sequence[torch.Tensor],
        t5_text_embeddings=None,
        neg_t5_text_embeddings=None,
    ) -> List[dict]:
        num_chunks = len(cam_chunks)
        assert len(intr_chunks) == num_chunks
        all_metas: List[dict] = []

        for k in range(self.K):
            pipe = self.pipeline_factory(self._seed(scene, k))
            for c in range(num_chunks):
                captured: dict = {}
                pipe._rollout_sink = captured.update
                try:
                  with torch.no_grad():  # sampling is inference; no autograd graph
                    pipe.autoregressive_step(
                        cam_w2c_chunk=cam_chunks[c],
                        intrinsics_chunk=intr_chunks[c],
                        t5_text_embeddings=t5_text_embeddings,
                        neg_t5_text_embeddings=neg_t5_text_embeddings,
                        is_last_step=(c == num_chunks - 1),
                    )
                finally:
                    pipe._rollout_sink = None
                assert "gen_chunk" in captured, "rollout sink produced no payload"
                accum = _accum_clip(pipe, self.strip_init_prefix) if self.save_accum_clip else None
                if os.environ.get("NFT_DEBUG_FRAMES"):
                    _hf = getattr(pipe, "history_frames", None)
                    print(f"[frames-dbg] scene={scene} roll={k} chunk={c} "
                          f"save_accum={self.save_accum_clip} "
                          f"hf={'None' if _hf is None else tuple(_hf.shape)} "
                          f"start_index={getattr(pipe, 'start_index', None)} "
                          f"accum={'None' if accum is None else tuple(accum.shape)}", flush=True)
                meta = self.writer.write_sample(
                    scene, rollout=k, chunk=c, payload=captured, accum_pixels=accum,
                    group_id=f"{scene}/chunk{c:03d}",
                )
                if os.environ.get("NFT_DEBUG_FRAMES"):
                    import glob as _g
                    cp = meta.get("clip_path") if isinstance(meta, dict) else None
                    npng = len(_g.glob(os.path.join(cp, "*.png"))) if cp else -1
                    print(f"[frames-dbg]   -> clip_path={cp} n_pngs={npng}", flush=True)
                all_metas.append(meta)
        return all_metas

# --------------------------------------------------------------------------- #
# Branching rollout sampler (OPTIONAL: shared-prefix mode, cleaner per-chunk credit)
# --------------------------------------------------------------------------- #
class BranchingRolloutSampler:
    """Branching tree rollout over a stateful Lyra2InferencePipeline."""

    def __init__(
        self,
        pipeline,
        writer: RolloutStoreWriter,
        k_branches: int = 4,
        base_seed: int = 0,
        commit_rng: Optional[torch.Generator] = None,
        save_accum_clip: bool = True,
        strip_init_prefix: bool = True,
    ):
        assert k_branches >= 1
        self.pipeline = pipeline
        self.writer = writer
        self.K = int(k_branches)
        self.base_seed = int(base_seed)
        self._rng = commit_rng or torch.Generator().manual_seed(base_seed + 1234567)
        self.save_accum_clip = save_accum_clip
        self.strip_init_prefix = strip_init_prefix

    def _branch_seed(self, scene: str, chunk: int, k: int) -> int:
        # Deterministic, distinct per (scene, chunk, branch).
        return self.base_seed + (_stable_hash(f"{scene}|{chunk}") % 1_000_000) * self.K + k

    def _commit_index(self) -> int:
        return int(torch.randint(0, self.K, (1,), generator=self._rng).item())

    def run_scene(
        self,
        scene: str,
        cam_chunks: Sequence[torch.Tensor],
        intr_chunks: Sequence[torch.Tensor],
        t5_text_embeddings=None,
        neg_t5_text_embeddings=None,
    ) -> List[dict]:
        """Roll out one scene as a branching tree; return the branch metas written."""
        num_chunks = len(cam_chunks)
        assert len(intr_chunks) == num_chunks
        all_metas: List[dict] = []

        for c in range(num_chunks):
            j = self._commit_index()  # uniform-random commit, decided up front
            order = [k for k in range(self.K) if k != j] + [j]  # committed branch last

            self.pipeline.save_snapshot()
            group_cond_written = False
            for pos, k in enumerate(order):
                self.pipeline.args.seed = self._branch_seed(scene, c, k)
                captured: dict = {}
                self.pipeline._rollout_sink = captured.update
                try:
                  with torch.no_grad():  # sampling is inference; no autograd graph
                    self.pipeline.autoregressive_step(
                        cam_w2c_chunk=cam_chunks[c],
                        intrinsics_chunk=intr_chunks[c],
                        t5_text_embeddings=t5_text_embeddings,
                        neg_t5_text_embeddings=neg_t5_text_embeddings,
                        is_last_step=(c == num_chunks - 1),
                    )
                finally:
                    self.pipeline._rollout_sink = None
                assert "gen_chunk" in captured, "rollout sink produced no payload"

                if not group_cond_written:
                    self.writer.write_group_cond(scene, c, captured)
                    group_cond_written = True

                accum = _accum_clip(self.pipeline, self.strip_init_prefix) if self.save_accum_clip else None
                meta = self.writer.write_branch(
                    scene, c, k, captured, accum, committed=(k == j)
                )
                all_metas.append(meta)

                is_committed_last = pos == len(order) - 1
                if not is_committed_last:
                    # Return to the shared base and re-arm the (single-level) snapshot.
                    self.pipeline.revert_to_snapshot()
                    self.pipeline.save_snapshot()
                # else: leave state at S_c (committed branch advances the tree)

        return all_metas

# --------------------------------------------------------------------------- #
# Scene-driven entry glue (reuses the run_lyra2_sample construction)
# --------------------------------------------------------------------------- #
def slice_trajectory_chunks(
    camera_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    num_new_video_frames: int,
    num_chunks: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Slice a full trajectory into per-chunk camera/intrinsics windows.

    Matches the per-ar_idx slicing in ``run_lyra2_sample``:
    ``[:, 1 + c*F_new : 1 + (c+1)*F_new]``. Returns (cam_chunks, intr_chunks).
    """
    cam_chunks, intr_chunks = [], []
    for c in range(num_chunks):
        start = 1 + c * num_new_video_frames
        end = start + num_new_video_frames
        cam_chunks.append(camera_w2c[:, start:end])
        intr_chunks.append(intrinsics[:, start:end])
    return cam_chunks, intr_chunks

def run_nft_sampling(
    model,
    data_batch: dict,
    args,
    writer: RolloutStoreWriter,
    scene: str,
    k_rollouts: int = 4,
    base_seed: int = 0,
    da3_model=None,
    process_group=None,
    strip_init_prefix: bool = True,
) -> List[dict]:
    """Sample one scene: K independent rollouts via the AR pipeline.

    Builds a ``pipeline_factory(seed)`` that constructs a fresh
    :class:`Lyra2InferencePipeline` from the prepared ``data_batch`` (mirroring
    ``run_lyra2_sample``), slices the trajectory into per-chunk windows, and runs
    :class:`IndependentRolloutSampler`. Requires the model + scene tensors (GPU).
    """
    from lyra_2._src.inference.lyra2_ar_inference import Lyra2InferencePipeline

    model._normalize_video_databatch_inplace(data_batch)
    init_frame = data_batch["video"][:, :, :1]
    first_depth = data_batch["depth"][:, 0]
    first_cam_w2c = data_batch["camera_w2c"][:, 0]
    first_intrinsics = data_batch["intrinsics"][:, 0]
    pos_text = data_batch.get("t5_text_embeddings", None)
    neg_text = data_batch.get("neg_t5_text_embeddings", None)

    F_new = int(model.framepack_num_new_video_frames)
    num_frames = int(args.num_frames)
    tokens_per_step = int(model.framepack_num_new_latent_frames)
    frames_per_latent = int(model.framepack_num_frames_per_latent)
    tokens_needed = (num_frames - 1 + frames_per_latent - 1) // frames_per_latent
    num_chunks = (tokens_needed + tokens_per_step - 1) // tokens_per_step
    cam_chunks, intr_chunks = slice_trajectory_chunks(
        data_batch["camera_w2c"], data_batch["intrinsics"], F_new, num_chunks
    )

    def pipeline_factory(seed: int):
        args.seed = int(seed)
        return Lyra2InferencePipeline(
            model=model,
            args=args,
            first_frame=init_frame,
            first_depth=first_depth,
            first_cam_w2c=first_cam_w2c,
            first_intrinsics=first_intrinsics,
            da3_model=da3_model,
            cp_group=process_group,
            base_t5_text_embeddings=pos_text,
            base_neg_t5_text_embeddings=neg_text,
            padding_mask=data_batch.get("padding_mask", None),
            fps=data_batch.get("fps", None),
        )

    sampler = IndependentRolloutSampler(
        pipeline_factory, writer, k_rollouts=k_rollouts, base_seed=base_seed,
        strip_init_prefix=strip_init_prefix,
    )
    return sampler.run_scene(scene, cam_chunks, intr_chunks, pos_text, neg_text)
