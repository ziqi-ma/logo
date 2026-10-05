"""One definition of "the torch-distributed environment", and how to keep it out of scorers.

torchrun exports MASTER_ADDR / MASTER_PORT / RANK / WORLD_SIZE / ... into the environment, and
every subprocess inherits them. The hpsv3 stack reacts to those vars by standing up its own c10d
store on MASTER_PORT -- which on global rank 0 is already bound by the training job's own
rendezvous, so only rank 0 dies:

    RuntimeError: The server socket has failed to listen on any local network address.
                  useIpv6: 0, code: -98, name: EADDRINUSE, message: address already in use

Without it, rank 0 loses all of its rollouts on
every hpsv3 phase (kept_frac 0.969) and all 4 of its val clips (hpsv3 paired tests ran at n=21,
not 25). On the DiffusionNFT side the same failure was worse -- train_pass's ReduceOp.MIN turned
one rank's loss into a skipped step on all ranks, which is what the board's "(hpsv3 dead)" rows are.

The fix used to live in the *callers* (rl/grpo/loop.py, rl/grpo/valid.py each had their own copy of
a `_scorer_env`), which left `rl/grpo/tune_noise.py` spawning the scorer with a raw `os.environ`
and left every DiffusionNFT caller unprotected. A scorer needs none of these vars, so the durable
place to drop them is inside the scorer processes -- then no caller can reintroduce the bug.
"""
from __future__ import annotations

import os

# torchrun / torchelastic exports. A scorer subprocess needs none of them.
DIST_ENV_VARS = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK",
                 "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE",
                 "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT",
                 "TORCHELASTIC_MAX_RESTARTS", "TORCHELASTIC_ERROR_FILE")

def strip_dist_env() -> list:
    """Drop the distributed vars from this process's environ. Call at the top of a scorer main()
    so anything it imports (and anything it spawns) can never see them. Returns what was removed,
    so a caller can log it. Safe to call twice."""
    removed = []
    for k in DIST_ENV_VARS:
        if os.environ.pop(k, None) is not None:
            removed.append(k)
    return removed

def scorer_env(local_rank=None, base=None) -> dict:
    """`base` (default os.environ) minus the distributed vars, optionally pinned to one GPU.
    For callers that build an explicit env= for subprocess.run."""
    env = {k: v for k, v in (base if base is not None else os.environ).items()
           if k not in DIST_ENV_VARS}
    if local_rank is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return env
