"""Frozen few-step flow-matching schedule shared by sampling and NFT training.

UniWorld-View samples with FlowMatchEulerDiscreteScheduler (shift=5.0) at 8 steps.
The scheduler's `step` is stateful, so RL code never calls it — sampling and
re-noising both read this frozen (timesteps, sigmas) table instead.

Convention (diffusers flow matching): x_t = (1-sigma)*x0 + sigma*noise,
net predicts v = noise - x0, x0 = x_t - sigma*v, timestep fed to the net is
sigma*1000 (float).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

SCHEDULER_CONFIG = {
    "num_train_timesteps": 1000,
    "shift": 5.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}

@dataclass(frozen=True)
class RLSchedule:
    timesteps: torch.Tensor  # [N] float, e.g. [1000.0, 967.9, ..., 24.4]
    sigmas: torch.Tensor     # [N+1] float, terminal 0.0 appended
    num_steps: int

    def to(self, device) -> "RLSchedule":
        return RLSchedule(self.timesteps.to(device), self.sigmas.to(device), self.num_steps)

def build_rl_schedule(num_steps: int = 8, device="cpu") -> RLSchedule:
    from diffusers import FlowMatchEulerDiscreteScheduler
    sched = FlowMatchEulerDiscreteScheduler.from_config(SCHEDULER_CONFIG)
    sched.set_timesteps(num_steps, device="cpu")
    ts = sched.timesteps.clone().float().to(device)
    sig = sched.sigmas.clone().float().to(device)
    assert sig.shape[0] == num_steps + 1 and float(sig[-1]) == 0.0
    assert torch.allclose(ts, sig[:-1] * 1000.0, atol=1e-3)
    return RLSchedule(ts, sig, num_steps)
