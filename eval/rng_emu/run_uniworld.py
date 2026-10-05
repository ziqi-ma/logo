"""eval_gen with the initial latent noise drawn in a foreign GPU's RNG layout."""
import os as _os_boot, sys as _sys_boot
_HERE = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
_ROOT = _os_boot.path.dirname(_os_boot.path.dirname(_HERE))  # the repo root
if _HERE not in _sys_boot.path:
    _sys_boot.path.insert(0, _HERE)

import os, sys
import numpy as np, torch

REPO = os.path.join(_ROOT, "models", "uniworld-view")
SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)
os.chdir(REPO)

import philox as P
import model.pipeline_uniview as PU

MPC = int(os.environ["EMU_MPC"])
SEED = int(os.environ.get("EMU_SEED", "42"))
_orig = PU.randn_tensor

def patched(shape, generator=None, device=None, dtype=None, layout=None):
    shape = tuple(shape)
    if len(shape) == 5 and shape[1] == 16:
        t = P.cuda_randn(shape, SEED, MPC, device=str(device))
        print(f"[emu] noise {shape} seed={SEED} mpc={MPC} "
              f"grid={P.grid_for(int(np.prod(shape)), MPC)}", flush=True)
        return t.to(device=device, dtype=dtype or torch.float32)
    return _orig(shape, generator=generator, device=device, dtype=dtype, layout=layout)

PU.randn_tensor = patched

from rl.inference.eval_gen import main
sys.argv = ["eval_gen"] + sys.argv[1:]
main()
