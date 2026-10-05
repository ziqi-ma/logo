"""gen_trajbench with every CUDA-generator randn draw re-created in a foreign
GPU's RNG layout (EMU_MPC multiProcessorCount).

The generator's philox offset is read from / written back to its own state
(a CUDA generator's state is [seed:int64, offset:int64]), so the running
offset stays exactly what CUDA would have, call after call, and any draw we
do not intercept still keeps the stream aligned.

EMU_MPC=0 -> pure instrumentation (log shapes, no substitution).
"""
import os as _os_boot, sys as _sys_boot
_HERE = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
_ROOT = _os_boot.path.dirname(_os_boot.path.dirname(_HERE))  # the repo root
if _HERE not in _sys_boot.path:
    _sys_boot.path.insert(0, _HERE)

import os, sys
import torch

REPO = os.path.join(_ROOT, "models", "lingbot-world-v2")
SCRATCH = _HERE
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)
os.chdir(REPO)

import philox as P

MPC = int(os.environ["EMU_MPC"])
LOG = os.environ.get("EMU_LOG", "1") == "1"

_randn = torch.randn
_randn_like = torch.randn_like
_n = [0]


def _state(g):
    return g.get_state().view(torch.int64)


def _get_off(g):
    return int(_state(g)[1].item())


def _set_off(g, off):
    s = _state(g).clone()
    s[1] = off
    g.set_state(s.view(torch.uint8))


def _emulate(shape, generator, device, dtype):
    numel = 1
    for d in shape:
        numel *= int(d)
    seed = generator.initial_seed()
    off = _get_off(generator)
    adv = P.advance_for(numel, MPC)
    t = P.cuda_randn(shape, seed, MPC, offset=off, device=str(device),
                     dtype=dtype or torch.get_default_dtype())
    _set_off(generator, off + adv)
    if LOG:
        print(f"[emu] #{_n[0]} randn {tuple(shape)} n={numel} dtype={dtype} "
              f"seed={seed} off={off} adv={adv} mpc={MPC} "
              f"grid={P.grid_for(numel, MPC)}", flush=True)
    return t


def randn(*size, **kw):
    if len(size) == 1 and not isinstance(size[0], int):
        shape = tuple(size[0])
    else:
        shape = tuple(int(s) for s in size)
    g = kw.get("generator")
    dev = kw.get("device")
    if g is not None and g.device.type == "cuda" and kw.get("out") is None:
        _n[0] += 1
        if MPC == 0:
            if LOG:
                numel = 1
                for d in shape:
                    numel *= int(d)
                print(f"[log] #{_n[0]} randn {shape} n={numel} "
                      f"dtype={kw.get('dtype')} device={dev} "
                      f"seed={g.initial_seed()} off={_get_off(g)}", flush=True)
            return _randn(*size, **kw)
        return _emulate(shape, g, dev if dev is not None else g.device,
                        kw.get("dtype"))
    if LOG and MPC == 0:
        print(f"[log] randn NO-CUDA-GEN {shape} gen={g}", flush=True)
    return _randn(*size, **kw)


def randn_like(inp, **kw):
    if LOG and MPC == 0:
        print(f"[log] randn_like {tuple(inp.shape)} dev={inp.device} "
              f"dtype={inp.dtype} (default generator)", flush=True)
    return _randn_like(inp, **kw)


torch.randn = randn
torch.randn_like = randn_like

from wan.rl.inference.gen_trajbench import main
sys.argv = ["gen_trajbench"] + sys.argv[1:]
main()
