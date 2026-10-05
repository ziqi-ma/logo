"""Lyra-2 custom-traj inference with the WHOLE default-CUDA-generator philox
stream re-created in a foreign GPU's RNG layout (EMU_MPC multiProcessorCount).

Two jobs:
  1. VALUES: torch.randn / torch.randn_like on a CUDA generator are replaced by
     the Triton philox emulation (philox) for the foreign mpc.
     This is what the DMD 4-step loop's `torch.randn_like(...)` consumes.
  2. OFFSETS: every other default-generator RNG op (uniform_/normal_ weight and
     LoRA init, DA3's randperm-based quantile subsample, randint) advances the
     philox offset by an mpc-DEPENDENT amount.  Their values are discarded
     (overwritten by the checkpoint) but their offset advance is not, so the
     real call is left alone and the generator offset is rewritten to what the
     foreign mpc would have produced.  Without this the DMD draws would start
     from the wrong philox counter and the emulation would be meaningless.

Each op's kernel shape (the `unroll` in torch's
counter_offset = ((numel-1)/(block*grid*unroll)+1)*unroll) is INFERRED from the
observed offset delta on this GPU, then re-evaluated at EMU_MPC -- so a wrong
model is reported, never silently assumed.  Unattributed offset drift (a draw
we do not intercept) is detected and reported too.

EMU_MPC=0 -> pure instrumentation, no substitution, no offset rewrite.
"""
import os as _os_boot, sys as _sys_boot
_HERE = _os_boot.path.dirname(_os_boot.path.abspath(__file__))
_ROOT = _os_boot.path.dirname(_os_boot.path.dirname(_HERE))  # the repo root
if _HERE not in _sys_boot.path:
    _sys_boot.path.insert(0, _HERE)

import os, sys, runpy
import torch

REPO = os.path.join(_ROOT, "models", "lyra2", "Lyra-2")
SCRATCH = _HERE
sys.path.insert(0, SCRATCH)
sys.path.insert(0, REPO)
os.chdir(REPO)

import philox as P
import philox_int as R

MPC = int(os.environ["EMU_MPC"])
LOG = os.environ.get("EMU_LOG", "1") == "1"
VERIFY = os.environ.get("EMU_VERIFY") == "1"  # draw real AND emulated, compare, return real
LATE = os.environ.get("EMU_LATE") == "1"      # stay passive until the pipeline's torch.manual_seed
_active = [not LATE]
NOSET = os.environ.get("EMU_NOSET") == "1"   # never write the generator offset
NOSUB = os.environ.get("EMU_NOSUB") == "1"   # never substitute randn values
NOPERM = os.environ.get("EMU_NOPERM") == "1"  # never substitute randperm/randint values
# EMU_FORCESUB substitutes randperm/randint values even when the grid (and hence
# the values) cannot differ -- the mpc=108 gate: forced substitution must leave
# the clip byte-identical to the unpatched run.
FORCESUB = os.environ.get("EMU_FORCESUB") == "1"
REAL = None            # this GPU's mpc, filled on first cuda use
_n = [0]
_expect = [None]        # offset we last wrote; drift detector
_stats = {"drift": 0, "drift_events": 0, "unmatched": 0, "ops": 0,
          "perm_sub": 0, "rint_sub": 0}


# ---------------------------------------------------------------- generators
def _st(g):
    return g.get_state().view(torch.int64)


def _get(g):
    return int(_st(g)[1].item())


def _set(g, off):
    if NOSET:
        return
    s = _st(g).clone()
    s[1] = off
    g.set_state(s.view(torch.uint8))


def _gen_for(device, explicit):
    if explicit is not None:
        return (explicit, "explicit") if explicit.device.type == "cuda" else (None, None)
    if device is None or torch.device(device).type != "cuda":
        return (None, None)
    d = torch.device(device)
    i = d.index if d.index is not None else torch.cuda.current_device()
    return (torch.cuda.default_generators[i], "default")


def _real():
    global REAL
    if REAL is None:
        REAL = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    return REAL


# ------------------------------------------------------------- offset models
def _adv_plain(n, mpc, U, itemsize=4):
    """torch's distribution_nullary_kernel reserves a philox offset at EVERY
    level of its with_32bit_indexing() split recursion: the outer call reserves
    for the full numel, then recurses into two halves when the iterator's byte
    offsets exceed INT_MAX, each of which reserves again."""
    g = P.grid_for(n, mpc)
    a = ((n - 1) // (256 * g * U) + 1) * U
    if (n - 1) * itemsize <= 2 ** 31 - 1:
        return a
    h = n // 2
    return a + _adv_plain(h, mpc, U, itemsize) + _adv_plain(n - h, mpc, U, itemsize)


def _adv_perm(n, mpc, U, itemsize=8):
    """randperm on CUDA: philox_cuda_state(n) for the duplicate-key handling
    (mpc-independent, rounded up to a multiple of 4) plus the key draw, whose
    reservation is 4*ceil(n/(256*grid*2)).  Fitted exactly on 13 measured n
    from 3.9e5 to 8e6 on this A100."""
    g = P.grid_for(n, mpc)
    return 4 * ((n + 3) // 4) + 4 * ((n - 1) // (256 * g * 2) + 1)


_MODEL = {"plain": _adv_plain, "perm": _adv_perm}


def _drift(gen, before, tag):
    if _expect[0] is not None and before != _expect[0]:
        d = before - _expect[0]
        if d < 0:
            _active[0] = True
        _stats["drift"] += d
        _stats["drift_events"] += 1
        if LOG:
            print(f"[drift] +{d} before {tag} (expected off {_expect[0]}, got {before})",
                  flush=True)


def _rewrite(gen, before, n, kind, tag, extra="", itemsize=4):
    """Real call already happened.  Observe its delta, infer the kernel model,
    rewrite the offset to the EMU_MPC value."""
    _stats["ops"] += 1
    obs = _get(gen) - before
    f = _MODEL[kind]
    U = None
    for u in (4, 2, 1, 8, 16, 32):
        if f(n, _real(), u, itemsize) == obs:
            U = u
            break
    if U is None:
        _stats["unmatched"] += 1
        target = obs
        note = f"UNMATCHED obs={obs}"
    else:
        target = f(n, MPC, U, itemsize) if MPC else obs
        note = f"U={U} obs={obs} emu={target}"
    if MPC and _active[0]:
        _set(gen, before + target)
        _expect[0] = before + target
    else:
        _expect[0] = before + obs
    if LOG:
        print(f"[{'emu' if MPC else 'log'}] #{_n[0]} {tag} n={n} off={before} "
              f"{note} mpc={MPC or _real()} {extra}", flush=True)
    return target


# --------------------------------------------------------------- randn/-like
_randn = torch.randn
_randn_like = torch.randn_like


def _numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _mem_order(shape, stride):
    """torch fills a tensor in MEMORY order, and randn_like preserves the input's
    (possibly permuted) strides.  Return the memory-order shape plus the permute
    that maps it back to `shape`, or None if the layout is not dense."""
    nd = len(shape)
    order = sorted(range(nd), key=lambda i: -stride[i])
    mem_shape = [int(shape[i]) for i in order]
    exp = 1
    for i in reversed(order):
        if int(shape[i]) != 1 and int(stride[i]) != exp:
            return None
        exp *= int(shape[i])
    inv = [0] * nd
    for pos, i in enumerate(order):
        inv[i] = pos
    return mem_shape, inv


def _do_randn(shape, gen, which, device, dtype, tag, real_call, like=None):
    _n[0] += 1
    before = _get(gen)
    _drift(gen, before, tag)
    if MPC == 0 or NOSUB or not _active[0]:
        out = real_call()
        _rewrite(gen, before, _numel(shape), "plain", tag,
                 extra=f"shape={tuple(shape)} dtype={dtype} gen={which}",
                 itemsize=out.element_size())
        return out
    n = _numel(shape)
    isz = torch.empty((), dtype=dtype or torch.get_default_dtype()).element_size()
    adv = _adv_plain(n, MPC, 4, isz)
    seed = int(_st(gen)[0].item())
    ref = real_call() if VERIFY else None
    mo = None
    if like is not None and not like.is_contiguous():
        mo = _mem_order(tuple(like.shape), like.stride())
    if mo is not None:
        mem_shape, inv = mo
        out = P.cuda_randn(mem_shape, seed, MPC, offset=before, device=str(device),
                           dtype=dtype or torch.get_default_dtype()).permute(inv)
    else:
        out = P.cuda_randn(shape, seed, MPC, offset=before, device=str(device),
                           dtype=dtype or torch.get_default_dtype())
    if VERIFY:
        bad = int((ref != out).sum().item())
        print(f"[verify] #{_n[0]} {tag} n={n} off={before} seed={seed} mpc={MPC} "
              f"mismatches={bad}/{n} real_adv={_get(gen) - before} emu_adv={adv} "
              f"ref_contig={ref.is_contiguous()} ref_stride={ref.stride()} "
              f"out_stride={out.stride()} ref_dtype={ref.dtype} "
              f"memorder={mo is not None}", flush=True)
        _expect[0] = _get(gen)
        return ref
    _set(gen, before + adv)
    _expect[0] = before + adv
    _stats["ops"] += 1
    if LOG:
        print(f"[emu] #{_n[0]} {tag} n={n} off={before} U=4 emu={adv} mpc={MPC} "
              f"shape={tuple(shape)} dtype={dtype} gen={which} "
              f"grid={P.grid_for(n, MPC)}", flush=True)
    return out


def randn(*size, **kw):
    shape = tuple(size[0]) if (len(size) == 1 and not isinstance(size[0], int)) \
        else tuple(int(s) for s in size)
    gen, which = _gen_for(kw.get("device"), kw.get("generator"))
    if gen is None or kw.get("out") is not None:
        return _randn(*size, **kw)
    dev = kw.get("device") or gen.device
    return _do_randn(shape, gen, which, dev, kw.get("dtype"), "randn",
                     lambda: _randn(*size, **kw))


def randn_like(inp, **kw):
    dev = kw.get("device", inp.device)
    gen, which = _gen_for(dev, kw.get("generator"))
    if gen is None:
        return _randn_like(inp, **kw)
    return _do_randn(tuple(inp.shape), gen, which, dev,
                     kw.get("dtype", inp.dtype), "randn_like",
                     lambda: _randn_like(inp, **kw), like=inp)


torch.randn = randn
torch.randn_like = randn_like


# ------------------------------------------- offset-only ops (values kept)
def _wrap_out(name, orig, kind="plain"):
    def f(*a, **kw):
        gen, which = _gen_for(kw.get("device"), kw.get("generator"))
        if gen is None:
            return orig(*a, **kw)
        _n[0] += 1
        before = _get(gen)
        _drift(gen, before, name)
        out = orig(*a, **kw)
        n = int(a[0]) if kind == "perm" else out.numel()
        _rewrite(gen, before, n, kind, name, extra=f"gen={which}",
                 itemsize=out.element_size() if kind != "perm" else 8)
        return out
    return f


def _wrap_inplace(name, orig):
    def f(self, *a, **kw):
        if self.device.type != "cuda":
            return orig(self, *a, **kw)
        gen, which = _gen_for(self.device, kw.get("generator"))
        if gen is None:
            return orig(self, *a, **kw)
        _n[0] += 1
        before = _get(gen)
        _drift(gen, before, name)
        out = orig(self, *a, **kw)
        _rewrite(gen, before, self.numel(), "plain", name,
                 extra=f"shape={tuple(self.shape)} dtype={self.dtype} gen={which}",
                 itemsize=self.element_size())
        return out
    return f


# ------------------------------------------- randperm / randint VALUES
# Both are grid-stride philox draws, so their values are SM-count dependent
# exactly when grid.x = min(mpc*8, ceil(n/256)) differs, i.e. when n >= mpc*2048.
# randperm additionally draws int64 KEYS (unroll 2) and stable-radix-sorts their
# low perm_bits(n) bits; see philox_int.py.  Verified bit-identical to this
# GPU's own randperm/randint at mpc=108 by verify_rand.py.
_randperm = torch.randperm
_randint = torch.randint


def _mpc_dep(n):
    return FORCESUB or P.grid_for(n, MPC) != P.grid_for(n, _real())


def randperm(n, *a, **kw):
    gen, which = _gen_for(kw.get("device"), kw.get("generator"))
    if gen is None or kw.get("out") is not None:
        return _randperm(n, *a, **kw)
    n = int(n)
    _n[0] += 1
    before = _get(gen)
    _drift(gen, before, "randperm")
    sub = (MPC and _active[0] and not NOSUB and not NOPERM
           and _mpc_dep(n) and R.perm_bits(n) > 32)
    if not sub:
        out = _randperm(n, *a, **kw)
        _rewrite(gen, before, n, "perm", "randperm",
                 extra=f"gen={which} sub=0 dep={int(_mpc_dep(n))} "
                       f"bits={R.perm_bits(n)}", itemsize=8)
        return out
    seed = int(_st(gen)[0].item())
    ref = _randperm(n, *a, **kw) if VERIFY else None
    dev = kw.get("device") or gen.device
    out, nisl = R.cuda_randperm(n, seed, MPC, offset=before, device=str(dev),
                                dtype=kw.get("dtype") or torch.int64)
    adv = R.perm_adv(n, MPC)
    if VERIFY:
        print(f"[verify] #{_n[0]} randperm n={n} off={before} seed={seed} mpc={MPC} "
              f"mismatches={int((ref != out).sum().item())}/{n} "
              f"real_adv={_get(gen) - before} emu_adv={adv} islands={nisl}", flush=True)
        _expect[0] = _get(gen)
        return ref
    _set(gen, before + adv)
    _expect[0] = before + adv
    _stats["ops"] += 1
    _stats["perm_sub"] += 1
    if LOG:
        print(f"[emu] #{_n[0]} randperm n={n} off={before} emu={adv} mpc={MPC} "
              f"bits={R.perm_bits(n)} islands={nisl} grid={P.grid_for(n, MPC)} "
              f"gen={which}", flush=True)
    return out


def randint(*a, **kw):
    gen, which = _gen_for(kw.get("device"), kw.get("generator"))
    if gen is None or kw.get("out") is not None or len(a) < 2:
        return _randint(*a, **kw)
    if len(a) >= 3 and not isinstance(a[2], int):
        low, high, size = int(a[0]), int(a[1]), tuple(a[2])
    elif not isinstance(a[1], int):
        low, high, size = 0, int(a[0]), tuple(a[1])
    else:
        return _randint(*a, **kw)
    numel = 1
    for d in size:
        numel *= int(d)
    dt = kw.get("dtype") or torch.int64
    _n[0] += 1
    before = _get(gen)
    _drift(gen, before, "randint")
    sub = (MPC and _active[0] and not NOSUB and not NOPERM and _mpc_dep(numel)
           and (high - low) < 2 ** 32 and dt in (torch.int64, torch.int32, torch.int16))
    if not sub:
        out = _randint(*a, **kw)
        _rewrite(gen, before, out.numel(), "plain", "randint",
                 extra=f"gen={which} sub=0 dep={int(_mpc_dep(numel))}",
                 itemsize=out.element_size())
        return out
    seed = int(_st(gen)[0].item())
    ref = _randint(*a, **kw) if VERIFY else None
    dev = kw.get("device") or gen.device
    out = R.cuda_randint(low, high, size, seed, MPC, offset=before,
                         device=str(dev), dtype=dt)
    adv = R.randint_adv(numel, MPC)
    if VERIFY:
        print(f"[verify] #{_n[0]} randint n={numel} off={before} seed={seed} "
              f"mpc={MPC} mismatches={int((ref != out).sum().item())}/{numel} "
              f"real_adv={_get(gen) - before} emu_adv={adv}", flush=True)
        _expect[0] = _get(gen)
        return ref
    _set(gen, before + adv)
    _expect[0] = before + adv
    _stats["ops"] += 1
    _stats["rint_sub"] += 1
    if LOG:
        print(f"[emu] #{_n[0]} randint n={numel} off={before} emu={adv} mpc={MPC} "
              f"range={high - low} gen={which}", flush=True)
    return out


torch.randperm = randperm
torch.randint = randint
for _nm in ("rand", "rand_like", "randint_like", "multinomial",
            "bernoulli", "normal", "poisson"):
    if hasattr(torch, _nm):
        setattr(torch, _nm, _wrap_out(_nm, getattr(torch, _nm)))
for _nm in ("uniform_", "normal_", "random_", "bernoulli_", "exponential_",
            "cauchy_", "log_normal_", "geometric_"):
    if hasattr(torch.Tensor, _nm):
        setattr(torch.Tensor, _nm, _wrap_inplace(_nm, getattr(torch.Tensor, _nm)))


def _report():
    print(f"[emu-summary] mpc={MPC} ops={_stats['ops']} "
          f"unmatched={_stats['unmatched']} drift_events={_stats['drift_events']} "
          f"drift_total={_stats['drift']} "
          f"perm_sub={_stats['perm_sub']} rint_sub={_stats['rint_sub']}", flush=True)


import atexit
atexit.register(_report)

# Which entry point to wrap. The RNG interception is entry-agnostic, so the trajbench
# generator is the default and the single-clip inference script is available for a one-off.
_ENTRY = {
    "gen_trajbench": "lyra_2._src.rl.inference.gen_trajbench",
    "custom_traj": "lyra_2._src.inference.lyra2_custom_traj_inference",
}[os.environ.get("EMU_ENTRY", "gen_trajbench")]

sys.argv = [_ENTRY.rsplit(".", 1)[-1]] + sys.argv[1:]
runpy.run_module(_ENTRY, run_name="__main__", alter_sys=True)
