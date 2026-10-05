"""Philox emulation of torch's CUDA randint and randperm for a foreign
multiProcessorCount, companion to philox.py (randn).

randint(0, n, shape) with n < 2**32 and an int64 output goes through
random_from_to_kernel's 32-bit branch:
    distribution_nullary_kernel<int64_t, uint32_t, 4>(iter, gen, curand4,
        (val) -> (val % range) + base)
i.e. exactly the same grid-stride / lane layout as randn, but with the RAW
philox words instead of Box-Muller.

randperm(n) for n large enough that `bits` == 64 does
    keys  = empty(n, int64).random_(INT64_MIN, INT64_MAX)
    perm  = stable_argsort(keys)                       (cub radix_sort_pairs)
    randperm_handle_duplicate_keys(...)                (offset-only for large n)
random_(INT64_MIN, INT64_MAX) has range == 2**64-1 >= 2**32 on an int64 tensor,
so it takes the 64-bit branch:
    distribution_nullary_kernel<int64_t, uint64_t, ulonglong2::size()==2>
        curand4 -> ret.x = (w0<<32)|w1, ret.y = (w2<<32)|w3
        val -> (uint64)(val % (2**64-1)) + INT64_MIN
Since +2**63 (mod 2**64) is a flip of the sign bit, the resulting signed key
order is identical to the unsigned order of the raw 64-bit word.
"""
import torch, triton, triton.language as tl

from philox import grid_for

A_ = tl.constexpr(0xD2511F53)
B_ = tl.constexpr(0xCD9E8D57)
KA = tl.constexpr(0x9E3779B9)
KB = tl.constexpr(0xBB67AE85)


@triton.jit
def _rounds(c0, c1, c2, c3, k0, k1):
    for _ in range(10):
        a = tl.full(c0.shape, A_, tl.uint32)
        b = tl.full(c0.shape, B_, tl.uint32)
        _c0, _c2 = c0, c2
        c0 = tl.umulhi(b, _c2) ^ c1 ^ k0
        c2 = tl.umulhi(a, _c0) ^ c3 ^ k1
        c1 = b * _c2
        c3 = a * _c0
        k0 = k0 + tl.full(c0.shape, KA, tl.uint32)
        k1 = k1 + tl.full(c0.shape, KB, tl.uint32)
    return c0, c1, c2, c3


@triton.jit
def _k_u32(out_ptr, numel, T, k0v, k1v, off4, BLOCK: tl.constexpr):
    """unroll 4: element li takes philox word (li//T)%4 of round (li//T)//4."""
    pid = tl.program_id(0)
    li = pid * BLOCK + tl.arange(0, BLOCK)
    m = li < numel
    idx = (li % T).to(tl.uint32)
    s = li // T
    j = ((s // 4) + off4).to(tl.uint32)
    lane = (s % 4).to(tl.int32)
    k0 = tl.full(j.shape, k0v, tl.uint32)
    k1 = tl.full(j.shape, k1v, tl.uint32)
    c0, c1, c2, c3 = _rounds(j, tl.zeros_like(j), idx, tl.zeros_like(j), k0, k1)
    v = tl.where(lane == 0, c0, tl.where(lane == 1, c1, tl.where(lane == 2, c2, c3)))
    tl.store(out_ptr + li, v, mask=m)


@triton.jit
def _k_u64(out_ptr, numel, T, k0v, k1v, off4, BLOCK: tl.constexpr):
    """unroll 2: element li takes the (li//T)%2-th 64-bit pair of round (li//T)//2."""
    pid = tl.program_id(0)
    li = pid * BLOCK + tl.arange(0, BLOCK)
    m = li < numel
    idx = (li % T).to(tl.uint32)
    s = li // T
    j = ((s // 2) + off4).to(tl.uint32)
    lane = (s % 2).to(tl.int32)
    k0 = tl.full(j.shape, k0v, tl.uint32)
    k1 = tl.full(j.shape, k1v, tl.uint32)
    c0, c1, c2, c3 = _rounds(j, tl.zeros_like(j), idx, tl.zeros_like(j), k0, k1)
    hi = tl.where(lane == 0, c0, c2).to(tl.uint64)
    lo = tl.where(lane == 0, c1, c3).to(tl.uint64)
    v = (hi << 32) | lo
    tl.store(out_ptr + li, v, mask=m)


def _launch(kern, numel, seed, mpc, offset, device, dtype):
    assert offset % 4 == 0, offset
    T = 256 * grid_for(numel, mpc)
    out = torch.empty(numel, dtype=dtype, device=device)
    BLOCK = 1024
    kern[(triton.cdiv(numel, BLOCK),)](out, numel, T, seed & 0xFFFFFFFF,
                                       (seed >> 32) & 0xFFFFFFFF, offset // 4,
                                       BLOCK=BLOCK)
    return out


def raw_u32(numel, seed, mpc, offset=0, device="cuda:0"):
    return _launch(_k_u32, numel, seed, mpc, offset, device, torch.int32)


def raw_u64(numel, seed, mpc, offset=0, device="cuda:0"):
    return _launch(_k_u64, numel, seed, mpc, offset, device, torch.int64)


def cuda_randint(low, high, shape, seed, mpc, offset=0, device="cuda:0",
                 dtype=torch.int64):
    """torch.randint(low, high, shape) for int64/int32 output and high-low < 2**32."""
    numel = 1
    for d in shape:
        numel *= int(d)
    rng = int(high) - int(low)
    w = raw_u32(numel, seed, mpc, offset, device).to(torch.int64) & 0xFFFFFFFF
    return ((w % rng) + int(low)).to(dtype).view(*shape)


def randint_adv(numel, mpc):
    g = grid_for(numel, mpc)
    return ((numel - 1) // (256 * g * 4) + 1) * 4


import math


def perm_bits(n):
    """randperm_out_cuda: bits = min(64, ceil(log2(n - (6n^2+1)/(log(0.9)*12))))."""
    lt = math.log(0.9) * 12
    nd = float(n)
    return min(64, int(math.ceil(math.log2(nd - (6 * nd * nd + 1) / lt))))


def perm_adv(n, mpc):
    """keys draw (unroll 2, 8-byte) + randperm_handle_duplicate_keys' philox_cuda_state(n)."""
    g = grid_for(n, mpc)
    return ((n - 1) // (256 * g * 2) + 1) * 4 + 4 * ((n + 3) // 4)


def perm_keys_adv(n, mpc):
    g = grid_for(n, mpc)
    return ((n - 1) // (256 * g * 2) + 1) * 4


# ------------------------------------------------- CPU philox, for the rare
# duplicate-key island shuffle (a handful of words, not worth a kernel)
_M0, _M1 = 0xD2511F53, 0xCD9E8D57
_K0, _K1 = 0x9E3779B9, 0xBB67AE85
_U32 = 0xFFFFFFFF


def philox_words(c0, c2, k0, k1):
    c1 = c3 = 0
    for _ in range(10):
        p0 = _M0 * c0
        p1 = _M1 * c2
        n0, n1 = (p1 >> 32) ^ c1 ^ k0, p1 & _U32
        n2, n3 = (p0 >> 32) ^ c3 ^ k1, p0 & _U32
        c0, c1, c2, c3 = n0 & _U32, n1, n2 & _U32, n3
        k0 = (k0 + _K0) & _U32
        k1 = (k1 + _K1) & _U32
    return c0, c1, c2, c3


def _curand_seq(seed, tid, offset, count):
    """curand() called `count` times on curand_init(seed, tid, offset) (offset%4==0)."""
    out = []
    k0, k1 = seed & _U32, (seed >> 32) & _U32
    r = 0
    while len(out) < count:
        out += list(philox_words((offset // 4 + r) & _U32, tid & _U32, k0, k1))
        r += 1
    return out[:count]


def _handle_duplicate_keys(perm, sorted_keys, n, seed, off_after_keys):
    """Reproduce randperm_handle_duplicate_keys_kernel: a Fisher-Yates shuffle
    inside every run of equal masked keys, seeded per island-start index."""
    eq = sorted_keys[1:] == sorted_keys[:-1]
    if not bool(eq.any()):
        return perm, 0
    e = eq.cpu().numpy()
    starts = [i for i in range(len(e)) if e[i] and (i == 0 or not e[i - 1])]
    p = perm.cpu().numpy().copy()
    for tid in starts:
        m = 1
        while tid + m < n and e[tid + m - 1]:
            m += 1
        w = _curand_seq(seed, tid, off_after_keys, m - 1)
        for j, i in enumerate(range(m - 1, 0, -1)):
            r = w[j] % (i + 1)
            if i != r:
                p[tid + i], p[tid + r] = p[tid + r], p[tid + i]
    import torch as _t
    return _t.from_numpy(p).to(perm.device), len(starts)


def perm_masked_keys(n, seed, mpc, offset=0, device="cuda:0"):
    """The key array randperm's radix sort actually orders by: the low `bits`
    bits of the int64 key (the sign-bit twiddle is outside that range)."""
    b = perm_bits(n)
    raw = raw_u64(n, seed, mpc, offset, device)
    return (raw & ((1 << b) - 1)) if b < 64 else (raw ^ -(1 << 63))


def cuda_randperm(n, seed, mpc, offset=0, device="cuda:0", dtype=torch.int64):
    """torch.randperm(n, device=cuda) as a GPU with `mpc` SMs would produce it.
    Only valid on the bits>32 path, i.e. n such that perm_bits(n) > 32
    (n >= ~30000); that covers every mpc-DEPENDENT n (>= 221184 for mpc>=108)."""
    assert perm_bits(n) > 32, f"n={n} takes randperm's int32-key path"
    k = perm_masked_keys(n, seed, mpc, offset, device)
    perm = torch.argsort(k, stable=True)
    perm, nisl = _handle_duplicate_keys(perm, k.gather(0, perm), n, seed,
                                       offset + perm_keys_adv(n, mpc))
    return perm.to(dtype), nisl
