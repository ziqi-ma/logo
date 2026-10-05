"""philox extended with a philox OFFSET, so a sequence of draws from one
CUDA generator can be emulated for a foreign multiProcessorCount.

torch's distribution_elementwise_grid_stride_kernel does
    curand_init(seed, idx, offset, &state)
curand_init sets ctr={0,0,0,0}, skipahead_sequence(idx) -> ctr.z += idx,
skipahead(offset) -> STATE = offset&3, ctr.x += offset/4.
Each curand_normal4 call then does ctr.x += 1.  torch's per-call offset advance
is always a multiple of 4, so STATE is always 0 and the only change vs. the
offset-0 kernel is ctr.x = offset/4 + round.
"""
import torch, triton, triton.language as tl

A_ = tl.constexpr(0xD2511F53)
B_ = tl.constexpr(0xCD9E8D57)
KA = tl.constexpr(0x9E3779B9)
KB = tl.constexpr(0xBB67AE85)
INV = tl.constexpr(2.3283064e-10)
INV2PI = tl.constexpr(2.3283064e-10 * 6.2831855)
INVH = tl.constexpr(2.3283064e-10 * 0.5)
INV2PIH = tl.constexpr(2.3283064e-10 * 6.2831855 * 0.5)


@triton.jit
def _kernel(out_ptr, numel, T, k0v, k1v, off4, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    li = pid * BLOCK + tl.arange(0, BLOCK)
    m = li < numel
    idx = (li % T).to(tl.uint32)
    s = li // T
    j = (s // 4).to(tl.uint32) + off4.to(tl.uint32)
    lane = (s % 4).to(tl.int32)

    c0 = j
    c1 = tl.zeros_like(j)
    c2 = idx
    c3 = tl.zeros_like(j)
    k0 = tl.full(c0.shape, k0v, tl.uint32)
    k1 = tl.full(c0.shape, k1v, tl.uint32)
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

    x = tl.where(lane < 2, c0, c2)
    y = tl.where(lane < 2, c1, c3)
    xf = x.to(tl.float32)
    yf = y.to(tl.float32)
    u = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                  [xf, tl.full(xf.shape, INV, tl.float32),
                                   tl.full(xf.shape, INVH, tl.float32)],
                                  dtype=tl.float32, is_pure=True, pack=1)
    v = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                  [yf, tl.full(yf.shape, INV2PI, tl.float32),
                                   tl.full(yf.shape, INV2PIH, tl.float32)],
                                  dtype=tl.float32, is_pure=True, pack=1)
    neg2log = -2.0 * tl.log(u)
    sc = tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", "=f,f", [neg2log],
                                   dtype=tl.float32, is_pure=True, pack=1)
    sn = tl.inline_asm_elementwise("sin.approx.f32 $0, $1;", "=f,f", [v],
                                   dtype=tl.float32, is_pure=True, pack=1)
    cs = tl.inline_asm_elementwise("cos.approx.f32 $0, $1;", "=f,f", [v],
                                   dtype=tl.float32, is_pure=True, pack=1)
    even = (lane == 0) | (lane == 2)
    val = tl.where(even, sn * sc, cs * sc)
    tl.store(out_ptr + li, val, mask=m)


def grid_for(numel, mpc, max_threads_per_sm=2048, block=256):
    """torch's calc_execution_policy grid.x."""
    return min(mpc * (max_threads_per_sm // block), (numel + block - 1) // block)


def advance_for(numel, mpc, block=256, unroll=4):
    """torch's counter_offset = ((numel-1)/(block*grid*unroll)+1)*4."""
    g = grid_for(numel, mpc, block=block)
    return ((numel - 1) // (block * g * unroll) + 1) * unroll


def cuda_randn(shape, seed, mpc, offset=0, device="cuda:0", dtype=torch.float32):
    numel = 1
    for d in shape:
        numel *= int(d)
    assert offset % 4 == 0, offset
    T = 256 * grid_for(numel, mpc)
    out = torch.empty(numel, dtype=torch.float32, device=device)
    BLOCK = 1024
    _kernel[(triton.cdiv(numel, BLOCK),)](out, numel, T,
                                          seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF,
                                          offset // 4, BLOCK=BLOCK)
    return out.view(*shape).to(dtype)
