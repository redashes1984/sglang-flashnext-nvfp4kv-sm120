"""QSA prefill-MQA starvation test v2 — production-realistic shapes.

Late-chunk reality: rows=128 (128MB logits budget ÷ 1MB/row), window≈4200
contiguous keys, keys pool=250244. grid=ceildiv(rows,bq) CTAs vs 148 SMs.

Tests:
  A) baseline tiles at prod rows=128 and 512/2048 — does wall time scale
     with CTA count (starvation) or per-row cost stay flat (saturated)?
  B) KEY-SPLIT variant: grid=(row_blocks, SPLIT), each y-CTA takes a
     contiguous slice of the block's [start_min,end_max] — non-overlapping
     logits columns, no atomics needed.
Correctness of split variant checked vs non-split on a small shape.
"""
import sys, time, statistics
import torch
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
import tilelang
import tilelang.language as T

HEADS, DIM, KEYS = 4, 128, 250_244
WIN = 4200
PC = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

def make_kernel(block_n, block_q, num_stages, threads, split=1):
    @tilelang.jit(pass_configs=PC)
    def kern(heads, head_dim, block_n=block_n, block_q=block_q,
             num_stages=num_stages, threads=threads, split=split):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),
            K: T.Tensor([keys, head_dim], T.bfloat16),
            Logits: T.Tensor([rows, keys], T.float32),
            Starts: T.Tensor([rows], T.int32),
            Ends: T.Tensor([rows], T.int32),
        ):
            with T.Kernel(T.ceildiv(rows, block_q), split, threads=threads) as (bx, by):
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                nkeys = T.alloc_var(T.int32)
                lo = T.alloc_var(T.int32)
                hi = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                nkeys = T.max(0, end_max - start_min)
                lo = start_min + nkeys * by // split
                hi = start_min + nkeys * (by + 1) // split
                lo = lo // block_n * block_n
                hi = (hi + block_n - 1) // block_n * block_n
                hi = T.min(hi, T.ceildiv(nkeys + start_min % block_n + block_n, block_n) * block_n)
                if lo < hi:
                    T.copy(Q[row_base * heads, 0], q_shared)
                    for ni in T.Pipelined(T.ceildiv(hi - lo, block_n), num_stages=num_stages):
                        T.copy(K[lo + ni * block_n, 0], k_shared)
                        T.gemm(k_shared, q_shared, scores, transpose_B=True,
                               clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                        for n, qi, head in T.Parallel(block_n, block_q, heads):
                            scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                        T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                        for qi, n in T.Parallel(block_q, block_n):
                            Logits[row_base + qi, lo + ni * block_n + n] = reduced[n, qi]
        return kernel
    return kern(heads=HEADS, head_dim=DIM)

def inputs(rows):
    q = torch.randn(rows * HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(KEYS, DIM, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(rows, KEYS, device="cuda", dtype=torch.float32)
    base = torch.randint(KEYS - WIN - rows, (1,)).item()
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) + base
    ends = starts + WIN
    return q, k, logits, starts, ends

def bench(rows, cfg, split=1, iters=5):
    try:
        kern = make_kernel(*cfg, split=split)
    except Exception as e:
        print(f"rows={rows} cfg={cfg} split={split} JIT-FAIL {str(e)[:80]}", flush=True); return
    q, k, logits, starts, ends = inputs(rows)
    bq = cfg[1]
    def call():
        for rs in range(0, rows, bq):
            re_ = min(rs + bq, rows)
            kern(q[rs*HEADS:re_*HEADS], k, logits[rs:re_], starts[rs:re_], ends[rs:re_])
    try:
        call(); torch.cuda.synchronize()
        ts=[]
        for _ in range(iters):
            t0=time.perf_counter(); call(); torch.cuda.synchronize(); ts.append(time.perf_counter()-t0)
        med=statistics.median(ts)
        ctas = (rows + bq - 1)//bq * split
        print(f"rows={rows:5d} cfg bn={cfg[0]:3d} bq={cfg[1]:3d} th={cfg[3]:3d} split={split} "
              f"CTAs={ctas:4d}: {med*1000:7.2f} ms  ({med/rows*1e6:6.1f} us/row)", flush=True)
    except Exception as e:
        print(f"rows={rows} cfg={cfg} split={split} RUN-FAIL {str(e)[:80]}", flush=True)
    finally:
        del q,k,logits,starts,ends; torch.cuda.empty_cache()

print("=== A: baseline, prod-late rows=128 (4 CTAs @bq32) vs 512 vs 2048 ===", flush=True)
for rows in (128, 512, 2048):
    bench(rows, (64, 32, 3, 512))
print("=== A2: smaller block_q => more CTAs at fixed rows=128 ===", flush=True)
for bq in (8, 16, 32):
    bench(128, (64, bq, 3, 256))
print("=== B: key-split at rows=128 ===", flush=True)
for sp in (1, 2, 4, 8, 16):
    bench(128, (64, 32, 3, 512), split=sp)
for sp in (1, 4, 8):
    bench(2048, (64, 32, 3, 512), split=sp)

print("=== C: correctness of split vs split=1 (small) ===", flush=True)
rows = 256
q, k, logits, starts, ends = inputs(rows)
ref = None
for cfg, sp in [((64,32,3,512),1), ((64,32,3,512),8)]:
    kern = make_kernel(*cfg, split=sp)
    out = torch.full_like(logits, float("nan"))
    for rs in range(0, rows, 32):
        re_ = min(rs+32, rows)
        kern(q[rs*HEADS:re_*HEADS], k, out[rs:re_], starts[rs:re_], ends[rs:re_])
    if sp == 1: ref = out.clone()
    else:
        m = ~torch.isnan(ref)
        print("  split8 vs split1 maxdiff:", (out[m]-ref[m]).abs().max().item(),
              " nan-mismatch:", int((torch.isnan(out)!=torch.isnan(ref)).sum()), flush=True)
