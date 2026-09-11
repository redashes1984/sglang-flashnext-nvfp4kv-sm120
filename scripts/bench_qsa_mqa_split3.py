"""v3: production-realistic indexer starvation test.

REAL late-chunk geometry (from mqa.py get_prefill_mqa_inputs):
  row_chunk = 128 rows (128MB logits budget / (250K keys x 4B))
  per-row window = causal compressed prefix ~ full 244K keys at late depth
  bq=32 -> grid 4 CTAs -> starvation candidate.
Sweep: key-split y-dim (2..64), bn, threads. Correctness re-checked.
"""
import sys, time, statistics
import torch
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
import tilelang
import tilelang.language as T

HEADS, DIM, KEYS = 4, 128, 250_244
WIN = 244_000          # real late-chunk causal window
ROWS = 128             # real production row_chunk
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
                hi = T.min(hi, keys)
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

def inputs(rows, win):
    q = torch.randn(rows * HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(KEYS, DIM, device="cuda", dtype=torch.bfloat16)
    logits = torch.full((rows, KEYS), float("nan"), device="cuda", dtype=torch.float32)
    base = KEYS - win - rows
    starts = torch.arange(rows, device="cuda", dtype=torch.int32) + base
    ends = starts + win
    return q, k, logits, starts, ends

def call(kern, bq, q, k, logits, starts, ends, rows):
    for rs in range(0, rows, bq):
        re_ = min(rs + bq, rows)
        kern(q[rs*HEADS:re_*HEADS], k, logits[rs:re_], starts[rs:re_], ends[rs:re_])

def bench(cfg, split=1, iters=5, verify=False):
    bq = cfg[1]
    try:
        kern = make_kernel(*cfg, split=split)
    except Exception as e:
        print(f"cfg={cfg} split={split} JIT-FAIL {str(e)[:70]}", flush=True); return
    q, k, logits, starts, ends = inputs(ROWS, WIN)
    try:
        call(kern, bq, q, k, logits, starts, ends, ROWS)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            call(kern, bq, q, k, logits, starts, ends, ROWS)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        med = statistics.median(ts)
        ctas = (ROWS + bq - 1) // bq * split
        fl = ROWS * WIN * HEADS * DIM * 2
        print(f"cfg bn={cfg[0]:3d} bq={bq:3d} st={cfg[2]} th={cfg[3]:3d} split={split:2d} CTAs={ctas:4d}: "
              f"{med*1000:7.2f} ms  {fl/med/1e12:5.1f} TF/s", flush=True)
        if verify:
            return logits.clone()
    except Exception as e:
        print(f"cfg={cfg} split={split} RUN-FAIL {str(e)[:70]}", flush=True)
    finally:
        if not verify:
            del q, k, logits, starts, ends
            torch.cuda.empty_cache()

print("=== production late-chunk iteration: rows=128 win=244K ===", flush=True)
base = bench((64, 32, 3, 512), 1, verify=True)
for cfg, sp in [
    ((64, 32, 3, 512), 2),
    ((64, 32, 3, 512), 4),
    ((64, 32, 3, 512), 8),
    ((64, 32, 3, 512), 16),
    ((64, 32, 3, 512), 32),
    ((64, 32, 3, 512), 64),
    ((64, 16, 3, 512), 16),   # finer rows x split
    ((128, 32, 2, 512), 8),
    ((64, 32, 4, 512), 16),
    ((64, 32, 3, 256), 32),
]:
    r = bench(cfg, sp, verify=True)
    if r is not None and base is not None:
        m = ~torch.isnan(r)
        d = (r[m] - base[m]).abs().max().item() if int(m.sum()) else float("nan")
        cov = int(m.sum()) / r.numel()
        print(f"    ^ covered={cov:.2%} maxdiff={d:.3e}", flush=True)
        del r

