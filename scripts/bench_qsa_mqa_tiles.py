"""Sweep tilelang QSA prefill MQA tile configs at the real cold-prefill shape.

Shape per iter (from E3-grade probe of live service): rows=512 (chunked by
128MB logits budget), keys=250244 compressed blocks, heads=4, dim=128,
topk-window per row ~4096 compressed blocks.

CUDA-event timing (no per-iter host sync) over 3 full row-loops.
"""
import sys, math, statistics, time
import torch
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
import tilelang
import tilelang.language as T

HEADS, DIM = 4, 128
ROWS, KEYS = 512, 250_244
TOPK_WINDOW = 4096  # average (end-start)

def make_kernel(block_n, block_q, num_stages, threads):
    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        }
    )
    def kern(heads: int, head_dim: int, block_n: int = block_n,
             block_q: int = block_q, num_stages: int = num_stages, threads: int = threads):
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
            with T.Kernel(T.ceildiv(rows, block_q), threads=threads) as bx:
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                T.copy(Q[row_base * heads, 0], q_shared)
                for ni in T.Pipelined(T.ceildiv(end_max - start_min, block_n), num_stages=num_stages):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(k_shared, q_shared, scores, transpose_B=True,
                           clear_accum=True, policy=T.GemmWarpPolicy.FullCol)
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[n, qi]
        return kernel
    return kern(heads=HEADS, head_dim=DIM)

def bench(cfg):
    bn, bq, st, thr = cfg
    try:
        kern = make_kernel(bn, bq, st, thr)
    except Exception as e:
        print(f"  cfg={cfg} JIT-FAIL {str(e)[:90]}", flush=True)
        return
    q = torch.randn(ROWS * HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(KEYS, DIM, device="cuda", dtype=torch.bfloat16)
    logits = torch.empty(ROWS, KEYS, device="cuda", dtype=torch.float32)
    starts = torch.randint(0, KEYS - TOPK_WINDOW, (ROWS,), device="cuda", dtype=torch.int32)
    ends = starts + TOPK_WINDOW
    def run_all():
        for rs in range(0, ROWS, bq):
            re_ = min(rs + bq, ROWS)
            kern(q[rs*HEADS:re_*HEADS], k, logits[rs:re_], starts[rs:re_], ends[rs:re_])
    try:
        run_all(); torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t0 = time.perf_counter(); run_all(); torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        wall = statistics.median(ts)
        # useful flops for the whole row-loop (window-only, not padded):
        fl = ROWS * TOPK_WINDOW * HEADS * DIM * 2 * 2  # gemm*2ops × relu+sum pass ~ 2x flops lower bound; report raw gemm too
        raw = ROWS * TOPK_WINDOW * HEADS * DIM * 2
        print(f"  cfg={str(cfg):22s} rowloop={wall*1000:8.1f} ms  windowed-gemm={raw/wall/1e12:5.1f}TF", flush=True)
    except Exception as e:
        print(f"  cfg={cfg} RUN-FAIL {str(e)[:90]}", flush=True)
    finally:
        del q, k, logits, starts, ends
        torch.cuda.empty_cache()

print("baseline=(64,32,3,512); note current runtime calls per-iter with host sync", flush=True)
print("=== window scan (bq=32, 512thr) ===", flush=True)
for cfg in [(64,32,3,512),(64,32,4,512),(128,32,2,512),(128,32,3,512),(128,32,4,512),(128,32,5,512),(256,32,2,512),(64,64,3,512),(128,64,3,512),(128,64,4,512),(64,32,3,256),(128,32,3,256)]:
    bench(cfg)
