"""QSA sparse-GQA kernel config sweep for SM120 (RTX PRO 6000) — TTFT cold-prefill.

Imports the two Triton kernels from the patched tree directly (no sglang runtime
needed). Synthesizes the 1M cold-chunk shape: total_q=4096, 24 q-heads / 2 kv-heads
(GROUP=12), HEAD_DIM=256, topk=2048, fp8 e4m3 K/V over a 1M-token pool (matches
late-chunk gather dispersion). Baseline = current (16,1,2) L20-table pick.

Run only while the inference service is STOPPED (needs ~1.1GB device mem).
"""
import itertools, statistics, sys, time
import torch
import triton

sys.path.insert(0, "/opt/sglang-patch/sglang/python")
from sglang.srt.layers.attention.qsa.sparse_attn import (
    _sparse_gqa_prefill, _sparse_gqa_chunk_prefill,
)

DEV = "cuda"
TOTAL_Q = 4096          # chunked-prefill chunk size
NQB, NKB, HD = 24, 2, 256
GROUP = NQB // NKB
BLOCK_M = max(16, triton.next_power_of_2(GROUP))
TOPK = 2048
KV_LEN = 1_000_976      # late-chunk depth (probe measured pt)
B = 1                   # single 1M chain during cold prefill

def make_inputs():
    torch.manual_seed(0)
    q = torch.randn(TOTAL_Q, NQB, HD, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(KV_LEN, NKB, HD, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(KV_LEN, NKB, HD, device=DEV, dtype=torch.bfloat16)
    # NOTE: real serving passes fp8 K/V through the pool buffer; timing here uses
    # bf16 storage to isolate config effects (bandwidth term slightly optimistic
    # vs prod, ranking of configs unaffected).
    idx = torch.randint(0, KV_LEN, (TOTAL_Q, NKB, TOPK), device=DEV, dtype=torch.int32)
    out = torch.empty_like(q)
    cu_q = torch.tensor([0, TOTAL_Q], device=DEV, dtype=torch.int32)
    kv_lens = torch.tensor([KV_LEN], device=DEV, dtype=torch.int32)
    cu_k = torch.tensor([0, KV_LEN], device=DEV, dtype=torch.int32)
    scale = 1.0 / (HD ** 0.5)
    return q, k, v, out, idx, cu_q, cu_k, kv_lens, scale

def timed(fn, iters=15):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)

def run(kind, bn, w, s, inp):
    q, k, v, out, idx, cu_q, cu_k, kv_lens, scale = inp
    sm, sh, sd = q.stride(); kn, kh, kd = k.stride()
    vn, vh, vd = v.stride(); on, oh, od = out.stride()
    im, ig = idx.stride(0), idx.stride(1)
    in_ = idx.stride(2) if idx.ndim == 3 else idx.stride(1)
    if kind == "dense":
        grid = (TOTAL_Q, B * NKB)
        def fn():
            _sparse_gqa_prefill[grid](q, k, v, out, idx, cu_q, scale, TOPK,
                sm, sh, sd, kn, kh, kd, vn, vh, vd, on, oh, od, im, ig, in_,
                NUM_KV_HEADS=NKB, GROUP_SIZE=GROUP, BLOCK_M=BLOCK_M,
                BLOCK_N=bn, HEAD_DIM=HD, num_warps=w, num_stages=s)
    else:
        grid = (TOTAL_Q, B * NKB)
        def fn():
            _sparse_gqa_chunk_prefill[grid](q, k, v, out, idx, cu_q, cu_k, kv_lens,
                scale, TOPK, sm, sh, sd, kn, kh, kd, vn, vh, vd, on, oh, od,
                im, ig, in_, NUM_KV_HEADS=NKB, GROUP_SIZE=GROUP, BLOCK_M=BLOCK_M,
                BLOCK_N=bn, HEAD_DIM=HD, num_warps=w, num_stages=s)
    return timed(fn)

def main():
    inp = make_inputs()
    combos = [(16,1,2)] + [(bn,w,s) for bn,w,s in itertools.product((16,32,64,128),(2,4,8),(2,3))]
    results = {}
    for kind in ("dense", "ck"):
        print(f"=== {kind} ===", flush=True)
        base = None
        for bn,w,s in combos:
            try:
                ms = run(kind, bn, w, s, inp)
            except Exception as e:
                print(f"  BN={bn:3d} w={w} s={s}  FAIL {str(e)[:80]}", flush=True)
                continue
            tag = " <== BASELINE(current)" if (bn,w,s)==(16,1,2) else ""
            if (bn,w,s)==(16,1,2): base = ms
            speedup = f" {base/ms:5.2f}x" if base else ""
            print(f"  BN={bn:3d} w={w} s={s}  {ms:7.2f} ms/chunk{speedup}{tag}", flush=True)
            results[(kind,bn,w,s)] = ms
        best = sorted([x for x in results if x[0]==kind], key=lambda x: results[x])[:5]
        print(f"BEST-TOP5 {kind}:")
        for kk in best:
            bn,w,s = kk[1:]
            print(f"  BN={bn:3d} w={w} s={s} -> {results[kk]:7.2f} ms  (~{TOTAL_Q/results[kk]*1e3:6.0f} t/s) "
                  f"est 1M-chain late-phase share")
if __name__ == "__main__":
    assert torch.cuda.get_device_capability()[0] == 12, "run on CT112 only"
    main()
