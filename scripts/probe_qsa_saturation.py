"""Go/no-go probe for QSA kernel rewrite: is the ck kernel parallelism-starved?

If 10.2ms @ B=1 (8192 CTA) drops toward ~7.7ms dense-cost @ B>=4 with near-same
per-chunk wall time, the SMs are starved -> topk-split rewrite has headroom.
If time grows ~linearly with B, the kernel is already saturated -> rewrite is dead.
"""
import sys, time, statistics
import torch, triton
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
from sglang.srt.layers.attention.qsa.sparse_attn import _sparse_gqa_chunk_prefill

HD, NKB, TOPK, KV = 256, 2, 2048, 1_000_976
BM = 16
def bench(B, total_q, cfg):
    bn, w, s = cfg
    NQB = 24
    q = torch.randn(B*total_q, NQB, HD, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B*KV, NKB, HD, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B*KV, NKB, HD, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, KV, (B*total_q, NKB, TOPK), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    cu_q = torch.arange(0, B+1, device="cuda", dtype=torch.int32) * total_q
    cu_k = torch.arange(0, B+1, device="cuda", dtype=torch.int32) * KV
    kvl = torch.full((B,), KV, device="cuda", dtype=torch.int32)
    grid = (total_q, B * NKB)
    def fn():
        _sparse_gqa_chunk_prefill[grid](q,k,v,out,idx,cu_q,cu_k,kvl,0.0625,TOPK,
            *q.stride(),*k.stride(),*v.stride(),*out.stride(),idx.stride(0),idx.stride(1),idx.stride(2),
            NUM_KV_HEADS=NKB,GROUP_SIZE=NQB//NKB,BLOCK_M=BM,BLOCK_N=bn,HEAD_DIM=HD,num_warps=w,num_stages=s)
    fn(); torch.cuda.synchronize()
    ts=[]
    for _ in range(9):
        t0=time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter()-t0)*1e3)
    del q,k,v,idx,out; torch.cuda.empty_cache()
    return statistics.median(ts)

for cfg in [(16,1,2),(64,4,2)]:
    for B in (1,2,4,8):
        ms = bench(B, 4096, cfg)
        # per-request-chunk cost if serialized; ideal parallel cost ~ dense floor 7.7ms
        print(f"cfg={cfg} B={B}: {ms:7.2f} ms  (per-req-chunk {ms/B*1000:6.0f}us-equiv)", flush=True)
