"""Gate A2: EXTENDED bitwise equality split=64 vs split=1 — full shape lattice.

Covers the shape classes gate A missed (early chunks with budget-full rows,
multi-request interleaved windows = the split-boundary stress case, degenerate
edges). Production shapes only (heads=4 dim=128 compressed kv_heads=1).
"""
import sys, math
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
import torch
import sglang.srt.layers.attention.qsa.mqa as mqa

KEYS = 250_244
HEADS, DIM = 4, 128

def logits(rows_spec, seed, split_override):
    """rows_spec: list of (count, base_offset, win) interleaved segments.
    Mirrors production select_prefill_tokens row-chunk loop (128MB budget)."""
    torch.manual_seed(seed)
    dev = "cuda"
    rows = sum(c for c, _, _ in rows_spec)
    q = torch.randn(rows, HEADS, DIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(KEYS, 1, DIM, device=dev, dtype=torch.bfloat16)
    starts, ends = [], []
    for count, base, win in rows_spec:
        s = torch.arange(count, device=dev, dtype=torch.int32) + base
        starts.append(s)
        ends.append(s + win)
    starts = torch.cat(starts)
    ends = torch.cat(ends)
    block_q = max(1, 128 // HEADS)
    # _qsa_prefill_row_chunk_size: 128MB / (KEYS*4B) aligned down to block_q
    rc = max(block_q, (128 * 1024 * 1024 // (KEYS * 4)) // block_q * block_q)
    kern = mqa._tilelang_qsa_mqa_prefill_kernel(
        heads=HEADS, head_dim=DIM, block_q=block_q, split=split_override)
    kb = k[:, 0].to(torch.bfloat16).contiguous()
    out = torch.empty((rows, KEYS), dtype=torch.float32, device=dev)
    for rs in range(0, rows, rc):
        re_ = min(rs + rc, rows)
        n = re_ - rs
        padding = (-n) % block_q
        padded = n + padding
        buf = torch.zeros((padded, KEYS), dtype=torch.float32, device=dev)
        qp = q[rs:re_].to(torch.bfloat16).contiguous()
        st, en = starts[rs:re_].clone(), ends[rs:re_].clone()
        if padding:
            qp = torch.cat([qp, qp.new_zeros(padding, HEADS, DIM)])
            st = torch.cat([st, st[-1:].expand(padding)])
            en = torch.cat([en, en[-1:].expand(padding)])
        kern(qp.reshape(-1, DIM), kb, buf, st, en)
        out[rs:re_] = buf[:n]
        del buf
    out.div_(math.sqrt(DIM))
    mqa._tilelang_qsa_mqa_mask_kernel()(out, starts, ends)
    return out

CASES = [
    # (name, rows_spec, seed)
    ("early-budget-full r4096 w8K",  [(4096, KEYS-9000, 8192)], 7),
    ("early r2048 w32K",             [(2048, KEYS-40000, 32768)], 8),
    ("interleave-2seq",              [(2048, 10_000, 50_000), (2048, 100_000, 150_000)], 9),
    ("interleave-extreme",           [(64, 0, 64), (4032, KEYS-5000, 4999)], 10),
    ("full-window",                  [(128, 10, KEYS-20)], 12),
    ("zero-len-segs r1",             [(1, KEYS//2, 1)], 13),
    ("all-blocks-partial r512",      [(512, KEYS-600, 599)], 14),  # win not multiple of 64
    ("late-1M control",              [(128, KEYS-244100, 244000)], 15),
    ("multi-batch 4x1024",           [(1024, 5_000, 120_000), (1024, 60_000, 90_000),
                                      (1024, 120_000, 60_000), (1024, 200_000, 30_000)], 16),
]
fails = 0
for name, spec, seed in CASES:
    try:
        a = logits(spec, seed, 64)
        a2 = logits(spec, seed, 64)
        det = torch.equal(a, a2)   # temporal determinism: catches overlapping-CTA
        del a2                     # write races (the silent-rot failure mode)
        torch.cuda.empty_cache()
        b = logits(spec, seed, 1)
        eq = torch.equal(a, b) and det
        # split=32 cross-check on the two fattest cases
        if "budget" in name or "multi-batch" in name:
            c = logits(spec, seed, 32)
            eq32 = torch.equal(a, c)
            del c
            eq = eq and eq32
            print(f"   split32-vs-split64 bitwise: {eq32}", flush=True)
        if not eq:
            fails += 1
            idx = (a != b).nonzero()[:5]
            for r, c in idx.tolist():
                print(f"   diff@[{r},{c}] s64={a[r,c].item()} s1={b[r,c].item()}")
        print(f"{name}: bitwise-equal={eq}", flush=True)
        del a, b
    except Exception as e:
        fails += 1
        print(f"{name}: EXC {str(e)[:90]}", flush=True)
    torch.cuda.empty_cache()
print("GATE-A2:", "PASS bitwise" if fails == 0 else f"FAIL {fails}", flush=True)
