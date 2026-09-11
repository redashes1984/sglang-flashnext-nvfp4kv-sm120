"""Gate A: bitwise equality of split-K vs split=1 logits (production mqa.py).

Runs the REAL patched kernel twice in one process: once with the shipped
launch (split=64 via _tilelang_qsa_mqa_prefill_kernel kwargs) and once with
split forced to 1, on identical seeded inputs, then asserts torch.equal on
the full logits matrix. Shapes = production late-chunk:
  rows=128 block_q=32 grid=(4,) ; keys=250244 compressed ; causal starts.
Also a mid-depth shape (rows=512, win=60K) and an edge shape (rows=130 with
padding -> 5 row-blocks) to hit uneven-split remainders.
"""
import sys, math
sys.path.insert(0, "/opt/sglang-patch/sglang/python")
import torch
import sglang.srt.layers.attention.qsa.mqa as mqa

KEYS = 250_244
HEADS, DIM = 4, 128

def logits_with_split(rows, win, seed, split_override):
    torch.manual_seed(seed)
    dev = "cuda"
    q = torch.randn(rows, HEADS, DIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(KEYS, 1, DIM, device=dev, dtype=torch.bfloat16)
    base = KEYS - win - rows - 100
    starts = (torch.arange(rows, device=dev, dtype=torch.int32) + base)
    ends = starts + win
    # replicate tilelang_qsa_mqa_prefill internals with split pinned
    block_q = max(1, 128 // HEADS)
    padding = (-rows) % block_q
    padded = rows + padding
    out = torch.zeros((padded, KEYS), dtype=torch.float32, device=dev)
    qp = q.to(torch.bfloat16).contiguous()
    st, en = starts.clone(), ends.clone()
    if padding:
        qp = torch.cat([qp, qp.new_zeros(padding, HEADS, DIM)])
        st = torch.cat([st, st[-1:].expand(padding)])
        en = torch.cat([en, en[-1:].expand(padding)])
    kern = mqa._tilelang_qsa_mqa_prefill_kernel(
        heads=HEADS, head_dim=DIM, block_q=block_q, split=split_override)
    kern(qp.reshape(-1, DIM), k[:, 0].to(torch.bfloat16).contiguous(), out, st, en)
    out = out[:rows]
    out.div_(math.sqrt(DIM))
    mqa._tilelang_qsa_mqa_mask_kernel()(out, starts, ends)
    return out

CASES = [(128, 244_000, "late-1M"), (512, 60_000, "mid"), (130, 999, "uneven-pad"), (37, 4096, "tiny-rows")]
fails = 0
for rows, win, tag in CASES:
    for seed in (11, 23):
        a = logits_with_split(rows, win, seed, 64)
        b = logits_with_split(rows, win, seed, 1)
        eq = torch.equal(a, b)
        md = (a - b).abs().max().item()
        # compare only in-window (mask sets -inf outside on both)
        print(f"{tag} rows={rows} win={win} seed={seed}: bitwise-equal={eq} maxdiff={md:.3e}", flush=True)
        if not eq:
            fails += 1
            nz = (a != b)
            # -inf == -inf is True, so diffs are real
            idx = nz.nonzero()[:5]
            for r, c in idx.tolist():
                print(f"   diff@[{r},{c}] split64={a[r,c].item()} split1={b[r,c].item()}")
del a, b
torch.cuda.empty_cache()
print("GATE-A:", "PASS bitwise" if fails == 0 else f"FAIL {fails} cases", flush=True)
