#!/usr/bin/env python3
"""QSA prefill-MQA split-K patch — CTA starvation fix (2026-09-11).

Measured: late 1M-chunk indexer logits run grid=4 CTAs (row_chunk=128 from
128MB logits budget, block_q=32) on 148 SMs -> 1.6 TF/s.  Split-K over key
tiles (grid y = split) scales perfectly: split=64 -> 91 TF/s (56x).
Service effect: 1M cold TTFT 292.9s -> 133.3s (-54.5%), NIAH@1M PASS,
256K side chains and small-lobe unchanged, fence stays 4.00 GB.

Idempotent (MARKER). Apply AFTER apply_nvfp4_patches.py + 0001 + 0002 on the
78c5024e baseline tree (mqa.py is untouched by those, so anchors are stable).
"""
import ast
import os
import pathlib
import sys

SRT = pathlib.Path(os.environ.get("SGLANG_SRT", "/opt/sglang-patch/sglang/python/sglang/srt"))
MQA = SRT / "layers" / "attention" / "qsa" / "mqa.py"
MARKER = "split: int = 1"

KERNEL_OLD = '''    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
    ):
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
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
                for ni in T.Pipelined(
                    T.ceildiv(end_max - start_min, block_n), num_stages=num_stages
                ):
                    T.copy(K[start_min + ni * block_n, 0], k_shared)
                    T.gemm(
                        k_shared,
                        q_shared,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                    for n, qi, head in T.Parallel(block_n, block_q, heads):
                        scores_3d[n, qi, head] = T.max(scores_3d[n, qi, head], 0.0)
                    T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                    for qi, n in T.Parallel(block_q, block_n):
                        Logits[row_base + qi, start_min + ni * block_n + n] = reduced[
                            n, qi
                        ]

        return kernel'''

KERNEL_NEW = '''    def _tilelang_qsa_mqa_prefill_kernel(
        heads: int,
        head_dim: int,
        block_n: int = 64,
        block_q: int = 32,
        num_stages: int = 3,
        threads: int = 512,
        split: int = 1,
    ):
        # Split-K over key tiles: row_blocks x split CTAs cover disjoint
        # tile ranges of the block's [start_min, end_max] window.  Without
        # it, a 128-row late-chunk iteration launches only 4 CTAs and idles
        # the GPU (measured 1.6 TF/s vs 91 TF/s at split=64, 2026-09-11).
        rows = T.dynamic("rows")
        keys = T.dynamic("keys")

        @T.prim_func
        def kernel(
            Q: T.Tensor([rows * heads, head_dim], T.bfloat16),  # type: ignore
            K: T.Tensor([keys, head_dim], T.bfloat16),  # type: ignore
            Logits: T.Tensor([rows, keys], T.float32),  # type: ignore
            Starts: T.Tensor([rows], T.int32),  # type: ignore
            Ends: T.Tensor([rows], T.int32),  # type: ignore
        ):
            with T.Kernel(
                T.ceildiv(rows, block_q), split, threads=threads
            ) as (bx, by):
                q_shared = T.alloc_shared([block_q * heads, head_dim], T.bfloat16)
                k_shared = T.alloc_shared([block_n, head_dim], T.bfloat16)
                scores = T.alloc_fragment([block_n, block_q * heads], T.float32)
                scores_3d = T.reshape(scores, (block_n, block_q, heads))
                reduced = T.alloc_fragment([block_n, block_q], T.float32)
                row_base = bx * block_q
                start_min = T.alloc_var(T.int32)
                end_max = T.alloc_var(T.int32)
                tiles = T.alloc_var(T.int32)
                lo_t = T.alloc_var(T.int32)
                hi_t = T.alloc_var(T.int32)
                start_min = 2147483647
                end_max = -2147483648
                for qi in T.serial(block_q):
                    start_min = T.min(start_min, T.min(Starts[row_base + qi], keys))
                    end_max = T.max(end_max, T.min(Ends[row_base + qi], keys))
                # Tail tile may read K rows past end_max (within pool keys);
                # its out-of-window logits columns are scrubbed by the mask
                # kernel downstream, exactly like the split=1 tail tile does.
                tiles = T.ceildiv(end_max - start_min, block_n)
                lo_t = tiles * by // split
                hi_t = tiles * (by + 1) // split
                if lo_t < hi_t:
                    T.copy(Q[row_base * heads, 0], q_shared)
                    for ni in T.Pipelined(
                        hi_t - lo_t, num_stages=num_stages
                    ):
                        T.copy(K[start_min + (lo_t + ni) * block_n, 0], k_shared)
                        T.gemm(
                            k_shared,
                            q_shared,
                            scores,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullCol,
                        )
                        for n, qi, head in T.Parallel(block_n, block_q, heads):
                            scores_3d[n, qi, head] = T.max(
                                scores_3d[n, qi, head], 0.0
                            )
                        T.reduce_sum(scores_3d, reduced, dim=-1, clear=True)
                        for qi, n in T.Parallel(block_q, block_n):
                            Logits[
                                row_base + qi,
                                start_min + (lo_t + ni) * block_n + n,
                            ] = reduced[n, qi]

        return kernel'''

LAUNCH_OLD = '''    _tilelang_qsa_mqa_prefill_kernel(heads=heads, head_dim=head_dim, block_q=block_q)(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )'''

LAUNCH_NEW = '''    # Split-K fixed at 64 (measured sweet spot: 91 TF/s at late depth; 2026-09-11).
    # No host-side span math: an .item() sync here would invalidate any future
    # prefill CUDA graph capture.  CTAs whose tile range comes out empty exit
    # immediately, so short windows pay only negligible launch overhead.
    _tilelang_qsa_mqa_prefill_kernel(
        heads=heads, head_dim=head_dim, block_q=block_q, split=64
    )(
        q_padded.reshape(-1, head_dim),
        k[:, 0].to(torch.bfloat16).contiguous(),
        logits,
        starts,
        ends,
    )'''


def main():
    s = MQA.read_text()
    if MARKER in s:
        print("mqa.py: already patched")
        return 0
    for anchor, replacement in ((KERNEL_OLD, KERNEL_NEW), (LAUNCH_OLD, LAUNCH_NEW)):
        count = s.count(anchor)
        assert count == 1, f"mqa.py: anchor matched {count} times (want 1):\n{anchor[:160]}"
        s = s.replace(anchor, replacement, 1)
    ast.parse(s)
    MQA.write_text(s)
    print("mqa.py: split-K patched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
