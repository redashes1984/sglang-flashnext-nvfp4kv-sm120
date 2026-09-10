"""Functional gate for the PLE file backend on this box.

The upstream check demands cudaDevAttr...UsesHostPageTables(100)=1 (GB10
route). This x86 box reports attr 88 (PageableMemoryAccess, IOMMU route) = 1
and 100 = 0. Enum debates aside, the only thing that matters: can the *real*
production gather kernel dereference an mmap'd pageable host pointer and read
the right rows? That is exactly what allocate_ple_host_table('file') hands
the kernel. Pass -> the check is over-strict here; fail -> file backend would
silently corrupt and the ValueError saved us.

Uses the production kernel _gather_ple_embedding_from_pinned_kernel verbatim
(is_fp8=False -> bf16 row path; the fp8 path differs only in element type,
the host-pointer dereference under test is identical).

    PYTHONPATH=/opt/sglang-patch/sglang/python /opt/sglang-env/bin/python3 test_pageable_gather.py
"""
import os
import sys
import tempfile

import torch

from sglang.srt.models.qwen4_exp import (
    _gather_ple_embedding_from_pinned_kernel,
)

ROWS, DIM = 4096, 160


def run_case(table_host, idx):
    out = torch.zeros(idx.numel(), DIM, dtype=torch.bfloat16, device="cuda")
    try:
        _gather_ple_embedding_from_pinned_kernel[(idx.numel(),)](
            table_host.data_ptr(),
            idx,
            out,
            embedding_dim=DIM,
            tp_vocab_start=0,
            tp_vocab_end=ROWS,
            is_fp8=False,
            BLOCK_D=256,
        )
        torch.cuda.synchronize()
    except Exception as e:
        return f"ERROR {type(e).__name__}: {e}"
    want = table_host.to("cuda")[idx.long()]
    return "PASS" if torch.equal(out, want) else "MISMATCH"


def main():
    if not torch.cuda.is_available():
        print("NO-CUDA"); sys.exit(2)
    print("device:", torch.cuda.get_device_name(0))

    src = torch.randn(ROWS, DIM, dtype=torch.bfloat16)
    idx = torch.randint(0, ROWS, (256,), dtype=torch.int64, device="cuda")

    # A: plain pageable (malloc) host tensor
    print("  pageable-malloc:", run_case(src.clone(), idx))

    # B: shared file mmap via torch.from_file — the file backend's allocator
    nbytes = ROWS * DIM * 2
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ple")
    tmp.close()
    with open(tmp.name, "wb") as f:
        f.truncate(nbytes)
    storage = torch.from_file(tmp.name, shared=True, size=nbytes, dtype=torch.uint8)
    mmap_table = storage.view(torch.bfloat16).view(ROWS, DIM)
    mmap_table.copy_(src)
    print("  pageable-mmap-file:", run_case(mmap_table, idx))
    os.unlink(tmp.name)


main()
