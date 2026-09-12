from __future__ import annotations

from typing import Optional

import torch

_FAST_TOPK_SUPPORTED_K = (512, 2048)


def _check_topk(topk: int) -> None:
    if topk not in _FAST_TOPK_SUPPORTED_K:
        raise RuntimeError(
            f"Unsupported topk {topk}. Supported: {_FAST_TOPK_SUPPORTED_K}"
        )


def fast_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Per-row top-k selection over a fp32 score matrix.

    Row b selects the `topk` largest values in
    ``score[b, row_starts[b] : row_starts[b] + lengths[b]]`` and returns their
    indices relative to ``row_starts[b]``. Slots beyond ``lengths[b]`` are -1.
    Output order is deterministic for fixed inputs but otherwise unspecified.

    Parameters
    ----------
    score      : CUDA fp32 tensor [B, L]
    lengths    : CUDA int32 tensor [B]
    topk       : number of indices per row; 512 or 2048
    row_starts : optional CUDA int32 tensor [B]; defaults to zeros

    Returns
    -------
    CUDA int32 tensor [B, topk]
    """
    _check_topk(topk)
    batch = score.shape[0]
    offsets = torch.zeros(batch, dtype=torch.int32, device=score.device)
    starts = None
    if row_starts is not None:
        starts = row_starts.to(device=score.device, dtype=torch.int32).contiguous()

    from flashinfer import top_k_ragged_transform

    return top_k_ragged_transform(
        score.contiguous(),
        offsets,
        lengths.to(device=score.device, dtype=torch.int32).contiguous(),
        topk,
        deterministic=True,
        row_starts=starts,
    )
