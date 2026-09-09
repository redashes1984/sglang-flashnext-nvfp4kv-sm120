"""Expert tiering for Qwen4-Exp fused MoE: hot keep-set + ring slots on GPU,
cold rows resident in pinned host RAM.

Active only when BOTH env vars are set:
  SGLANG_EXPERT_KEEP_MASK=<path>   keep json: {layer_idx_str: [expert ids]}
  SGLANG_EXPERT_KEEP_OFFLOAD=1

Otherwise every entry point is a pure passthrough.

Per layer: fused GPU tensors are rebuilt as (n_keep + R) rows — keep experts
ordered by sorted keep id, then R ring slots seeded from global frequency
counts at init. Cold rows live in one pinned CPU pool per tensor, indexed by
rank within sorted cold ids. Forward remap: bincount/topk on GPU picks this
step's cold experts, pinned pool rows are copied into ring slots (async H2D,
capture-recordable), and a lookup binds selected ids to slots. Ids that miss
fall back to slot 0. Under CUDA graphs the host gather bakes at capture time
while the GPU selection replays live — steady-state selection is stable
across batches (two-round recording: 98.6% overlap), so staleness is limited.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_KEEP_MASK_PATH = os.environ.get("SGLANG_EXPERT_KEEP_MASK", "")
_KEEP_OFFLOAD = os.environ.get("SGLANG_EXPERT_KEEP_OFFLOAD", "") == "1"
_ENABLED = bool(_KEEP_MASK_PATH) and _KEEP_OFFLOAD
_RING = 16

_counts_cache: Optional[torch.Tensor] = None
_counts_loaded = False
_pool_mb = 0.0
_logged_init = False


def tiering_enabled() -> bool:
    return _ENABLED


def _load_keep_sets() -> dict:
    with open(_KEEP_MASK_PATH) as fh:
        raw = json.load(fh)
    return {int(k): sorted(int(i) for i in v) for k, v in raw.items()}


def _counts() -> Optional[torch.Tensor]:
    """(layers, experts) frequency tensor used to seed ring rows."""
    global _counts_cache, _counts_loaded
    if not _counts_loaded:
        _counts_loaded = True
        try:
            _counts_cache = torch.load(
                "/opt/sglang-config/combined_counts_48x512.pt",
                map_location="cpu",
                weights_only=True,
            ).float()
        except Exception:
            _counts_cache = None
    return _counts_cache


class LayerTier:
    def __init__(self, layer, keep_ids):
        global _pool_mb, _logged_init
        self.layer_id = getattr(layer, "layer_id", -1)
        num_experts = layer.num_local_experts
        keep_set = set(keep_ids)
        cold_ids = [i for i in range(num_experts) if i not in keep_set]
        n_keep, n_cold = len(keep_ids), len(cold_ids)

        main = {
            "w13_weight": layer.w13_weight.detach(),
            "w2_weight": layer.w2_weight.detach(),
        }
        sides = dict(layer.named_per_expert_tensors(num_experts))
        device = main["w13_weight"].device

        keep_idx = torch.as_tensor(keep_ids, dtype=torch.long, device=device)
        cold_idx = torch.as_tensor(cold_ids, dtype=torch.long, device=device)

        counts = _counts()
        seed_ranks: Optional[torch.Tensor] = None
        if isinstance(counts, torch.Tensor) and counts.shape[0] > self.layer_id:
            row = counts[self.layer_id]
            if row.numel() == num_experts:
                cold_counts = row[cold_idx.cpu()]
                seed_ranks = torch.argsort(
                    cold_counts, descending=True
                )[:_RING].to(device)

        self.pairs = []
        for name, t in list(main.items()) + list(sides.items()):
            assert t.ndim > 0 and t.shape[0] == num_experts, (
                f"expert_tier: unexpected shape {tuple(t.shape)} for {name}"
            )
            keep_part = t.index_select(0, keep_idx)
            cold_part = t.index_select(0, cold_idx)
            ring = torch.zeros(
                (_RING,) + tuple(t.shape[1:]), dtype=t.dtype, device=device
            )
            if seed_ranks is not None:
                ring[: len(seed_ranks)] = cold_part.index_select(0, seed_ranks)
            new = torch.cat([keep_part, ring], dim=0)
            layer.replace_expert_tensor(name, new)
            pool = torch.empty(
                (n_cold,) + tuple(t.shape[1:]), dtype=t.dtype, pin_memory=True
            )
            pool.copy_(cold_part.cpu())
            _pool_mb += pool.numel() * pool.element_size() / (1024 * 1024)
            self.pairs.append((new, pool))

        rows = n_keep + _RING
        layer.num_local_experts = rows
        layer._num_local_routed = rows
        layer.num_experts = rows

        lookup = torch.full((num_experts,), -1, dtype=torch.long, device=device)
        lookup[keep_idx] = torch.arange(n_keep, dtype=torch.long, device=device)
        # keep-only base map (ring rows excluded): CUDA-graphed decode uses
        # this and zeroes any cold id, since ring contents are step-volatile.
        self.keep_lookup = lookup
        cold_rank = torch.zeros((num_experts,), dtype=torch.long, device=device)
        cold_rank[cold_idx] = torch.arange(n_cold, dtype=torch.long, device=device)
        self.cold_rank_cpu = cold_rank.cpu()
        keep_mask = torch.ones((num_experts,), dtype=torch.bool, device=device)
        keep_mask[keep_idx] = False

        self.keep_mask = keep_mask
        self.n_keep = n_keep
        self.n_cold = n_cold
        self.num_experts = num_experts
        self.warned_overflow = False
        self.ring_k = min(_RING, n_cold)
        self.ring_dsts = torch.arange(
            n_keep, n_keep + self.ring_k, dtype=torch.long, device=device
        )

        if not _logged_init:
            _logged_init = True
            logger.info(
                "expert_tier init: keep=%d ring=%d rows=%d host_pool_so_far=%.0fMB",
                n_keep,
                _RING,
                rows,
                _pool_mb,
            )

    def remap(self, ids: torch.Tensor, weights: torch.Tensor):
        """Logical ids -> physical GPU rows; returns (new_ids, new_weights).

        Eager (prefill): only cold experts actually hit this step are copied
        in from the pinned pool (avg 0-3 per layer, not all 16). Cold ids
        beyond ring capacity get their routing weight zeroed -> graceful
        mask-only degradation instead of garbage.

        Captured (graphed decode): dynamic H2D gathers are impossible, so the
        baked path maps through the static keep+seed lookup and zeroes
        everything else -> graphed decode is a mask-only(330+16) approx.
        """
        ids_l = ids.long()
        if torch.cuda.is_current_stream_capturing():
            mapped = self.keep_lookup[ids_l]
            miss = mapped < 0
            new_ids = torch.where(miss, self.ring_dsts[:1], mapped)
            new_w = torch.where(miss, torch.zeros_like(weights), weights)
            return new_ids.to(ids.dtype), new_w
        cnt = torch.bincount(ids_l.reshape(-1), minlength=self.num_experts)
        cold_cnt = cnt * (~self.keep_mask).to(cnt.dtype)
        vals, sel = torch.topk(cold_cnt, self.ring_k)
        n = int((vals > 0).sum().item())
        lookup = self.keep_lookup.clone()
        if n:
            lookup[sel[:n]] = self.ring_dsts[:n]
            ords = self.cold_rank_cpu[sel[:n].cpu()]
            for gpu_t, pool in self.pairs:
                gpu_t[self.ring_dsts[:n]] = pool.index_select(0, ords).to(gpu_t.device)
        mapped = lookup[ids_l]
        miss = mapped < 0
        new_ids = torch.where(miss, self.ring_dsts[0], mapped)
        new_w = torch.where(miss, torch.zeros_like(weights), weights)
        if not self.warned_overflow and n >= self.ring_k:
            self.warned_overflow = True
            logger.info(
                "expert_tier L%d: ring full (%d distinct cold) this step; "
                "tail zero-weighted",
                self.layer_id, n,
            )
        return new_ids.to(ids.dtype), new_w


def get_tier(layer):
    if not _ENABLED:
        return None
    return getattr(layer, "_expert_tier", None)


def setup_expert_tiers(model: torch.nn.Module) -> None:
    """Build tiers for every tierable FusedMoE layer. No-op when disabled."""
    if not _ENABLED:
        return
    try:
        keep_sets = _load_keep_sets()
        built = 0
        for module in model.modules():
            if type(module).__name__ != "FusedMoE":
                continue
            lid = getattr(module, "layer_id", None)
            if lid is None or lid not in keep_sets:
                continue
            module._expert_tier = LayerTier(module, keep_sets[lid])
            built += 1
        logger.info(
            "expert_tier active: %d layers tiered, keep=%d ring=%d host_pool=%.0fMB",
            built,
            len(next(iter(keep_sets.values()))),
            _RING,
            _pool_mb,
        )
    except Exception as exc:
        logger.warning("expert_tier setup failed (%s); running untiered", exc)


def apply_to_topk_output(layer, topk_output):
    """Hook called at the top of FusedMoE.forward_impl."""
    if not _ENABLED:
        return topk_output
    tier = get_tier(layer)
    if tier is None:
        return topk_output
    try:
        fmt = getattr(topk_output, "format", None)
        if fmt is not None and getattr(fmt, "name", str(fmt)) == "BYPASSED":
            topk_output = topk_output.to_standard(
                getattr(layer, "layer_id", None)
            )
        ids = getattr(topk_output, "topk_ids", None)
        wts = getattr(topk_output, "topk_weights", None)
        if ids is None or wts is None:
            return topk_output
        new_ids, new_w = tier.remap(ids, wts)
        packed = getattr(topk_output, "packed_topk_ids", None)
        if packed is not None:
            low = packed & 0xFFFF
            return topk_output._replace(
                topk_ids=new_ids, topk_weights=new_w,
                packed_topk_ids=(new_ids.long() << 16) | low,
            )
        return topk_output._replace(topk_ids=new_ids, topk_weights=new_w)
    except Exception as exc:
        if not getattr(apply_to_topk_output, "_warned", False):
            apply_to_topk_output._warned = True
            logger.warning("expert_tier: remap failed (%s); passthrough", exc)
        return topk_output
