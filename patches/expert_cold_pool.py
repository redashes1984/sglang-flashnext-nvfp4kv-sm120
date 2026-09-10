# expert_cold_pool.py — dynamic expert load/evict v2 (NVFP4 modelopt_fp4, flashinfer_cutlass)
#
# Design: docs/expert-dynamic-v2-design.md. Reference: ranxianglei/sglang@ours/main
# (keep-mask 3b7a2ba0c, offload c91c5c09d, cold-pool d464389af), corrected & adapted to
# modelopt_fp4 tensor inventory and our load-time (post-load_weights) shrink point.
#
# Layers (all env-gated; module fully inert without SGLANG_EXPERT_KEEP_MASK):
#   L1 router gating  — per-layer persistent bias column (keep/staged=0.0, else finfo.min)
#       and persistent remap table (global id -> physical row). Built ONCE at shrink time;
#       select_experts only ever reads the cached tensors. Both are graph-safe: capture
#       bakes the pointer, we mutate contents outside replay.
#   L2 physical shrink — at end of Qwen4Exp.load_weights (after process_weights_after_loading
#       in the loader), cold expert rows -> pinned host; every expert-sharded GPU tensor is
#       rebuilt to (keep_len + slots) rows. Pool shape never changes again. fp4-only: layers
#       without w13_weight_scale (e.g. unquantized draft) are skipped.
#   L3 dynamic staging — select_experts stashes RAW top-k global ids + true sigmoid scores
#       (computed from pre-mask logits — renormalized/masked weights would poison the
#       strong-demand band test); model_runner.forward tail calls after_forward_hook()
#       strictly outside graph replay; LRU rows; one-step lookahead; mask-before-overwrite.
#   L4 online keep<->cold — SGLANG_COLD_DEGRADE=1 (default OFF): starved keep rows D2H'd
#       into the host pool and their rows join the free list; sustained-demand staged rows
#       become LRU-protected. Content + tables only; tensors never resized.
#
# nvfp4 inventory (verified against checkpoint + modelopt_quant.py 2026-09-09):
#   w13_weight[E,2N,K/2]u8  w2_weight[E,K,N/2]u8
#   w13_weight_scale[E,2N,K/16]f8e4m3  w2_weight_scale[E,K,N/16]f8e4m3   (post-swizzle;
#       swizzle_blockscale treats dim0 as an independent batch dim — row-slicing == per-expert)
#   w13_weight_scale_2[E,2]f32  w2_weight_scale_2[E]f32  w13_input_scale[E,2]f32  w2_input_scale[E]f32
#   g1_alphas / g1_alphas_up / g2_alphas [E]f32 (derived; move by index_select too)
#   w*_input_scale collapse to 0-d scalars on the cutlass path -> dim0!=E -> auto-skipped.
#   ALIAS: w*_blockscale_swizzled may bind the SAME Parameter object as w*_weight_scale —
#   replacement sets all aliases to the new object (stale alias would leak ~150MB/layer and
#   alias readers would get the discarded full-E storage).

from __future__ import annotations

import gc
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

ROW_PARAMS = (
    "w13_weight",
    "w2_weight",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_input_scale",
    "w2_input_scale",
)
DERIVED_PARAMS = ("g1_alphas", "g1_alphas_up", "g2_alphas")
ALL_PARAMS = ROW_PARAMS + DERIVED_PARAMS


def _env_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name, default):
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def keep_mask_path():
    return os.environ.get("SGLANG_EXPERT_KEEP_MASK", "")


def offload_requested():
    return os.environ.get("SGLANG_EXPERT_KEEP_OFFLOAD", "") == "1" and bool(keep_mask_path())


def cold_slots():
    """Default 16 (v1-comparable). slots=0 => pure static keep-only pool (their default 32
    is for 296-keep W4A16; we run keep330)."""
    return _env_int("SGLANG_EXPERT_COLD_POOL_SLOTS", 16)


def dynamic_mode():
    return cold_slots() > 0


def strong_alpha():
    return _env_float("SGLANG_COLD_STRONG_ALPHA", 0.8)


def need_hits():
    return _env_int("SGLANG_COLD_NEED_HITS", 2)


def max_stage_per_tick():
    return _env_int("SGLANG_COLD_MAX_STAGE", 4)


def ema_decay():
    return _env_float("SGLANG_COLD_EMA_DECAY", 0.75)


def degrade_enabled():
    return _STATE["shrunk"] and dynamic_mode() and os.environ.get("SGLANG_COLD_DEGRADE", "") == "1"


def degrade_window():
    return _env_int("SGLANG_COLD_DEGRADE_WINDOW", 2048)


def debug_on():
    return os.environ.get("SGLANG_COLD_DEBUG", "") == "1"


_KEEP = None
_KEEP_LOADED = False
_LEAK_DUMPED = [0]  # v2.5b forensic budget (storage-referrer dumps, first 2 misses)
_BIAS_CACHE = {}    # (layer, dtype) -> persistent bias column (len = full-E = 512)
_REMAP_CACHE = {}   # layer -> persistent remap table (len = full-E)
_DIRTY = False
_STASH = {}         # layer -> {"ids","scores"} ring buffers (consume-once)
_CFG_LIDS = {}      # v2.9: id(topk_config) -> layer_id, ONLY for shrunk target modules
_CFG_REFS = []      # keeps those TopKConfig objects alive so id() is never recycled
_STATE = {
    "shrunk": False,       # shrink ran in THIS process (gates ALL runtime hooks)
    "dynamic": False,      # staging active (L3)
    "layers": {},
    "stats": None,
}


def _keep_mask():
    global _KEEP, _KEEP_LOADED
    if not _KEEP_LOADED:
        _KEEP_LOADED = True
        p = keep_mask_path()
        if p:
            with open(p) as fh:
                raw = json.load(fh)
            _KEEP = {int(k): sorted(int(i) for i in v) for k, v in raw.items()}
            logger.info("[COLD-POOL] keep mask loaded: %d layers, keep/layer=%s",
                        len(_KEEP), next(iter((len(v) for v in _KEEP.values())), 0))
    return _KEEP


def offload_active():
    """True only AFTER the shrink ran — pre-shrink forwards (warmup passes that happen
    before weight commit, draft process) must never gate/remap: the physical pool still
    holds all experts until shrink completes."""
    return _STATE["shrunk"]


# ------------------------------------------------------------------ tables


def _bias_col_or_none(layer_id, width, dtype, device):
    """Persistent router bias. Normally pre-built at shrink; lazy path covers a missing
    cache key — must run eager (list->CUDA indexing is H2D, illegal in capture)."""
    km = _keep_mask()
    if km is None:
        return None
    keep = km.get(int(layer_id))
    if not keep:
        return None
    key = (int(layer_id), dtype)
    col = _BIAS_CACHE.get(key)
    if col is None:
        if torch.cuda.is_current_stream_capturing():
            logger.error("[COLD-POOL] bias col for layer %s dtype %s missing during capture — "
                         "falling back to NO mask this call (staged pool stays routable)",
                         layer_id, dtype)
            return None
        col = torch.full((width,), torch.finfo(dtype).min, dtype=dtype, device=device)
        col[keep] = 0.0
        _BIAS_CACHE[key] = col
    return col


def _remap_or_none(layer_id, device):
    if not _STATE["shrunk"]:
        return None
    tbl = _REMAP_CACHE.get(int(layer_id))
    if tbl is not None and tbl.device != device:
        return None
    return tbl


def build_tables(layer_id, keep, cold_len, slots, width, dev):
    """Allocate persistent bias/remap for one layer at shrink time. Rows:
    keep gid -> its keep index; everything else -> row 0 with bias=-finfo.min (never
    selected AND if captured stale, row0 holds a valid expert). Staged cold gids get
    keep_len+row assigned by the prefetcher."""
    keep_len = len(keep)
    phys = keep_len + slots
    for dtype in (torch.bfloat16, torch.float32, torch.float16):
        col = torch.full((width,), torch.finfo(dtype).min, dtype=dtype, device=dev)
        col[keep] = 0.0
        _BIAS_CACHE[(int(layer_id), dtype)] = col
    tbl = torch.zeros(width, dtype=torch.long, device=dev)
    for row, gid in enumerate(keep):
        tbl[gid] = row
    _REMAP_CACHE[int(layer_id)] = tbl
    return phys


def mark_dirty():
    global _DIRTY
    _DIRTY = True


def consume_dirty():
    global _DIRTY
    d, _DIRTY = _DIRTY, False
    return d


def _unmask(layer_id, gid, phys_row):
    tbl = _REMAP_CACHE.get(int(layer_id))
    if tbl is not None:
        tbl[gid] = phys_row
    # v2.4 (clue B): update EVERY cached bias column of this layer, not just
    # pool.bias_dtype — apply_keep_mask picks the column by RUNTIME logits dtype,
    # a one-column update left the others permanently masking the staged gid.
    lid = int(layer_id)
    for (bl, _dt), col in _BIAS_CACHE.items():
        if bl == lid:
            col[gid] = 0.0


def _mask(layer_id, gid):
    tbl = _REMAP_CACHE.get(int(layer_id))
    if tbl is not None:
        tbl[gid] = 0
    lid = int(layer_id)
    for (bl, dt), col in _BIAS_CACHE.items():
        if bl == lid:
            col[gid] = torch.finfo(dt).min


def _return_row(pool, row):
    """v2.4 (clue A): a failed stage must give the row back. _pick_row already
    popped it from free_rows (and LRU-evicted its old owner); without this the
    row belongs to nobody — silent slot leak accumulating toward zero capacity."""
    if row is not None and row not in pool.row_gid and row not in pool.free_rows:
        pool.free_rows.append(row)


# ------------------------------------------------------------------ topk integration


def register_managed(module, layer_id, topk_obj=None):
    """v2.9 IDENTITY GATE — mark exactly the FusedMoE modules this process shrank.
    MTP/EAGLE draft models reuse decoder layer_id 0..47 in the SAME process
    (qwen2_moe block's TopK carries layer_id too). A layer_id-only gate would
    apply the TARGET's keep-bias + physical remap to draft routers: garbage
    drafts, poisoned demand stats, and remapped ids pointing at wrong GPU rows.
    select_experts is therefore gated on the TopKConfig OBJECT identity instead.
    The TopK lives on the PARENT sparse block (sibling of .experts), so callers
    pass it via topk_obj when the shrunk FusedMoE module has no own .topk."""
    tk = getattr(module, "topk", None) if getattr(module, "topk", None) is not None else topk_obj
    cfg = getattr(tk, "topk_config", None)
    if cfg is None:
        raise RuntimeError(
            f"[COLD-POOL] layer {layer_id}: no topk.topk_config on the shrunk module "
            "or its owner block — the identity gate cannot be armed; refusing "
            "(shrunk pool + ungated router would remap every id blindly)"
        )
    _CFG_LIDS[id(cfg)] = int(layer_id)
    _CFG_REFS.append(cfg)  # pin the object so its id() can never be recycled


def is_managed_config(topk_config):
    """layer_id of a registered target TopKConfig, or None (draft/other model)."""
    if not _STATE["shrunk"] or topk_config is None:
        return None
    return _CFG_LIDS.get(id(topk_config))


def apply_keep_mask(layer_id, router_logits):
    """select_experts entry, BEFORE raw topk: returns (maybe-biased logits, stash_ctx).
    stash_ctx = pre-mask logits for true-score extraction, or None when inactive."""
    if not _STATE["shrunk"] or layer_id is None:
        return router_logits, None
    width = int(router_logits.shape[-1])
    col = _bias_col_or_none(layer_id, width, router_logits.dtype, router_logits.device)
    if col is None:
        return router_logits, None
    return router_logits + col, router_logits


def remap_topk_ids(layer_id, topk_ids):
    tbl = _remap_or_none(layer_id, topk_ids.device) if layer_id is not None else None
    if tbl is None:
        return topk_ids
    # padded/masked rows carry id=-1 (_mask_topk_ids_padded_region fill_value=-1):
    # tbl[-1] would SILENTLY map them to the last expert's row. Keep -1 verbatim.
    safe = topk_ids.clamp(min=0).long()
    mapped = tbl[safe].to(topk_ids.dtype)
    return torch.where(topk_ids >= 0, mapped, topk_ids)


def remap_packed_ids(layer_id, packed):
    """StandardTopKOutputPacked carries ids in the high 16 bits (v1 convention).
    int32 in / int32 out — dtype stability matters under graph replay.
    Negative packed values (id=-1 padding, (id<<16)|low) pass through untouched:
    (& 0xFFFF) on them would index the table out of bounds -> device assert."""
    tbl = _remap_or_none(layer_id, packed.device) if layer_id is not None else None
    if tbl is None:
        return packed
    ids = ((packed >> 16) & 0xFFFF).long()
    low = packed & 0xFFFF
    new_ids = tbl[ids.clamp(max=tbl.numel() - 1)].to(packed.dtype)
    return torch.where(packed >= 0, (new_ids << 16) | low, packed)


_STASH_ROWS = 819  # subsample budget (rows of top-k per forward step)


def stash_demand(layer_id, pre_mask_logits, topk_ids, k):
    """Demand signal = raw ids + TRUE sigmoid scores from pre-mask logits.
    capture-time skipped (no host path anyway; previous eager demand carries over)."""
    if not _STATE["dynamic"] or layer_id is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    lay = int(layer_id)
    with torch.no_grad():
        rows = int(topk_ids.shape[0])
        idx = topk_ids.reshape(rows, -1).long()
        width = int(pre_mask_logits.shape[-1])
        neg = idx < 0  # padded/masked rows: never feed them as demand for expert 0
        idx = idx.clamp(min=0, max=width - 1)  # fused-shared-expert ids above width: masked later anyway
        # true scores from PRE-mask logits (renormalized/masked weights would poison
        # the strong-demand band test); raw-logit sigmoid — the scale factor is a
        # monotone constant on the softmax-free router and cancels in alpha comparisons
        logits2 = pre_mask_logits.reshape(rows, -1).float()
        sc = torch.gather(logits2, 1, idx).sigmoid().masked_fill(neg, -1.0)
        idx = idx.masked_fill(neg, -1)
        iff = idx.reshape(-1)
        sflat = sc.reshape(-1)
        row_valid = (idx >= 0).any(dim=1)
        if rows > _STASH_ROWS:
            stride = (rows + _STASH_ROWS - 1) // _STASH_ROWS
            iff = idx[::stride].reshape(-1)
            sflat = sc[::stride].reshape(-1)
            lg = logits2[::stride]
            row_valid = row_valid[::stride]
        else:
            lg = logits2
        # hook-side flat rows are the ids-filtered survivors: keep lg row-aligned
        # by dropping fully-invalid rows HERE (a fully -1 row contributes to
        # neither hot_min nor the probe).
        lg = lg[row_valid]
    cap = _STASH_ROWS * k
    buf = _STASH.get(lay)
    if buf is None:
        buf = {
            "ids": torch.full((cap,), -1, dtype=torch.long, device=iff.device),
            "scores": torch.full((cap,), -1.0, dtype=torch.float32, device=iff.device),
            "logits": None, "logits_n": 0,
        }
        _STASH[lay] = buf
    m = min(int(iff.numel()), cap)
    with torch.no_grad():
        buf["ids"][:m] = iff[:m]
        buf["ids"][m:].fill_(-1)
        buf["scores"][:m] = sflat[:m]
        buf["scores"][m:].fill_(-1.0)
        # v2.8: the MASKED router can never re-discover cold demand (bias -inf
        # removes them from topk) — stash the raw sampled logits so the hook
        # probes the alpha band directly. rows>=m/k are all -1-invalid, so the
        # hook aligns lg with the surviving valid rows by row order.
        buf["logits"] = lg
        buf["logits_n"] = int(lg.shape[0])
    mark_dirty()


# ------------------------------------------------------------------ pool


class LayerPool:
    __slots__ = (
        "layer_id", "keep", "keep_len", "cold", "slots", "module", "E", "phys",
        "host", "gid_row", "row_gid", "row_tick", "protected", "free_rows",
        "tick", "bias_dtype", "cold_ids_gpu", "cold_mask_gpu", "keep_ids_gpu",
        "need_counts", "seen_tick",
    )

    def __init__(self, layer_id, keep, cold, slots, module, E):
        self.layer_id = int(layer_id)
        self.keep = list(keep)
        self.keep_len = len(keep)
        self.cold = list(cold)
        self.slots = slots
        self.module = module
        self.E = int(E)
        self.phys = self.keep_len + slots
        self.host = {}
        self.gid_row = {}
        self.row_gid = {}
        self.row_tick = {}
        self.protected = set()
        self.free_rows = list(range(self.keep_len, self.phys))
        self.tick = 0
        self.bias_dtype = torch.bfloat16
        dev = module.w13_weight.device
        self.cold_ids_gpu = torch.tensor(self.cold, dtype=torch.long, device=dev)
        maskv = torch.zeros(self.E, dtype=torch.bool, device=dev)
        maskv[self.cold_ids_gpu] = True
        self.cold_mask_gpu = maskv
        self.keep_ids_gpu = torch.tensor(self.keep, dtype=torch.long, device=dev)
        self.need_counts = None
        self.seen_tick = None

    def host_bytes(self):
        return sum(t.numel() * t.element_size() for d in self.host.values() for t in d.values())


def _alias_names(module, target):
    """All attribute names bound to the same Parameter object as `target`.
    NOTE: must iterate _parameters directly — named_parameters() dedups
    (remove_duplicate=True) so a two-name alias would only yield ONE name."""
    names = [nm for nm, p in module._parameters.items() if p is target]
    return names or [None]


def _replace_with_aliases(module, name, new_t):
    """Replace `name` and bind every alias attribute to the SAME new Parameter.
    MF1 (star audit, 69/FAIL): replace_parameter lives in
    sglang.srt.layers.quantization.utils (NOT layers.utils.common — that module
    has copy_or_rebind_param/alias_or_bind_derived_param). Wrong import = the
    first cold-pool shrink raises AttributeError = zero delivery."""
    from sglang.srt.layers.quantization.utils import replace_parameter as _rp

    target = module._parameters.get(name)
    if target is None:
        # plain attribute (non-Parameter alias like a derived buffer) — direct set
        setattr(module, name, new_t)
        return new_t
    aliases = _alias_names(module, target)
    new_p = torch.nn.Parameter(new_t, requires_grad=False)
    for nm in aliases:
        # real impl re-registers when storage/dtype differ (our shrink always does);
        # trailing setattr pins every alias to the ONE new_p object (shared storage)
        _rp(module, nm, new_p)
        setattr(module, nm, new_p)
    return new_p


def _inventory_gate(module, E):
    """R3: dim0==E params outside the whitelist abort startup. 0-d and non-expert tensors pass."""
    bad = []
    for name, t in module.named_parameters(recurse=False):
        if t is None or t.dim() == 0 or t.shape[0] != E:
            continue
        if name in ALL_PARAMS:
            continue
        if name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            # v2.7 audit guard: safe ONLY while alias_or_bind_derived_param took
            # the same-object path (true iff swizzle needs no M/K padding, which
            # holds for this checkpoint: 1280x160 / rows%128==0, cols%4==0).
            # If a future config lands on the copy_or_rebind fallback, the
            # swizzled scale becomes a SEPARATE 512-row Parameter that staging
            # never refreshes -> silent kernel corruption; refuse to arm instead.
            src = getattr(module, name.replace("blockscale_swizzled", "weight_scale"), None)
            if t is not src:
                bad.append((name, tuple(t.shape), str(t.dtype) + " (unaliased swizzled scale)"))
            continue
        bad.append((name, tuple(t.shape), str(t.dtype)))
    return bad


def maybe_shrink_after_process(model):
    """Shrink entry — MUST run AFTER process_weights_after_loading (v2.1 fix).
    Hooked in model_runner.load_model right after the loader returns, which is
    loader-agnostic and strictly post-pwal: host pool then snapshots the FINAL
    runtime layout (deinterleaved w13, swizzled scales, derived g1_alphas).
    The old qwen4_exp end-of-load_weights point ran pre-pwal and would have
    staged pre-pwal bytes into post-pwal rows (v1's silent-corruption
    candidate root cause). Full pool exists until here; cold rows -> pinned host;
    GPU tensors rebuilt as keep+slots; persistent tables built; gating armed.
    Idempotent and process-local (draft workers are guarded out at the hook)."""
    if _STATE["shrunk"] or not offload_requested():
        return
    km = _keep_mask()
    if km is None:
        return
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    slots = cold_slots()
    shrunk, total_host = 0, 0
    # v2.9 identity gate: FusedMoE has no own .topk — TopK lives on the parent
    # sparse block (Qwen2MoeSparseMoeBlock.topk, sibling of .experts). Map each
    # FusedMoE to the unique sibling TopK in its direct parent container.
    def _owner_topk(mm):
        for parent in model.modules():
            for attr in dir(parent):
                try:
                    v = getattr(parent, attr)
                except Exception:
                    continue
                if v is mm:
                    tk = getattr(parent, "topk", None)
                    if tk is not None and getattr(tk, "topk_config", None) is not None:
                        return tk
        return None

    for m in model.modules():
        if not isinstance(m, FusedMoE):
            continue
        layer_id = getattr(m, "layer_id", None)
        if layer_id is None or int(layer_id) not in km:
            continue
        ref = getattr(m, "w13_weight", None)
        scale_ref = getattr(m, "w13_weight_scale", None)
        if ref is None or scale_ref is None or ref.dim() == 0:
            continue  # unquantized (draft) or unexpected layout — never shrink blind
        E = int(ref.shape[0])
        del scale_ref  # v2.5: holding it across the param loop pins one GPU tensor
        if E <= 1:
            continue
        keep = km[int(layer_id)]
        keep_set = set(keep)
        cold_ids = [g for g in range(E) if g not in keep_set]
        if not cold_ids:
            continue
        bad = _inventory_gate(m, E)
        if bad:
            raise RuntimeError(
                f"[COLD-POOL] layer {layer_id}: unclassified per-expert tensors {bad} — "
                "extend the whitelist (design §3 R3 gate) before enabling"
            )
        dev = ref.device
        pool = LayerPool(layer_id, keep, cold_ids, slots, m, E)
        keep_idx_cpu = torch.tensor(keep, dtype=torch.long)
        cold_idx_cpu = torch.tensor(cold_ids, dtype=torch.long)
        # v2.6e — v1's own mechanism, learned from expert_tier.py: it survived
        # real load via replace_expert_tensor => `param.data = new` (layer.py:527)
        # — the Parameter OBJECT stays, its storage is swapped. Any alias or
        # closure holding the OBJECT (not the old storage) sees the new one
        # automatically; the old GPU storage loses its last tensor ref instantly.
        # Peak trick: point param.data at the CPU keep slice FIRST (frees the GPU
        # storage with zero new GPU bytes), then copy back — never above the old
        # allocation size. v2.5's register_parameter(None) route failed because a
        # closure cell pinned the old *storage object* (LEAK-CAP: cell(STORAGE)).
        phys = pool.phys
        # v2.7 FLAT PINNED BUFFER — CT112 census proved the pin cost is 40.5GB
        # for 24.2GB of data: CachingHostAllocator rounds every >=1MB request UP
        # TO THE NEXT POWER OF 2 (292MB->512, 146->256, 36.5->64, 18.3->32 per
        # layer across 4 params = 864MB pinned per 493MB real). One per-param
        # pin_memory() per layer did that 384 times. Fix: gather cold rows into
        # ONE exact-size uint8 buffer per layer, pin once (493MB->512MB bucket),
        # hand out row views. pool.host[gid][name] keeps the SAME tensor-row
        # interface, so _row_from_host / Phase B / tests are untouched.
        # keep/cold are ARBITRARY subsets (profile order) — row slices MUST go
        # through index_select with keep_idx/cold_idx, never cpu[k:] tails.
        param_cpu = {}   # id(tensor) -> (name, full cpu copy)
        seen_ids = set()
        cold_bytes = 0
        for name in ALL_PARAMS:
            t = getattr(m, name, None)
            if not isinstance(t, torch.Tensor) or t.dim() == 0 or t.shape[0] != E:
                continue  # absent / 0-d / already swapped through a shared object
            if id(t) in seen_ids:
                continue  # two names aliasing one Parameter: pin/copy it once
            seen_ids.add(id(t))
            cpu = t.detach().to("cpu")  # D2H before release; no GPU alloc
            param_cpu[id(t)] = (name, t, cpu)
            cold_bytes += (cpu[0].numel() * cpu.element_size()) * len(cold_ids)
        buf = torch.empty(cold_bytes, dtype=torch.uint8)
        off = 0
        _cold_slices = {}
        for tid, (name, t, cpu) in param_cpu.items():
            cr = cpu.index_select(0, cold_idx_cpu)  # (cold_len, ...)
            nb = cr.numel() * cr.element_size()
            buf[off:off + nb].view(cr.dtype).reshape(cr.shape).copy_(cr)
            _cold_slices[name] = (off, nb, cr.shape, cr.dtype)
            off += nb
            del cr
        pbuf = buf.pin_memory()  # ONE power-of-2 bucket per layer (~512MB)
        del buf
        for name, (o, nb, shape, dt) in _cold_slices.items():
            flat = pbuf[o:o + nb].view(dt).reshape(shape)
            for i, gid in enumerate(cold_ids):
                pool.host.setdefault(gid, {})[name] = flat[i]
        # GPU swap: free old 512-row storage via param.data=cpu-view (v2.6e),
        # rebuild phys-row tensor, H2D the keep rows.
        for tid, (name, t, cpu) in param_cpu.items():
            keep_cpu = cpu.index_select(0, keep_idx_cpu)
            _sh = t.shape
            t.data = keep_cpu  # GPU storage refcount -> 0 (if nothing pins it)
            torch.cuda.empty_cache()
            t.data = torch.empty((phys,) + tuple(_sh[1:]), dtype=t.dtype, device=dev)
            t.data[: pool.keep_len].copy_(keep_cpu)  # H2D of the SMALL tensor
            del keep_cpu
        param_cpu.clear()
        del param_cpu
        # num_local_experts metadata must agree with the physical pool
        m.num_local_experts = pool.phys
        build_tables(layer_id, keep, len(cold_ids), slots, E, dev)
        register_managed(m, layer_id, topk_obj=_owner_topk(m))  # v2.9 identity gate
        _STATE["layers"][int(layer_id)] = pool
        total_host += pool.host_bytes()
        shrunk += 1
        torch.cuda.empty_cache()

    if shrunk:
        _STATE["shrunk"] = True
        _STATE["dynamic"] = slots > 0
        _STATE["stats"] = {
            "calls": 0, "strong": 0, "staged": 0, "evicted": 0,
            "h2d_bytes": 0, "demoted": 0, "promoted": 0, "stage_fail": 0,
        }
        gc.collect()
        torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[1] / 1e9
        klens = sorted({p.keep_len for p in _STATE["layers"].values()})
        logger.info(
            "[COLD-POOL] SHRUNK %d layers keep=%s +%d slots -> host pinned %.2f GB; "
            "dynamic=%s degrade=%s; gpu free now %.2f GB",
            shrunk, klens, slots, total_host / 1e9, _STATE["dynamic"], degrade_enabled(), free,
        )
    else:
        logger.info("[COLD-POOL] offload requested but nothing to shrink in this process "
                    "(draft worker or keep covers all experts) — inert here")


# ------------------------------------------------------------------ staging


def _row_from_host(pool, gid, row):
    rows = pool.host.get(gid)
    if not rows:
        return False
    nbytes = 0
    for name, pinned in rows.items():
        gpu = getattr(pool.module, name, None)
        if gpu is None or gpu.dim() == 0 or row >= gpu.shape[0]:
            logger.error("[COLD-POOL] layer %d: stage gid=%d row=%d failed (param %s)",
                         pool.layer_id, gid, row, name)
            if _STATE["stats"]:
                _STATE["stats"]["stage_fail"] += 1
            return False
        gpu.data[row].copy_(pinned, non_blocking=False)
        nbytes += pinned.numel() * pinned.element_size()
    pool.gid_row[gid] = row
    pool.row_gid[row] = gid
    pool.row_tick[row] = pool.tick
    _STATE["stats"]["h2d_bytes"] += nbytes
    _unmask(pool.layer_id, gid, row)
    return True


def _evict_row(pool, row):
    """mask FIRST, then the row is free to be overwritten by the caller."""
    old = pool.row_gid.pop(row, None)
    if old is not None:
        pool.gid_row.pop(old, None)
        _mask(pool.layer_id, old)
        _STATE["stats"]["evicted"] += 1


def _pick_row(pool):
    if pool.free_rows:
        return pool.free_rows.pop(0)
    cands = [r for r in pool.row_tick if r not in pool.protected]
    if not cands:
        return None
    victim = min(cands, key=lambda r: pool.row_tick[r])
    _evict_row(pool, victim)
    return victim


# ------------------------------------------------------------------ forward hook


def _checksum_selfcheck():
    bad = 0
    checked = 0
    for layer_id, pool in _STATE["layers"].items():
        for gid, row in list(pool.gid_row.items()):
            for name, pinned in pool.host[gid].items():
                gpu = getattr(pool.module, name, None)
                checked += 1
                if gpu is None or not torch.equal(gpu.data[row].cpu().contiguous(), pinned.contiguous()):
                    bad += 1
                    logger.error("[COLD-POOL] CHECKSUM MISMATCH layer=%d gid=%d row=%d param=%s",
                                 layer_id, gid, row, name)
    logger.info("[COLD-POOL] checksum selfcheck: %d rows %s", checked, "OK" if bad == 0 else f"{bad} MISMATCHES")


def _phase_b(pool, flat):
    """L4: demote starved keep rows, protect sustained-demand rows. Default OFF."""
    dev = flat.device
    if pool.seen_tick is None:
        pool.seen_tick = torch.full((pool.E,), -1e9, dtype=torch.float32, device=dev)
    pool.seen_tick[flat.reshape(-1)] = float(pool.tick)
    starved = (pool.tick - pool.seen_tick[pool.keep_ids_gpu]) > float(degrade_window())
    doomed = pool.keep_ids_gpu[starved.nonzero(as_tuple=True)[0]].tolist()
    st = _STATE["stats"]
    for gid in doomed[:2]:  # 2 demotions / layer / tick — slow churn by design
        row = int(_REMAP_CACHE[pool.layer_id][gid].item())
        if row >= pool.keep_len:
            continue  # already demoted
        rows = {}
        for name in ALL_PARAMS:
            t = getattr(pool.module, name, None)
            if isinstance(t, torch.Tensor) and t.dim() > 0 and row < t.shape[0]:
                rows[name] = t.data[row].to("cpu").contiguous().pin_memory()
        pool.host[gid] = rows
        _mask(pool.layer_id, gid)
        pool.row_gid.pop(row, None)
        pool.row_tick[row] = -1e9  # oldest -> first LRU victim
        st["demoted"] += 1
    for gid, row in list(pool.gid_row.items()):
        if row in pool.protected:
            continue
        if pool.need_counts is not None and float(pool.need_counts[gid].item()) >= 2.0:
            pool.protected.add(row)
            st["promoted"] += 1
            logger.info("[COLD-POOL] promote layer=%d gid=%d row=%d", pool.layer_id, gid, row)


def after_forward_hook():
    """model_runner.forward tail — OUTSIDE graph replay. One-step lookahead staging."""
    if not _STATE["dynamic"]:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    st = _STATE["stats"]
    st["calls"] += 1
    if st["calls"] % 8 != 0 or not consume_dirty():
        return
    alpha = strong_alpha()
    need = float(need_hits())
    cap = max_stage_per_tick()
    k = 10
    try:
        k = _env_int("SGLANG_COLD_TOPK", 10) or 10
    except Exception:
        pass
    staged_now = 0
    for layer_id, pool in _STATE["layers"].items():
        buf = _STASH.get(layer_id)
        if buf is None:
            continue
        ids = buf["ids"]
        flat = ids[ids >= 0]
        n = int(flat.numel())
        pool.tick += 1
        if n < k:
            continue
        n = (n // k) * k
        flat = flat[:n].reshape(-1, k)
        # .clone(): basic slicing returns a VIEW — the buffer fill below would
        # otherwise erase sc (found by T3; strong filter was silently dead)
        sc = buf["scores"][:n].clone().reshape(-1, k)
        ids.fill_(-1)
        buf["scores"].fill_(-1.0)
        is_cold = pool.cold_mask_gpu[flat]
        hot_min = torch.where(~is_cold, sc, torch.full_like(sc, float("inf"))).amin(dim=-1, keepdim=True)
        strong = is_cold & (sc >= alpha * hot_min)
        # v2.8 COLD DEMAND PROBE: masked cold experts can never appear in
        # topk output (bias -inf), so the selected-ids path alone starves the
        # pool forever. Re-check the alpha band on RAW pre-mask logits: any
        # cold expert scoring >= alpha*hot_min in the row is demand.
        lg = buf.get("logits")
        lgn = int(buf.get("logits_n") or 0)
        probe_rows = min(lgn, flat.shape[0])
        if lg is not None and probe_rows > 0:
            with torch.no_grad():
                sc_full = lg[:probe_rows].sigmoid()  # (rows, E) fp32
                rowmin = hot_min[:probe_rows]
                band = sc_full >= alpha * rowmin
                coldband = band & pool.cold_mask_gpu.unsqueeze(0)
                # exclude ids already staged (in gid_row) — mask their columns
                if pool.gid_row:
                    staged_cols = torch.tensor(list(pool.gid_row.keys()), dtype=torch.long, device=flat.device)
                    coldband[:, staged_cols] = False
                if coldband.any():
                    hit = coldband.float()  # (rows, E)
                    vals, argmax_g = hit.mul(sc_full).max(dim=-1)
                    sel_g = argmax_g[vals > 0]
                    if sel_g.numel():
                        pool.need_counts = pool.need_counts if pool.need_counts is not None else torch.zeros(pool.E, dtype=torch.float32, device=flat.device)
                        pool.need_counts.index_add_(0, sel_g, torch.ones(sel_g.numel(), dtype=torch.float32, device=flat.device))
                        st["strong"] += int(sel_g.numel())
        if pool.need_counts is None:
            pool.need_counts = torch.zeros(pool.E, dtype=torch.float32, device=flat.device)
        pool.need_counts.mul_(ema_decay())
        sel = flat[strong]
        if sel.numel():
            pool.need_counts.index_add_(0, sel.reshape(-1),
                                        torch.ones(sel.numel(), dtype=torch.float32, device=flat.device))
            st["strong"] += int(sel.numel())
        cands = pool.cold_ids_gpu[pool.need_counts[pool.cold_ids_gpu] >= need]
        if cands.numel():
            order = torch.argsort(pool.need_counts[cands], descending=True)
            for gid in cands[order[:cap]].tolist():
                pool.need_counts[gid] = 0.0
                if gid in pool.gid_row:
                    continue
                row = _pick_row(pool)
                if row is None:
                    break
                if _row_from_host(pool, gid, row):
                    st["staged"] += 1
                    staged_now += 1
                else:
                    _return_row(pool, row)  # v2.4: never leak the slot on failure
        if degrade_enabled():
            _phase_b(pool, flat)
    if staged_now and st["calls"] % 64 < 8:
        logger.info("[COLD-POOL] calls=%d strong=%d staged=%d evicted=%d demoted=%d promoted=%d h2d=%.2fGB",
                    st["calls"], st["strong"], st["staged"], st["evicted"], st["demoted"],
                    st["promoted"], st["h2d_bytes"] / 1e9)
    elif debug_on() and st["calls"] % 128 == 0:  # TEMP probe: was 4096
        # v2.7b: with keep-330 covering all synthetic demand, staged stays 0 and
        # the gated line above never prints -> strong/hits invisible. Unconditional
        # DEBUG cadence so soak can PROVE whether the demand path is dead or idle.
        logger.info("[COLD-POOL] calls=%d strong=%d staged=%d evicted=%d h2d=%.2fGB (idle-cadence)",
                    st["calls"], st["strong"], st["staged"], st["evicted"], st["h2d_bytes"] / 1e9)
    if debug_on() and st["calls"] % 256 == 0:
        _checksum_selfcheck()
