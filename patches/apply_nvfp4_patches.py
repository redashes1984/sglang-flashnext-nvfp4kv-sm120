#!/usr/bin/env python3
"""Apply dspark-style NVFP4-KV QSA patches to our sglang source build.

Ported from Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks apply_nvfp4_patches.py
(MIT License, MiaAI Lab), adapted for our PR#36497+78c5024e build:
  * extra call site: _forward_trtllm_sparse (SM120 trtllm-gen sparse decode,
    which dspark's SM121 build never reaches)
  * no server_args patch needed (our build already allows nvfp4 on SM120)
  * no SM121 fp8 tl.dot patch needed (we dequant to bf16 before the dots)
All patches are inert unless --kv-cache-dtype nvfp4.
"""
import ast
import os
import pathlib
import sys

SRT = pathlib.Path(os.environ.get("SGLANG_SRT", "/opt/sglang-src/sglang/python/sglang/srt"))
MARKER = "qsa_nvfp4_kv"

def patch(path, replacements):
    s = path.read_text()
    if MARKER in s:
        print(f"{path.name}: already patched")
        return
    for anchor, replacement in replacements:
        count = s.count(anchor)
        assert count == 1, f"{path.name}: anchor matched {count} times (want 1):\n{anchor[:200]}"
        s = s.replace(anchor, replacement, 1)
    ast.parse(s)
    path.write_text(s)
    print(f"{path.name}: patched")

BACKEND = SRT / "layers" / "attention" / "qwen_sparse_attn_backend.py"

IMPORT_ANCHOR = """from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_valid_counts_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
"""
IMPORT_REPLACEMENT = IMPORT_ANCHOR + (
    "from sglang.srt.layers.attention import qsa_nvfp4_kv  # dspark: NVFP4 KV cache\n"
)

PAGED_HEAD_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""
PAGED_HEAD_REPLACEMENT = """        pool = self.token_to_kv_pool
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is None:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
        else:
            k_buffer = v_buffer = None
        if not q.is_cuda:
            metadata = self._resolve_metadata(forward_batch)
            if fp4_kv is not None:
                k_buffer = pool.get_key_buffer(layer.layer_id)
                v_buffer = pool.get_value_buffer(layer.layer_id)
            slots = self._logical_to_physical(topk_indices, metadata)
            output = qsa_sparse_attention(q, k_buffer, v_buffer, slots, layer.scaling)
            return output.reshape(q.shape[0], -1)
"""

# --- our SM120 addition: trtllm-gen sparse decode path ---
TRTLLM_HEAD_ANCHOR = """        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
"""
TRTLLM_HEAD_REPLACEMENT = """        fp4_kv = qsa_nvfp4_kv.try_fp4_view(self.token_to_kv_pool, layer.layer_id)
        batch, topk = topk_indices.shape
        page = _TRTLLM_SPARSE_PAGE_SIZE
"""

TRTLLM_EXTRACT_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            max(capacity_rows, batch) * stride,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_strided,
            packed_k,
            packed_v,
            batch,
            topk,
        )
        num_kv_heads = k_buffer.shape[1]
        head_dim = k_buffer.shape[2]
"""
TRTLLM_EXTRACT_REPLACEMENT = """        if fp4_kv is not None:
            # dspark NVFP4: gather-dequant to BF16 rows; trtllm-gen decode
            # then consumes the packed scratch as a bf16 paged cache.
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                max(capacity_rows, batch) * stride,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                batch,
                topk,
            )
            num_kv_heads = fp4_kv.head_num
            head_dim = fp4_kv.head_dim
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                max(capacity_rows, batch) * stride,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_strided,
                packed_k,
                packed_v,
                batch,
                topk,
            )
            num_kv_heads = k_buffer.shape[1]
            head_dim = k_buffer.shape[2]
"""

# --- dspark varlen fallback path ---
EXTRACTION_ANCHOR = """        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
        qwen_sparse_kv_extraction_compact_triton(
            k_buffer,
            v_buffer,
            self.req_to_token_pool.req_to_token,
            (
                metadata.row_req_pool_indices
                if metadata.row_req_pool_indices is not None
                else forward_batch.req_pool_indices
            ),
            topk_indices,
            sequence_lens,
            cu_seqlens_k,
            packed_k,
            packed_v,
            batch,
            topk,
        )
"""
EXTRACTION_REPLACEMENT = """        if fp4_kv is not None:
            packed_k, packed_v = qsa_nvfp4_kv.compact_and_dequant(
                self,
                fp4_kv,
                scratch_capacity,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                batch,
                topk,
            )
        else:
            packed_k, packed_v = self._get_fa2_scratch(
                scratch_capacity,
                k_buffer.shape[1],
                k_buffer.shape[2],
                k_buffer.dtype,
                k_buffer.device,
            )
            qwen_sparse_kv_extraction_compact_triton(
                k_buffer,
                v_buffer,
                self.req_to_token_pool.req_to_token,
                (
                    metadata.row_req_pool_indices
                    if metadata.row_req_pool_indices is not None
                    else forward_batch.req_pool_indices
                ),
                topk_indices,
                sequence_lens,
                cu_seqlens_k,
                packed_k,
                packed_v,
                batch,
                topk,
            )
"""

EXTEND_CHUNK_ANCHOR = """        pool = self.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        k_parts = [
            k_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
        v_parts = [
            v_buffer.index_select(
                0, req_to_token[req_indices[i], : sequence_lens[i]].long()
            )
            for i in range(len(sequence_lens))
        ]
"""
EXTEND_CHUNK_REPLACEMENT = """        pool = self.token_to_kv_pool
        req_to_token = self.req_to_token_pool.req_to_token
        req_indices = forward_batch.req_pool_indices.tolist()
        fp4_kv = qsa_nvfp4_kv.try_fp4_view(pool, layer.layer_id)
        if fp4_kv is not None:
            k_cat, v_cat = qsa_nvfp4_kv.gather_history_fp4(
                fp4_kv, req_to_token, req_indices, sequence_lens
            )
        else:
            k_buffer = pool.get_key_buffer(layer.layer_id)
            v_buffer = pool.get_value_buffer(layer.layer_id)
            k_parts = [
                k_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            v_parts = [
                v_buffer.index_select(
                    0, req_to_token[req_indices[i], : sequence_lens[i]].long()
                )
                for i in range(len(sequence_lens))
            ]
            k_cat = torch.cat(k_parts)
            v_cat = torch.cat(v_parts)
"""

EXTEND_CK_ANCHOR = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            torch.cat(k_parts),
            torch.cat(v_parts),
"""
EXTEND_CK_REPLACEMENT = """        output = sparse_gqa_fwd_interface_triton_ck(
            q.contiguous(),
            k_cat,
            v_cat,
"""

FP4_METHOD = SRT / "layers" / "quantization" / "fp4_kv_cache_quant_method.py"
FP4_METHOD_ANCHOR = """    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""
FP4_METHOD_REPLACEMENT = """    if name == "nvfp4":
        # dspark: QSA models consume the NVFP4 pool via plain BF16 dequant
        # reads - no FP8 dequant workspace, no native-FP4 decode.
        from sglang.srt.layers.attention.qsa_nvfp4_kv import QSANVFP4KVCacheMethod

        return QSANVFP4KVCacheMethod(**kwargs)
    return KV_CACHE_QUANT_REGISTRY[name](**kwargs)
"""

POOL_CFG = SRT / "model_executor" / "pool_configurator.py"
POOL_CFG_ANCHOR = """                # FP4 prefill uses one shared FP8 dequant workspace across layers.
                cell_size += n * k * 2 * kv_size
"""
POOL_CFG_REPLACEMENT = """                # FP4 prefill uses one shared FP8 dequant workspace across
                # layers - except on the QSA path (dspark qsa_nvfp4_kv),
                # whose method allocates no FP8 workspace.
                _is_qsa_kv4 = False
                try:
                    from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

                    _is_qsa_kv4 = is_qwen_qsa(model_config.hf_config)
                except Exception:
                    pass
                if not _is_qsa_kv4:
                    cell_size += n * k * 2 * kv_size
"""

POOL = SRT / "mem_cache" / "memory_pool.py"
QUANT_SCALES_ANCHOR = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        if k_scale is None and hasattr(self.quant_method, "k_scales_gpu"):
"""
QUANT_SCALES_REPLACEMENT = """    def _quantized_scales(self, global_layer_id: int, k_scale, v_scale):
        # dspark qsa_nvfp4_kv: HybridTokenToKVPool.set_kv_buffer defaults
        # k_scale=v_scale=1.0 (Python floats). That skips the on-device
        # per-layer NVFP4 global scales, and NVFP4KVQuantizeUtil.quantize
        # then does torch.tensor(..., device=cuda) - illegal during CUDA
        # graph capture. Host scalars are treated as unset on nvfp4 only.
        use_gpu = hasattr(self.quant_method, "k_scales_gpu")
        nvfp4 = getattr(self.quant_method, "name", None) == "nvfp4"
        host_scalar = nvfp4 and not (
            torch.is_tensor(k_scale) and k_scale.is_cuda
        )
        if use_gpu and (k_scale is None or host_scalar):
"""

def main():
    patch(BACKEND, [
        (IMPORT_ANCHOR, IMPORT_REPLACEMENT),
        (PAGED_HEAD_ANCHOR, PAGED_HEAD_REPLACEMENT),
        (TRTLLM_HEAD_ANCHOR, TRTLLM_HEAD_REPLACEMENT),
        (TRTLLM_EXTRACT_ANCHOR, TRTLLM_EXTRACT_REPLACEMENT),
        (EXTRACTION_ANCHOR, EXTRACTION_REPLACEMENT),
        (EXTEND_CHUNK_ANCHOR, EXTEND_CHUNK_REPLACEMENT),
        (EXTEND_CK_ANCHOR, EXTEND_CK_REPLACEMENT),
    ])
    patch(FP4_METHOD, [(FP4_METHOD_ANCHOR, FP4_METHOD_REPLACEMENT)])
    patch(POOL_CFG, [(POOL_CFG_ANCHOR, POOL_CFG_REPLACEMENT)])
    patch(POOL, [(QUANT_SCALES_ANCHOR, QUANT_SCALES_REPLACEMENT)])
    print("ALL-PATCHED")

if __name__ == "__main__":
    sys.exit(main())
