# Qwen3.8-Flash-Next × NVFP4 KV cache on one RTX PRO 6000 (SM120)

English | **[简体中文](README_zh.md)**

**Patches, dual-scheme deployment configs and MTP calibration data for serving Qwen3.8-Flash-Next (180B MoE, NVFP4 weights) with `--kv-cache-dtype nvfp4` on a single RTX PRO 6000 Blackwell (96 GB, sm120) — QSA sparse attention included.**

Upstream sglang could not run NVFP4 *KV cache* together with Qwen Sparse Attention (QSA): the Triton gather path receives packed fp4 buffers and dies on `KeyError: 'float4_e2m1fn_x2'`. This repo ships the working fix (a port of the [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) patch, MIT, adapted for the SM120 trtllm-gen sparse-decode path that dspark's SM121 build never reaches), plus everything we learned tuning the result: a sampler OOM fix ([#37962](https://github.com/sgl-project/sglang/issues/37962)-class), the HiCache hard constraint, radix-cache economics under fp4 KV, and a full speculative-decoding steps calibration.

Built on the groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) and [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — both fp8-KV; **this repo is the nvfp4-KV datapoint** they don't have.

## Results

| | fp8kv scheme | **nvfp4kv scheme** |
|---|---|---|
| KV cache dtype | fp8_e4m3 | **nvfp4 (packed e2m1 + per-block scales)** |
| Context | 512K (YaRN ×2) | **768K (YaRN ×3)** |
| KV pool @ 786432 tokens | 9.4 GB (fp8, cannot fit) | **~5.1 GB** |
| Concurrency | 4 | **6** |
| Decode C1 | ~200 tok/s | 136 tok/s (finalized config, see below) |
| Decode C6 aggregate | — | **550–575 tok/s** |
| Prefill | ~11K tok/s | ~11K tok/s |
| MTP (NEXTN) steps | 3 | 2 (calibrated, see below) |
| Radix prefix reuse | on | **on — 99.96% hit, 6.3s → 0.6s on a 58K shared prefix** |
| HiCache L2 | on | **must be OFF** (see Constraints) |

Quality gates all green on the nvfp4 KV path: NIAH 200K, needle-in-haystack at 6×107K concurrent pool pressure (645K/786K tokens), post-eviction prefix re-query correctness, grammar JSON ×6, tool calls, zero retractions, zero errors.

## Why nvfp4 KV

The fp4 KV pool halves KV memory (packed e2m1 + tiny per-block scales), which on a 96 GB card is the difference between 512K and 768K context at conc 6 — the weights, the ~44 GB pinned PLE n-gram table and MTP graphs eat everything else. The single-stream decode tax is the price (gather-dequant adds two Triton launches + one dequant kernel per step) — early builds measured −34%, the finalized stack narrows it to ~−32% at a higher absolute bar (C1 ≈136); at concurrency it amortizes to a net capacity win (C6 550–575 tok/s).

## Contents

```
patches/
  apply_nvfp4_patches.py     idempotent patcher: QSA×NVFP4-KV support for sglang
                             (anchor-count==1 checks + ast.parse + py_compile)
  0001-sampler-37962-tp1-sync-skip.diff
                             TP=1 sampler fix: skip the cross-TP token sync whose
                             first NCCL op lazily allocates 512 MB → OOM on a full
                             pool (grammar requests only). Crash root-caused live.
config/
  dealignai-qwen4exp-nvfp4kv.yaml   nvfp4 KV scheme — SECOND-ROUND TUNED (current): conc6 / 768K /
                                    MTP2 / mamba24 pinned / KV pool 851968 / prefill CG full /
                                    FR-Spec token map / SAM=decode / extra_buffer_lazy (CLI-only alias)
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV scheme (conc4 / 512K / MTP3 / mamba24 / HiCache ON)
  baseline/                         nvfp4kv BASELINE snapshot (pre second-round tuning): mamba32 auto-sized,
                                    KV pool 786432, no FR-Spec map, SAM unset, extra_buffer. Kept as the
                                    rollback point and the tuning before/after evidence.
  systemd/                          units per scheme + warmups (same port, same model name,
                                    mutually exclusive — stop one, start the other)
scripts/
  warmup_qsa_nvfp4kv.py       closes the late-device-load OOM window (long prefill,
                              grammar bitmask, bs6 graph buckets) before real traffic
  warmup_qsa_fp8kv.py         fp8 variant (prefix-cache hit path)
docs/
  RESULTS.md                  full stress matrix, MTP steps 1–3 calibration table,
                              mamba slot economics, every dead-end and root cause
```

## Key findings (the short version)

1. **QSA × nvfp4 KV needs a patch.** Upstream declares nvfp4 access as FlashInfer-prefill / TRT-LLM-native-decode; QSA's Triton gather path is neither. The patcher routes QSA to PLAIN BF16 access and gather-dequants (packed data + scale buffers, both viewed as uint8) before the dots. CUDA-graph safe, inert unless `--kv-cache-dtype nvfp4`.
2. **HiCache × fp4 KV = silent corruption.** Host transfer moves packed data buffers but not the per-block scale buffers; eviction/load-back then serves garbage that *looks* plausible. sglang PR #36121 proposes the fail-fast; until it lands, `enable-hierarchical-cache: false` is mandatory.
3. **Radix cache × fp4 KV is safe on current builds** — `_slot_move_pointer_buffers` already moves `k/v_scale_buffer` alongside the data. Verified with a 553K-token eviction-pressure probe: needles stay correct after forced eviction and re-hit.
4. **TP=1 sampler OOM (fixed here).** With grammars enabled the sampler runs a cross-TP token sync even at world_size=1; the first NCCL op allocates 512 MB lazily — fatal when the pool is 99.3% full. One-line guard, upstream #37962.
5. **MTP steps must be re-calibrated per KV dtype.** Under fp4 KV, steps=2 beats both 1 and 3 (dequant cost scales with verify tokens; steps=3's accept-rate gain doesn't pay for it). fp8-era "more steps is better" does not transfer.
6. **Mamba slot economics flip with radix.** Radix OFF: ~1 slot/request (conc6 fits in 16). Radix ON: completed-request states linger for prefix reuse → ~3 slots/request, 32 needed. The two knobs are coupled.
7. **Late Triton device-loads are a real OOM class.** First-touch kernel specializations (GDN chunk prefill, sparse-GQA, xgrammar bitmask) allocate *after* the pool is full. Warm up every shape family before traffic; extreme shapes can still trigger new specializations — monitor `device-loaded` in the journal.

## Reproduce

```bash
# 0. sglang built from the qwen4-main-squashed branch (PR #36497 head) for sm120,
#    e.g. CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0" — see jpezzulli/gabrielolympie repos.
export SGLANG_SRT=/opt/sglang-src/sglang/python/sglang/srt

# 1. patches (idempotent; re-run after any source sync)
python3 patches/apply_nvfp4_patches.py
git -C /opt/sglang-src/sglang apply ../patches/0001-sampler-37962-tp1-sync-skip.diff

# 2. config — edit MODEL_DIR in the unit, drop YAMLs in place
systemctl start sglang-dealignai-qwen4exp-nvfp4kv
systemctl enable --now sglang-dealignai-nvfp4kv-warmup   # one-shot, waits for /health

# 3. verify
curl -s localhost:8000/v1/models          # → Qwen3.8-Flash-Next-NVFP4
curl -s localhost:8000/get_server_info | jq '.kv_cache_dtype, .disable_radix_cache'
```

The nvfp4 KV path is gated entirely by `kv-cache-dtype: nvfp4` — flip it back to `fp8_e4m3` (use the fp8kv scheme files) and every patch in here is inert.

## Constraints & honest caveats

- Finalized nvfp4kv stack (2026-09-08): GDN flashinfer both prefill+decode, mamba slots pinned at 24 + KV pool 851968, prefill CUDA graph forced full (max_bs 4096), FR-Spec 64K token map rebuilt on own corpus (coverage 1.0), MTP steps=2, `--mamba-radix-cache-strategy=extra_buffer_lazy` (CLI-only, alias arg), `speculative-attention-mode: decode`. Hot-state: C1≈136 / C6≈550–575 / prefill≈10K tok/s / accept len≈2.0–2.3.
- **Two recorded nvfp4kv versions** (both in this repo, both Apache-2.0): baseline in `config/baseline/` (mamba32 auto-sized, KV pool 786432, extra_buffer, no FR-Spec/SAM/CG-full — hot-state C1≈122 / C6≈409), second-round tuned above (current, in `config/`). The gap between them is the tuning evidence: lazy slot strategy + pinned mamba cap + decode-path spec attention + rebuilt token map ≈ +11% C1 / +36% C6, at the cost of a one-time prefix-cache namespace reset after the token-map swap. Roll back by copying the baseline files over the current ones and restarting the unit.
- Single consumer GPU + 44 GB pinned PLE table: the two schemes are mutually exclusive; expect ~4 min cold start.
- Decode at C1 still trails fp8 (gather-dequant is inherent to the approach today; the finalized stack closes most of the gap: 136 vs ~200 tok/s). Upstream native fp4 QSA decode pools (#37798) would remove it.
- `Restart=no` on the experiment unit is deliberate — preserve the crash scene.
- Numbers are from one RTX PRO 6000 (96 GB), sglang `0.5.19.dev6+g78c5024e` + local patches. SM121/GB10 is a different story (see gabrielolympie's notes on silent long-context corruption there).

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — `qwen4-main-squashed` branch (PR #36497, #36806)
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) (MiaAI Lab, MIT) — the dspark QSA×nvfp4 patch this repo ports and extends to SM120
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — sm120 groundwork
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — the fp8-KV performance ceiling this repo's KV trade-off is measured against

## License

Apache-2.0 (matches the sglang base these patches apply to). The dspark-derived patcher retains MIT attribution to MiaAI Lab in its header. See [LICENSE](LICENSE).
