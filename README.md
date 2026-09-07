# Qwen3.8-Flash-Next × NVFP4 KV cache on one RTX PRO 6000 (SM120)

English | **[简体中文](README_zh.md)**

**Patches, deployment configs and calibration data for serving Qwen3.8-Flash-Next (180B MoE, NVFP4 weights) with `--kv-cache-dtype nvfp4` on a single RTX PRO 6000 Blackwell (96 GB, sm120) — QSA sparse attention included. Both shipped schemes are fully tuned: `nvfp4kv` (`--kv-cache-dtype nvfp4`, current mainline, second-round tuned) and `fp8kv` (`fp8_e4m3`, the rollback scheme, carrying the portable half of the same tuning). Each keeps its pre-tuning baseline under `config/baseline/` as rollback anchor and before/after evidence.**

Upstream sglang could not run NVFP4 *KV cache* together with Qwen Sparse Attention (QSA): the Triton gather path receives packed fp4 buffers and dies on `KeyError: 'float4_e2m1fn_x2'`. This repo ships the working fix (a port of the [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) patch, MIT, adapted for the SM120 trtllm-gen sparse-decode path that dspark's SM121 build never reaches), plus everything we learned tuning the result: a sampler OOM fix ([#37962](https://github.com/sgl-project/sglang/issues/37962)-class), the HiCache hard constraint, radix-cache economics under fp4 KV, and a full speculative-decoding steps calibration.

Built on the groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) and [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — both fp8-KV; **this repo is the nvfp4-KV datapoint** they don't have.

## Results (second-round tuned stack)

| | fp8kv scheme | baseline nvfp4kv | **tuned nvfp4kv (current)** |
|---|---|---|---|
| KV cache dtype | fp8_e4m3 | nvfp4 (packed e2m1 + per-block scales) | **nvfp4** |
| Context | 512K (YaRN ×2) | 768K (YaRN ×3) | **768K (YaRN ×3)** |
| KV pool @ tuned | 552,960 | 786,432 | **851,968** (+64K from reclaimed mamba slots) |
| Concurrency | 4 | 6 | **6** |
| Decode C1 | ≈165 tok/s | ~122 tok/s | **≈136 tok/s** |
| Decode C6 aggregate | ≈355–365 tok/s | ~409 tok/s | **≈550–575 tok/s** |
| Prefill | ~11K tok/s | ~10K tok/s | **≈10K tok/s** |
| MTP (NEXTN) steps | 3 | 2 | **2** (calibrated, see below) |
| Radix prefix reuse | on | on | **on — 99.96% hit, 6.3s → 0.6s on a 58K shared prefix** |
| HiCache L2 | **on** (safe under fp8) | off | **off** (mandatory under fp4 KV, see Constraints) |

Quality gates all green on the nvfp4 KV path: NIAH 200K, needle-in-haystack at 6×107K concurrent pool pressure, post-eviction prefix re-query correctness, grammar JSON ×6, tool calls, zero retractions, zero errors. Accept-len ≈2.0–2.3, accept-rate ≈0.5–0.66.

## What changed vs baseline, why, and what it bought

The tuned version differs from the baseline snapshot (`config/baseline/`) on exactly seven points. Aggregate effect: **+11% C1 / +36% C6**; the tuning evidence lives in this table.

| # | Change (baseline → tuned) | Why | Effect |
|---|---|---|---|
| 1 | GDN linear-attn backend: decode-only flashinfer → **flashinfer on both prefill+decode** | Auto-resolution on SM120 falls through an SM100-only check back to triton for prefill; pin explicitly to reach FlashInferGDNKernel | Faster GDN prefill; base for the later wins |
| 2 | `max-mamba-cache-size`: 32 (auto-sized) → **pin at 24** | Measured peak radix usage 0.62 ≈ 20 slots with lazy eviction; surplus slots are dead VRAM. Pennyroyal-class pinning | Freed VRAM converted into KV pool (+64K → 851,968), more session prefixes resident; multi-session C6 capacity up |
| 3 | `--mamba-radix-cache-strategy`: extra_buffer → **extra_buffer_lazy** | Completed-request states linger for prefix reuse but don't need eager extra slots; lazy cuts ~5 → ~4 slots/request | **C6 +38%** (the single biggest win); −3% C1, net positive at concurrency |
| 4 | Prefill CUDA graph: auto → **forced full capture** (`cuda-graph-config.prefill.backend: full`, `max_bs: 4096`) | Breakable×multimodal auto-disable leaves prefill batches eager (#28386-class); forcing full capture avoids it. Capture cost ~0.56 GB / ~30 s one-time | Prefill stable ≈10K tok/s; no eager-path jitter |
| 5 | `speculative-attention-mode`: unset (prefill path) → **decode** | Verify/draft batches are decode-shaped; routing them to trtllm_mha/XQA matches the real batch shape. Does NOT touch real prefill batches (stay on triton) | +2–4% on both C1 and C6; accept distribution unchanged |
| 6 | FR-Spec speculative token map: none → **self-built 64K hot table** (`speculative-token-map: frspec_map_64k.pt`, coverage 1.0) | Draft acceptance improved by seeding from our own corpus (obsidian notes + skill files → 65,536 IDs); map hash enters the cache namespace so old prefixes are auto-isolated | accept len ≈2.0 → 2.0–2.3; caveat: one-time prefix-cache namespace reset after swapping the table (warm 2–3 rounds) |
| 7 | MTP steps: 3 → **2** (with draft=3/topk=1) | steps=3's accept-rate gain doesn't pay for the linear dequant cost under fp4 KV; measured steps=3: C1 drops to ~133–135, accept rate noisy 0.29–0.62 | steps=2 is the sweet spot; carried into baseline too but re-verified here |

Operational note baked into the units: `mamba-radix-cache-strategy` and `ple-offload-embedding` are alias/BooleanOptionalAction args that the YAML ConfigArgumentMerger rejects (`DeprecatedAliasStoreAction`) — they must stay as CLI flags in the systemd `ExecStart`, never in the YAML.

## The fp8kv scheme (rollback stack)

fp8kv is not the untouched old config — on 2026-09-08 it absorbed the portable half of the nvfp4kv tuning, so the two schemes swap on one GPU without losing the generic wins.

| Item | fp8kv state | Why |
|---|---|---|
| GDN flashinfer both ends | ported | Backend routing, independent of KV dtype |
| Prefill CUDA graph forced full | ported | Same eager-jitter avoidance (#28386-class) |
| SAM=decode | ported | Pure backend routing for verify/draft batches |
| FR-Spec 64K hot map | ported, same `.pt` | The map is tokenizer-scoped, not KV-dtype-scoped |
| extra_buffer_lazy (CLI-only alias) | ported | Same slot economics |
| MTP steps | kept **3** (nvfp4kv: 2) | fp8 accept-len runs 2.08–2.50 — the deeper chain still pays |
| KV pool | kept **552,960** | nvfp4kv's +64K is bought by fp4 halving KV bytes; fp8 has no equivalent headroom |
| HiCache L2 | kept **ON** | Works under fp8; nvfp4kv requires it OFF (#36121) |

Hot-state after the port (two-pass, second run): C1 ≈165 / C6 aggregate ≈355–365 tok/s. An earlier single-pass ~200 C1 reading did not reproduce post-port — trust the two-pass numbers. Note the crossover: fp8kv now beats nvfp4kv on C1 (≈165 vs ≈136) while losing on aggregate concurrency and prefix-cache depth. Pick fp8kv for latency-sensitive single-stream work; pick nvfp4kv for gateway-style concurrent traffic. Same port, same served-model-name — swapping = stop one unit, start the other. Rollback anchor: the fp8kv pair in `config/baseline/`.

## Why nvfp4 KV

The fp4 KV pool halves KV memory (packed e2m1 + tiny per-block scales), which on a 96 GB card is the difference between 512K and 768K context at conc 6 — the weights, the ~44 GB pinned PLE n-gram table and MTP graphs eat everything else. The single-stream decode tax is inherent to gather-dequant (two Triton launches + one dequant kernel per step); the tuned stack lifts C1 from ~122 to ≈136, and at concurrency it's a clear net win (C6 ≈550–575 tok/s).

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
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV scheme — tuned rollback stack (conc4 / 512K / MTP3 / mamba24 / HiCache ON; portable items ported 2026-09-08: GDN dual-end flashinfer, CG-full prefill, SAM=decode, FR-Spec map, ABL=lazy — steps kept 3, accept-len 2.08-2.50 favors depth under fp8 batch shapes)
  baseline/                         Pre-tuning BASELINE snapshots for BOTH schemes, kept as
                                    rollback points and before/after evidence:
                                    nvfp4kv — mamba32 auto-sized, KV pool 786432, no FR-Spec map,
                                              SAM unset, extra_buffer.
                                    fp8kv   — pre-port state (2026-09-08 tuning port excluded):
                                              GDN prefill still triton, no CG-full, no SAM, no
                                              FR-Spec map, extra_buffer (non-lazy).
  systemd/                          units per scheme + warmups (same port, same model name,
                                    mutually exclusive — stop one, start the other)
scripts/
  frspec_map_64k.pt           FR-Spec speculative token-map artifact (65,536 IDs; sha256
                              598b0dc4… matches the manifest; shipped so the tuned stack is reproducible)
  frspec_map_64k.manifest.json  build provenance: tokenizer sha256, corpus file list with per-file
                              sha256, coverage 1.0, size/base_count/special-ids
  build_token_map.py            rebuild the map from corpus + model tokenizer
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
6. **Mamba slot economics flip with radix.** Radix OFF: ~1 slot/request (conc6 fits in 16). Radix ON: completed-request states linger for prefix reuse → ~3 slots/request; with extra_buffer_lazy that drops to ~4/request and 24 pinned slots suffice. The knobs are coupled — tune them together.
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

The nvfp4 KV path is gated entirely by `kv-cache-dtype: nvfp4` — flip it back to `fp8_e4m3` (use the fp8kv scheme files) and every patch in here is inert. To run the fp8kv scheme instead, start its own unit pair (`sglang-dealignai-qwen4exp-fp8kv` + warmup) after stopping the nvfp4kv one — see §The fp8kv scheme for the expected values.

## Constraints & honest caveats

- Finalized tuned stack (2026-09-08): GDN flashinfer both ends, mamba pinned 24 + KV pool 851968, prefill CG forced full, FR-Spec 64K map (coverage 1.0), NEXTN steps=2/draft=3/topk=1, extra_buffer_lazy (CLI-only alias), SAM=decode. Hot-state: C1≈136 / C6≈550–575 / prefill≈10K / accept len≈2.0–2.3. Rollback: copy `config/baseline/` files over the current ones and restart the unit.
- Finalized fp8kv stack (same date): portable items ported (§The fp8kv scheme), steps kept 3, HiCache ON, conc4 / YaRN×2 512K / KV pool 552,960, FR-Spec map shared. Hot-state: C1≈165 / C6≈355–365 / accept len≈2.08–2.50. Rollback = its own `config/baseline/` pair over the current files.
- Single consumer GPU + 44 GB pinned PLE table: the nvfp4kv and fp8kv schemes are mutually exclusive; expect ~4 min cold start. The unit deliberately ships unenabled to avoid boot-time GPU contention.
- Decode at C1 still trails fp8 (~136 vs ~165 tok/s post-port — inherent to gather-dequant today). Upstream native fp4 QSA decode pools (#37798) would remove the gap.
- `Restart=no` on the experiment unit is deliberate — preserve the crash scene.
- Numbers are from one RTX PRO 6000 (96 GB), sglang `0.5.19.dev6+g78c5024e` + local patches. SM121/GB10 is a different story (see gabrielolympie's notes on silent long-context corruption there).
- Swapping the FR-Spec token map resets the prefix-cache namespace once — warm 2–3 rounds after any table change.

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — `qwen4-main-squashed` branch (PR #36497, #36806)
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) (MiaAI Lab, MIT) — the dspark QSA×nvfp4 patch this repo ports and extends to SM120
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — sm120 groundwork
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — the fp8-KV performance ceiling this repo's KV trade-off is measured against

## License

Apache-2.0 (matches the sglang base these patches apply to). The dspark-derived patcher retains MIT attribution to MiaAI Lab in its header. See [LICENSE](LICENSE).
