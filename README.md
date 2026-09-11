# Qwen3.8-Flash-Next × NVFP4 KV cache on one RTX PRO 6000 (SM120)

English | **[简体中文](README_zh.md)**

**Patches, deployment configs and calibration data for serving Qwen3.8-Flash-Next (180B MoE, NVFP4 weights) with `--kv-cache-dtype nvfp4` on a single RTX PRO 6000 Blackwell (96 GB, sm120) — QSA sparse attention included. Both shipped schemes are fully tuned: `nvfp4kv` (`--kv-cache-dtype nvfp4`, current mainline; the live instance runs the round-5 frozen stack: expert cold pool + per-path checkpoint cap) and `fp8kv` (`fp8_e4m3`, the rollback scheme, carrying the portable half of the same tuning). Each keeps its pre-tuning baseline under `config/baseline/` as rollback anchor and before/after evidence.**

Upstream sglang could not run NVFP4 *KV cache* together with Qwen Sparse Attention (QSA): the Triton gather path receives packed fp4 buffers and dies on `KeyError: 'float4_e2m1fn_x2'`. This repo ships the working fix (a port of the [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) patch, MIT, adapted for the SM120 trtllm-gen sparse-decode path that dspark's SM121 build never reaches), plus everything we learned tuning the result: a sampler OOM fix ([#37962](https://github.com/sgl-project/sglang/issues/37962)-class), the HiCache hard constraint, radix-cache economics under fp4 KV, and a full speculative-decoding steps calibration.

Built on the groundwork of [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) and [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — both fp8-KV; **this repo is the nvfp4-KV datapoint** they don't have.

## Results (stack generations: baseline → round-2 tuned → nvfp4kv-1m trial → expert cold pool → round-5 freeze)

| | fp8kv scheme | baseline nvfp4kv | tuned nvfp4kv (round 2) | nvfp4kv-1m trial (round 3) | **frozen stack (round 4 cold pool + round 5 cache fixes, live)** |
|---|---|---|---|---|---|
| KV cache dtype | fp8_e4m3 | nvfp4 (packed e2m1 + per-block scales) | nvfp4 | nvfp4 | **nvfp4** |
| Context | 512K (YaRN ×2) | 768K (YaRN ×3) | 768K (YaRN ×3) | 1M (YaRN ×4 explicit) | **1M (YaRN ×4 explicit)** |
| KV pool | 552,960 | 786,432 | 851,968 (+64K reclaimed mamba slots) | 1,179,648 (after funding int8 ckpt pool + boot headroom) | **2,359,296** (keep330+16 cold pool frees ~30 GB VRAM → +1.18M tokens) |
| Concurrency | 4 | 6 | 6 | 6 | **12** (mamba 64 = 4/req × 12 + 16 anchor slack) |
| Decode C1 | ≈165 tok/s | ~122 tok/s | ≈136 tok/s | ≈75–90 tok/s warm-state | **≈122–129 tok/s** (spec-on steps=2, cold pool live; post-warmup acceptance bench 2026-09-10) |
| Decode C6 aggregate | ≈355–365 tok/s | ~409 tok/s | ≈550–575 tok/s | ≈72–91 tok/s | **≈704–710 tok/s** at C12 (same acceptance harness) |
| Prefill | ~11K tok/s | ~10K tok/s | ≈10K tok/s | ≈9.7K tok/s fresh-prefix; higher on radix hits | **≈9.7K tok/s** fresh-prefix; higher on radix hits |
| MTP (NEXTN) steps | 3 | 2 | 2 (calibrated sweet spot) | OFF (steps≥1 OOMs at 1M on one GPU) | **ON, steps=2** (cold pool's freed VRAM funds spec again) |
| Routed experts on GPU | all 512 | all 512 | all 512 | all 512 | **346/512** (330 keep + 16 dynamic slots; 166 cold experts in 24.15 GB pinned host, demand-staged) |
| Int8 mamba ckpt pool | — | — | — | ON (patch 0002 × PLE mirrors, PR #38619) | **ON** (128 slots / 3.74 GB; per-path cap `mamba-max-states-per-path 16`, round 5) |
| Radix prefix reuse | on | on | on — 99.96% hit, 6.3s → 0.6s on a 58K shared prefix | on | **on** — 2×549K long chains coexist, re-ask 0.9 s full hit (round 5) |
| HiCache L2 | **off** (fail-fast vs int8 mamba ckpt, `server_args.py:6333`) | off | off | off (mandatory under fp4 KV, see Constraints) | **off** (mandatory under fp4 KV, see Constraints) |

Quality gates all green on the nvfp4 KV path: NIAH 200K, needle-in-haystack at 6×107K concurrent pool pressure, post-eviction prefix re-query correctness, grammar JSON ×6, tool calls, zero retractions, zero errors. Round-4 adds: 8448-row staging checksum byte-exact under CUDA graphs, 30-min acceptance soaks, 12×66K long-context concurrent stress — zero MISMATCH / stage_fail / OOM. Accept-len ≈2.0–2.6, accept-rate ≈0.5–0.8.

## What changed vs baseline, why, and what it bought

The tuned version differs from the baseline snapshot (`config/baseline/`) on eight points — items 1–7 are the round-2 tuning, item 8 is the round-3 nvfp4kv-1m trial layer on top. Aggregate effect of items 1–7: **+11% C1 / +36% C6**; the tuning evidence lives in this table.

| # | Change (baseline → tuned) | Why | Effect |
|---|---|---|---|
| 1 | GDN linear-attn backend: decode-only flashinfer → **flashinfer on both prefill+decode** | Auto-resolution on SM120 falls through an SM100-only check back to triton for prefill; pin explicitly to reach FlashInferGDNKernel | Faster GDN prefill; base for the later wins |
| 2 | `max-mamba-cache-size`: 32 (auto-sized) → **pin at 24** | Measured peak radix usage 0.62 ≈ 20 slots with lazy eviction; surplus slots are dead VRAM. Pennyroyal-class pinning | Freed VRAM converted into KV pool (+64K → 851,968), more session prefixes resident; multi-session C6 capacity up |
| 3 | `--mamba-radix-cache-strategy`: extra_buffer → **extra_buffer_lazy** | Completed-request states linger for prefix reuse but don't need eager extra slots; lazy cuts ~5 → ~4 slots/request | **C6 +38%** (the single biggest win); −3% C1, net positive at concurrency |
| 4 | Prefill CUDA graph: auto → **forced full capture** (`cuda-graph-config.prefill.backend: full`, `max_bs: 4096`) | Breakable×multimodal auto-disable leaves prefill batches eager (#28386-class); forcing full capture avoids it. Capture cost ~0.56 GB / ~30 s one-time | Prefill stable ≈10K tok/s; no eager-path jitter |
| 5 | `speculative-attention-mode`: unset (prefill path) → **decode** | Verify/draft batches are decode-shaped; routing them to trtllm_mha/XQA matches the real batch shape. Does NOT touch real prefill batches (stay on triton) | +2–4% on both C1 and C6; accept distribution unchanged |
| 6 | FR-Spec speculative token map: none → **self-built 64K hot table** (`speculative-token-map: frspec_map_64k.pt`, coverage 1.0) | Draft acceptance improved by seeding from our own corpus (obsidian notes + skill files → 65,536 IDs); map hash enters the cache namespace so old prefixes are auto-isolated | accept len ≈2.0 → 2.0–2.3; caveat: one-time prefix-cache namespace reset after swapping the table (warm 2–3 rounds) |
| 7 | MTP steps: 3 → **2** (with draft=3/topk=1) | steps=3's accept-rate gain doesn't pay for the linear dequant cost under fp4 KV; measured steps=3: C1 drops to ~133–135, accept rate noisy 0.29–0.62 | steps=2 is the sweet spot; carried into baseline too but re-verified here |
| 8 | *(round-3 layer)* int8 mamba checkpoint pool: OFF (upstream ValueError-guarded against PLE side states) → **ON** via patch 0002 PLE mirrors | At 1M ctx the BF16 ckpt slots are the memory frontier; mirrors cost <10 MB/slot vs ~27 MB/slot temporal — see §Third round and PR #38619 | ckpt pool 48 slots / 1.42 GB funded by pool 1441792→1179648; soak 326/326; eviction headroom for many distinct long prefixes |
| 8b | *(round-5 layer)* `mamba-max-states-per-path`: -1 (uncapped) → **16** | the int8 pool is 128 slots; under extra_buffer_lazy chunked prefill donates **1 ckpt per 4096 tok** → one 549K chain ≈134 slots → two chains mutually LRU-evict each other's checkpoints → `cached=0` (see §Fifth round) | 2×549K and 3×495K re-asks 89 s → 0.9–1.5 s full hit; C12 unchanged; zero VRAM cost |

Operational note baked into the units: `mamba-radix-cache-strategy` and `ple-offload-embedding` are alias/BooleanOptionalAction args that the YAML ConfigArgumentMerger rejects (`DeprecatedAliasStoreAction`) — they must stay as CLI flags in the systemd `ExecStart`, never in the YAML.

## Third round (2026-09-09): int8 mamba checkpoint × PLE side-states, enabled

Context: with a 1M-context nvfp4 pool the mamba BF16 checkpoint slots are the next memory frontier. Upstream's `maybe_init_int8_mamba_checkpoint_pool` **refused** to coexist with Qwen4-Exp PLE side states (`raise ValueError` when `int8_ckpt_pool` and `ShortConvPool`/`NGramPool` are both active) — the int8 pool frees the BF16 active slot after donating its state, but the PLE pools index their rows by that slot, so donated checkpoints would orphan the side states. No upstream issue/PR covered this combination at the time (searched via `gh search`).

What `patches/0002-mamba-ckpt-ple.diff` does instead of the guard:

1. `MambaCheckpointPool.__init__` gains `ple_side_states` — builds one mirror buffer per enabled side-state pool (ShortConv → bf16 rows, NGram → int64 rows), slot-axis sized `ckpt_slots + 1`, exact copies (quantization stays on the temporal state only).
2. `store_from_active` / `load_to_active` carry the mirror rows alongside temporal/conv state; `clear()` intentionally keeps mirrors alive across flush (same discipline as qdata).
3. `estimate_mem_usage_bytes` gains `ple_extra_bytes`; overhead measured <10 MB/slot vs ~27 MB/slot for temporal.
4. `memory_pool.py` replaces the ValueError with a real `ple_side_states=[...]` pass-through.

Enablement on this machine: `PYTHONPATH=/opt/sglang-patch/sglang/python` overlay + `--enable-int8-mamba-checkpoint` appended to the unit's `ExecStart`. The ckpt pool (48 slots, 1.42 GB: qdata 1.29 + scale 0.02 + conv 0.10 + ple mirrors 0.01) is funded **outside** the token-accounting formula, so `max-total-tokens` came down 1441792 → 1310720 → **1179648** to keep ~3.2 GB boot headroom. First soak at pool=1310720 failed mid-run with a genuine `torch.OutOfMemoryError` in the GDN extend path (late-load JIT + 6-concurrency activations on top of the ckpt pool) — the second drop to 1179648 fixed it. Second gotcha: after restarting the main service, `systemctl start` on the warmup oneshot can silently no-op if the unit already recorded a `Finished` state — use `systemctl restart sglang-dealignai-nvfp4kv-warmup` and confirm `[warmup-fp4kv] warmup complete` in its journal.

Soak result on the final stack (`scripts/test_ckpt_ple_patch.py` is the CPU integration test, T1–T6 incl. the factory path that caught a kwargs-leak TypeError; the service soak is `/tmp/stress_int8ple.py`): **326 requests / 326 ok / 0 fail** across Phase A (64 distinct prefixes ×3 passes, overflows the 48-slot ckpt pool → evict → reload), Phase B (same-prefix reload through int8 + PLE mirrors), Phase C (6×200K KV crunch). Latency p50=23.7s p95=45.4s max=135.7s, journal clean after cutoff (zero OOM, zero leak/invariant lines), VRAM steady 96.8 GB, `NRestarts=0`. Hot-state on this final pool: C1 ≈87 / C6 ≈91 / prefill ≈9.7K tok/s — slightly below the spec-on peaks, which is the price of the 1M pool. Submitted upstream as **sgl-project/sglang#38619**.

Patch-tree hygiene note: `/opt/sglang-patch` is built on the pinned baseline commit `78c5024e9`, not latest HEAD — newer `memory_pool.py` imports `set_mla_kv_buffer_dcp_sharded_triton` which doesn't exist in the baseline's `utils.py`, breaking full service boot with an ImportError. Re-apply `0002` onto a fresh checkout of that commit, don't copy HEAD files in.

## Fourth round (2026-09-10): dynamic expert cold pool — 346/512 experts on GPU, spec + 12-way concurrency reborn

Context: at 1M context the routed-expert NVFP4 weights (~68 GB of the ~86 GB static footprint) are the wall — round 3 had to kill spec decoding to fund the pool. The mechanism reference is [ranxianglei/sglang `ours/main`](https://github.com/ranxianglei/sglang) (keep-mask ablation + keep-only offload + cold-pool v2 dynamic staging, W4A16/GPTQ); `patches/expert_cold_pool.py` re-implements the architecture for **NVFP4 / `modelopt_fp4` / `flashinfer_cutlass`** — parameter names, scale handling and graph constraints all differ, nothing was copied verbatim.

How it works (all env-gated; inert unless `SGLANG_EXPERT_KEEP_MASK` + `SGLANG_EXPERT_KEEP_OFFLOAD=1` are set):

1. **Static keep-mask** — a profiled keep-set (330/512 per layer, `config/expert_keep_330_final.json` from router-frequency census) masks cold experts out of router logits via a persistent cached bias column (`torch.finfo.min`, built on first eager call, cached by `(layer_id, dtype)` — no list→CUDA indexing inside graph capture).
2. **Physical shrink after load** — `model_runner.load_model` hook, strictly *after* `process_weights_after_loading`: per-expert tensor inventory gate (NVFP4 row params + `weight_scale_2`/`input_scale` scalars + derived `g1_alphas`/`g1_alphas_up`/`g2_alphas` all whitelisted explicitly; unknown per-expert tensor **refuses to arm**), cold rows copied to host, GPU tensors rebuilt at `keep+slots=346` rows. An **alias guard** rejects unaliased `w*_blockscale_swizzled` (same-object alias only holds while `swizzle_blockscale` needs no padding — config drift fails loud instead of silently corrupting kernels).
3. **Flat pin** — `CachingHostAllocator` rounds every ≥1 MB pin request to the next power of two; per-parameter pinning cost 40.5 GB for 24.2 GB of data. One exact-size `uint8` buffer per layer (48 × ~512 MB = 24.15 GB, 3.7% waste) with row views handed out per expert.
4. **Demand staging outside graph replay** — `select_experts` stashes row-sampled raw **pre-mask** logits (masked selection can never discover cold demand — v2.8's cold-demand probe reads the alpha band `sigmoid ≥ 0.8 × hot_min` from them, needs 2 repeat hits); `model_runner` after-forward hook stages winners into free/LRU slots (H2D on the current stream), updates persistent remap+bias tables by **content** (shapes never change → graphs stay valid), evicts mask-before-overwrite.
5. **v2.9 identity gate** — MTP draft layers reuse decoder `layer_id 0..47` *in the same process*; a layer_id-only gate silently applied the target's mask+remap to draft routers (accept rate 0.21, garbage drafts). Gating now requires the caller's `TopKConfig` **object identity** to have been registered at shrink — drafts (unregistered) pass through untouched.

Measured on the live stack (cold pool ON + decode graphs + NEXTN steps=2, conc 12, pool 2,359,296):

| Gate | Result |
|---|---|
| Staging correctness | 8448-row host↔GPU checksum byte-exact, under eager **and** CUDA graphs; zero MISMATCH / stage_fail / CUDA error across soaks |
| Staging activity | 30-min soak: staged 4997, h2d 13.82 GB; forced-alpha proof: staged 1618 / evicted 850 |
| Decode C1 | **122–129 tok/s** post-warmup acceptance bench, steps=2/draft3 (vs ~116 steps=1 and 104–105 graphs-only no-spec — shallow-MTP A/B falsified the "free 2.8 GB at no cost" reading: steps2 wins accept len 2.07–2.31 vs 1.74–1.79, so steps=2 stays default and steps=1 is demoted to a memory-pressure fallback) |
| Decode C12 aggregate | **704–710 tok/s** (two independent runs on the same harness; earlier 773–825 figures came from warm-cache windows and are superseded) |
| KV pool headroom | OFF-control at 1,572,864 **OOMs** → keep330+16 is what unlocks the pool; 2,359,296 boots with 7.1 GB torch-avail, 12×66K long-context stress peaks at 92,694/97,887 MiB with 5.1 GB real fence left, zero OOM/retract |
| Host RAM | pinned 24.15 GB cold pool + 64 GB PLE table on a 112 GB box — MemAvailable ~20 GB at soak end, swap stable |

Honest caveats: the demand probe is torch-only (≈0.3 ms/layer/tick in eager — invisible under graphs, Triton kernel deferred to the speed phase); keep=330/slots=16 is a starting point, not a tuned optimum; the reference author's own conclusion (static keep-sets Pareto-optimal, dynamic staging WIP) is a live hypothesis we still have to beat with Phase B data; and greedy byte-level nondeterminism exists **with the cold pool fully OFF** (atomics/cuBLAS runtime property, OFF-control proven) — quality gates are checksum + semantics + error counters, never raw byte equality.

## Fifth round (2026-09-11): cached=0 solved — per-path mamba checkpoint cap

**Symptom**: re-asking a huge prompt returned `cached_tokens: 0` and re-prefilled cold (≈89 s at 549K). The radix tree was still there and so were its KV pages — what was missing was the **mamba** side state: a prefix match requires the deepest node carrying a valid mamba checkpoint, so no checkpoints means no reuse. Five hypotheses died under the evidence matrix (template wording, single context size, pool occupancy, mamba anchor starvation, the int8 path itself — D2 still double-missed with int8 checkpoints *disabled*).

**Forensics (E3)**: three probes — match exit, split event, donate site (with the int8 pool watermark read out per event) — env-gated behind `SGLANG_E3_PROBE=1`, mounted on the live service for one day, then fully removed (source md5s, zero greps, clean process `environ`, zero journal hits). The smoking gun: under `extra_buffer_lazy`, chunked prefill **donates one int8 checkpoint per 4096-token chunk** (PREP `cache_len` marching 12544 → 16640 → 20736 …). A 549K chain ≈ **134 checkpoints > the 128-slot pool** (pool = 2× `max_mamba_cache_size` 64; `mamba_max_states_per_path` defaults to `-1`, uncapped). Two long chains together = 268 → mutual LRU eviction; the loser's re-ask matches nothing. The miss re-prefills and donates again — self-amplifying (this killed the three-chain cases, and equally fp8kv's old 3×336K all-miss result).

**Fix**: `--mamba-max-states-per-path 16` — one flag, zero VRAM, both schemes. A chain keeps 16 evenly spread anchors instead of 134 dense ones; whole-document re-ask is untouched, mid-document deep branching just recomputes a middle segment (self-healing, never a correctness failure). Pool watermarks went from slow-motion exhaustion to steady `ckpt_free 82/128`. Before → after:

| Shape | cap -1 | cap 16 |
|---|---|---|
| D2 · 2×549K re-ask | 88.8 / 91.0 s, cached=0 | **0.9 / 1.0 s, cached=548,864 — both hit** |
| hw77 · 3×495K re-ask | 74.5–74.8 s, cached=0 | **1.1 / 1.5 s, cached=494,848 — hit** |
| g2x · 3×233K re-ask | already hit (under the cliff) | unchanged |
| C12 burst | 704–710 tok/s | 606–703 tok/s, accept 2.18–2.31 — noise band |

**Rejected**: growing the pool to 256 slots (+3.7 GB against 5.29 GB of headroom = permanent OOM tightrope) — and checkpoint count scales linearly with document length, so 2×549K would blow through it anyway; the cap removes the cliff instead of moving it. Deferred lever: coarsening the donate interval 4096→16384 (≈34 ckpt/chain, at ~16K extra recompute cost on partial hits).

**Attribution correction**: this is **not** upstream #22935 (split-tombstone) — the #38625-style interior checkpoints are behavior `extra_buffer_lazy` already implements. Pure capacity economics on our side of the fence.

## The fp8kv scheme (rollback stack)

fp8kv is not the untouched old config — on 2026-09-08 it absorbed the portable half of the nvfp4kv tuning, so the two schemes swap on one GPU without losing the generic wins.

| Item | fp8kv state | Why |
|---|---|---|
| GDN flashinfer both ends | ported | Backend routing, independent of KV dtype |
| Prefill CUDA graph forced full | ported | Same eager-jitter avoidance (#28386-class) |
| SAM=decode | ported | Pure backend routing for verify/draft batches |
| FR-Spec 64K hot map | ported, same `.pt` | The map is tokenizer-scoped, not KV-dtype-scoped |
| extra_buffer_lazy (CLI-only alias) | ported | Same slot economics |
| int8 mamba ckpt + `mamba-max-states-per-path 16` | added (round 5, CLI flags) | Same donate-storm fix; fp8kv's old 3×336K all-miss case belongs to this family (96-slot pool, ~82 ckpt/chain uncapped) |
| MTP steps | 3 → **2** (round-4 A/B) | steps2 won concurrency (C4 aggregate 431–438 tok/s); the 09-08 "keep 3" reading is superseded |
| KV pool | 552,960 → **1,310,720** | B+ rung: 2× full 524K chains co-resident — the expert cold pool migrated to fp8kv too, funding it |
| mamba cache | 24 → **48** | anchor-slot ladder; int8 pool auto-sizes 2× (96 slots) |
| HiCache L2 | **OFF** | Upstream fail-fast against int8 mamba checkpoints (`server_args.py:6333`) — same-side trade as nvfp4kv's #36121 ban |
| Expert cold pool | added (round 4+) | `switch_fp8kv_coldpool.sh`, same keep330+16 staging as nvfp4kv |

Hot-state (round-4 era): fp8kv at steps=2 runs C4 aggregate ≈431–438 tok/s (the A/B winner; steps=3 measured slower under concurrency despite deeper accept chains). Both schemes share one GPU and are mutually exclusive — since 2026-09-10 fp8kv sits stopped (nvfp4kv live; fp8kv in `failed` state after a manual KILL, so cap16 activates on its next bring-up). Pick fp8kv for latency-sensitive single-stream work; pick nvfp4kv when context capacity matters. Same port, same served-model-name — swapping = stop one unit, start the other. Rollback anchor: the fp8kv pair in `config/baseline/`.

## Why nvfp4 KV

The fp4 KV pool halves KV memory (packed e2m1 + tiny per-block scales), which on a 96 GB card is what makes 1M context at conc 12 possible at all — the weights, the ~44 GB pinned PLE n-gram table and the CUDA graphs eat everything else. The single-stream decode tax is inherent to gather-dequant (two Triton launches + one dequant kernel per step); round 4's cold pool freed the VRAM that re-funds spec (C1 ≈122–129 tok/s with steps=2), and rounds 3+5 gave the checkpoint pools the real eviction headroom. Pick fp8kv (512K) when raw decode speed matters; pick nvfp4kv when context capacity matters.

## Contents

```
patches/
  apply_nvfp4_patches.py     idempotent patcher: QSA×NVFP4-KV support for sglang
                             (anchor-count==1 checks + ast.parse + py_compile)
  0001-sampler-37962-tp1-sync-skip.diff
                             TP=1 sampler fix: skip the cross-TP token sync whose
                             first NCCL op lazily allocates 512 MB → OOM on a full
                             pool (grammar requests only). Crash root-caused live.
  0002-mamba-ckpt-ple.diff   int8 mamba checkpoint pool mirrors PLE side states
                             (short-conv bf16 / ngram int64 rows) instead of the
                             upstream ValueError guard. Upstream PR #38619.
  expert_tier.py             v1 swap-in prototype — kept VERBATIM as the failure
                             baseline (corrupted output after ~10 min of traffic);
                             source of the R1-R3 red lines in expert_cold_pool.py
  expert_cold_pool.py        round-4 dynamic expert cold pool (see §Fourth round):
                             keep-mask + post-pwal shrink + flat pin + demand
                             staging + v2.9 TopKConfig identity gate
config/
  dealignai-qwen4exp-nvfp4kv.yaml   nvfp4 KV scheme — ROUND 5 FROZEN (current): conc12 /
                                    1M / spec ON steps=2 / mamba64 + per-path ckpt cap 16 /
                                    KV pool 2359296 / int8 mamba checkpoint ON (128 slots) /
                                    cold pool keep330+16 /
                                    decode CG bs[1,2,4,6,8,10,12] / prefill CG disabled
  expert_keep_330_final.json        per-layer keep-set (48 × 330 global expert ids) from the
                                    router-frequency census — cold pool's arm input
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV scheme — rollback stack (conc4 / 512K / MTP steps=2 / mamba48 + per-path cap 16 / int8 ckpt ON / HiCache OFF / KV pool 1310720 / expert cold pool; 2026-09-08 portable port + round-4 steps2 A/B + round-5 cached0 cap)
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
  test_ckpt_ple_patch.py     CPU integration test T1–T6 for patch 0002 (mirror
                             roundtrip, dtype fidelity, factory-path kwargs check)
  deploy_expert_cold_pool.py idempotent hook installer for the cold pool (5 anchors
                             across topk.py/model_runner.py + v2.8→v2.9 upgrade pass;
                             anchor-count checks + py_compile)
  switch_coldpool.sh         mode switcher: eager (cold pool ON, decode CG off) /
                             graphs (cold pool ON, CG on) / mtp (cold pool + graphs +
                             NEXTN uncommented) / off (full rollback)
  soak_watch_v2.sh           30-min mixed-traffic soak + journal watch + greedy probes
  soak_final.sh              rotating-topic acceptance soak + built-in log triage
  log_triage.sh              journal ERROR/WARNING sweep + classification
  stress_conc12.sh           12-way concurrent short-context stress + GPU peak sampler
  stress_long12.sh           12-way ~66K-token long-context concurrent stress (fence probe)
  frspec_map_64k.pt           FR-Spec speculative token-map artifact (65,536 IDs; sha256
                              598b0dc4… matches the manifest; shipped so the tuned stack is reproducible)
  frspec_map_64k.manifest.json  build provenance: tokenizer sha256, corpus file list with per-file
                              sha256, coverage 1.0, size/base_count/special-ids
  build_token_map.py            rebuild the map from corpus + model tokenizer
  warmup_qsa_nvfp4kv.py       closes the late-device-load OOM window (long prefill,
                              grammar bitmask, bs6 graph buckets) before real traffic
  warmup_qsa_fp8kv.py         fp8 variant (prefix-cache hit path)
tests/
  test_cold_pool_logic.py    CPU logic tests T1–T9 for the cold pool (shrink+stage+
                             remap+bias, packed remap, demand stash, weak-demand
                             rejection, inventory gate, free-row accounting, bias
                             cache refresh, mask-starvation probe, identity gate)
  probe_cached0_discrim.py   round-5 discriminator: D1 1×517K vs D2 2×549K branch
                             chains, per-line cached_tokens hit/miss
  probe_highwater_77.py      3×495K high-water probe (cold fill + A/B/C re-asks)
  probe_g2x.py / probe_nvfp4_mamba_gate.py
                             mamba-slot / anchor-gate probes (G2X: 3×233K all-hit)
docs/
  RESULTS.md                  full stress matrix, MTP steps 1–3 calibration table,
                              mamba slot economics, every dead-end and root cause
  expert-dynamic-v2-design.md cold-pool v2 design: R1-R3 red lines from the v1
                              incident, Phase 0 tensor-inventory gate, eager-first
                              validation order, rollback gates
bootstrap/
  MANIFEST.md                 FULL-RESTORE checklist for a bare same-hardware box:
                              verified base tarball (official sgl-project commit
                              78c5024e9, sha256-pinned), per-file md5 table of the
                              7-file overlay delta, NVIDIA/CUDA/Python stack versions
  pip-freeze-20260911.txt     205-package environment lock (install --no-deps)
  ninja-wrapper               /opt/fakebin/ninja shim — flashinfer JIT -j1 memory guard
```

## Key findings (the short version)

1. **QSA × nvfp4 KV needs a patch.** Upstream declares nvfp4 access as FlashInfer-prefill / TRT-LLM-native-decode; QSA's Triton gather path is neither. The patcher routes QSA to PLAIN BF16 access and gather-dequants (packed data + scale buffers, both viewed as uint8) before the dots. CUDA-graph safe, inert unless `--kv-cache-dtype nvfp4`.
2. **HiCache × fp4 KV = silent corruption.** Host transfer moves packed data buffers but not the per-block scale buffers; eviction/load-back then serves garbage that *looks* plausible. sglang PR #36121 proposes the fail-fast; until it lands, `enable-hierarchical-cache: false` is mandatory.
3. **Radix cache × fp4 KV is safe on current builds** — `_slot_move_pointer_buffers` already moves `k/v_scale_buffer` alongside the data. Verified with a 553K-token eviction-pressure probe: needles stay correct after forced eviction and re-hit.
4. **TP=1 sampler OOM (fixed here).** With grammars enabled the sampler runs a cross-TP token sync even at world_size=1; the first NCCL op allocates 512 MB lazily — fatal when the pool is 99.3% full. One-line guard, upstream #37962.
5. **MTP steps must be re-calibrated per KV dtype.** Under fp4 KV, steps=2 beats both 1 and 3 (dequant cost scales with verify tokens; steps=3's accept-rate gain doesn't pay for it). fp8-era "more steps is better" does not transfer.
6. **Mamba slot economics flip with radix.** Radix OFF: ~1 slot/request (conc6 fits in 16). Radix ON: completed-request states linger for prefix reuse → ~3 slots/request; with extra_buffer_lazy that drops to ~4/request and 24 pinned slots suffice. The knobs are coupled — tune them together.
7. **Late Triton device-loads are a real OOM class.** First-touch kernel specializations (GDN chunk prefill, sparse-GQA, xgrammar bitmask) allocate *after* the pool is full. Warm up every shape family before traffic; extreme shapes can still trigger new specializations — monitor `device-loaded` in the journal.
8. **NVFP4 per-expert side tensors defeat `dim0 == E` heuristics.** Beyond `w13/w2_weight` + block scales, each expert owns `weight_scale_2`/`input_scale` scalars and derived `g1_alphas`/`g1_alphas_up`/`g2_alphas`; `process_weights_after_loading` deinterleaves w13, swizzles scales and may alias derived params into source storage. The shrink must snapshot **post-pwal** state (hook in `model_runner.load_model`, not the weight loader), round-trip every whitelisted tensor, and refuse to arm on anything unknown. `w*_blockscale_swizzled` is only safe while it aliases its source Parameter — guard for it.
9. **A keep-masked router cannot feed its own demand signal.** Cold experts are `-inf`-masked, so selection-path demand is structurally zero and the pool starves forever (observed: 30 min traffic, staged=0). Demand must be probed from raw **pre-mask** logits in the alpha band. Related trap class: any stash-buffer slice read before refill must be `.clone()`d — a view erased by `fill_` silently killed the strong-demand filter in v1 of the probe.
10. **Spec-decoding drafts share the decoder `layer_id` namespace in-process.** The NEXTN draft model is the same class with `num_hidden_layers=1` → its layer 0 collides with target layer 0. Any per-layer runtime mechanism (mask, remap, stats) gated on `layer_id` alone will silently hijack drafts — gate on **object identity** of the module's TopKConfig registered at arm time (v2.9; symptom was accept rate 0.21 with drafts masked+remapped).
11. **Under `extra_buffer_lazy` + int8 mamba checkpoints, prefix reuse is capped by the checkpoint pool, not the KV tree.** Chunked prefill donates one int8 checkpoint per 4096-token chunk; with the default per-path cap of −1, a single 549K chain eats ≈134 of the 128 pool slots, so two chains evict each other into `cached=0` (tree and KV pages present, match dead). `--mamba-max-states-per-path 16` is the zero-VRAM fix, on both schemes; the re-ask cliff (89 s → 0.9 s) is the proof. Upstream #22935 looks similar but is a different family (`no_buffer`) — don't chase its split-tombstone fix.

## Reproduce

```bash
# 0. sglang built from the qwen4-main-squashed branch (PR #36497 head) for sm120,
#    e.g. CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0" — see jpezzulli/gabrielolympie repos.
#    VERIFIED for full restore: official sgl-project commit 78c5024e9 builds it;
#    pinned tarball sha256, env lock, ninja shim, chat template → bootstrap/MANIFEST.md
export SGLANG_SRT=/opt/sglang-src/sglang/python/sglang/srt

# 1. patches (idempotent; re-run after any source sync)
python3 patches/apply_nvfp4_patches.py
git -C /opt/sglang-src/sglang apply ../patches/0001-sampler-37962-tp1-sync-skip.diff

# 2. config — edit MODEL_DIR in the unit, drop YAMLs in place
systemctl start sglang-dealignai-qwen4exp-nvfp4kv
systemctl enable --now sglang-dealignai-nvfp4kv-warmup   # one-shot, waits for /health

# 3. (round 4, optional) expert cold pool — overlay tree + hooks + keep-set + env
cp -a /opt/sglang-src /opt/sglang-patch            # fork isolation, pinned base commit
/opt/sglang-env/bin/python scripts/deploy_expert_cold_pool.py   # 5 anchors, idempotent
# ship patches/expert_cold_pool.py -> $S/layers/moe/expert_cold_pool.py
# ship config/expert_keep_330_final.json -> /opt/sglang-config/
# unit needs: Environment=PYTHONPATH=/opt/sglang-patch/sglang/python
#             Environment=SGLANG_EXPERT_KEEP_MASK=… SGLANG_EXPERT_KEEP_OFFLOAD=1
#             Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=16
/opt/sglang-env/bin/python tests/test_cold_pool_logic.py       # T1–T9, CPU-only
bash scripts/switch_coldpool.sh eager    # validate staging in eager first
bash scripts/switch_coldpool.sh graphs   # then CUDA graphs
bash scripts/switch_coldpool.sh mtp      # then NEXTN spec
bash scripts/switch_coldpool.sh off      # full rollback at any step

# 3b. (round 5, both schemes) cached0 cap — CLI flags on the units:
#   --enable-int8-mamba-checkpoint --mamba-max-states-per-path 16
#   int8 pool auto-sizes 2× max_mamba_cache_size; the cap keeps N long
#   chains co-resident (16 ckpt/path vs ≈134 per 549K chain uncapped)

# 4. verify
curl -s localhost:8000/v1/models          # → Qwen3.8-Flash-Next-NVFP4
curl -s localhost:8000/get_server_info | jq '.kv_cache_dtype, .disable_radix_cache'
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager | grep "COLD-POOL"  # SHRUNK line
```

The nvfp4 KV path is gated entirely by `kv-cache-dtype: nvfp4` — flip it back to `fp8_e4m3` (use the fp8kv scheme files) and every patch in here is inert. To run the fp8kv scheme instead, start its own unit pair (`sglang-dealignai-qwen4exp-fp8kv` + warmup) after stopping the nvfp4kv one — see §The fp8kv scheme for the expected values.

## Constraints & honest caveats

- Finalized tuned stack (round 2, memory knobs superseded by round 3): GDN flashinfer both ends, mamba pinned 24, extra_buffer_lazy (CLI-only alias), SAM=decode. Round-3 state: ctx 1M / YaRN ×4 explicit / spec OFF / KV pool 1,179,648 / int8 ckpt ON via patch 0002 overlay. **Round-5 frozen live state: cold pool keep330+16 / spec ON steps=2 / conc 12 / mamba 64 / per-path ckpt cap 16 (int8 pool 128 slots) / KV pool 2,359,296 / decode CG bs[1,2,4,6,8,10,12].** Hot-state round 4: C1 ≈122–129 / C12 aggregate ≈704–710 / prefill ≈9.7K fresh-prefix (post-warmup acceptance bench, steps=2/draft3; supersede earlier warm-cache-window figures 146–171 / 773–825). Rollback: `switch_coldpool.sh off`, or copy `config/baseline/` files over the current ones and restart the unit.
- Finalized fp8kv stack: portable items ported (§The fp8kv scheme), MTP steps=2 (round-4 A/B superseded the 09-08 "keep 3"), HiCache OFF (int8-ckpt fail-fast), conc4 / YaRN×2 512K / KV pool 1,310,720 / mamba 48 + cap 16 / expert cold pool. Hot-state: C4 ≈431–438 tok/s aggregate. Currently stopped (single-card mutex, nvfp4kv live) — cap16 activates on next bring-up. Rollback = its own `config/baseline/` pair over the current files.
- Single consumer GPU + 44 GB pinned PLE table: the nvfp4kv and fp8kv schemes are mutually exclusive; expect ~4-5 min cold start (round 4 adds ~3.5 min weight load + shrink). The unit deliberately ships unenabled to avoid boot-time GPU contention.
- Round-4 host-RAM budget is tight by design: 24.15 GB cold-pool pins + 64 GB PLE pinned table (power-of-two rounded from 47.7 GB — a `file`-backend A/B is deferred) on a 112 GB box → MemAvailable ~20 GB under soak. Watch `Shmem` and swap before adding any more pinned consumers.
- C1 decode with the cold pool + spec (≈122–129) sits just below fp8kv (≈165) — the round-3 "fp4 KV is slower" gap was mostly the spec-off tax, not gather-dequant alone. Upstream native fp4 QSA decode pools (#37798) remain the path to close or invert the gap.
- `Restart=no` on the experiment unit is deliberate — preserve the crash scene.
- Numbers are from one RTX PRO 6000 (96 GB), sglang `0.5.19.dev6+g78c5024e` + local patches. SM121/GB10 is a different story (see gabrielolympie's notes on silent long-context corruption there).
- Swapping the FR-Spec token map resets the prefix-cache namespace once — warm 2–3 rounds after any table change.

## Credits

- [sgl-project/sglang](https://github.com/sgl-project/sglang) — `qwen4-main-squashed` branch (PR #36497, #36806)
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) (MiaAI Lab, MIT) — the dspark QSA×nvfp4 patch this repo ports and extends to SM120
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) — sm120 groundwork
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) — the fp8-KV performance ceiling this repo's KV trade-off is measured against
- [ranxianglei/sglang `ours/main`](https://github.com/ranxianglei/sglang) + [ranxianglei/sglang-expert-profile](https://github.com/ranxianglei/sglang-expert-profile) — the keep-mask / keep-offload / cold-pool v2 mechanism architecture (W4A16) that round 4 re-implements for NVFP4

## License

Apache-2.0 (matches the sglang base these patches apply to). The dspark-derived patcher retains MIT attribution to MiaAI Lab in its header. See [LICENSE](LICENSE).
