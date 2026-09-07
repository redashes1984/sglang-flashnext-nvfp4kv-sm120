# Results & methodology — the full ledger

Everything below was measured on one RTX PRO 6000 Blackwell (96 GB, sm120), sglang
`0.5.19.dev6+g78c5024e` (PR #36497 head + #36806), editable install at `/opt/sglang-src`,
model `dealignai Qwen3.8-Flash-Next-ABLITERATED-NVFP4` (RadixArk checkpoint, abliterated
variant), PLE n-gram table offloaded to host RAM (~44 GB pinned).

## 1. The crash that started it: bs8 / 768K OOM

First nvfp4-KV attempt at `max-running-requests: 8` started fine, passed functional
tests, then died under traffic:

```
Triton device-load of apply_token_bitmask_inplace_kernel at free=0.23 GiB
Failed to CUDA calloc 536870912 bytes
scheduler SIGKILL in sampler.py:_sync_token_ids_across_tp → dist.all_reduce
```

Root cause chain:
- grammar-enabled requests take the cross-TP token sync path in the sampler
  (`SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars`)
- **even at TP=1** — the first NCCL op on that group lazily allocates 512 MB
- the fp4 pool at 786K tokens leaves < 1 GB free; the late kernel load + 512 MB
  rendezvous together blow it

Fix: `patches/0001-sampler-37962-tp1-sync-skip.diff` — guard the sync with
`dist.get_world_size(self.tp_sync_group) > 1`. (Same class as upstream #37962.)
After the fix, conc6/768K survives everything we threw at it.

## 2. Stress matrix (conc6 / 768K / nvfp4 KV / MTP steps2)

| Test | Result |
|---|---|
| T1 C1 decode 512 tok ×3 | 119.5 / 116.9 / 123.6 tok/s |
| T2 C6 decode 256 tok | wall 1.9s, aggregate 342 tok/s |
| T3 C6 × ~40K long ctx | prefill 11.2K tok/s, 6/6 needles correct |
| T4 grammar JSON ×6 | all valid, wall 0.4s |
| T5 mixed long+short pressure | zero retractions |
| T6 pool pressure 6 × ~107K | 645K/786K tokens resident, 6/6 needles correct |
| NIAH 200K | hit |
| tool calls | `finish=tool_calls` + correct arguments |

## 3. MTP steps calibration (fp4 KV, radix OFF, same battery per step)

| steps | C1 median | C6 agg | accept len | accept rate | free VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 123.2 | ~430 | 1.57 | 0.57 | 4185 MiB |
| **2** | **126.2** | **~434** | **1.95** | 0.48 | 2791 MiB |
| 3 | 117.9 | ~405 | 2.09 | 0.36 | 2585 MiB |

**steps=2 wins under fp4 KV** — steps=3 is strictly dominated: highest accept length,
lowest throughput, worst VRAM. The gather-dequant cost scales with verify tokens, so
longer draft chains pay more than in the fp8 era (where steps=3 was optimal).
**Re-calibrate steps whenever you change KV dtype.**

The tempting "steps=1 saves 1 GB" is false: itemized graph savings (−0.66 verify,
−0.32 draft-decode) are eaten by a +0.10 draft-extend graph and +0.55 pre-capture
overhead → **net 0.32 GB** (capture-end `avail mem` 5.01 vs 4.69). Always reconcile
against the final `avail mem` log line, never against itemized sums.

## 4. Radix cache × fp4 KV

Source review: no hard blocker (server_args only rejects `--prefill-only-disable-kv-cache`
for nvfp4); slot moves go through `move_kv_cache`, and current builds already include
`k/v_scale_buffer` in `_slot_move_pointer_buffers` — the scale-follows-data requirement
is handled.

Live validation (radix ON, mamba 32):

| Probe | Result |
|---|---|
| cold 58K prefix + needle | 6.3s, correct |
| same prefix, new question | **0.6s, `#cached-token 58368/58392` = 99.96%, correct** |
| 30K prefix, follow-up | correct |
| **eviction pressure**: flood 553K tokens, re-query original prefix | needle still correct |
| throughput cost | C1 −3%, C6 −6% |
| mamba peak usage | 0.62 × 32 ≈ 20 slots in use |

The eviction-pressure re-query is the important one: it exercises exactly the path
(#36121's corruption scenario) that HiCache fails, and fp4 KV passes it.

## 5. Mamba slot economics

| regime | slots/request observed | sizing at conc6 |
|---|---|---|
| radix OFF, `gdn-mtp-cache-mode: none` | ~1 (usage 0.12 @ cap 48) | 16 |
| radix ON | ~3 (usage 0.62 @ cap 32) | 32 |

Unit cost ≈ 56 MB/slot (conv 0.10 GB + ssm 2.58 GB ÷ 48). Radix retention reverses
the economics — the two knobs move together or you get silent scheduler stalls.

## 6. Late device-load watchlist

Kernels observed loading *after* ready on the fp4 path: GDN chunk prefill
(`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`), fused gate/sigmoid epilogues,
`_sparse_gqa_prefill` / `_sparse_gqa_chunk_prefill`, xgrammar
`apply_token_bitmask_inplace_kernel`. The warmup script covers the typical shapes;
extreme shapes (107K prefill, bs6 verify) still triggered 22 late loads in one stress
window — all survived post-sampler-fix, but keep `journalctl | grep device-loaded`
in your monitoring.

## 7. Dead-ends (so you don't repeat them)

- **nvfp4 + HiCache**: silent corruption after eviction (scale buffers not transferred,
  host capacity formula assumes full-width dtype). PR #36121 adds fail-fast; our build
  predates it. Hard no.
- **stock upstream nvfp4 + QSA**: `KeyError: float4_e2m1fn_x2` in Triton
  `canonicalize_dtype` during CUDA graph capture. The patcher exists because of this.
- **widening the trtllm-gen sparse decode gate to all SM12x**: SM121/GB10 can silently
  corrupt long-context decode. Keep it exact-SM120.
- **moving every CLI flag into YAML**: `BooleanOptionalAction` and
  `DeprecatedAliasStoreAction` args are not merger-safe — `--ple-offload-embedding`
  and `--mamba-radix-cache-strategy` must stay in the unit.
- **served-model-name suffixes for experiments**: routing keys break silently.
  Distinguish instances by port, never by model name.
