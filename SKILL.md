---
name: deploy-qwen-flash-next-nvfp4kv-sm120
description: Reproduce both tuned Qwen3.8-Flash-Next sglang schemes on one RTX PRO 6000 (SM120) — nvfp4kv mainline (round-4 expert cold-pool state) plus fp8kv rollback, each with baseline snapshots.
---

# Deploy Qwen3.8-Flash-Next with NVFP4 KV on one RTX PRO 6000 (SM120)

Self-contained recipe for any agent (or human) to stand up this exact stack, verify it, tune it later, or roll it back. Only four things need editing for a different machine: model dir, sglang env path, config dir, unit name.

## What this is

Qwen3.8-Flash-Next is a 180B hybrid MoE (GDN linear attention + QSA sparse attention + native NEXTN MTP) whose official FP8 checkpoint weighs ~173 GB and needs multi-GPU. A quantized community checkpoint (~102 GB NVFP4 with BF16 PLE table, or the smaller FP8-PLE variant) fits one RTX PRO 6000 Blackwell 96 GB (SM120). This stack runs it with `nvfp4` KV cache + QSA on sglang `qwen4-main-squashed` builds. The live nvfp4kv instance is the **round-4 expert cold-pool state**: 1M context at conc 12, spec decoding ON (NEXTN steps=2), KV pool 2,359,296, with only 346/512 routed experts physically on GPU (keep 330 + 16 dynamic slots; 166 cold experts demand-staged from 24.15 GB pinned host). The fp8kv rollback scheme carries the portable subset of the tuning with spec ON (§Switching below).

Stack generations, each documented in README §Results as its own column: baseline → round-2 tuned (C1 ≈136 / C6 ≈550–575, spec-on 768K) → round-3 nvfp4kv-1m trial (spec-off at 1M, int8 ckpt pool) → round-4 expert cold pool (live). Do not overwrite a generation's numbers with the next one's — the chain is the evidence.

Each scheme ships tuned-plus-baseline: active files in `config/`, pre-tuning snapshots as same-suffix `.baseline.*` under `config/baseline/` (rollback anchors + before/after evidence). `nvfp4kv` is the current mainline; `fp8kv` is the tuned rollback scheme — see §Switching to the fp8 KV scheme.

## Prerequisites

- Linux host, one RTX PRO 6000 Blackwell (SM120), ≥96 GB VRAM, ≥128 GB system RAM (PLE n-gram table lives pinned in CPU RAM ~44–51 GB depending on variant; the round-4 cold pool adds another ~24–25 GB of pinned host memory).
- sglang built from the `qwen4-main-squashed` branch head (PR #36497) compiled for SM120: `CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0"`, PyTorch cu13x. See `jpezzulli/sglang-rtxpro6000` and `gabrielolympie/sglang-flashnext-sm120` for build notes. Verify: `python3 -c "import sglang; print(sglang.__version__)"` in `/opt/sglang-env`.
- **Patch-tree base pin:** `/opt/sglang-patch` (the overlay for patch 0002) must be built on the pinned baseline commit `78c5024e9`, not latest HEAD. Newer HEAD's `memory_pool.py` imports `set_mla_kv_buffer_dcp_sharded_triton`, absent from the baseline's `utils.py` → ImportError on full boot. Re-apply `patches/0002-mamba-ckpt-ple.diff` onto a fresh checkout of the pinned commit; don't merge HEAD files into the overlay.
- A quantized checkpoint with its chat template in the model dir (NVFP4 experts + BF16 PLE works; the FP8-PLE variant fits tighter RAM budgets). The tokenizer must be the Qwen BPE — the FR-Spec token map is tokenizer-specific.
- Model dir, config dir, python env chosen per site. Below assumes `/opt/models/…`, `/opt/sglang-config/`, `/opt/sglang-env/`.

## Install steps

1. **Patch sglang source (idempotent).** The QSA×fp4 fix and the sampler guard are mandatory; without them QSA crashes on Triton `KeyError: 'float4_e2m1fn_x2'` and grammar requests OOM at TP=1.
   ```bash
   export SGLANG_SRT=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))" )/srt
   /opt/sglang-env/bin/python patches/apply_nvfp4_patches.py
   git -C "${SGLANG_SRT}/.." apply patches/0001-sampler-37962-tp1-sync-skip.diff
   ```
   Re-run after any source sync. The patcher checks anchor counts and compiles each edited file.
2. **Set up the int8 ckpt overlay (round 3).** Copy the source tree out of `/opt/sglang-src` (`cp -a`, fork isolation so the live env stays untouched), apply `patches/0002-mamba-ckpt-ple.diff` inside `/opt/sglang-patch`. The patch lets the int8 mamba checkpoint pool coexist with Qwen4-Exp PLE side states (ShortConv bf16 / NGram int64 rows) by mirroring them into ckpt-slot-indexed buffers instead of the upstream ValueError guard. Verify CPU-side (no GPU needed, works while the card is busy):
   ```bash
   CUDA_VISIBLE_DEVICES= /opt/sglang-env/bin/python scripts/test_ckpt_ple_patch.py   # T1–T6, expects all-ok
   ```
3. **Place configs.** Copy `config/dealignai-qwen4exp-nvfp4kv.yaml` to `/opt/sglang-config/`. Replace `${MODEL_DIR}` inside the YAML manually (systemd does NOT expand variables inside YAML content). Keep `served-model-name` verbatim if you have downstream routers keyed on it — renaming breaks routing silently.
4. **Place the FR-Spec artifact.** Copy `scripts/frspec_map_64k.pt` (+ manifest for audit) next to the YAML. Optional but recommended — it carries the +accept-len gain; without it the tuned stack loses most of its edge.
5. **Install units.** Copy `config/systemd/sglang-dealignai-qwen4exp-nvfp4kv.service` and `sglang-dealignai-nvfp4kv-warmup.service` to `/etc/systemd/system/`, adjust paths, then:
   ```bash
   systemctl daemon-reload
   systemctl start sglang-dealignai-qwen4exp-nvfp4kv
   systemctl --no-pager status sglang-dealignai-qwen4exp-nvfp4kv   # wait until /health responds
   systemctl restart sglang-dealignai-nvfp4kv-warmup   # restart, NOT start — see warmup gotcha
   ```
   The unit carries three CLI-only items: `--ple-offload-embedding`, `--mamba-radix-cache-strategy=extra_buffer_lazy` (alias/BooleanOptionalAction args the YAML merger rejects), and `--enable-int8-mamba-checkpoint` (+ `Environment=PYTHONPATH=/opt/sglang-patch/sglang/python` overlay). Everything else lives in the YAML.
   Cold start is ~4 min (weight load + graph capture). Only one scheme runs at a time — stop the other before starting this one (shared GPU).
6. **Verify.**
   ```bash
   curl -s localhost:8000/v1/models | head -c 200          # expects Qwen3.8-Flash-Next-NVFP4
   curl -s localhost:8000/get_server_info | python3 -c "import json,sys;d=json.load(sys.stdin);print('ctx',d.get('context_length'),'conc',d.get('max_running_requests'),'KV',d.get('max_total_tokens'),'spec',d.get('speculative_algorithm'),'ckpt',bool(d.get('enable_int8_mamba_checkpoints')),'gdn',d.get('linear_attn_prefill_backend'))"
   journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --since -10m | grep 'int8 mamba checkpoint pool'
   ```
   Expected live values on the round-4 stack: ctx 1048576, conc 12, KV pool 2359296, spec EAGLE/NEXTN steps=2, ckpt True (96 slots, 2.81GB), gdn flashinfer. Then run one real chat completion and watch the journal for late `device-loaded` lines (`journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --since -10m | grep -c 'device-loaded'` → should trend to zero after warmup).
7. **(Round 4, optional) Expert cold pool.** The cold pool turns the ~30 GB of GPU expert-weight savings into the bigger KV pool + revived spec decoding. Install into the same `/opt/sglang-patch` overlay:
   ```bash
   cp patches/expert_cold_pool.py $S/layers/moe/expert_cold_pool.py   # $S = overlay srt dir
   /opt/sglang-env/bin/python scripts/deploy_expert_cold_pool.py       # 5 anchors, idempotent
   /opt/sglang-env/bin/python tests/test_cold_pool_logic.py            # T1–T9, CPU-only, expects ALL PASS
   ```
   Ship `config/expert_keep_330_final.json` next to the YAML. Unit env: `SGLANG_EXPERT_KEEP_MASK=<that json>`, `SGLANG_EXPERT_KEEP_OFFLOAD=1`, `SGLANG_EXPERT_COLD_POOL_SLOTS=16`. Use `scripts/switch_coldpool.sh {eager|graphs|mtp|off}` to walk the validation ladder — **eager first** (proves staging correctness without graph variables), then `graphs`, then `mtp`. Arm proof in journal: `[COLD-POOL] SHRUNK 48 layers keep=[330] +16 slots -> host pinned 24.15 GB`. Under `SGLANG_COLD_DEBUG=1`, `checksum selfcheck: 8448 rows OK` with zero MISMATCH is the staging-correctness gate. Red lines (from the v1 incident, enforced by design): shrink strictly after `process_weights_after_loading`; staging only in the after-forward hook, never inside graph replay; gate tables updated by content, shapes constant; NVFP4 per-expert tensors explicitly inventoried — unknown tensor refuses to arm.

## Bench

```bash
python3 scripts/bench_gdn.py            # or your own harness: C1 latency sweep + C6 aggregate
```
Reference hot-state numbers for the round-4 cold-pool stack (same GPU, this sglang build): **C1 ≈146–171 tok/s, C12 aggregate ≈773–825 tok/s**, prefill ≈9.7K tok/s fresh-prefix (higher on radix hits), accept len 2.15–2.27 / rate 0.5–0.8. Run twice — first pass warms caches. If your numbers are far off, check the eight tuning items (below) and the cold-pool arm line are all present. Round-3 reference (spec-off, conc6, pool 1.18M): C1 ≈75–90 / C6 ≈72–91. Round-2 (spec-on 768K): C1 ≈136 / C6 ≈550–575. fp8kv scheme: C1 ≈165 / C6 aggregate ≈355–365 tok/s (conc4, spec-on, steps=3).

Why round-4 C1 (≈146–171) is now level with fp8kv (≈165): the round-3 gap was mostly the spec-off tax, not gather-dequant alone — the cold pool's freed VRAM funds NEXTN again at 1M. Log-reading gotchas: `gen throughput` in the first decode line after an idle gap (or a new request's first window) is polluted by prefill/sleep-on-idle in the denominator — it can read 0.5 tok/s while the true rate is 136; trust consecutive-window values or a wall-clock curl timing, never the first line. Greedy byte-level output is nondeterministic **even with the cold pool fully OFF** (atomics/cuBLAS runtime property, proven by OFF-control) — never gate on raw byte equality.

## Tuning items vs baseline (what changed, why, effect)

See README §What changed vs baseline — the table there is the authority. Items 1–7 (round 2): GDN flashinfer both ends; mamba cap pinned 24; radix strategy lazy; prefill CUDA graph forced full; `speculative-attention-mode: decode`; FR-Spec token map; NEXTN steps 3→2. Net **+11% C1 / +36% C6** for round-2 vs baseline. Item 8 (round-3 layer on top): int8 mamba ckpt pool ON via patch 0002 PLE mirrors — funded outside the token formula, so `max-total-tokens` steps 1441792 → **1179648** (~3.2 GB boot headroom); buys eviction headroom, costs the spec-off-at-1M decode tax. Round-2 used prefill CG=full; round-3's spec-off at 1M forces both CG sections to decode-only `{bs:[1,2,4,6], max_bs:6}` + `prefill: disabled` (QSA graph state is initialized on the spec chain — spec-off + prefill CG crashes at boot).

## Soak expectation

Service-level soak (`stress_int8ple.py`: Phase A 64 distinct prefixes ×3 passes overflowing the 48-slot ckpt pool → evict → reload, Phase B same-prefix replays through int8+mirror path, Phase C 6×200K KV crunch): expect **all-ok, zero fail**, latency p50 ≈24s p95 ≈45s, VRAM steady ~96.8 GB, journal clean after cutoff (zero OOM, zero leak/invariant lines), `NRestarts=0`. If Phase A OOMs, the pool budget is too high — step `max-total-tokens` down further (the 1441792→1310720→1179648 chain exists because late JIT + conc-6 activations stack on top of the static ckpt pool).

## Rollback

```bash
cp config/baseline/dealignai-qwen4exp-nvfp4kv.baseline.yaml       /opt/sglang-config/dealignai-qwen4exp-nvfp4kv.yaml
cp config/baseline/sglang-dealignai-qwen4exp-nvfp4kv.baseline.service /etc/systemd/system/sglang-dealignai-qwen4exp-nvfp4kv.service
systemctl daemon-reload && systemctl restart sglang-dealignai-qwen4exp-nvfp4kv
```
Baseline is expected to measure C1 ≈122 / C6 ≈409 (768K, spec-on, no ckpt overlay). Rollback within round 3 = re-drop only the ckpt items: delete `--enable-int8-mamba-checkpoint` + `PYTHONPATH=` lines, restore `max-total-tokens: 1441792`.

## Switching to the fp8 KV scheme

Use it when single-stream decode matters most — fp8kv is ahead at C1 (≈165 vs ≈75–90), nvfp4kv trades that for context capacity. Economics: conc4 / YaRN×2 512K / HiCache ON. Swap in `config/dealignai-qwen4exp-fp8kv.yaml` + matching unit pair (main + warmup); rollback to its own pair in `config/baseline/`. The fp8kv files carry the portable round-2 items (GDN dual-end flashinfer, prefill CG full, SAM=decode, FR-Spec map sharing the same `.pt`, extra_buffer_lazy CLI flag); they differ where fp8 economics differ: steps kept 3 (fp8 accept-len 2.08–2.50 favors deeper chains), HiCache ON (safe under fp8, mandatory-off under nvfp4 per #36121), KV pool stays 552,960. The two schemes are mutually exclusive on one GPU — same port, same served-model-name, stop one then start the other.

## Pitfalls checklist

- **Concurrency 12 is a three-knob change, and mamba clamps silently.** Raise `max-running-requests`, the decode-CG `bs` list + `max_bs`, AND `max-mamba-cache-size` together. Under spec mode each request eats **4** mamba state slots (not the 2 visible in `mamba num` log lines) — 32 slots silently clamped conc 12 back to 8 with `max_running_requests is capped to 8 by the mamba state cache` in the journal; 48 slots = 12×4 is what makes 12 stick. The int8 ckpt pool doubles alongside (96 slots, 2.81 GB). Verify `max_running_requests=12` in the boot summary, not just the YAML.
- **First decode line lies.** `gen throughput` on the first window after idle (or a request's first window) divides by a denominator polluted with prefill/sleep-on-idle wait — read 0.5 tok/s once while wall-clock measured 136. Use consecutive-window values or curl timing.
- **MTP drafts share decoder `layer_id` in-process.** Any per-layer runtime hook gated on `layer_id` alone will hijack the NEXTN draft model (same class, layer 0 collides) — cold pool v2.9 gates on TopKConfig object identity registered at shrink. Symptom if you reintroduce the bug: accept rate collapses to ~0.21 with drafts masked+remapped.
- **A keep-masked router starves its own cold pool.** If staging never fires (`staged=0` after real traffic), the demand signal is coming from selection-path ids that the -inf mask already excluded — probe raw **pre-mask** logits in the alpha band instead. And any stash-buffer slice read before refill must be `.clone()`d: a view erased by `fill_` silently killed the strong-demand filter once.
- **Cold-pool pinned RAM is real RAM:** 24.15 GB flat pins (48 × ~512 MB exact-size uint8 buffers — never pin per-parameter, `CachingHostAllocator` power-of-two rounding wasted 40.5 GB doing that) + PLE's 64 GB on a 112 GB box. Check MemAvailable/swap before adding any other pinned consumer.
- **Warmup oneshot silent no-op:** after restarting the main service, `systemctl start` on the warmup unit can no-op if it already recorded `Finished`. Use `systemctl restart sglang-dealignai-nvfp4kv-warmup` and confirm `[warmup-fp4kv] warmup complete` in its journal before benchmarking. Skipping warmup on the tight round-3 pool is how late-load OOM bites on first real traffic.
- **Overlay must load:** if boot log lacks the `int8 mamba checkpoint pool:` line, the PYTHONPATH overlay didn't take (check `/proc/<pid>/environ` after daemon-reload+restart — a stale process keeps old env).
- HiCache MUST be off with nvfp4 KV (scale buffers skipped on host transfer → silent corruption; upstream #36121). fp8 KV can keep HiCache ON.
- FR-Spec map hash enters the prefix-cache namespace: swapping tables resets cached prefixes once — warm 2–3 rounds after any table change.
- `mamba-radix-cache-strategy` and `ple-offload-embedding` belong on the CLI line only; putting alias keys into YAML fails with `DeprecatedAliasStoreAction`-style errors from the merger.
- `mamba-ssm-dtype: bfloat16` is a hard dependency for the flashinfer GDN backend — missing it raises at startup.
- Keep `--model-path` on the CLI, everything else in YAML (matches this deployment's convention and the merger's behavior).
- Never fork served-model-name for experiments — distinguish instances by port only.
- Watch for OOM from late Triton specializations on extreme shapes (>100K prefill); extend the warmup script's shape families if your traffic has new ones.
- The unit intentionally ships unenabled (manual `systemctl start`) so it does not fight other GPU services at boot.

## One-command restore for our own box

On the deployment host the stack is already in place; refresh = re-copy these files, `systemctl restart sglang-dealignai-qwen4exp-nvfp4kv`, then restart the warmup unit, rerun verify steps. Nothing else is machine-specific beyond `${MODEL_DIR}` and the env path.
