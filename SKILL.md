---
name: deploy-qwen-flash-next-nvfp4kv-sm120
description: Reproduce both tuned Qwen3.8-Flash-Next sglang schemes on one RTX PRO 6000 (SM120) — nvfp4kv mainline (round-3 1M trial state) plus fp8kv rollback, each with baseline snapshots.
---

# Deploy Qwen3.8-Flash-Next with NVFP4 KV on one RTX PRO 6000 (SM120)

Self-contained recipe for any agent (or human) to stand up this exact stack, verify it, tune it later, or roll it back. Only four things need editing for a different machine: model dir, sglang env path, config dir, unit name.

## What this is

Qwen3.8-Flash-Next is a 180B hybrid MoE (GDN linear attention + QSA sparse attention + native NEXTN MTP) whose official FP8 checkpoint weighs ~173 GB and needs multi-GPU. A quantized community checkpoint (~102 GB NVFP4 with BF16 PLE table, or the smaller FP8-PLE variant) fits one RTX PRO 6000 Blackwell 96 GB (SM120). This stack runs it with `nvfp4` KV cache + QSA on sglang `qwen4-main-squashed` builds. The live nvfp4kv instance is the **round-3 nvfp4kv-1m trial state**: 1M context at conc 6 with the int8 mamba checkpoint pool enabled (patch 0002), spec decoding OFF. The fp8kv rollback scheme carries the portable subset of the tuning with spec ON (§Switching below).

Stack generations, each documented in README §Results as its own column: baseline → round-2 tuned (C1 ≈136 / C6 ≈550–575, spec-on 768K) → round-3 nvfp4kv-1m trial (live). Do not overwrite a generation's numbers with the next one's — the chain is the evidence.

Each scheme ships tuned-plus-baseline: active files in `config/`, pre-tuning snapshots as same-suffix `.baseline.*` under `config/baseline/` (rollback anchors + before/after evidence). `nvfp4kv` is the current mainline; `fp8kv` is the tuned rollback scheme — see §Switching to the fp8 KV scheme.

## Prerequisites

- Linux host, one RTX PRO 6000 Blackwell (SM120), ≥96 GB VRAM, ≥128 GB system RAM (PLE n-gram table lives pinned in CPU RAM ~44–51 GB depending on variant).
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
   Expected live values on the round-3 stack: ctx 1048576, conc 6, KV pool 1179648, spec None (OFF), ckpt True, gdn flashinfer; journal shows `int8 mamba checkpoint pool: 48 slots, 1.42GB (qdata 1.29 + scale 0.02 + conv 0.10 + ple 0.01); active mamba pool 24 slots`. Then run one real chat completion and watch the journal for late `device-loaded` lines (`journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --since -10m | grep -c 'device-loaded'` → should trend to zero after warmup).

## Bench

```bash
python3 scripts/bench_gdn.py            # or your own harness: C1 latency sweep + C6 aggregate
```
Reference hot-state numbers for the round-3 nvfp4kv-1m trial stack (same GPU, this sglang build): C1 ≈75–90 tok/s, C6 aggregate ≈72–91 tok/s, prefill ≈9.7K tok/s fresh-prefix (higher on radix hits). Run twice — first pass warms caches. If your numbers are far off, check the eight tuning items (below) are all present. Reference hot-state for the fp8kv scheme instead: C1 ≈165 / C6 aggregate ≈355–365 tok/s (conc4, spec-on, steps=3).

Why nvfp4kv C1 looks low next to fp8kv: spec-off at 1M plus gather-dequant overhead. The trade is deliberate — 2× context capacity (1M vs 512K) and eviction headroom for many distinct long prefixes. If single-stream latency dominates, switch schemes instead of tuning around it.

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
