---
name: deploy-qwen-flash-next-nvfp4kv-sm120
description: Reproduce both tuned Qwen3.8-Flash-Next sglang schemes on one RTX PRO 6000 (SM120) — nvfp4kv mainline (round-5 frozen: expert cold pool + per-path checkpoint cap) plus fp8kv rollback, each with baseline snapshots.
---

# Deploy Qwen3.8-Flash-Next with NVFP4 KV on one RTX PRO 6000 (SM120)

Self-contained recipe for any agent (or human) to stand up this exact stack from a bare same-hardware box, verify it, tune it later, or roll it back. Machine-specific edits are only: model dir, sglang env path, config dir, unit name.

## What this is

Qwen3.8-Flash-Next is a 180B hybrid MoE (GDN linear attention + QSA sparse attention + native NEXTN MTP) whose official FP8 checkpoint weighs ~173 GB and needs multi-GPU. A quantized community checkpoint (~102 GB NVFP4 with BF16 PLE table, or the smaller FP8-PLE variant) fits one RTX PRO 6000 Blackwell 96 GB (SM120). This stack runs it with `nvfp4` KV cache + QSA on sglang `qwen4-main-squashed` builds. The live nvfp4kv instance is the **round-5 frozen stack**: 1M context at conc 12, spec decoding ON (NEXTN steps=2), KV pool 2,359,296, 346/512 routed experts on GPU (keep 330 + 16 dynamic slots; 166 cold experts demand-staged from 24.15 GB pinned host), int8 mamba checkpoint pool ON (128 slots / 3.74 GB) with **per-path cap `mamba-max-states-per-path 16`** and mamba cache 64. The fp8kv rollback scheme carries the same cap (unit + YAML shipped) but currently sits stopped — single-card mutex; it comes up capped.

Stack generations, each a column in README §Results: baseline → round-2 tuned → round-3 nvfp4kv-1m trial (int8 ckpt pool via patch 0002) → round-4 expert cold pool (spec + conc12 reborn) → round-5 cached=0 cap (D2/hw77 re-ask 89 s → 0.9–1.5 s full hit). Do not overwrite a generation's numbers with the next one's — the chain is the evidence.

Each scheme ships tuned-plus-baseline: active files in `config/`, pre-tuning snapshots as same-suffix `.baseline.*` under `config/baseline/` (rollback anchors + before/after evidence). `nvfp4kv` is the current mainline; `fp8kv` is the tuned rollback scheme — see §Switching below.

## Prerequisites

- Linux host, one RTX PRO 6000 Blackwell (SM120), ≥96 GB VRAM, ≥112 GB system RAM (PLE n-gram table pinned ~64 GB after power-of-two rounding + cold pool 24.15 GB pins; see README §Constraints before sizing down).
- Frozen environment (full lock in `bootstrap/`): Python 3.13 (Debian 13.1 system), torch 2.13.0, triton 3.7.1, flashinfer-python 0.6.17, sglang-kernel 0.4.6.post1 — `pip-freeze-20260911.txt` (205 pkgs, install `--no-deps`). NVIDIA open kernel module 595.71.05, CUDA 13.2.
- **Base tree (verified 2026-09-11):** `sgl-project/sglang` **official repo**, branch `qwen4-main-squashed` (PR #36497 head), pinned commit `78c5024e9` ("fix(qsa): restore SM121 correctness…#36845", 2026-08-30) — no third-party fork needed. Tarball `https://github.com/sgl-project/sglang/archive/78c5024e9.tar.gz`, sha256 `25a9af10ee3a0b4a3173e074b26b07789b4ec1aa64c9c4241bf168533f33c68e`. All shipped patches apply green on it (gold-standard rehearsed). Build SM120: `CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0"`, editable install into `/opt/sglang-env`.
- **Chat template:** `config/chat_template_froggeric_v22.5.jinja` (md5 `5c928b42…`) — community template, NOT shipped inside the HF checkpoint; both YAMLs' `chat-template:` point at it inside the model dir. Copy it there during setup or you serve with the wrong formatter.
- **ninja shim:** `bootstrap/ninja-wrapper` → `/opt/fakebin/ninja` (mode 755; units prepend `/opt/fakebin` to PATH). Forces `-j1` so flashinfer JIT never blows the cgroup memory. Missing it = sporadic boot OOM on memory-tight hosts.
- Tokenizer must be Qwen BPE — the FR-Spec token map is tokenizer-specific.

## Install steps

1. **Patch the base tree (idempotent).** Without these, QSA crashes on Triton `KeyError: 'float4_e2m1fn_x2'` and grammar requests OOM at TP=1.
   ```bash
   export SGLANG_SRT=/opt/sglang-src/sglang/python/sglang/srt
   /opt/sglang-env/bin/python patches/apply_nvfp4_patches.py   # ALL-PATCHED + py_compile each
   git -C /opt/sglang-src/sglang apply /path/patches/0001-sampler-37962-tp1-sync-skip.diff
   ```
   Re-run after any source sync. The patcher checks anchor counts.
2. **Overlay tree (rounds 3–5).** `cp -a /opt/sglang-src /opt/sglang-patch` (fork isolation), then inside the patch tree:
   ```bash
   git -C /opt/sglang-patch apply --check -p1 ../patches/0002-mamba-ckpt-ple.diff  # NOTE: 0002 paths
   git -C /opt/sglang-patch apply -p1 ../patches/0002-mamba-ckpt-ple.diff          # are repo-parent-relative
   CUDA_VISIBLE_DEVICES= /opt/sglang-env/bin/python scripts/test_ckpt_ple_patch.py  # T1–T6 all-ok
   ```
   0002 lets the int8 mamba checkpoint pool coexist with Qwen4-Exp PLE side states by mirroring them into ckpt-slot buffers instead of the upstream ValueError guard (upstream PR #38619). Path-prefix gotcha: 0001 targets `python/…` (apply inside the repo root dir), 0002 targets `sglang/python/…` (apply one level up) — wrong level = clean "No such file" rejection, harmless but confusing.
3. **Place configs.** YAMLs → `/opt/sglang-config/` (replace `${MODEL_DIR}` manually — systemd does NOT expand inside YAML content). Copy the chat template into the model dir. Ship `scripts/frspec_map_64k.pt` + `config/expert_keep_330_final.json` next to the YAMLs. Keep `served-model-name` verbatim if downstream routers key on it.
4. **Install units** (`config/systemd/`, main + warmup per scheme; adjust paths). Each unit ships its CLI-only items the YAML merger rejects: `--ple-offload-embedding`, `--mamba-radix-cache-strategy=extra_buffer_lazy`, plus `--enable-int8-mamba-checkpoint --mamba-max-states-per-path 16` and `Environment=PYTHONPATH=/opt/sglang-patch/sglang/python`. `systemctl daemon-reload`; start is manual (units ship unenabled by design — boot-time GPU contention).
   ```bash
   systemctl start sglang-dealignai-qwen4exp-nvfp4kv          # cold start ~4-5 min (weights + shrink + capture)
   systemctl restart sglang-dealignai-nvfp4kv-warmup          # restart, NOT start — see warmup gotcha
   ```
5. **Expert cold pool (round 4).** Into the same overlay:
   ```bash
   cp patches/expert_cold_pool.py $S/layers/moe/expert_cold_pool.py   # $S = overlay srt dir
   /opt/sglang-env/bin/python scripts/deploy_expert_cold_pool.py       # 5 anchors, idempotent
   /opt/sglang-env/bin/python tests/test_cold_pool_logic.py           # T1–T9 CPU-only, ALL PASS
   ```
   Unit env: `SGLANG_EXPERT_KEEP_MASK=<json>`, `SGLANG_EXPERT_KEEP_OFFLOAD=1`, `SGLANG_EXPERT_COLD_POOL_SLOTS=16`, `SGLANG_COLD_DEBUG=1`. Walk the ladder `scripts/switch_coldpool.sh {eager|graphs|mtp|off}` — **eager first**. Arm proof: `[COLD-POOL] SHRUNK 48 layers keep=[330] +16 slots -> host pinned 24.15 GB`; correctness gate: `checksum selfcheck: 8448 rows OK`, zero MISMATCH. Red lines (v1 incident, enforced by design): shrink strictly after `process_weights_after_loading`; stage only in after-forward hook, never inside graph replay; remap/bias tables updated by content, shapes constant; unknown per-expert tensor refuses to arm.
6. **Verify (round-5 frozen expectations).**
   ```bash
   curl -s localhost:8000/get_server_info | python3 -c "import json,sys;d=json.load(sys.stdin);print('ctx',d.get('context_length'),'conc',d.get('max_running_requests'),'KV',d.get('max_total_tokens'),'spec',d.get('speculative_algorithm'),'mamba',d.get('max_mamba_cache_size'))"
   journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --since -30m | grep -E "int8 mamba checkpoint pool|COLD-POOL|SHRUNK"
   ```
   Expected: ctx 1048576, conc 12, KV 2359296, EAGLE steps=2, `int8 mamba checkpoint pool: 128 slots, 3.74GB`, SHRUNK line, mamba 64. Then a cached=0 smoke: `python3 tests/probe_cached0_discrim.py` — D1 and D2 re-asks must both hit (0.5–1 s, cached ≈ prompt length). Watch late `device-loaded` lines trend to zero after warmup.

## Round-5 cached=0 cap (why it's mandatory)

Under `extra_buffer_lazy` + int8 checkpoints, chunked prefill donates **one int8 checkpoint per 4096-token chunk**; uncapped (`mamba_max_states_per_path: -1` default), a single 549K chain eats ≈134 of the 128 pool slots → two long chains mutually LRU-evict each other's checkpoints → `cached=0` on re-ask (tree + KV pages present, match dead), self-amplifying on misses. Cap 16 keeps N chains co-resident at zero VRAM cost; whole-document reuse untouched, mid-doc branching recomputes a segment (self-heals). Do not "simplify" the flag away and do not expand the pool instead (256 slots = +3.7 GB and still linear-scaling). Full forensics: README §Fifth round + `~/.hermes` skill `sglang-service-operations` ref `cached0-rootcause-e3-20260911.md`.

## Bench

```bash
python3 scripts/bench_gdn.py            # or your harness: C1 sweep + aggregate
```
Round-4/5 acceptance-bench reference (same GPU, this build, post-warmup, steps=2/draft3/thr1.0): **C1 ≈122–129 tok/s, C12 aggregate 606–710 tok/s** (band widened by the cap-era rerun; accept len 2.07–2.31 / rate 0.54–0.65), prefill ≈9.7K fresh-prefix. Superseded: warm-cache-window 146–171 / 773–825. fp8kv (steps2 era): C4 aggregate ≈431–438 tok/s. Measure after warmup, run twice. Log-reading: first `gen throughput` window after idle/request-entry is polluted (prefill/sleep-on-idle denominator) — trust consecutive windows or wall-clock curl. Greedy byte-level nondeterminism exists **even with cold pool OFF** — gate on checksum + semantics + error counters, never raw bytes.

## Rollback

```bash
cp config/baseline/dealignai-qwen4exp-nvfp4kv.baseline.yaml           /opt/sglang-config/dealignai-qwen4exp-nvfp4kv.yaml
cp config/baseline/sglang-dealignai-qwen4exp-nvfp4kv.baseline.service /etc/systemd/system/sglang-dealignai-qwen4exp-nvfp4kv.service
systemctl daemon-reload && systemctl restart sglang-dealignai-qwen4exp-nvfp4kv
```
Baseline = C1 ≈122 / C6 ≈409 (768K, spec-on, no ckpt overlay). Within-round knobs: cold-pool ladder `switch_coldpool.sh off`; round-5 rollback = delete the cap flag (reintroduces multi-chain cached=0 — only useful as a diagnostic control).

## Switching to the fp8 KV scheme

fp8kv frozen state (in-repo YAML+unit): conc4 / YaRN×2 512K / **steps=2** (round-4 A/B beat steps=3 under concurrency) / KV pool **1,310,720** (B+ rung2: 2× full 524K co-resident) / mamba 48 + **cap 16** / int8 ckpt ON / **HiCache OFF** (upstream fail-fast vs int8 ckpt, `server_args.py:6333`) / expert cold pool via `scripts/switch_fp8kv_coldpool.sh`. Same port, same served-model-name, mutually exclusive — stop one unit, start the other, restart its warmup. It is currently stopped on the fleet box; its cap16 activates on next bring-up (verify `mamba_max_states_per_path 16` in server args then). Rollback: `config/baseline/` fp8kv pair.

## Full restore on a fresh same-hardware box

`bootstrap/MANIFEST.md` is the authoritative checklist: base tarball + sha256, pip lock, ninja shim, chat template, per-file md5 table for the 7-file overlay delta (models/qwen4_exp.py, qwen4_exp_ple_table.py, model_executor/model_runner.py, layers/moe/topk.py, layers/moe/expert_cold_pool.py, mem_cache/mamba_checkpoint_pool.py, mem_cache/memory_pool.py — `runtime-src/` holds the exact live bytes; the diff-vs-clean-tree list there is complete and verified 2026-09-11). Order: env lock → base tarball build + patcher/0001 → overlay + 0002 → cold-pool deploy → configs/units/shim/template → start → verify step 6. The repo is the backup: frozen remote state == git HEAD ≥ `df987bc` (live YAMLs md5-verified against repo same day; intermediate `.bak-*` on the box deleted 2026-09-11, rollup tarball `CT112:/root/backups/sglang-frozen-round5-20260911.tar.gz` sha256 `80d88f36…` in MANIFEST).

## Pitfalls checklist

- **Concurrency 12 is a multi-knob change, mamba clamps silently.** Raise `max-running-requests`, decode-CG `bs` list + `max_bs`, AND `max-mamba-cache-size` together. Spec mode eats **4** mamba slots per request (not the 2 visible in `mamba num` logs); 64 = 12×4 + 16 anchor slack. The int8 ckpt pool auto-doubles alongside (128 slots at mamba 64). Verify in the boot summary, not just YAML.
- **No checkpoint cap = cached=0 on multi-chain workloads** (see §Round-5). Cap 16 ships in the unit CLI flags; a YAML-only restore silently drops it — `get_server_info` check after every bring-up.
- **First decode line lies.** See §Bench log-reading note.
- **MTP drafts share decoder `layer_id` in-process.** Per-layer runtime hooks gated on `layer_id` alone hijack the NEXTN draft (accept rate collapses to ~0.21). Cold pool v2.9 gates on TopKConfig object identity registered at shrink.
- **A keep-masked router starves its own cold pool.** `staged=0` under traffic ⇒ demand must be probed from raw **pre-mask** logits (alpha band). Any stash-buffer slice read before refill must be `.clone()`d — a `fill_`-erased view once silently killed the strong-demand filter.
- **Cold-pool pinned RAM is real RAM:** 24.15 GB flat pins (exact-size uint8 buffers per layer; never pin per-parameter — `CachingHostAllocator` power-of-two rounding wasted 40.5 GB doing that) + PLE 64 GB on a 112 GB box. Check MemAvailable/swap before any new pinned consumer.
- **Warmup oneshot silent no-op:** after restarting main, `systemctl start` on warmup can no-op if it already recorded `Finished` — use `restart` and confirm `[warmup-fp4kv] warmup complete` before benching. Skipping warmup on the tight pool = late-load OOM on first traffic.
- **Overlay must load:** no `int8 mamba checkpoint pool:` line in boot log ⇒ PYTHONPATH overlay didn't take (check `/proc/<pid>/environ` after daemon-reload+restart — stale processes keep old env; mtime lies, process env doesn't).
- **HiCache must be OFF on both schemes** — nvfp4 KV because scale buffers skip host transfer → silent corruption (#36121); fp8kv because the int8-ckpt path fail-fasts (`server_args.py:6333`).
- **ninja shim missing** ⇒ flashinfer JIT compiles at full parallelism and can blow the container cgroup. `/opt/fakebin` must be first in the unit PATH.
- FR-Spec map hash enters the prefix-cache namespace: swapping tables resets cached prefixes once — warm 2–3 rounds after any table change.
- `mamba-radix-cache-strategy` + `ple-offload-embedding` are CLI-line-only (alias/BooleanOptionalAction args the merger rejects).
- `mamba-ssm-dtype: bfloat16` is a hard dependency of the flashinfer GDN backend — missing raises at startup.
- Keep `--model-path` on the CLI, everything else in YAML; never fork served-model-name (distinguish instances by port only); units intentionally unenabled.
- Watch late Triton specializations on extreme shapes (>100K prefill); extend warmup shape families if traffic introduces new ones.
