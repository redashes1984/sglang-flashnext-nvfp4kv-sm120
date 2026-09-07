---
name: deploy-qwen-flash-next-nvfp4kv-sm120
description: Reproduce the nvfp4kv Qwen3.8-Flash-Next sglang deployment on a single RTX PRO 6000 (SM120), including the second-round tuned stack and baseline rollback.
---

# Deploy Qwen3.8-Flash-Next with NVFP4 KV on one RTX PRO 6000 (SM120)

Self-contained recipe for any agent (or human) to stand up this exact stack, verify it, tune it later, or roll it back. Only four things need editing for a different machine: model dir, sglang env path, config dir, unit name.

## What this is

Qwen3.8-Flash-Next is a 180B hybrid MoE (GDN linear attention + QSA sparse attention + native NEXTN MTP) whose official FP8 checkpoint weighs ~173 GB and needs multi-GPU. A quantized community checkpoint (~102 GB NVFP4 with BF16 PLE table, or the smaller FP8-PLE variant) fits one RTX PRO 6000 Blackwell 96 GB (SM120). This stack runs it with `nvfp4` KV cache + QSA on sglang `qwen4-main-squashed` builds, plus seven tuning changes that raise aggregate throughput ~36% at conc 6.

Two recorded config versions ship here: `config/baseline/` (pre-tuning snapshot, rollback anchor) and the current tuned files in `config/`. fp8kv is the second scheme (different trade-off, see `config/README-notes` below).

## Prerequisites

- Linux host, one RTX PRO 6000 Blackwell (SM120), ≥96 GB VRAM, ≥128 GB system RAM (PLE n-gram table lives pinned in CPU RAM ~44–51 GB depending on variant).
- sglang built from the `qwen4-main-squashed` branch head (PR #36497) compiled for SM120: `CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0"`, PyTorch cu13x. See `jpezzulli/sglang-rtxpro6000` and `gabrielolympie/sglang-flashnext-sm120` for build notes. Verify: `python3 -c "import sglang; print(sglang.__version__)"` in `/opt/sglang-env`.
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
2. **Place configs.** Copy `config/dealignai-qwen4exp-nvfp4kv.yaml` to `/opt/sglang-config/`. Replace `${MODEL_DIR}` inside the YAML manually (systemd does NOT expand variables inside YAML content). Keep `served-model-name` verbatim if you have downstream routers keyed on it — renaming breaks routing silently.
3. **Place the FR-Spec artifact.** Copy `scripts/frspec_map_64k.pt` (+ manifest for audit) next to the YAML. Optional but recommended — it carries the +accept-len gain; without it the tuned stack loses most of its edge.
4. **Install units.** Copy `config/systemd/sglang-dealignai-qwen4exp-nvfp4kv.service` and `sglang-dealignai-nvfp4kv-warmup.service` to `/etc/systemd/system/`, adjust paths, then:
   ```bash
   systemctl daemon-reload
   systemctl start sglang-dealignai-qwen4exp-nvfp4kv
   systemctl --no-pager status sglang-dealignai-qwen4exp-nvfp4kv   # wait until /health responds
   systemctl enable --now sglang-dealignai-nvfp4kv-warmup          # one-shot pre-warm, exits when done
   ```
   Cold start is ~4 min (weight load + graph capture). Only one scheme runs at a time — stop the other before starting this one (shared GPU).
   The two CLI-only flags in `ExecStart` are required because the YAML config merger rejects alias / `BooleanOptionalAction` args: `--ple-offload-embedding`, `--mamba-radix-cache-strategy=extra_buffer_lazy`. Everything else lives in the YAML.
5. **Verify.**
   ```bash
   curl -s localhost:8000/v1/models | head -c 200          # expects Qwen3.8-Flash-Next-NVFP4
   curl -s localhost:8000/get_server_info | python3 -c "import json,sys;d=json.load(sys.stdin);print('steps',d.get('speculative_num_steps'),'draft',d.get('speculative_num_draft_tokens'),'topk',d.get('speculative_eagle_topk'),'map',bool(d.get('speculative_token_map')),'conc',d.get('max_running_requests'),'pool',d.get('max_total_tokens'),'gdn',d.get('linear_attn_prefill_backend'))"
   ```
   Expected live values on the tuned stack: steps 2, draft 3, topk 1, map True, conc 6, pool 851968, gdn flashinfer. Then run one real chat completion and watch the journal for late `device-loaded` lines (`journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --since -10m | grep -c 'device-loaded'` → should trend to zero after warmup).

## Bench

```bash
python3 scripts/bench_gdn.py            # or your own harness: C1 latency sweep + C6 aggregate
```
Reference hot-state numbers (same GPU, this sglang build): C1 ≈136 tok/s, C6 aggregate ≈550–575 tok/s, prefill ≈10K tok/s, accept len ≈2.0–2.3 @ rate ≈0.5–0.66, mamba peak usage ≤12 slots. Run twice — first pass warms caches. If your numbers are far off, check the seven tuning items (below) are all present.

## Tuning items vs baseline (what changed, why, effect)

See README § What changed vs baseline — the table there is the authority; summary: GDN flashinfer both ends; mamba cap pinned 24; radix strategy lazy; prefill CUDA graph forced full; `speculative-attention-mode: decode`; FR-Spec token map; NEXTN steps 3→2. Net +11% C1 / +36% C6.

## Rollback

```bash
cp config/baseline/dealignai-qwen4exp-nvfp4kv.baseline.yaml       /opt/sglang-config/dealignai-qwen4exp-nvfp4kv.yaml
cp config/baseline/sglang-dealignai-qwen4exp-nvfp4kv.baseline.service /etc/systemd/system/sglang-dealignai-qwen4exp-nvfp4kv.service
systemctl daemon-reload && systemctl restart sglang-dealignai-qwen4exp-nvfp4kv
```
Baseline is expected to measure C1 ≈122 / C6 ≈409.

## Switching to the fp8 KV scheme

Use it when you need maximum single-stream decode (~200 tok/s C1) and can live with 512K ctx / conc 4. Swap in `config/dealignai-qwen4exp-fp8kv.yaml` + matching unit. As of 2026-09-08 the fp8kv files carry the same portable tuning items as nvfp4kv (GDN dual-end flashinfer, prefill CG full, SAM=decode, FR-Spec map, extra_buffer_lazy CLI flag); they differ only where fp8 economics differ: steps=3, HiCache ON, conc4/512K. The two schemes are mutually exclusive on one GPU.

## Pitfalls checklist

- HiCache MUST be off with nvfp4 KV (scale buffers skipped on host transfer → silent corruption; upstream #36121). fp8 KV can keep HiCache ON.
- FR-Spec map hash enters the prefix-cache namespace: swapping tables resets cached prefixes once — warm 2–3 rounds after any table change.
- `mamba-radix-cache-strategy` and `ple-offload-embedding` belong on the CLI line only; putting alias keys into YAML fails with `DeprecatedAliasStoreAction`-style errors from the merger.
- `mamba-ssm-dtype: bfloat16` is a hard dependency for the flashinfer GDN backend — missing it raises at startup.
- Keep `--model-path` on the CLI, everything else in YAML (matches this deployment's convention and the merger's behavior).
- Never fork served-model-name for experiments — distinguish instances by port only.
- Watch for OOM from late Triton specializations on extreme shapes (>100K prefill); extend the warmup script's shape families if your traffic has new ones.
- The unit intentionally ships unenabled (manual `systemctl start`) so it does not fight other GPU services at boot.

## One-command restore for our own box

On the deployment host the stack is already in place; refresh = re-copy these files, `systemctl restart sglang-dealignai-qwen4exp-nvfp4kv`, rerun verify steps. Nothing else is machine-specific beyond `${MODEL_DIR}` and the env path.
