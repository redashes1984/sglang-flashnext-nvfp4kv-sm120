# Bootstrap manifest — frozen environment for full restore (2026-09-11)

Everything a same-hardware new box needs that is NOT rebuildable from the patcher
alone. Verify each hash before trusting a restored file.

## Baseline source tree (the ground truth, verified 2026-09-11)

- Project: `sgl-project/sglang`, branch `qwen4-main-squashed` (PR #36497 head)
- Pinned commit: `78c5024e9` — **exists in the OFFICIAL repo**, commit date
  2026-08-30T16:05:24Z, message: "fix(qsa): restore SM121 correctness with
  Humanize and Kernel Design Agent (#36845)". No third-party fork needed.
- Source tarball: `https://github.com/sgl-project/sglang/archive/78c5024e9.tar.gz`
  sha256 `25a9af10ee3a0b4a3173e074b26b07789b4ec1aa64c9c4241bf168533f33c68e`
- Gold-standard verified on this date: `apply_nvfp4_patches.py` + `0001` + `0002`
  (with `-p2`) + cold-pool deploy all apply ALL-green on the clean tarball tree.
- Build for SM120: `CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0"`, editable install
  into `/opt/sglang-env` (see pip lock below).

## Python environment lock

- `pip-freeze-20260911.txt` — full 205-package freeze of `/opt/sglang-env`
  (Debian 13.1 system Python **3.13**, torch 2.13.0, triton 3.7.1,
  flashinfer-python 0.6.17, sglang-kernel 0.4.6.post1).
  Install with `--no-deps` (or a fresh venv + this exact lock) — do not let pip
  resolve sideways and drift versions.

## System-level pieces

| Item | Content | Where it goes |
|---|---|---|
| `ninja-wrapper` | ninja `-j1` shim (flashinfer JIT memory guard) | `/opt/fakebin/ninja` (mode 755); unit `Environment=PATH` prepends `/opt/fakebin` |
| `chat_template_froggeric_v22.5.jinja` | md5 `5c928b42a335c295c31789e97713c778` — community template, **not shipped in the HF checkpoint**; YAML `chat-template:` points at it | into the model dir |

## NVIDIA stack (host, not python)

- GPU: RTX PRO 6000 Blackwell Workstation Edition, 97887 MiB (SM120)
- Open Kernel Module `595.71.05` (from the driver's own /proc/driver/nvidia banner)
- CUDA toolkit `13.2` (nvcc V13.2.78)
- Host: Debian 13.1, kernel 6.x-pve class, ≥112 GB RAM, NVMe model volume

## Overlay delta — authoritative file inventory (src vs patch, verified 2026-09-11)

Build `/opt/sglang-patch` = `cp -a /opt/sglang-src` (the patched base tree), then
these exact files must match the hashes; anything else in the tree is a lie.

| File (under `sglang/python/sglang/srt/`) | md5 (live + repo `runtime-src/`) | Produced by |
|---|---|---|
| `layers/moe/topk.py` | `d150f8fde0a68d681d70c341ba201603` | deploy_expert_cold_pool.py (5 anchors) |
| `layers/moe/expert_cold_pool.py` | `e891bd62aa6261e1e3be6a588d008fc7` | copy patches/expert_cold_pool.py |
| `model_executor/model_runner.py` | `32e773e0111c9b305e30417c2d4648a8` | deploy_expert_cold_pool.py |
| `models/qwen4_exp.py` | `8aada4a8416447a87ad8781e2a9f27fb` | (runtime-src snapshot) |
| `models/qwen4_exp_ple_table.py` | `b788ee8fcb2969bb525cb0617c1d801f` | (runtime-src snapshot) |
| `mem_cache/mamba_checkpoint_pool.py` | `ccc608bb3faa7ac9dc253380af32db52` | patch 0002 |
| `mem_cache/memory_pool.py` | `90b27aad2ed8e2829a95ce968cfb47eb` | patcher + patch 0002 |

Base tree (before overlay): QSA patcher also creates `layers/attention/qsa_nvfp4_kv.py`
and edits `qwen_sparse_attn_backend.py`, `fp4_kv_cache_quant_method.py`,
`pool_configurator.py` in `/opt/sglang-src` itself — the patcher is idempotent,
re-run it after any base-tree sync.

## Artifacts carried elsewhere in the repo (already versioned)

- `config/expert_keep_330_final.json` md5 `bd8d6abb7bde8bab937db9887fdc58da`
- `scripts/frspec_map_64k.pt` md5 `49f23b7fb5a30459e10c817e1e76b1c5` (+ manifest for provenance)
- QSA-patched base files: created/verified by running the patcher; spot-check
  `layers/attention/qsa_nvfp4_kv.py` (live md5 `1197e264dfdd8b27e037c4f6a7dc7718`)
  and sampler guard marker `dspark-37962` in `layers/sampler.py` (both trees carry 0001).
- Model weights: `RadixArk`-lineage `dealignai_Qwen3.8-Flash-Next-ABLITERATED-NVFP4`
  — re-download from HF; record a sha256 ledger per-file into the shared-disk
  MODEL_CARD before serving.

## What this does NOT include (known boundaries)

- HF download credentials / model license acceptance (site-specific).
- The 24.15 GB cold-pool pin + 64 GB PLE pin need a ≥112 GB host **with the same
  overcommit discipline**; see README §Constraints before sizing down.

## Frozen-state archive on the deployment host (2026-09-11 cleanup)

All intermediate `.bak-*` litter (units, YAMLs, src-tree files, probe .out logs,
model-dir config backups) was deleted after this kit landed; the repo + git
history is the single source of truth. One rollup backup remains:

`CT112:/root/backups/sglang-frozen-round5-20260911.tar.gz`
sha256 `80d88f36893ef4eac69f8c84ce80814234304ab7fd8ef7b4a14eb983cf1223ab`
(14 entries: both YAMLs, all 4 units, keep-mask, FR-Spec map, both warmups,
expert_cold_pool.py — all md5-verified against the repo. Probe outputs are not
archived: their numbers live in README §Fifth round and the git log). Out of
scope and untouched: the 17 `vllm-*.bak` unit files on the same box (different
project).
