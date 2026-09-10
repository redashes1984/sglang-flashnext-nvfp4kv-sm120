#!/usr/bin/env bash
# switch_fp8kv_coldpool.sh — fp8kv round-5: expert cold pool (+ optional PLE SSD backend) on CT112
# Modes:
#   cold     : PLAN A — patch-tree PYTHONPATH + cold-pool env (keep330/slots16/debug)
#              + int8 mamba ckpt + HiCache OFF (memory prerequisite: 64 PLE + 24 pool
#              + 22 hicache would overflow). PLE stays pinned, SSD untouched.
#   plefile  : PLE -> SSD file backend only (--ple-offload-backend file, nvme1).
#   on       : PLAN B+ end state = cold + plefile (hicache stays OFF in arm; re-enabling
#              HiCache 2.0 with PLE on SSD is a post-green tuning step, script stays honest).
#   off      : strip everything cold/plefile (yaml hicache restored from .bak-pre-coldpool).
#   restore  : unit file <- .bak-pre-coldpool (pre-round-5 verbatim).
# YAML backed up once to .bak-pre-coldpool before the first hicache edit.
set -euo pipefail
U=/etc/systemd/system/sglang-dealignai-qwen4exp-fp8kv.service
C=/opt/sglang-config/dealignai-qwen4exp-fp8kv.yaml
PLEDIR=/mnt/GLOWAY_YCT2TNVMe_202732/sglang-cache/ple   # PCIe4.0 nvme0 (4.5GB/s); HYV1TBX3 is PCIe3.0 (2.4GB/s) — dimin corrected 2026-09-10
KEEP=/opt/sglang-config/expert_keep_330_final.json
MODE="${1:-}"

restart_pair() {
  systemctl daemon-reload
  systemctl restart sglang-dealignai-qwen4exp-fp8kv
  systemctl restart sglang-dealignai-fp8kv-warmup || true
}

strip_cold() {
  /opt/sglang-env/bin/python3 - "$U" <<'PY'
import re, sys
u = sys.argv[1]
s = open(u).read()
for key in ("SGLANG_EXPERT_KEEP_MASK", "SGLANG_EXPERT_KEEP_OFFLOAD",
            "SGLANG_EXPERT_COLD_POOL_SLOTS", "SGLANG_COLD_DEBUG"):
    s = re.sub(rf"^#?Environment={key}=.*\n", "", s, flags=re.M)
s = s.replace("Environment=PYTHONPATH=/opt/sglang-patch/sglang/python\n", "")
s = s.replace(" \\\n  --enable-int8-mamba-checkpoint", "")
open(u, "w").write(s)
PY
}

arm_cold() {
  /opt/sglang-env/bin/python3 - "$U" "$KEEP" <<'PY'
import re, sys
u, keep = sys.argv[1], sys.argv[2]
s = open(u).read()
if "PYTHONPATH=/opt/sglang-patch" not in s:
    s = s.replace("Environment=PATH=",
        "Environment=PYTHONPATH=/opt/sglang-patch/sglang/python\nEnvironment=PATH=", 1)
for line in (f"Environment=SGLANG_EXPERT_KEEP_MASK={keep}",
             "Environment=SGLANG_EXPERT_KEEP_OFFLOAD=1",
             "Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=16",
             "Environment=SGLANG_COLD_DEBUG=1"):
    key = line.split("=", 1)[0] + "="
    s = re.sub(rf"^#?Environment={key}.*\n", "", s, flags=re.M)
    s = s.replace("Environment=CUDA_HOME=", line + "\nEnvironment=CUDA_HOME=", 1)
if "--enable-int8-mamba-checkpoint" not in s:
    s = s.replace("--mamba-radix-cache-strategy=extra_buffer_lazy",
                  "--mamba-radix-cache-strategy=extra_buffer_lazy \\\n  --enable-int8-mamba-checkpoint")
open(u, "w").write(s)
PY
}

hicache_off() {
  [ -f "$C.bak-pre-coldpool" ] || cp "$C" "$C.bak-pre-coldpool"
  /opt/sglang-env/bin/python3 - "$C" <<'PY'
import re, sys
c = sys.argv[1]
s = open(c).read()
# comment out an ACTIVE enable-hierarchical-cache (any trailing note); tolerate
# lines already commented by earlier rounds (they carry their own reasons).
s = re.sub(r"^enable-hierarchical-cache: true(.*)$",
           r"#enable-hierarchical-cache: true\1   #coldpool-off", s, flags=re.M)
open(c, "w").write(s)
active = [l for l in open(c).read().splitlines()
          if l.strip() and not l.lstrip().startswith("#")
          and "enable-hierarchical-cache" in l]
assert not active, f"hicache disable failed: {active}"
PY
}

arm_plefile() {
  mkdir -p "$PLEDIR"
  /opt/sglang-env/bin/python3 - "$U" "$PLEDIR" <<'PY'
import re, sys
u, pledir = sys.argv[1], sys.argv[2]
s = open(u).read()
# idempotency must ignore comment lines: an explanatory comment mentioning
# --ple-offload-dir would otherwise short-circuit the arm forever.
active = "\n".join(l for l in s.splitlines() if not l.lstrip().startswith("#"))
if "--ple-offload-dir" not in active:
    s = re.sub(r"(--ple-offload-embedding)(?!.*--ple-offload-backend)",
               r"\1 --ple-offload-backend file "
               r"--ple-offload-dir " + pledir, s, count=1)
# The upstream attr-100 gate is GB10-only. This x86 box reads pageable host
# memory through the IOMMU (attr 88=1, attr 100=0) and the production gather
# kernel was functionally verified against malloc + file-mmap pointers
# (test_pageable_gather.py: both byte-exact). Skip the over-strict check.
if "SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK" not in active:
    s = s.replace("[Service]\n", "[Service]\nEnvironment=SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1\n", 1)
open(u, "w").write(s)
act2 = "\n".join(l for l in open(u).read().splitlines() if not l.lstrip().startswith("#"))
assert "--ple-offload-dir" in act2 and "SKIP_DEVICE_CHECK" in act2, "plefile arm failed"
PY
}

strip_plefile() {
  /opt/sglang-env/bin/python3 - "$U" <<'PY'
import re, sys
u = sys.argv[1]
s = open(u).read()
s = re.sub(r" --ple-offload-backend file --ple-offload-dir \S+", "", s)
s = re.sub(r"^Environment=SGLANG_QWEN4_PLE_FILE_SKIP_DEVICE_CHECK=1\n", "", s, flags=re.M)
s = re.sub(r"^Environment=SGLANG_QWEN4_PLE_FILE_RSS_BUDGET_GB=\d+(\.\d+)?\n", "", s, flags=re.M)
open(u, "w").write(s)
PY
}

case "$MODE" in
  cold)
    [ -f "$U.bak-pre-coldpool" ] || cp "$U" "$U.bak-pre-coldpool"
    arm_cold
    hicache_off
    strip_plefile
    restart_pair
    echo "FP8KV-COLD-ARMED (plan A: pinned PLE, HiCache off)"
    ;;
  plefile)
    [ -f "$U.bak-pre-coldpool" ] || cp "$U" "$U.bak-pre-coldpool"
    arm_plefile
    restart_pair
    echo "FP8KV-PLEFILE-SENT"
    ;;
  on)
    [ -f "$U.bak-pre-coldpool" ] || cp "$U" "$U.bak-pre-coldpool"
    arm_cold
    arm_plefile
    hicache_off
    restart_pair
    echo "FP8KV-ON (plan B+ end state: cold pool + PLE on SSD)"
    ;;
  off)
    strip_cold
    strip_plefile
    [ -f "$C.bak-pre-coldpool" ] && cp "$C.bak-pre-coldpool" "$C"
    restart_pair
    echo "FP8KV-OFF (pinned PLE + HiCache restored; cold pool inert)"
    ;;
  restore)
    cp "$U.bak-pre-coldpool" "$U"
    [ -f "$C.bak-pre-coldpool" ] && cp "$C.bak-pre-coldpool" "$C"
    restart_pair
    echo "FP8KV-RESTORED-TO-PRE-ROUND5"
    ;;
  *)
    echo "usage: $0 cold|plefile|on|off|restore"; exit 2
    ;;
esac
