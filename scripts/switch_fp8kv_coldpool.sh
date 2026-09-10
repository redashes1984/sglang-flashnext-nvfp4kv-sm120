#!/usr/bin/env bash
# switch_fp8kv_coldpool.sh — fp8kv scheme round-5: PLE->SSD file backend + expert cold pool (CT112)
# Modes: plefile | on | off
#   plefile : unit gains --ple-offload-backend file --ple-offload-dir <nvme> ONLY.
#             Frees the 64GB pinned PLE table (host RAM prerequisite for cold pool + HiCache).
#   on      : plefile + PYTHONPATH patch tree + cold-pool env trio (keep330+slots16,
#             DEBUG/ALPHA parity with live nvfp4kv round-4) + --enable-int8-mamba-checkpoint.
#   off     : restore unit to .bak-pre-coldpool (pre-round-5), i.e. pinned PLE + no cold pool.
# YAML is NEVER touched (pool 552,960 / conc4 / steps3 stay as accepted 2026-09-08).
# Idempotent; backups: .bak-pre-coldpool (unit, first run only).
set -euo pipefail
U=/etc/systemd/system/sglang-dealignai-qwen4exp-fp8kv.service
PLEDIR=/mnt/HYV1TBX3_Pro_001173/sglang-cache/ple
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
            "SGLANG_EXPERT_COLD_POOL_SLOTS", "SGLANG_COLD_DEBUG",
            "SGLANG_COLD_STRONG_ALPHA"):
    s = re.sub(rf"^Environment=?#?{key}=.*\n", "", s, flags=re.M)
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
    s = re.sub(rf"^#?{key}.*\n", "", s, flags=re.M)
    s = s.replace("Environment=CUDA_HOME=", line + "\nEnvironment=CUDA_HOME=", 1)
if "--enable-int8-mamba-checkpoint" not in s:
    s = s.replace("--mamba-radix-cache-strategy=extra_buffer_lazy",
                  "--mamba-radix-cache-strategy=extra_buffer_lazy \\\n  --enable-int8-mamba-checkpoint")
open(u, "w").write(s)
PY
}

arm_plefile() {
  mkdir -p "$PLEDIR"
  /opt/sglang-env/bin/python3 - "$U" <<'PY'
import re, sys
u = sys.argv[1]
s = open(u).read()
if "--ple-offload-dir" not in s:
    s = re.sub(r"(--ple-offload-embedding)(?!.*--ple-offload-backend)",
               r"\1 --ple-offload-backend file "
               r"--ple-offload-dir /mnt/HYV1TBX3_Pro_001173/sglang-cache/ple", s, count=1)
open(u, "w").write(s)
PY
}

strip_plefile() {
  /opt/sglang-env/bin/python3 - "$U" <<'PY'
import re, sys
u = sys.argv[1]
s = open(u).read()
s = re.sub(r" --ple-offload-backend file --ple-offload-dir \S+", "", s)
open(u, "w").write(s)
PY
}

case "$MODE" in
  plefile)
    [ -f "$U.bak-pre-coldpool" ] || cp "$U" "$U.bak-pre-coldpool"
    arm_plefile
    restart_pair
    echo "FP8KV-PLEFILE-SENT"
    ;;
  on)
    [ -f "$U.bak-pre-coldpool" ] || cp "$U" "$U.bak-pre-coldpool"
    arm_plefile
    arm_cold
    restart_pair
    echo "FP8KV-COLDPOOL-ON-SENT"
    ;;
  off)
    strip_cold
    strip_plefile
    restart_pair
    echo "FP8KV-COLDPOOL-OFF-SENT (ple pinned restored; cold pool inert)"
    ;;
  restore)
    cp "$U.bak-pre-coldpool" "$U"
    restart_pair
    echo "FP8KV-UNIT-RESTORED"
    ;;
  *)
    echo "usage: $0 plefile|on|off|restore"; exit 2
    ;;
esac
