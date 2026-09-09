#!/usr/bin/env bash
# switch_coldpool.sh — CT112 冷池配置切换器（在 CT112 本机跑）
# 用法:  ./switch_coldpool.sh eager|graphs|off
#   eager  = Phase-2 判别实验第一道门：冷池开 + decode 图关（30min soak + 探针）
#   graphs = 第二道门：冷池开 + decode 图恢复 full bs=[1,2,4,6]
#   off    = 回滚：pre-tier 备份 yaml + env 注释（生产原态）
# 幂等、有备份、每步可回退。绝不手动 sed 生产文件。
set -euo pipefail

CFG=/opt/sglang-config/dealignai-qwen4exp-nvfp4kv.yaml
BAK=/opt/sglang-config/dealignai-qwen4exp-nvfp4kv.yaml.bak-pre-tier
UNIT=/etc/systemd/system/sglang-dealignai-qwen4exp-nvfp4kv.service
KEEP=/opt/sglang-config/expert_keep_330_final.json
SVC=sglang-dealignai-qwen4exp-nvfp4kv
WARM=sglang-dealignai-nvfp4kv-warmup

MODE=${1:?usage: switch_coldpool.sh eager|graphs|off}

# 先备份当前运行配置（只留最新一份，带时间戳的存档另议）
if [[ ! -f $BAK ]]; then
  cp $CFG $BAK
  echo "[bak] $CFG -> $BAK"
fi

case "$MODE" in
  eager)
    # 1) env 三行：取消注释/追加
    grep -q "^Environment=SGLANG_EXPERT_KEEP_MASK=" "$UNIT" || \
      sed -i "/^#Environment=SGLANG_EXPERT_KEEP_MASK=/s/^#//" "$UNIT"
    grep -q "^Environment=SGLANG_EXPERT_KEEP_OFFLOAD=" "$UNIT" || \
      sed -i "/^#Environment=SGLANG_EXPERT_KEEP_OFFLOAD=/s/^#//" "$UNIT"
    grep -q "^Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=" "$UNIT" || \
      sed -i "/^Environment=SGLANG_EXPERT_KEEP_OFFLOAD=/a Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=16" "$UNIT"
    # 2) yaml：decode 图关 + KV 池开大（tier 态 1572864）
    python3 - "$CFG" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"(cuda-graph-config:\n  decode:\n)((?:[ ]{4,}.*\n)+)",
           r"\1    backend: disabled   # coldpool-eager (switch_coldpool.sh)\n", s)
s = re.sub(r"^max-total-tokens: \d+", "max-total-tokens: 1572864", s, flags=re.M)
open(p, "w").write(s)
assert "backend: disabled   # coldpool-eager" in s, "decode block rewrite failed"
print("[yaml] decode=disabled, max-total-tokens=1572864")
PY
    ;;
  graphs)
    grep -q "^Environment=SGLANG_EXPERT_KEEP_MASK=" "$UNIT" || \
      sed -i "/^#Environment=SGLANG_EXPERT_KEEP_MASK=/s/^#//" "$UNIT"
    grep -q "^Environment=SGLANG_EXPERT_KEEP_OFFLOAD=" "$UNIT" || \
      sed -i "/^#Environment=SGLANG_EXPERT_KEEP_OFFLOAD=/s/^#//" "$UNIT"
    grep -q "^Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=" "$UNIT" || \
      sed -i "/^Environment=SGLANG_EXPERT_KEEP_OFFLOAD=/a Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=16" "$UNIT"
    # 从 pre-tier 基线恢复（decode 图 bs 块原样回来），再放大 KV 池
    cp $BAK $CFG
    python3 - "$CFG" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"^max-total-tokens: \d+", "max-total-tokens: 1572864", s, flags=re.M)
open(p, "w").write(s)
assert "bs: [1, 2, 4, 6]" in s, "decode bs block missing after restore"
print("[yaml] decode graphs restored, max-total-tokens=1572864")
PY
    ;;
  off)
    sed -i "s/^Environment=SGLANG_EXPERT_KEEP_MASK=/#&/; s/^Environment=SGLANG_EXPERT_KEEP_OFFLOAD=/#&/; s/^Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=/#&/" "$UNIT"
    cp $BAK $CFG
    echo "[yaml] restored $BAK"
    ;;
  *) echo "bad mode"; exit 1;;
esac

systemctl daemon-reload
systemctl restart $SVC
echo "[wait] restart $SVC ..."
for i in $(seq 1 24); do
  sleep 30
  ST=$(systemctl is-active $SVC || true)
  HC=$(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || echo 000)
  echo "check$i: svc=$ST health=$HC"
  if [[ "$HC" == "200" ]]; then break; fi
  if [[ "$ST" == "failed" ]]; then
    echo "SERVICE FAILED — journal tail:"
    journalctl -u $SVC --no-pager -n 40
    exit 1
  fi
done
[[ "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" == "200" ]] || { echo "NOT READY"; exit 1; }

systemctl restart $WARM  # warmup 是一次性 unit，必须 restart
sleep 5
echo "[done] mode=$MODE"
journalctl -u $SVC --no-pager --since "-3min" | grep -E "COLD-POOL|SHRUNK|inventory|KV Cache|max_total" | tail -12 || true
