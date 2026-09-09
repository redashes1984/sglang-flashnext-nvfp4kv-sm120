#!/bin/bash
# log triage: ALL errors/warnings since cold-pool boot 01:58, classified
U=sglang-dealignai-qwen4exp-nvfp4kv
echo "=== ERROR-ish lines ==="
journalctl -u $U --no-pager --since "01:58" | grep -iE "error|exception|traceback|fatal|assert|crash|abort|corrupt|failed" | grep -v "0 error" | sort | uniq -c | sort -rn | head -25
echo ""
echo "=== WARNING lines (grouped) ==="
journalctl -u $U --no-pager --since "01:58" | grep -iE "warn" | sed -E 's/^[A-Za-z0-9 :]+ sglang\[[0-9]+\]: //; s/[0-9]{2,}/N/g' | sort | uniq -c | sort -rn | head -30
echo ""
echo "=== COLD-POOL lines (all) ==="
journalctl -u $U --no-pager --since "01:58" | grep "COLD-POOL" | tail -12
echo ""
echo "=== counts ==="
journalctl -u $U --no-pager --since "01:58" | grep -icE "error|exception|traceback|assert"
journalctl -u $U --no-pager --since "01:58" | grep -icE "warn"
LOGTRIAGE-DONE
