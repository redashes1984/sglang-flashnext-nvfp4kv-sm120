# A2 探针证据链整改 — 验证记录

日期:2026-09-12 | 执行:tester | 对应卡:t_c86b0fbc(R4 四条 🟡)

## 改动清单

| 文件 | 整改项 |
|---|---|
| tests/probe_cached0_discrim.py | 1+2:cached_pair 双源(meta_info + journal #cached-token),marker 关联,缺失判 INCONCLUSIVE |
| tests/probe_highwater_77.py | 同上 |
| tests/probe_g2x.py | 同上 |
| tests/probe_d2_i8off.py | 同上 |
| tests/probe_nvfp4_mamba_gate.py | 1+2+3+4:双源、marker、journal gen throughput 为头条(awk 降级为 reference)、显式 exit code |
| scripts/bench_steps3.sh | 2+3+4:jslice marker 切片替换固定 --since(无 marker 时回退 --since @$START 运行起点窗,绝不用陈旧固定窗);headline=journal gen throughput;curl 失败计数入 exit code;大 body 走 --data @file |
| scripts/stress_conc12.sh | 2+4:同法 marker 切片;失败计数;nvidia-smi 缺失时显式提示而非静默 |
| scripts/soak_final.sh | 4:/opt/sglang-env/bin/python3 或 probe_cp.py 缺失时打印 QUALITY-PROBE SKIP 并置非零 exit,不再静默降级 |

约束遵守:探针无新增 sync(CUDA-graph capture-safe);marker 以 `[a2mk-*]` 追加在 prompt 尾部一行内,不改语义;A2_MARK/HERMES_JOURNAL_FILE 仅为自证注入点。

## 语法门禁

`python3 -m py_compile` ×5 + `bash -n` ×3 全过(SYNTAX-OK)。

## 最小自证(mock HTTP + 伪 journal)

本机无 sglang,用 http.server mock(轮转三种 meta_info:含 cached_tokens / 缺字段 / 故意不一致)+ /tmp/a2_journal_fixture.txt 伪 journal,验证四类分支:

1. **双源一致**:probe_d2_i8off B reask → `cached_meta=1230 cached_journal=1230.0 verdict=consistent` ✓
2. **单源缺失 → INCONCLUSIVE(非 FAIL)**:probe_nvfp4_mamba_gate G1 → `cached_meta=None ... verdict=INCONCLUSIVE (source missing)`,gate 终态仍 `ANCHOR-GATE: PASS (inconclusive=1)` rc=0 ✓
3. **双值不一致 → MISMATCH**:probe_g2x B reask2 → `cached_meta=999 cached_journal=1230.0 verdict=MISMATCH` ✓
4. **marker 关联切片**:bench_steps3.sh C1 headline 取到 marker 行之后的 `446.14`(running-req: 1 首条),C12 headline 取到 running-req: 12 的 `452.30`——fixture 里 marker 前的 999.99 陈旧样本被正确排除,"两次字节级相同"的陈窗复读模式不复现 ✓
5. **吞吐口径**:journal 值为头条,awk 标称值标注 `(reference only)` ✓
6. **stress_conc12.sh** 无 GPU 环境下仍产出 token usage / error count / curl failures=0,rc=0 ✓

soak_final.sh 的 SKIP 分支逻辑为纯 shell 条件,本机无 /opt/sglang-env 环境,按 bash -n + 阅读验证(缺文件 → QUALITY-PROBE SKIP + exit 1)。

## 结论

四条 🟡 整改全部落实并通过 mock 自证;证据链从单源升级为双源交叉,时间窗升级为 marker 归属,假 FAIL 路径(cached=None→0)消除。
