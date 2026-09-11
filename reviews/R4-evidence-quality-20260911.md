# R4 证据可证性审计 — 探针与 gate 质量

日期:2026-09-11 | 审查者:reviewer | 评分:**88 / 100(有条件通过)**

审查范围:tests/ 下 9 个探针 + 3 个单测 + 2 个调试脚本;scripts/ 下 6 个 soak/bench 脚本 + 4 个 kernel bench;gateA/gateA2。
审查标准:这些脚本产出的证据能否支撑既有的命中/驱逐、NIAH、吞吐、bitwise gate 结论。只看可证性,不看风格。

---

## 🔴 阻断问题(必须修复)

无。证据链主体成立。

## 🟡 重要问题(强烈建议)

1. **cached_tokens 单源依赖,违背双源要求。**
   - tests/probe_cached0_discrim.py:27,33-34;tests/probe_highwater_77.py:30-32;tests/probe_g2x.py:23-30;tests/probe_d2_i8off.py:17-18 —— 全部只打印 `meta_info.cached_tokens`,无 journal `#cached-token` 交叉读数。若服务未暴露该字段,`.get()` 返回 None → 日志显示 `cached=None/缺失`,易被误读为"驱逐导致 miss",而实际可能是字段缺失的假失败。
   - tests/probe_nvfp4_mamba_gate.py:36 `m2.get("cached_tokens") or 0`:`or 0` 会把字段缺失直接折算成 cached=0 → G1 假 FAIL 路径。
   - **修复**:每次 reask 读数旁路同时抓 journal `#cached-token:` 行,双值一起打印;单值缺失时判 INCONCLUSIVE 而非 FAIL。

2. **cold→reask 之间无排他窗口机制。**
   所有命中探针依赖"操作员保证服务空闲",脚本自身不验证。共享服务上若 soak/Hindsight 后台流量在 cold 与 reask 之间落入,驱逐可把 hit 翻成 miss(假失败),warm 残留可把 miss 翻成 hit(假通过)。tests/probe_branch_coexist.py:18-25 顺序合理但仍是裸时间假设。
   - **修复**:探针 prompt 内嵌唯一 marker(时间戳/随机串),事后 `journalctl | grep marker` 框定本请求的 batch 行,用行归属替代固定时间窗。scripts/stress_conc12.sh:22 和 scripts/bench_steps3.sh:20 的 `--since "-2/-3min"` 固定窗正是历史"两次字节级相同 446.14"假象的来源模式,同法改造。

3. **bench_qsa_configs.py 用 bf16 存储代理 fp8 KV,绝对值不可信。**
   scripts/bench_qsa_mqa_split… 系列的 configs 版自述:b33-35 注明 bf16 计时,"ranking unaffected"只是假设,未验证。合成 shape total_q=4096(scripts/bench_qsa_configs.py:20)也与生产 late-chunk rows=128 不同构。
   - **结论**:仅相对排序可用;绝对 ms/TF/s 需补一轮 fp8_e4m3 实测才可入头条。

4. **聚合吞吐按标称 token 数计算,系统性偏高。**
   scripts/bench_steps3.sh:18 用固定 3600 tok 除墙钟;tests/probe_nvfp4_mamba_gate.py:55 用固定 2400 tok。temperature>0 下实际输出常少于 max_new_tokens → tok/s 虚高。journal 的 `gen throughput (token/s)` 才是可信值。
   - **修复**:头条吞吐一律取 journal 行聚合;awk 值只作粗略参考。

## 🟢 改进建议(可选)

- tests/bench_prefix_benefit.py:30 用 `len(ln)*0.9` 估 CJK token,低于真实比值(CJK ~1–1.5 token/字符),容量结论偏低估;好在每行都打印实测 pt,可事后校正。建议系数 ≥1.0。
- tests/probe_nvfp4_mamba_gate.py:59 exit code 未被上层脚本检查;soak_final.sh:30 引用外部路径 `/opt/sglang-test/probe_cp.py`,部署机上文件缺失时质量检查静默降级(仅打印错误文本)。
- tests/dbg_niah_answer_empty.py、tests/debug_hook_trace.py:调试脚本残留风险低——只写 /tmp 临时文件,无副作用;`torch.equal(out, want)`(tests/test_pageable_gather.py:47)是功能正确性断言而非仅计时,设计良好。

## ℹ️ 风格提示(仅供参考)

- gen()/ctx() helper 在 ~9 个探针里重复;一致性强,可接受。gateA 末尾 `del a, b`(tests/gateA_mqa_bitwise.py:63)落在循环外缩进,无害。

---

## 值得肯定的部分(证据链的可靠环节)

- **bitwise gate 覆盖真实**:tests/gateA_mqa_bitwise.py:46 含 uneven-pad(130 行)与 tiny rows;gateA2:56-67 补齐 early-budget-full、多请求 interleaved windows、win=1、非对齐 win=599、multi-batch。且 gateA2:73-74 用"同配置跑两次"的时序确定性检查抓 overlapping-CTA 写竞争——对静默漂移的抓取力是真的。split=32 cross-check(gateA2:81-82)合理。
- **单测是真断言**:test_ple_sidecar 覆盖指纹漂移、mtime/inode 换文件、malformed JSON;test_cold_pool_logic.py:236-238 记录了一次"恒真断言"的历史教训并改为实断言;test_pageable_gather.py 走生产 kernel 本体。
- **NIAH 端点姿势正确**:probe_fp8kv_1m_shape.py 用 chat 端点 + enable_thinking=false;needle 词唯一(紫罗兰/琥珀/青金带编号后缀),假阴性风险低。bringup 版仍走 /generate——两者结论冲突时以 shape 版为准。
- **失败不被吞**:HTTP 异常走 urlopen 默认抛出(未被 try/pass);curl 侧 `-s` 吞 stdout 但错误计数走 journal grep,链路可见。

## 我认为不可信 / 需重测的结论清单

1. **"cached=0 ⇒ 槽位不足/驱逐"类结论**:全部来自 meta_info 单源。需补 journal `#cached-token` 旁证后复核一次。
2. **bench_qsa_configs 的绝对毫秒与 TF/s**:bf16 代理 + 合成 shape,只采信相对排序。
3. **awk 聚合吞吐(bench_steps3 / mamba_gate G3)**:标称 token 偏高,以 journal gen throughput 为准重算。
4. **bringup 版 NIAH FAIL**:未禁思考链,/generate 下 64 tok 预算可能截断答案,以 shape 版为准重跑确认。
5. **固定 `--since` 窗 + tail 的历史数值**:同窗口重叠会复读上一轮样本(已发生过的假象模式),带 marker 关联重采一轮。

## 评分依据

🔴 0 ×-40 / 🟡 4 ×约-2.5 ≈ -10 / 🟢·ℹ️ 共约 -2 → **88**。
有条件通过:上述 🟡 四项(双源、marker 隔离、configs 绝对值、awk 聚合口径)落实后即可升为可信基线。
