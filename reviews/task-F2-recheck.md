复审卡 F2：对 A2 修复（commit 6cad2f9）做回归审查，核验 R4 四条 🟡 是否真闭环。只读，不改文件、不跑生产探针。

审查对象：
1. 双源 cached_tokens：五个 probe 的 cached_pair/等价逻辑——meta is None or journal is None → INCONCLUSIVE（非 FAIL 非 0）；一致容差 ±8 是否合理（journal #cached-token 与 meta 的微差来源是页对齐，评估容差是否会漏报真 MISMATCH）。
2. marker 关联：探针 MARK 随机串进 prompt、journal_stats 按最末 MARK 行切窗；bench_steps3.sh / stress_conc12.sh 的 jslice——无 marker 回退 --since @$START 的降级路径是否还有陈旧窗复读风险。
3. 吞吐口径：journal gen throughput 为头条、awk 标称降级 reference only——检查 mamba_gate 的聚合是否正确（并发多流时取首条/末条/p90 的语义是否写清）。
4. 失败路径显式化：soak_final.sh SKIP+exit 1、curl 失败计数——确认无新静默分支。
5. reviews/A2-verification.md 六条自证与代码是否名实相符（星野已抽查 cached_pair 源码一致）。
6. 约束回归：无新增 sync、marker 不越行、大 body @file。

输出：reviews/F2-recheck-A2-20260912.md，结论行：R4-🟡1..4 各判 [闭环|未闭环]，闭环则给出 R4 报告回分（原 88/100 重算）。问题按 🔴/🟡/🟢 分级 file:line。完成后 kanban complete + metadata 报告路径。
