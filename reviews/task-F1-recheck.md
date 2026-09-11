复审卡 F1：对 A1 修复（commit 2249c5f）做回归审查，核验 R1-🟡1 与 R2-🟡1 是否真闭环。只读，不改文件、不跑生产。

审查对象（git show 2249c5f）：
1. patches/expert_cold_pool.py remap_topk_ids：新增 ok=(safe<tbl.numel())&(topk_ids>=0) + clamp(max=numel-1) 透传。对照你 R1 报告 §🟡1 药方（reviews/R1-coldpool-20260911.md）逐点核验：越界是否根除、-1 padding 语义是否保留、与 packed :322 路是否对称、to(dtype) 截断风险是否仍在（你原来标 🟢-4 的那条，看是否顺带修掉）。
2. stash_demand：neg=(idx<0)|(idx>=width) 过滤共享列。核验 last-expert 需求统计是否干净、clamp 顺序是否引入新 alias。
3. config/dealignai-qwen4exp-nvfp4kv.yaml 注释：8 深链前提 + fp8kv conc8×16=128 闭合表述是否与你 R2 §🟡1 的算术一致，有无误导。
4. tests/test_cold_pool_logic.py T10/T11：星野已在 CPU-torch venv 实测 11 passed。审用例覆盖面是否足以证明"融合共享列透传不改语义"，缺什么补什么（可写进报告建议，不改代码）。

输出：reviews/F1-recheck-A1-20260912.md，开头给结论行：R1-🟡1 [闭环|未闭环]、R2-🟡1 [闭环|未闭环]，若闭环给出 R1 报告回分（原 60/100，按协议 🔴-40/🟡-20/🟢-10/ℹ️-10合计 重算）。发现问题按 🔴/🟡/🟢 分级，file:line 定位。完成后 kanban complete + metadata 报告路径。
