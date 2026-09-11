审查对象：测试/探针资产质量与结论可证性。仓库 /root/sglang-fl-nvfp4。

必读文件：
- tests/probe_cached0_discrim.py、probe_highwater_77.py、probe_g2x.py、probe_d2_i8off.py、probe_nvfp4_mamba_gate.py、probe_fp8kv_1m_shape.py、probe_fp8kv_1m_bringup.py、probe_branch_coexist.py、bench_prefix_benefit.py
- tests/test_pageable_gather.py、test_ple_sidecar.py、test_cold_pool_logic.py（单测有效性）
- tests/dbg_niah_answer_empty.py、debug_hook_trace.py（调试脚本是否有残留污染风险）
- scripts/soak_final.sh、soak_watch_v2.sh、stress_conc12.sh、stress_long12.sh、bench_fp8kv_steps.sh、bench_steps3.sh
- scripts/bench_qsa_mqa_split.py、bench_qsa_mqa_split3.py、bench_qsa_mqa_tiles.py、bench_qsa_configs.py

要审的不是代码风格，而是"这些脚本给出的证据能不能支撑这些天的结论"：
① cached_tokens 读取路径是否可靠（meta_info.cached_tokens + journal #cached-token 双源；/server_info hit/evict 字段可能返回 {}）
② 冷/热判定：cold 与 reask 之间是否有驱逐干扰（探针顺序、时间窗、并发污染）；Hindsight/后台流量污染是否被排除
③ NIAH 探针：是否强制 chat 端点 + enable_thinking=false（/generate 会被思考链吃满 token 预算导致假失败）；needle 题目与答案是否唯一（答案多解会造成假阴性）
④ 大 body 是否走 --data @file 而非内联（HTTP 400 风险）；估算 token 时是否保守（CJK ~1 token/char）
⑤ 吞吐测量：是否避开 idle/进场首行（sleep-on-idle 等待污染 + warm-cache 窗口），journal grep 固定窗口是否产生重复/陈旧样本（历史教训：两次字节级相同的 446.14 是 grep 固定窗口假象）
⑥ CUDA graph capture 安全：探针/脚本中是否隐含 torch.cuda.synchronize() 或 .item() 进入 forward 路径
⑦ bitwise gate：gateA/A2 的 3-way 对比是否真能捕获静默漂移（是否覆盖 tail/非对齐 shape）
⑧ 失败路径：脚本超时/异常是否被静默吞掉（|| true、2>/dev/null 过度使用）
输出：报告写 reviews/R4-evidence-quality-20260911.md，四层分类 + file:line 证据 + 0-100 评分，并明确列出"你认为不可信/需重测的结论"清单。kanban 评论贴结论。
硬约束：只读，禁止修改任何文件，禁止启动探针或调用推理服务。