审查对象：CT112 生产运行时的专家冷池（expert cold pool v2.9）代码栈。仓库 /root/sglang-fl-nvfp4，live 与仓库 md5 一致性已由星野核验（见 reviews/audit-brief-20260911.md），审仓库快照即审生产。

必读文件（按 git diff 增量看，勿读整个 upstream 基线）：
- runtime-src/qwen4_exp_ple_table.py（新增 622 行，PLE sidecar + pinned/file 后端）
- patches/expert_cold_pool.py（冷池主体 v2.7a：flat pinned buffer、alias guard、identity gate 配套）
- runtime-src/topk.py（v2.9 identity gate：register_managed/is_managed_config、FusedMoE→owner-block.topk 解析；git diff 只看改动 hunks）
- runtime-src/model_runner.py（v2.1 shrink timing：SHRUNK 必须在 graph capture 之前完成）
- scripts/deploy_expert_cold_pool.py（锚点 patcher，v2.8→v2.9 升级 pass + pristine-anchor skip）
- tests/test_cold_pool_logic.py（381 行单测）

背景：v1 曾发生流量后输出损坏（已回滚，红线来源）；v2 系列引入 keep-mask 常驻 + 16 动态槽 + 校验和（8448 rows）+ staging/eviction 不变量 staged-evicted=768。

重点审查：① 线程/异步安全（capture 与 shrink 时序、staging worker 竞态）② keep-mask 与 draft 层共享 layer_id 的 identity 判定是否有 id() 复用漏洞 ③ pinned buffer 生命周期与泄漏 ④ fail-loud 路径是否存在静默降级 ⑤ patcher 锚点漂移风险。
硬约束：只读，禁止修改任何文件、禁止重启服务、禁止执行写操作。
输出：报告写 reviews/R1-coldpool-20260911.md，四层分类（🔴阻断/🟡重要/🟢建议/ℹ️风格），每条附 file:line 证据，末尾 0-100 评分（95 分门槛）。完成后 kanban 评论贴结论。