审查对象：运维面（配置/服务/切换脚本）与状态一致性。仓库 /root/sglang-fl-nvfp4。

必读文件：
- config/dealignai-qwen4exp-nvfp4kv.yaml + config/dealignai-qwen4exp-fp8kv.yaml（KV 池、mamba 参数、spec 参数、cuda-graph-config、sleep-on-idle、prefill graph intent=full 注释）
- systemd/ 下 unit 快照（PYTHONPATH 挂载、cold-pool env trio、ExecStart 续行格式）
- scripts/switch_fp8kv_coldpool.sh（atomic arm 函数、5 模式、idempotency、strip_plefile/hicache_off）
- scripts/deploy_expert_cold_pool.py（与 R1 重叠但重点在部署幂等、锚点缺失 fail-loud）
- bootstrap/MANIFEST.md + baseline 记录（冻结点可复现性）

已知事实须独立核验：
① nvfp4kv 池最终 2,202,048（≥2,202,010 向上 64 页对齐）；2,477,312 曾实炸（late-load kernel 税 ~4.25-5.3GB 吃穿 boot fence 4.39GB）
② mamba 参数（mamba64/cap16/int8 池 128 slots）不随 KV 池联动；但任何池调整须复核 mem-fraction 余量是否覆盖 int8 池 + late-load kernel 税 + activation + fence
③ fp8kv 池 1,651,520 = (1x1M+2x256K)×1.05，64 页对齐
④ 单卡互斥：nvfp4kv 与 fp8kv 不可同时运行
⑤ sleep-on-idle: true 已验证生效（IdleSleeper 仅依赖 get_device().sleep_on_idle，与 enable_memory_saver 无关）
⑥ HiCache 与 int8 mamba checkpoint 互斥（upstream server_args.py:6333 fail-fast），YAML 注释是否准确表达
重点审查：① 两 YAML 与 remote /opt/sglang-config/ 一致性、注释中数字是否与代码/结论一致 ② 页对齐算术（64 的倍数）是否成立 ③ switch 脚本幂等性（env 重复堆叠类 bug 曾在 ba10b56 出现过，检查是否还有同类正则/去重缺陷）④ ExecStart 续行中注释导致参数吞掉的风险（历史上出现过 unrecognized arguments: Restart=no）⑤ fence/回滚触发条件是否被文档与配置一致表达 ⑥ 密钥/凭据是否泄漏进仓库（CREDENTIALS.md 应解耦）。
硬约束：只读，禁止修改文件、禁止 systemctl 任何操作。
输出：报告写 reviews/R3-config-ops-20260911.md，四层分类 + file:line 证据 + 0-100 评分。kanban 评论贴结论。