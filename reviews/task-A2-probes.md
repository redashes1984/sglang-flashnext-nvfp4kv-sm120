星鉴 R4 证据质量整改卡（测试资产改造，只动 tests/ 与 scripts/，不碰 runtime-src/ patches/ config/，禁 SSH 生产、禁 systemctl）。

背景：reviews/R4-evidence-quality-20260911.md 判 88/100 有条件通过，四条 🟡 整改项：

1. cached_tokens 双源化：probe_cached0_discrim.py / probe_highwater_77.py / probe_g2x.py / probe_d2_i8off.py 现在单读 meta_info.cached_tokens，字段缺失时 .get()→None 或 `or 0` 会把"字段缺失"误判成 cached=0（假 FAIL）。改法：每次 reask 旁路同时抓 journal "#cached-token:" 行（按 marker 关联，见第 2 条），双值一起打印；单源缺失判 INCONCLUSIVE 而非 FAIL。
2. 请求-日志关联 marker：所有命中探针在 prompt 内嵌唯一 marker（时间戳+随机串），事后 journalctl grep marker 框定本请求的 batch 行，替换固定 --since 时间窗（stress_conc12.sh:22、bench_steps3.sh:20 是重灾区——固定窗产生过"两次字节级相同 446.14"假象）。
3. 吞吐口径统一：bench_steps3.sh:18 与 probe_nvfp4_mamba_gate.py:55 用标称 token 数除墙钟（temperature>0 时实际输出常少 → tok/s 虚高）。改为输出 journal "gen throughput (token/s)" 聚合为头条值，awk 值降级为参考。
4. probe_nvfp4_mamba_gate.py:59 exit code 未被上层检查 + soak_final.sh:30 引用外部路径缺文件时静默降级——失败路径显式化。

约束：探针保持 CUDA-graph capture-safe（不引入 sync）；大 body 走 --data @file；marker 不得影响语义（追加在 prompt 尾部一行内）。
验收：改完后每个脚本 python3 -m py_compile / bash -n 全过；给出一个最小自证：本机无法起服务，就用 mock HTTP 服务（python3 http.server 假 meta_info 响应 + 伪造 journal 文本）证明双源解析、INCONCLUSIVE 分支、marker 关联逻辑正确。写 reviews/A2-verification.md 记录验证输出。
完成后 git commit（本地，不 push），kanban 评论列改动清单 + 验证摘要。
