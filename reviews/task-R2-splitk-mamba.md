审查对象：TTFT split-K 内核改动 + Mamba 缓存链路（cached0 根因修复）代码栈。仓库 /root/sglang-fl-nvfp4。

必读文件：
- runtime-src/mqa.py（live md5 d04fb0a5；split-K 实现：_QSA_PREFILL_LOGITS_BUDGET_BYTES=128MB、split=64、tilelang prefill MQA kernel over key tiles、每 CTA 写不相交 logits 列、launcher 无 .item() host sync）
- patches/mqa_splitk.py（patcher）
- runtime-src/mamba_checkpoint_pool.py + runtime-src/memory_pool.py（int8 mamba checkpoint pool：128 slots/3.74GB，从 mem-fraction 余量出资，不进 max_total_num_tokens）
- runtime-src/mqa.py 与 upstream 基线对比用 git diff 看 hunks
- scripts/warmup_qsa_fp8kv.py / warmup_qsa_nvfp4kv.py（warmup 形状阶梯）
- tests/gateA_mqa_bitwise.py、gateA2_mqa_bitwise_extended.py、gateB_indexer_canary.py、gateD_quality_splitk.py
- references 背景：/root/.hermes/skills/mlops/sglang-service-operations/references/ttft-qsa-splitk-20260911.md、cached0-rootcause-e3-20260911.md

已知待审结论（须独立证伪）：① cached0 根因=extra_buffer_lazy 下每 4096 token chunk 捐 1 个 int8 ckpt，549K 链=134>128 槽互相驱逐；cap16 后共存 8 长链 ② --mamba-max-states-per-path 是 upstream 已有参数（我们未改码，仅 YAML 设 16）③ split-K 无 atomics 无跨 CTA 归约，capture 安全。
重点审查：① split-K 索引/边界数学（tail tile、split 不整除、budget 分支回退路径）② CUDA graph capture 安全性（host sync、shape 依赖）③ cap16 驱逐顺序是否可能驱逐浅锚点导致整链 cached=0 复发 ④ int8 池分配失败→evict(mamba_num=1) 的降级链 ⑤ gate 测试是否真能捕获 bitwise 漂移。
硬约束：只读，禁止修改文件、禁止重启服务。
输出：报告写 reviews/R2-splitk-mamba-20260911.md，四层分类 + file:line 证据 + 0-100 评分（95 门槛）。kanban 评论贴结论。