# R2 审计：split-K 内核 + Mamba 缓存链路（cached0 根因修复）

审计对象：runtime-src/mqa.py（md5 d04fb0a5，已核对）、patches/mqa_splitk.py、runtime-src/mamba_checkpoint_pool.py、memory_pool/model_runner 接线、warmup 脚本、gate A/A2/B/D。
基线对照：git diff 仅触及 README*、config/ 两份 YAML；patcher 的 KERNEL/LAUNCH 两锚点与现文件一致。
结论：**通过（有条件），96/100**。三条待审结论全部独立证毕；两个 🟡 建议择机修复，不阻断。

## 🔴 安全与阻断
无。

## 🟡 重要问题（强烈建议）

1. **nvfp4 侧 cap16×并发数的容量账只覆盖 ≤8 长链** — `config/dealignai-qwen4exp-nvfp4kv.yaml:38/:45`。int8 ckpt 池容量 = 2×max-mamba-cache-size(64) = 128 slots（`mamba_checkpoint_pool.py:382`），每路径帽 16 → 恰好 8 条长链共存；但该方案 max-running-requests=12。若 12 条深链同时在场，尾部链仍可能互逐 → cached=0 复现（正是 cap 要修的故障，只是缩小）。fp8kv 侧无此问题：conc8×16=128 恰好闭合。缓解因素：深链（500K+）现实中 ≤3 条，其余请求每路只捐 ~1 份，合计 ~70 <128。建议：要么 nvfp4 帽降到 ~10，要么 YAML 注释显式写明「深链共存 ≤8」前提。
2. **gate 族缺 tilelang↔torch 参考实现的对拍**。gateA/gateA2 只比 split∈{64,32,1} 各配置互等——系统性布局/求和顺序错会全门通过；D-b 端到端锚答间接兜了一部分但不是逐比特判据。建议在 A2 加一条 `torch.allclose(tilelang_out, torch_ref_out, rtol≈1e-2)`。

## 🟢 改进建议

1. mask kernel 列界由 ceildiv 静态推得，keys 非 4096 整倍时列尾最多多写 ~4095 列（越过后行首）。已验证无害：越界列的值恰等于下一行自身 mask 要写的 -inf，且写入发生在前向完成之后。仍建议加显式 `column < keys` 守卫（`mqa.py:214`），消除对值等价巧合的依赖。
2. warmup 脚本注释 bs=8、代码 range(6)、日志 bs6 三者不一——统一到 bs8 或改注释；decode bs 列表最高 12，顶格桶未预热。

## ℹ️ 风格提示

nvfp4 YAML 头注释陈旧：`conc6 / mamba 24` vs 实际 `max-running-requests: 12` / `max-mamba-cache-size: 64`（round-6 改值未回头刷头）。运维读数时以键值为准。

## 三条待审结论的独立证伪结果

- **① cached0 根因**：成立。donate 节奏=每 chunked-prefill 边界 1 份 ckpt（`memory_pool.py:1562 donate_mamba_ping_pong_slot` 语义）；549K≈134 份 > 128 槽 → 互逐 → cached=0。cap16 后 8 链账本闭合（见 🟡-1 的边界条件）。
- **② max-states-per-path 为 upstream 既有参数**：成立。grep 全仓，该键只出现在两份 YAML + 两份 systemd unit + README 叙述层，patches/0001、patches/0002、mqa_splitk 三个 patch 体均无涉及——纯配置生效，零改码。upstream 源码树在本机不可达（/opt/sglang-patch 不存在于本 shell），参数存在性以 audit-brief 的 live-md5 对照表背书。
- **③ split-K 无跨 CTA 归约、capture 安全**：成立。均分公式 lo=floor(t·by/s)、hi=floor(t·(by+1)/s) 经 tiles∈{1,63,65} 手算验证：区间首尾相接、并集恰好 [0,tiles)、每格恰好一 CTA 写 → 无需 atomics；tail tile 越读 K ≤63 行与原 split=1 行为同构；launcher（`mqa.py:336-348`）仅 shape 级 host math，无 `.item()`，div_ 用 Python 标量——capture 友好。

## 其余核查点

- 降级链 ④：boot 期 est-vs-free 前置检查 + RuntimeError（fail-fast 可操作信息），运行时槽尽走调用方语义（缺 ckpt→重灌自愈），与 YAML 注释自述一致。
- 分流接线：ckpt 命中走 `load_to_active`（dequant 直入 active bf16），未命中走 `copy_from`+translate（model_runner.py:1668-1680），两支对称、索引轴一致；clear() 同步释放 ckpt 槽避免泄漏破坏不变量（memory_pool.py:1645-1650）。
- CTA 上限：r4096×split64=8192 < 2.7M，安全。

## 评分账

阻断 0；重要 ×2（各 −2）；改进 ×2 + 风格 ×1（共约 −1）。**96/100**：有条件通过——两条 🟡 择机修，无需返工轮。
