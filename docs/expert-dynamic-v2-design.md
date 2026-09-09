# 动态专家加载/驱逐机制设计 v2

> 状态：待批准 | 日期：2026-09-09 | 前置：v1 事故复盘（见 sglang-service-operations skill reference）
> 参考实现：ranxianglei/sglang@ours/main（topk.py keep-mask/offload、cold_pool.py v2、expert-tools）
> 部署目标：CT112 (10.10.4.12) dealignai_Qwen3.8-Flash-Next-ABLITERATED-NVFP4, 1×96GB, modelopt_fp4 + flashinfer_cutlass, TP=1, MoE A2A=none

## 0. 一句话架构

**静态 keep+offload 保容量，动态 cold-pool 保长尾；所有门控走持久化表（bias/remap）的"内容更新"，物理池形状终身不变，换入换出严格发生在 CUDA graph 重放之外的步后钩子里（一步前瞻）。**

## 1. v1 教训 → 设计红线

v1（`patches/expert_tier.py`，commit a19e6b1 留档）三条结构性缺陷，本设计逐条对应红线：

| # | v1 缺陷 | 事故证据 | v2 红线 |
|---|---------|---------|---------|
| R1 | forward **中途**同步 H2D 覆写 ring slot，decode graph replay 拿着 baked 的旧映射读半新半旧的 slot | 崩坏发生在流量+图执行 ~10min 后，非启动期；capture 分支 baked 静态 lookup | 换入只允许在 model forward 结束后、graph 外执行；为**下一步**生效（一步前瞻） |
| R2 | 门控用"每步重建 lookup + torch.where"，图里 baked 的是旧 tensor 内容/指针 | capture/eager 双分支语义不一致 | 所有门控表（bias_gpu/remap_gpu）注册为持久 buffer，图 baked 指针，运行期只做 in-place 内容更新 |
| R3 | side tensor 靠 `dim0==E` 启发式自动发现，scale 类张量是否同步搬运未验证 | HiCache 被迫 OFF 的先例（config 头注：fp4 scale buffer 驱逐回载静默腐坏）证明 fp4 scale 是独立且脆弱的存储类 | 逐专家张量**显式白名单** + Phase 0 运行时 inventory 门禁 + byte 级 round-trip 校验；名单外任何 dim0==E 张量存在即拒绝启动 |

## 2. 机制分层（对齐 ranxianglei 但按我们后端适配）

### 层 1：静态 keep-mask（路由器屏蔽）
- 拦截点 `select_experts()`（topk.py），topk 选择**之前**给 `router_logits` 加 bias 列：keep=0.0，非 keep=`finfo.min`
- bias 列在首次 eager 调用（warmup 期）构建，按 `(layer, dtype)` 缓存——list→CUDA 索引是 H2D copy，capture 期非法（他们注释原话，我们同样适用）
- 效果：被屏蔽专家永不出现在 top-k → 零计算路径、零显存读

### 层 2：keep-only offload（物理池收缩）
- loader 完成后钩子（`maybe_shrink_after_process` 模式，挂在 loader.py 的 process_weights_after_loading 之后）：
  - 逐专家参数按**白名单**切成 keep 部分（留在 GPU，index_select 保持行序 = keep 表序）+ cold 部分（`pin_memory()` 存 host）
  - GPU 张量重建为 `(keep_len + slots)` 行，cold slot 零填充
  - `num_local_experts` 同步改为 keep_len+slots
- 建两张 GPU 持久表（长度 = 全局专家数 512）：
  - `remap_gpu[gid]`：keep → 池内 slot（i<keep_len）；staged cold → keep_len+slot；其余 → 0（slot0，bias 已封死永不被选，越界安全值）
  - `bias_gpu[gid]`：可路由集 = 0.0，其余 = -inf 等效值
- topk 出口统一 `_maybe_remap_topk_ids`：`topk_ids = remap_gpu[topk_ids]`，一次 gather，capture 安全（表是普通指针）
- **收益预期**（v1 已实测的外科部分）：512→330 keep 释放 ~14.7GB，KV 池 1,179,648→1,572,864（+33%）

### 层 3：cold-pool 动态预取（一步前瞻）
- `select_experts` 内 stash 本步 raw topk 全局 id + sigmoid 分数（capture 期跳过；consume-once：读后清 -1）
- `model_runner` forward 出口调 `after_forward_hook()`（graph 外）：
  - 节流：每 8 步且 dirty-flag 置位才扫描
  - **强需求过滤**：冷专家 raw 分数 ≥ α×(同行热专家最低分)（α=0.8）才计一票；EMA 计数（×0.75 衰减）≥ need_hits(2) 才候选；每层每拍最多 stage 4 个
  - `_stage_expert`：从 host pinned 逐参数 `copy_`（非阻塞关闭，走当前流，此刻无任何在途读）进 LRU 最旧空 slot → `remap_gpu[gid]=keep_len+slot` → `bias_gpu[gid]=0`
  - **淘汰纪律（顺序不可反）**：LRU 踢出旧专家时先 `bias_gpu[old]=-inf`，再允许 slot 被覆写。mask-before-overwrite ⇒ 物理上不存在"读到被覆写中/已覆写 slot"的窗口
- **时序不变量**：第 N 步的需求信号 → 第 N+1 步才可路由。冷专家"第一脚"晚一步命中，换来的是无竞态。这比 v1 的同步换入慢一步，但一步的代价是 topk 权重重归一（router 天然处理），而 v1 的代价是输出崩坏。

### 层 4（Phase B，我们设想的完整闭环，默认关闭）：在线 keep↔cold 迁移
他们 README 承认静态 keep 集 pareto-optimal、动态信号"仍属实验"——所以 Phase A 完整移植其机制，我们的增量押在有把握的闭环上：
- **降级（evict）**：keep 成员连续 W 步（默认 2048 步）零需求且不在 stash 窗口 → 异步 D2H 存回 host pinned 池 → `bias=-inf` + `remap=0` → 该 keep slot 归入可 stage 池。物理池大小不变，图不需重录
- **晋升（promote）**：staged 冷专家 EMA 需求持续 ≥ 阈值 T 步 → 转正进 bias/remap 表（纯表更新）
- **离线再切片**：定期用流量画像重生成 keep.json（移植 expert-tools + 我们的 combined_counts 管线），滚动重启生效——这是他们的成熟路径，我们保留为运维兜底
- Phase B 全部逻辑只动表内容与 slot 内容，永不改池形状 → graph 语义安全是归纳式成立的

## 3. 张量映射（v2 决胜点，Phase 0 必须实测钉死）

checkpoint 实证（modelopt_fp4，layer0 expert0）：
```
{gate,up,down}_proj.weight        U8   (packed fp4)
{gate,up,down}_proj.weight_scale  F8_E4M3  块 scale（(out, in/16)，swizzle 在专家内部，dim0=专家轴切片安全）
{gate,up,down}_proj.weight_scale_2 F32 标量  → stacking 后 (E,) —— dim0==E，必须随迁！
{gate,up,down}_proj.input_scale    F32 标量  → (E,) —— 同上
```
process_weights_after_loading 后（flashinfer_cutlass 路径）实际参数名待 Phase 0 dump，预期形如：
`w13_weight_packed / w13_weight_scale / w13_weight_scale_2 / w13_input_scale / w2_*`（4 组 ×2）

**Phase 0 门禁（不通过不许进 Phase 1）**：
1. 起一个一次性脚本，加载完成后 dump 每个 FusedMoE 的全部参数/缓冲，凡 `dim0==num_experts` 者列表化 → 生成白名单，任何名单外的 dim0==E 张量 = 阻断项
2. round-trip 校验：任选 8 个冷专家，GPU→host pin→GPU，逐字节 `torch.equal`（含 fp8/e4m3 和标量组）
3. 实测每专家字节数（预期 ~1.9MB：w13 0.82+0.10 / w2 0.82+0.10 / 标量组 ~28B×6）与 48 层总 host pinned 需求（v1 实测 25.6GB，CT112 空闲内存门禁 ≥30GB，含 PLE pinned）

## 4. 物理布局参数（首版默认，可 env 调）

| 参数 | 值 | 依据 |
|------|-----|------|
| keep/层 | 330 | 已有画像 `expert_keep_330_final.json`（48×512 combined_counts），v1 实测 -14.7GB 成立 |
| slots/层 | 16 | 2.9GB→16 slot ≈1.4GB VRAM 代价；他们默认 32，我们用 v1 同值便于对比 |
| α / need_hits / EMA / 每拍stage上限 / 钩子节流 | 0.8 / 2 / 0.75 / 4 / 8步 | 照抄其验证过的默认，Phase B 再调参 |
| KV 池 | 1,572,864（tier 态）| v1 同款增益，作为收益锚点 |
| Phase B W(降级观察窗)/T(晋升窗) | 2048 / 待定 | 保守起步 |

## 5. CUDA graph 策略（我们侧的实况）

- 我们的 decode：`backend=full, bs=[1,2,4,6]` 图常开；prefill 图禁用 → 动态换入在 prefill 天然 eager，风险集中在 decode
- 图 baked 的东西清单（必须逐个确认"指针 baked、内容活"）：bias 列、remap_gpu、bias_gpu、各 slot 权重。池形状与 topk 出口 gather 的索引张量形状恒定 → 图合法
- **判别实验（写进 Phase 2 门禁）**：先以 `--cuda-graph-config decode=eager` 跑动态路径 soak ≥30min 无腐坏 → 再开 full 图复测。若 eager 也腐坏 ⇒ 真凶是张量映射（R3）而非图竞态（R1/R2），回炉 Phase 0；若 eager 好、图坏 ⇒ 表持久化哪里失配，dump capture 前后表指针定位。v1 根因判决**不靠猜**，靠这个实验
- 已知次生成本：swap-in 生效步数内 grouped-gemm 多算 slot 专家（slot 零行也进 gemm），预计 ≤ 现有动态池化的 6.4K tok/s prefill 惩罚，Phase 3 实测

## 6. 可观测性（v1 完全没有，这次全上）

- 结构化日志 `[COLD-POOL]`：stage/evict/mask 事件（层、gid、slot、触发分数）；每 512 步摘要：hook 调用数、strong 事件数、staged 数、slot 命中复用率、H2D 累计字节/延迟
- 周期自检（debug env 门控）：随机抽 slot 与 host pinned 源比对哈希
- `/metrics` 不需要，日志 + 探针足够原型期

## 7. 正确性验证套件（每个 Phase 出口全跑）

1. greedy 三连：`The capital of France is`→Paris / `1 2 3 4`→5 / 单词指令遵循；连续两次 miss 判腐坏（容忍批处理抖动单次）
2. 中文 chat：梯度下降一句话 → content 非空、finish=stop、无 `# In[ ]:` 残渣、无循环复读
3. **自愈探针**（移植他们的 anomaly self-heal 思路）：构造只在冷专家知识域出题的 prompt 集，静态 keep 版应有答不出→换入后应能答对（他们数据 0/3→2/3）；这是动态层存在意义的唯一证据，跑不出来就要诚实砍掉 Phase B
4. 流量后验：warmup pass5 + bench 各档 + soak 后再全跑 1/2/3
5. 回滚红线（任一条触发即回 pre-tier 备份）：Paris 探针连错、content 通道空、残渣出现、输出循环、expert_tier 异常栈、NaN

## 8. 实施与环境边界

- **测试实例隔离**：在 CT112 上开第二个 systemd unit（端口 8001，`--max-running-requests` 缩减，KV 池按需），生产 8000 实例全程不动——v1 直接在生产 unit 上试验是本次事故的流程错误，禁止重犯
- 补丁文件：`patches/expert_cold_pool.py`（层2/3/4）+ `patches/topk_keep_mask.py` 接线 diff + 部署脚本幂等 patch；钩子点 model_runner forward 出口 + loader after-process_weights
- 交付流水线（AGENTS.md）：本设计 → 批准 → 星码实现（单任务，≤3 并行限）→ 星鉴审（对照本文件红线 R1-R3 与 §3/§5 门禁）→ Phase 0/1 在测试实例执行 → 汇报后再批准 Phase 2
- 每步完成汇报，跳一步停一步由棣民控制

## 9. 决策点（等批准）

- **D1** keep=330/slots=16 起步（沿用 v1 画像，便于三方对比）——还是先用他们的 296 做纯移植基线？我建议 330，画像已就绪
- **D2** 判别实验顺序：eager-soak 先行（更保守）——OK？
- **D3** Phase B（在线 keep↔cold 迁移）实现进原型代码但默认 env 关闭，Phase A 全绿后才打开——OK？
- **D4** 实现主体：星码写、星鉴审，我出设计+验收门禁——还是我直接写补丁走同样的审计？（v1 是我写的，代码量 ~400 行且钩子点位极窄，我倾向自己写省去上下文传递，星鉴照审）
