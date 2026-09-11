# Qwen3.8-Flash-Next × NVFP4 KV —— 单卡 RTX PRO 6000（SM120）部署实录

[English](README.md) | **简体中文**

**在单张 RTX PRO 6000 Blackwell（96 GB，sm120）上跑通 Qwen3.8-Flash-Next（180B MoE，NVFP4 权重）的补丁、部署配置与校准数据 —— 含 QSA 稀疏注意力。两套方案都已完成调优：`nvfp4kv`（`--kv-cache-dtype nvfp4`，容量兜底方案；第五轮定档栈：专家冷池 + 每路径 checkpoint 帽）与 `fp8kv`（`fp8_e4m3`，**2026-09-11 起现役默认**：1M 上下文、并发 8、第六轮 QSA split-K 把冷灌 TTFT 砍半）。各方案的调优前基准快照保留在 `config/baseline/`，作回滚锚点与前后对照证据。**

上游 sglang 无法让 NVFP4 *KV cache* 与 Qwen 稀疏注意力（QSA）共存：Triton gather 路径拿到的是打包 fp4 缓冲区，直接死在 `KeyError: 'float4_e2m1fn_x2'`。本仓库给出可用的修复（移植自 [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) 的 MIT 补丁，并针对 dspark 的 SM121 环境根本走不到的 SM120 trtllm-gen 稀疏解码路径做了适配），外加整套调优经验：sampler OOM 修复（[#37962](https://github.com/sgl-project/sglang/issues/37962) 同类问题）、HiCache 硬性约束、fp4 KV 下的 radix cache 经济学、完整的投机解码 steps 校准。

前作基础：[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) 与 [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— 两家都是 fp8-KV；**本仓库补上的是他们没有的 nvfp4-KV 数据点**。

## 成果（栈的代际：基准 → 二次调优 → nvfp4kv-1m 试验版 → 专家冷池 → 第五轮定档 → 第六轮 fp8kv-1M 现役）

| | fp8kv 方案 | nvfp4kv 基准版 | nvfp4kv 二次调优版 | nvfp4kv-1m 试验版（第三轮） | 定档栈（第四轮冷池 + 第五轮缓存修复） | **fp8kv-1M + split-K（第六轮，现役）** |
|---|---|---|---|---|---|---|
| KV cache 类型 | fp8_e4m3 | nvfp4（打包 e2m1 + 分块 scale） | nvfp4 | nvfp4 | nvfp4 | **fp8_e4m3** |
| 上下文 | 512K（YaRN ×2） | 768K（YaRN ×3） | 768K（YaRN ×3） | 1M（YaRN ×4 显式） | 1M（YaRN ×4 显式） | **1M（YaRN ×4 显式）** |
| KV 池 | 552,960 | 786,432 | 851,968（回收 mamba 槽 +64K） | 1,179,648（为 int8 ckpt 池腾资 + 启动余量后） | 2,359,296（keep330+16 冷池腾出 ~30GB 显存 → +118 万 token） | **1,651,520** =（1×1M + 2×256K 工作集）× 1.05 防溢出余量；≈20.3 GB，fence 4.00 GB |
| 并发 | 4 | 6 | 6 | 6 | 12（spec 下 4 槽/请求 ×12 + 16 锚点余量 → mamba 64） | **8**（mamba 48；侧翼小请求与 1M 主会话并行） |
| 解码 C1 | ≈165 tok/s | ~122 tok/s | ≈136 tok/s | ≈75–90 tok/s（热态） | ≈122–129 tok/s（spec-on steps=2 + 冷池在线） | **≈165 tok/s 档**；C8 burst 616–668 tok/s 聚合（temp-0，split-K 后） |
| 解码聚合 | ≈355–365 tok/s（C6） | ~409 tok/s（C6） | ≈550–575 tok/s（C6） | ≈72–91 tok/s（C6） | ≈704–710 tok/s（C12，同 harness） | **C8 ≈428–650 tok/s**（accept 0.40–0.71 波动带） |
| Prefill | ~11K tok/s | ~10K tok/s | ≈10K tok/s | ≈9.7K tok/s（冷前缀；radix 命中更高） | ≈9.7K tok/s（冷前缀；radix 命中更高） | **1M 冷灌 133 s**（split-K 前 293 s，见第六轮）；256K ≈25–28 s |
| MTP（NEXTN）steps | 3 | 2 | 2（校准甜点） | OFF（1M 单卡下 steps≥1 即 OOM） | 开，steps=2（冷池腾出的显存重新养得起投机） | **开，steps=2/draft=3** |
| GPU 常驻路由专家 | 全部 512 | 全部 512 | 全部 512 | 全部 512 | 346/512（330 keep + 16 动态槽；166 个冷专家进 24.15 GB pinned 内存，按需求搬运） | **346/512**（冷池已移植，同款 keep330+16） |
| Int8 mamba ckpt 池 | — | — | — | 开（补丁 0002 × PLE 镜像，PR #38619） | 开（128 槽 / 3.74 GB；每路径帽 `mamba-max-states-per-path 16`，第五轮） | **开**（96 槽 / 2.81 GB，帽 16） |
| Radix 前缀复用 | 开 | 开 | 开 —— 58K 共享前缀命中 99.96%，6.3s → 0.6s | 开 | 开 —— 2×549K 长链共存，重问 0.9s 全命中（第五轮） | **开** —— 1M + 2×256K 三链共存，重问 0.8–2.9s 全命中 |
| HiCache L2 | **关**（与 int8 mamba ckpt fail-fast，`server_args.py:6333`） | 关 | 关 | 关（fp4 KV 硬约束，见约束一节） | 关（fp4 KV 硬约束，见约束一节） | **关**（int8-ckpt fail-fast） |

nvfp4 KV 路径的质量门禁全绿：NIAH 200K、6×107K 并发池压下的 needle 测试、驱逐后前缀重查正确性、grammar JSON ×6、工具调用、零 retract、零报错。第四轮追加：CUDA graph 下 8448 行搬运校验逐字节一致、30 分钟验收 soak、12×66K 长文并发压测 —— 零 MISMATCH / stage_fail / OOM。accept len ≈2.0–2.6，accept rate ≈0.5–0.8。

## 相对基准版改了什么、为什么、效果如何

调优版相对基准快照（`config/baseline/`）共八个改动点 —— 第 1–7 项是二次调优，第 8 项是叠加其上的第三轮 nvfp4kv-1m 试验层。第 1–7 项汇总效果：**C1 +11% / C6 +36%**，下表即调优证据。

| # | 改动（基准 → 调优） | 为什么 | 效果 |
|---|---|---|---|
| 1 | GDN linear-attn 后端：仅 decode flashinfer → **prefill+decode 双端 flashinfer** | SM120 上自动解析走 SM100-only 判断，prefill 会回落到 triton；显式钉死才能用上 FlashInferGDNKernel | GDN prefill 提速，也是后面几项收益的地基 |
| 2 | `max-mamba-cache-size`：32（自动 sizing）→ **钉死 24** | radix 峰值 usage 实测 0.62 ≈ 20 槽（lazy 驱逐下），多余槽是死显存 | 省出的显存转进 KV 池（+64K → 851,968），多会话前缀驻留更多，C6 容量直接受益 |
| 3 | `--mamba-radix-cache-strategy`：extra_buffer → **extra_buffer_lazy** | 完成请求的状态为前缀复用而驻留，但不需要预扩额外槽；lazy 把每请求槽开销从 ~5 降到 ~4 | **C6 +38%**（单项最大收益）；单流微亏 −3%，并发场景净赚 |
| 4 | prefill CUDA graph：auto → **强制 full 捕获**（`cuda-graph-config.prefill.backend: full` + `max_bs: 4096`） | Breakable×multimodal 自动禁用会把 prefill 批次留在 eager（#28386 一类问题）；强制 full 绕开。一次性成本 ~0.56 GB / ~30s | prefill 稳定 ≈10K tok/s，无 eager 抖动 |
| 5 | `speculative-attention-mode`：不设（prefill 路径）→ **decode** | verify/draft 批次形状就是 decode 形状，路由到 trtllm_mha/XQA 才对得上；真实 prefill 批次不受影响（仍走 triton） | C1/C6 各 +2–4%，accept 分布不变 |
| 6 | FR-Spec 投机热表：无 → **自建 64K 表**（`speculative-token-map: frspec_map_64k.pt`，coverage 1.0） | 用自有语料（obsidian 笔记 + skill 文件 → 65,536 IDs）重建草稿接受源，比通用表命中率更高；表 hash 进 cache namespace，旧前缀自动隔离 | accept len ≈2.0 → 2.0–2.3；注意：换表后 prefix cache namespace 一次性重置，需预热 2–3 轮 |
| 7 | MTP steps：3 → **2**（draft=3/topk=1） | fp4 KV 下 steps=3 的接受率增益填不平随 verify token 数线性涨的 dequant 开销；实测 steps=3：C1 掉到 ~133–135，accept rate 波动 0.29–0.62 | steps=2 是甜点；此结论在基准期已定，调优后复验依旧成立 |
| 8 | *（第三轮叠加）* int8 mamba ckpt 池：关（上游对 PLE side states 直接 ValueError 拒绝共存）→ 补丁 0002 PLE 镜像后**开** | 1M 上下文下 BF16 ckpt 槽是下一个显存前沿；镜像开销 <10 MB/槽 vs 时间态 ~27 MB/槽 —— 详见「第三轮」一节及 PR #38619 | ckpt 池 48 槽 / 1.42 GB，由 KV 池 1441792→1179648 腾资；soak 326/326；大量不同长前缀的驱逐余量 |
| 8b | *（第五轮叠加）* `mamba-max-states-per-path`：-1（无帽）→ **16** | int8 池 128 槽；extra_buffer_lazy 下 chunked prefill **每 4096 token 捐 1 份 ckpt** → 单条 549K 链 ≈134 槽 → 两条链互相 LRU 驱逐 checkpoint → `cached=0`（详见「第五轮」一节） | 2×549K 与 3×495K 重问 89s → 0.9–1.5s 全命中；C12 无回归；零显存成本 |

写进 unit 的操作性约束：`mamba-radix-cache-strategy` 和 `ple-offload-embedding` 是别名/BooleanOptionalAction 参数，YAML ConfigArgumentMerger 不认（`DeprecatedAliasStoreAction` 报错）—— 只能以 CLI flag 形式留在 systemd `ExecStart`，不进 YAML。

## 第三轮（2026-09-09）：int8 mamba checkpoint × PLE side-states 启用

背景：1M 上下文的 nvfp4 池下，mamba BF16 checkpoint 槽是下一个显存前沿。上游 `maybe_init_int8_mamba_checkpoint_pool` **拒绝**与 Qwen4-Exp PLE side states 共存（`int8_ckpt_pool` 与 `ShortConvPool`/`NGramPool` 同时激活时 `raise ValueError`）—— int8 池 donate 状态后释放 BF16 活动槽，但 PLE 池按该槽索引行，donate 后 side states 成孤儿。当时 `gh search` 上游无任何 issue/PR 覆盖此组合。

`patches/0002-mamba-ckpt-ple.diff` 的做法（取代守卫）：

1. `MambaCheckpointPool.__init__` 新增 `ple_side_states` —— 每个启用的 side-state 池建一个镜像缓冲（ShortConv → bf16 行，NGram → int64 行），槽轴尺寸 `ckpt_slots + 1`，精确拷贝（量化只作用于 temporal state）。
2. `store_from_active` / `load_to_active` 连同镜像行一起搬运；`clear()` 刻意让镜像跨 flush 存活（与 qdata 同一纪律）。
3. `estimate_mem_usage_bytes` 新增 `ple_extra_bytes`；实测开销 <10 MB/槽，对比 temporal 的 ~27 MB/槽。
4. `memory_pool.py` 把 ValueError 守卫换成真正的 `ple_side_states=[...]` 透传。

本机启用方式：`PYTHONPATH=/opt/sglang-patch/sglang/python` overlay + unit `ExecStart` 追加 `--enable-int8-mamba-checkpoint`。ckpt 池（48 槽，1.42 GB：qdata 1.29 + scale 0.02 + conv 0.10 + ple 镜像 0.01）在 token 记账公式**之外**筹资，因此 `max-total-tokens` 依次下调 1441792 → 1310720 → **1179648**，保住 ~3.2 GB 启动余量。第一轮 pool=1310720 的 soak 在运行中途死于真实的 `torch.OutOfMemoryError`（GDN extend 路径；ckpt 池之上叠 late-load JIT + 6 并发激活）—— 再降 131K 后修复。第二个坑：重启主服务后，warmup oneshot 若已是 `Finished` 状态，`systemctl start` 会静默 no-op —— 用 `systemctl restart sglang-dealignai-nvfp4kv-warmup`，并在其 journal 确认 `[warmup-fp4kv] warmup complete`。

最终栈 soak 结果（`scripts/test_ckpt_ple_patch.py` = CPU 集成测试 T1–T6，含抓到 kwargs 泄漏 TypeError 的 factory 路径；服务级压测见 `/tmp/stress_int8ple.py`）：**326 请求 / 326 ok / 0 fail**，覆盖 Phase A（64 个不同前缀 ×3 遍，打爆 48 槽 ckpt 池 → 驱逐 → 重载）、Phase B（同前缀重放，走 int8 + PLE 镜像回载）、Phase C（6×200K KV crunch）。延迟 p50=23.7s p95=45.4s max=135.7s，cutoff 后 journal 干净（零 OOM、零 leak/invariant），VRAM 稳态 96.8 GB，`NRestarts=0`。最终池热态：C1 ≈87 / C6 ≈91 / prefill ≈9.7K tok/s —— 略低于 spec-on 峰值，这是 1M 池的代价。已提交上游 **sgl-project/sglang#38619**。

补丁树卫生：`/opt/sglang-patch` 基于钉死的基线 commit `78c5024e9`，不是最新 HEAD —— 新版 `memory_pool.py` 引用了基线 `utils.py` 里不存在的 `set_mla_kv_buffer_dcp_sharded_triton`，混用会导致整服务启动 ImportError。正确做法是在该 commit 的干净检出上重新应用 `0002`，不要拷 HEAD 文件进去。

## 第四轮（2026-09-10）：动态专家冷池 —— GPU 只养 346/512 专家，投机解码与 12 并发复活

背景：1M 上下文下，路由专家的 NVFP4 权重（~86 GB 静态占用里约 68 GB）就是那堵墙 —— 第三轮为了养 KV 池不得不杀掉投机解码。机制参照 [ranxianglei/sglang `ours/main`](https://github.com/ranxianglei/sglang)（keep-mask 消融 + keep-only 卸载 + cold-pool v2 动态搬运，W4A16/GPTQ 栈）；`patches/expert_cold_pool.py` 是针对 **NVFP4 / `modelopt_fp4` / `flashinfer_cutlass`** 的重新实现 —— 参数名、scale 处理、graph 约束全都不同，没有照抄任何一个名字。

工作原理（全部 env 门控；不设 `SGLANG_EXPERT_KEEP_MASK` + `SGLANG_EXPERT_KEEP_OFFLOAD=1` 就完全惰性）：

1. **静态 keep-mask** —— 按路由频率普查产出的每层 keep 集（330/512，`config/expert_keep_330_final.json`），用常驻缓存 bias 列把冷专家从 router logits 里屏蔽掉（`torch.finfo.min`，首次 eager 调用时构建、按 `(layer_id, dtype)` 缓存 —— 图捕获内绝不做 list→CUDA 索引）。
2. **加载后物理收缩** —— 钩在 `model_runner.load_model`，严格位于 `process_weights_after_loading` *之后*：张量清单门（NVFP4 行参数 + `weight_scale_2`/`input_scale` 标量 + 派生 `g1_alphas`/`g1_alphas_up`/`g2_alphas` 全部显式白名单；出现未知每专家张量**拒绝武装**），冷行拷进 host，GPU 张量重建为 `keep+slots=346` 行。**别名守卫**拒绝非同一对象的 `w*_blockscale_swizzled`（只有 `swizzle_blockscale` 无需 padding 时别名才成立 —— 配置漂移要响亮报错，不能让 kernel 静默吃旧数据）。
3. **扁平 pin** —— `CachingHostAllocator` 把每个 ≥1MB 的 pin 请求向上取整到 2 的幂；按参数逐个 pin 时 24.2 GB 数据吃掉 40.5 GB。改为每层一个精确大小的 `uint8` 缓冲（48 × ~512 MB = 24.15 GB，浪费 3.7%），按专家发行视图。
4. **图回放之外的需求搬运** —— `select_experts` 抽样暂存**屏蔽前**的原始 logits（被 -inf 屏蔽的冷专家永远不可能从选择路径产生需求 —— v2.8 冷需求探针在 alpha 带 `sigmoid ≥ 0.8 × hot_min` 上读原始 logits，需 2 次重复命中）；`model_runner` forward 后钩子把赢家搬进空闲/LRU 槽（H2D 走当前流），按**内容**更新常驻 remap+bias 表（形状永不变 → 图始终有效），驱逐先屏蔽后覆写。
5. **v2.9 身份门** —— MTP 草稿层在**同一进程内**复用 decoder `layer_id 0..47`；只认 layer_id 的门控会把 target 的 mask+remap 静默套到草稿 router 上（accept rate 0.21，草稿全是垃圾）。现在要求调用方的 `TopKConfig` **对象身份**在收缩时被注册过 —— 草稿（未注册）原样放行。

现役栈实测（冷池 ON + decode graphs + NEXTN steps=2，conc 12，池 2,359,296）：

| 门禁 | 结果 |
|---|---|
| 搬运正确性 | 8448 行 host↔GPU 校验逐字节一致，eager 与 CUDA graph 双态通过；soak 全程零 MISMATCH / stage_fail / CUDA error |
| 搬运活跃度 | 30 分钟 soak：staged 4997、h2d 13.82 GB；强制 alpha 证明：staged 1618 / evicted 850 |
| 解码 C1 | **122–129 tok/s** 预热后验收 bench（steps=2/draft3）。对比：steps=1 约 116、纯 graphs 无投机 104–105。浅投机 A/B 推翻过"steps=1 免费省 2.8GB"的判断——steps2 的 accept len 2.07–2.31 压过 steps1 的 1.74–1.79，故 steps=2 保持默认，steps=1 降级为显存告急时的应急档 |
| 解码 C12 聚合 | **704–710 tok/s**（同一 harness 两轮独立采样；早先 773–825 出自热缓存窗口，已废弃） |
| KV 池余量 | 冷池 OFF 对照在 1,572,864 直接 **OOM** → keep330+16 正是解锁池子的钥匙；2,359,296 启动后 torch-avail 7.1 GB，12×66K 长文压测峰值 92,694/97,887 MiB、真实围栏仍留 5.1 GB，零 OOM/retract |
| 主机内存 | 112 GB 机器上：冷池 pinned 24.15 GB + PLE 表 64 GB —— soak 末 MemAvailable ~20 GB，swap 稳定 |

诚实声明：需求探针目前是纯 torch 实现（eager 下 ~0.3 ms/层/拍 —— graph 态下不可见，Triton kernel 留给提速阶段）；keep=330/slots=16 是起点不是调优终点；原作者自己的结论（静态 keep 集 Pareto 最优、动态搬运 WIP）仍是我们要用 Phase B 数据去推翻的活假设；另外 greedy 字节级不确定性在**冷池完全关闭**时同样存在（atomics/cuBLAS 运行时属性，OFF 对照已证）—— 质量门禁是校验和 + 语义 + 错误计数，永远不是裸字节相等。

## 第五轮（2026-09-11）：cached=0 破案 —— 每路径 mamba checkpoint 帽

**症状**：大 prompt 重问 `cached_tokens: 0`、整篇冷重算（549K ≈89 秒）。radix 树还在、KV 页也还在 —— 缺的是 **mamba** 侧状态：前缀匹配要求命中节点携带有效 mamba checkpoint，checkpoint 没了就没有复用。证据矩阵下五个假说全部阵亡（模板措辞、单上下文规模、池占用率、mamba 锚槽饥饿、int8 路径本身 —— D2 把 int8 checkpoint *关掉*依旧双 miss）。

**取证（E3）**：三个探针 —— match 出口、split 事件、donate 点（带 int8 池水位逐事件读数）—— 由 `SGLANG_E3_PROBE=1` 门控，现役服务上挂了一天，随后彻底撤净（源码 md5 比对、grep 零命中、进程 environ 干净、journal 计数归零）。实锤：`extra_buffer_lazy` 下 chunked prefill **每 4096 token 的 chunk 捐一份 int8 checkpoint**（PREP `cache_len` 以 12544 → 16640 → 20736 行进）。一条 549K 链 ≈ **134 份 > 128 槽池**（池 = 2× `max_mamba_cache_size` 64；`mamba_max_states_per_path` 默认 `-1` 无帽）。两条大链 = 268 份 → 互相 LRU 驱逐；被驱逐一方重问匹配不到任何东西。miss 又触发重新灌入、再捐一轮 —— 自我放大（三链全灭案与 fp8kv 旧 3×336K 全 miss 案皆此因）。

**修复**：`--mamba-max-states-per-path 16` —— 一个 flag、零显存、两方案通吃。每条链保留均匀分布的 16 个锚点而非 134 个密集锚；整篇复用完全不受影响，文档中段深分支只是多算中间一段（自愈型性能损失，永远不是正确性故障）。池水位从慢性耗尽变为稳定 `ckpt_free 82/128`。前后对照：

| 形态 | cap -1 | cap 16 |
|---|---|---|
| D2 · 2×549K 重问 | 88.8 / 91.0 s，cached=0 | **0.9 / 1.0 s，cached=548,864 双命中** |
| hw77 · 3×495K 重问 | 74.5–74.8 s，cached=0 | **1.1 / 1.5 s，cached=494,848 命中** |
| g2x · 3×233K 重问 | 本就命中（悬崖以内） | 不变 |
| C12 突发 | 704–710 tok/s | 606–703 tok/s，accept 2.18–2.31（噪声带内） |

**否决项**：池扩到 256 槽（+3.7 GB 对着 5.29 GB 余量走钢丝）—— 而且 checkpoint 数随文档长度线性涨，2×549K 照样撑爆；帽是拆悬崖，扩池只是把悬崖挪远。备选杠杆（暂缓）：donate 间隔 4096→16384（≈34 份/链，代价=部分命中时多算 ~16K token）。

**归因修正**：这不是上游 #22935（split 墓碑）—— #38625 式 interior checkpoint 正是 `extra_buffer_lazy` 已有的行为。纯粹是我们这边的池容量经济学。

## 第六轮（2026-09-11）：fp8kv 升现役 + QSA indexer split-K —— 1M 冷灌 TTFT 293s → 133s

**升职**：fp8kv 从回滚栈升为现役默认 —— 1M 上下文（YaRN ×4）、KV 池 1,651,520 =（1×1M + 2×256K 工作集）× 1.05 防溢出余量、并发 8 让 Hindsight/MoviePilot 侧翼小请求与主会话并行。开机验证：NIAH@1M 通过（10/50/90% 深度三针全中，pt=1,000,976）—— "YaRN×4 在 fp8 KV 下缺背书"这条悬案就此了结；三链共存驻留 91.6% 池，重问 0.8–2.9s 全命中；满池下小请求 0.66–0.81s；capture 后 fence 实测 4.00 GB（回退触发线 <2.5 GB → 砍 decode CG 桶到 [1,2,4,6]）。探针纪律是踩坑换来的：NIAH 必须走 chat 端点 + `enable_thinking=false` —— `/generate` 裸补全会把 token 预算全喂给思考链，报出假 FAIL。

**TTFT 取证**：现役服务上挂五段 env 门控计时账本（match 出口 → donate 点 → indexer select → 注意力 kernel → KV gather），把 293s 冷灌拆开：人人喊打的稀疏注意力 kernel 只占 16s；**indexer 的 MQA logits 打分占 ≈208s**。根因是 CTA 饥饿不是算法浪费：128MB logits 暂存预算 ÷ 1MB/行把晚期行块钳到 128 行，`block_q=32` 下 tilelang kernel 只开 **4 个 CTA 对阵 148 个 SM** —— 1.6 TF/s，而同一 kernel 并行化后能跑 91 TF/s。FLOP 算术与实测逐 chunk 时长闭合（预测 ≈1.07s vs 实测 1.12s），这是"嫌疑"变"实锤"的地方。

**修复 —— split-K**（`patches/mqa_splitk.py`，幂等，跑在 dspark patcher 之后）：prefill MQA kernel 的 grid 加 y 维，64 瓦片的键窗口切给 `split=64` 个 CTA 各写不相交的 logits 列（无原子操作、无跨 CTA 归约；mask kernel 照常兜尾部越界）。launcher 故意不做任何 host 侧数学 —— 一个 `.item()` 就会废掉未来 prefill CUDA-graph capture。微基准近完美线性：晚期单次迭代 19.6ms → 0.35ms（56×）。服务级：**1M 冷灌 292.9s → 133.3s（-54.5%）**，517K 80.1→58.9s，549K 89→62–66s；NIAH@1M 依旧 PASS；C8 burst 616–668 tok/s（temp-0，5 轮）对 split=1 对照组（578–634）无回归；并发预灌+解码反而更好（聚合 117 vs 102 tok/s）；fence 4.00 GB 不变。

**静默腐化审计**（本补丁可能藏的事故模式：不崩、零报错、topk 选错、模型自信答错）：① 逐比特门 —— split=64 自对拍（时间决定论，专杀重叠 CTA 竞态写）+ split=64 vs split=1 + split=32 交叉，覆盖 9 个生产形状类（早期满预算 r4096、多序列交错窗口、全窗、r1 退化、非 64 整除窗、多批 4×1024）：**全部逐比特相等**；② 45 分钟新鲜文档 soak（67 轮，indexer 窗口在 45K→250K 深度连续扫，唯一答案 NIAH + 决定论 + 冒烟）：**零违例、服务零报错**；③ `return_indexer_topk` 采集路线在本架构探过即死（capturer 的 `num_indexer_layers` 是 DSA/DSv4 专属配置，本模型自动禁用）—— 端到端 soak 顶替其职能。路上还枪毙了两个假设（triton sparse-GQA 配置表：全配置 ±1%；"kernel 已饱和"：形状中毒微基准的产物 —— 因果掩码、随机起点 padding、失真 launch 几何各自能扭曲 4–30 倍；账目算术是测谎仪）。

**fp8kv 就此成为容量+延迟双甜点位**；nvfp4kv 停机留作 conc-12 容量兜底（同一份打过补丁的源码树，split-K 在其下次启动自动生效）。

## fp8kv 方案（2026-09-11 起现役默认）

fp8kv 不是原始旧配置 —— 2026-09-08 吸收 nvfp4kv 可移植的一半调优，09-11 吃进第五轮 cap16 + 冷池，同日升为现役并带上 1M 上下文与第六轮 QSA split-K（见第六轮）。两方案同卡互换，nvfp4kv 留作 conc-12 容量兜底。

| 项 | fp8kv 状态 | 理由 |
|---|---|---|
| GDN flashinfer 双端 | 已移植 | 纯后端路由，与 KV 类型无关 |
| prefill CUDA graph 强制 full | 已移植 | 同样绕开 eager 抖动（#28386 一类） |
| SAM=decode | 已移植 | verify/draft 纯后端路由 |
| FR-Spec 64K 热表 | 已移植，共用同一 `.pt` | 表按 tokenizer 建，与 KV 类型无关 |
| extra_buffer_lazy（别名参数走 CLI） | 已移植 | 槽经济学相同 |
| int8 mamba ckpt + `mamba-max-states-per-path 16` | 已加（第五轮，CLI flag） | 同款 donate 风暴修复；fp8kv 旧 3×336K 全 miss 案属本病族（96 槽池，无帽时 ~82 份/链） |
| QSA prefill-MQA split-K | **已加（第六轮，`patches/mqa_splitk.py`）** | 晚期 indexer CTA 饥饿（4 CTA / 148 SM）；split=64 → 56× kernel 加速、1M 冷灌 TTFT -54.5%，逐比特审计静默腐化零违例 |
| MTP steps | 3 → **2**（第四轮 A/B） | steps2 并发胜出（C4 聚合 431–438 tok/s）；09-08 的「保持 3」结论已被取代 |
| 上下文 | 512K → **1M**（YaRN ×2 → ×4，第六轮） | NIAH@1M PASS 补上 fp8 KV 下的背书缺口；2×1M 双开被否决（fence 会砸到 1.75 GB） |
| KV 池 | 552,960 → 1,310,720（第四轮 B+）→ **1,651,520** | （1×1M + 2×256K 工作集）× 1.05 防溢出余量；64 页对齐；池 ≈20.3 GB @ 12,288 B/token（QSA：fp8 KV 只存 12 个 full-attn 层） |
| 并发 | 4 → **8**（第六轮） | 主 1M 会话进行时并行吃 Hindsight/MoviePilot 小请求；mamba 8×4=32 ≤ 48−16 余量成立 |
| mamba cache | 24 → **48** | 锚槽阶梯；int8 池自动 2×（96 槽） |
| HiCache L2 | **关** | 上游对 int8 mamba checkpoint fail-fast（`server_args.py:6333`）—— 与 nvfp4kv 的 #36121 禁令同侧取舍 |
| 专家冷池 | 已加（第四轮+） | `switch_fp8kv_coldpool.sh`，与 nvfp4kv 同款 keep330+16 搬运 |

现役实测（第六轮，2026-09-11）：C8 burst 616–668 tok/s temp-0（5 轮），accept 0.40–0.71 带；并发 C8×172K 冷灌聚合 117 tok/s；满池小请求 p50 ≈0.49s；1M 冷灌 133s / 重问 2.9s；fence 4.00 GB。两方案共享单卡互斥 —— 互换 = 停一对 unit 起另一对（`systemctl stop … && nvidia-smi` 确认归零再 start）。同端口、同 served-model-name —— 下游路由不管哪套方案应答都认 `Qwen3.8-Flash-Next-NVFP4`。回滚锚点：`config/baseline/` 里的 fp8kv 同名对。

## 为什么要 nvfp4 KV

fp4 KV 池把 KV 显存砍半（打包 e2m1 + 很小的分块 scale）。在 96 GB 卡上，这是 conc 12 下 1M 上下文能成立的前提 —— 权重、~44 GB 的 PLE n-gram pinned 表、CUDA graph 把其他显存全吃光了。gather-dequant 的单流代价是固有的（每步多两次 Triton launch + 一次反量化 kernel）；第四轮冷池腾出的显存重新养得起投机（steps=2 下 C1 ≈122–129 tok/s），第三、五轮的 checkpoint 池才真正给足了大量不同长前缀的驱逐余量。第六轮之后格局更新：fp8kv 拿到 1M + split-K 后成为现役默认（单流 ≈165 tok/s + 侧翼并发 8），nvfp4kv 留作需要 12 路长链并发时的容量兜底。

## 目录结构

```
patches/
  apply_nvfp4_patches.py     幂等 patcher：给 sglang 打上 QSA×NVFP4-KV 支持
                             （锚点 count==1 校验 + ast.parse + py_compile）
  0001-sampler-37962-tp1-sync-skip.diff
                             TP=1 sampler 修复：跳过跨 TP token 同步 —— 它的首个
                             NCCL op 会惰性分配 512 MB，池子接近打满时直接 OOM
                             （grammar 请求触发）。现场根因定位。
  0002-mamba-ckpt-ple.diff   int8 mamba checkpoint 池镜像 PLE side states
                             （short-conv bf16 / ngram int64 行），取代上游的
                             ValueError 守卫。对应上游 PR #38619。
  mqa_splitk.py              第六轮幂等 patcher：QSA prefill-MQA split-K
                             （grid.y=64 键瓦片均分互斥列区间；杀掉晚期 4-CTA
                             饥饿，1M 冷灌 TTFT -54.5%）。跑在 apply_nvfp4_patches
                             之后；与 split=1 逐比特对拍验证（tests/gateA2）。
                             回滚锚：原始 mqa.py = 基线 tarball（MANIFEST md5
                             813f5d65…）。
  expert_tier.py             v1 swap-in 原型 —— 原样保留作失败基线（约 10 分钟流量后
                             输出腐坏）；expert_cold_pool.py 的 R1-R3 红线来源
  expert_cold_pool.py        第四轮动态专家冷池（见「第四轮」一节）：keep-mask +
                             pwal 后收缩 + 扁平 pin + 需求搬运 + v2.9 TopKConfig 身份门
config/
  dealignai-qwen4exp-nvfp4kv.yaml   nvfp4 KV 方案 · 第五轮定档（容量兜底，停机待命）：conc12 / 1M /
                                    spec ON steps=2 / mamba64 + 每路径帽 16 /
                                    KV 池 2359296 / int8 ckpt ON（128 槽）/
                                    冷池 keep330+16 /
                                    decode CG bs[1,2,4,6,8,10,12] / prefill CG disabled
                                    （同源补丁树 —— split-K 在其下次启动自动生效）
  expert_keep_330_final.json        每层 keep 集（48 × 330 全局专家 id），路由频率普查产物
                                    —— 冷池的武装输入
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV 方案 · 第六轮定档（现役）：conc8 / 1M（YaRN×4）/
                                    MTP steps=2 / mamba48 + 每路径帽 16 / int8 ckpt ON（96 槽）/
                                    HiCache 关 / KV 池 1651520 = (1M+2×256K)×1.05 / 专家冷池 /
                                    decode CG bs[1,2,4,6,8]；unit 带 --enable-int8-mamba-checkpoint
                                    --mamba-max-states-per-path 16；源码树加打 mqa_splitk.py
                                    （09-08 移植可移植项 + 第四轮 steps2 A/B + 第五轮 cached0 帽 +
                                    第六轮 1M/split-K）
  baseline/                         双方案调优前基准快照，作回滚锚点与前后对照证据：
                                    nvfp4kv —— mamba32 自动 sizing、KV 池 786432、无 FR-Spec 表、
                                             SAM 不设、extra_buffer。
                                    fp8kv   —— 移植前状态（不含 2026-09-08 调优移植）：GDN prefill
                                             仍 triton、无 CG-full、无 SAM、无 FR-Spec 表、extra_buffer 非 lazy。
  systemd/                          每方案 main + warmup（同端口、同模型名，互斥 —— 停一个才能起另一个）
scripts/
  test_ckpt_ple_patch.py     补丁 0002 的 CPU 集成测试 T1–T6（镜像往返、dtype
                              保真、factory 路径 kwargs 检查）
  deploy_expert_cold_pool.py 冷池幂等钩子安装器（topk.py/model_runner.py 共 5 个锚点
                              + v2.8→v2.9 升级通道；锚点计数校验 + py_compile）
  switch_coldpool.sh         模式切换：eager（冷池 ON、decode CG 关）/ graphs（冷池
                              ON、CG 开）/ mtp（冷池 + graphs + NEXTN 解注释）/
                              off（完全回滚）
  soak_watch_v2.sh           30 分钟混合流量 soak + journal 监视 + greedy 探针
  soak_final.sh              轮换话题验收 soak + 内置日志分诊
  log_triage.sh              journal ERROR/WARNING 清扫 + 分类
  stress_conc12.sh           12 路短上下文并发压测 + GPU 峰值采样
  stress_long12.sh           12 路 ~66K token 长文并发压测（安全围栏探针）
  frspec_map_64k.pt           FR-Spec 投机热表产物（65,536 IDs；sha256 598b0dc4… 与 manifest
                              一致；随仓库分发，保证调优栈可复现）
  frspec_map_64k.manifest.json  构建溯源：tokenizer sha256、语料文件清单（逐文件 sha256）、
                              coverage 1.0、size/base_count/特殊 token 列表
  build_token_map.py            从语料 + 模型 tokenizer 重建热表
  warmup_qsa_nvfp4kv.py       在真实流量前进场加载，关掉 late-device-load OOM 窗口
                              （长 prefill、grammar bitmask、bs6 图桶）
  warmup_qsa_fp8kv.py         fp8 变体（含前缀缓存命中路径）
  bench_qsa_mqa_split3.py     第六轮 split-K 饥饿实锤：生产晚期几何
                              （rows=128 win=244K），56× 近线性加速
  bench_qsa_configs.py        证伪存档：triton sparse-GQA 配置表扫描
                              （全档 ±1% —— 不是杠杆，别再追）
  bench_qsa_mqa_tiles.py      证伪存档：随机起点窗口微基准
                              （病态 padding —— 形状教训的实物）
  probe_qsa_saturation.py     证伪存档：形状中毒的饱和探针
tests/
  test_cold_pool_logic.py    冷池 CPU 逻辑测试 T1–T9（收缩+搬运+remap+bias、打包
                              remap、需求暂存、弱需求拒绝、清单门、空闲行记账、bias
                              缓存刷新、屏蔽饥饿探针、身份门）
  probe_cached0_discrim.py   第五轮判别器：D1 1×517K vs D2 2×549K 分支链，
                             逐行输出 cached_tokens 命中/落空
  probe_highwater_77.py      3×495K 高水位探针（冷灌 + A/B/C 重问）
  probe_g2x.py / probe_nvfp4_mamba_gate.py
                             mamba 槽 / 锚点门探针（G2X：3×233K 全命中）
  probe_fp8kv_1m_bringup.py  第六轮开机验证：1M 冷灌 + 重问 + NIAH + 侧链
  gateA_mqa_bitwise.py       第六轮逐比特门：split=64 vs split=1 logits，
                             生产晚/中形状，锁 seed（需停服独占）
  gateA2_mqa_bitwise_extended.py
                             扩展逐比特格：9 形状类 ×（自对拍决定论 +
                             跨 split + split32 交叉）
  gateB_indexer_canary.py    indexer-topk 采集探针 —— 记录死路（capturer 在
                             本架构自动禁用）；保留作负面结果 + meta_info 字段
                             dump
  gateD_quality_splitk.py    端到端：grammar / 中段分支 / 工具调用
  （远端 /opt/sglang-test）   soak_splitk2.sh 新鲜文档静默腐化 soak（67 轮
                             零违例）、battery.sh A/B QoS 套件、c8rep.sh
                             temp-0 5 轮 burst 协议
docs/
  RESULTS.md                  完整压测矩阵、MTP steps 1–3 校准表、mamba 槽经济学、
                              每一条死路和根因
  expert-dynamic-v2-design.md 冷池 v2 设计：v1 事故的 R1-R3 红线、Phase 0 张量清单门、
                              eager 先行验证顺序、回滚门禁
bootstrap/
  MANIFEST.md                 裸机完整恢复清单（同硬件新机器）：已核验的基线 tarball
                              （官方 sgl-project commit 78c5024e9，sha256 钉死）、
                              overlay 7 文件逐文件 md5 表、NVIDIA/CUDA/Python 栈版本
  pip-freeze-20260911.txt     205 包环境锁（--no-deps 安装）
  ninja-wrapper               /opt/fakebin/ninja shim —— flashinfer JIT -j1 内存护栏
```

## 关键发现（省流版）

1. **QSA × nvfp4 KV 必须打补丁。** 上游把 nvfp4 的访问声明为 FlashInfer-prefill / TRT-LLM 原生 decode，QSA 的 Triton gather 路径两头都不是。patcher 把 QSA 路由到 PLAIN BF16 访问，在点积之前把打包数据和 scale 缓冲（都按 uint8 视图）gather-dequant 成 BF16。CUDA-graph 安全，且只要不传 `--kv-cache-dtype nvfp4` 补丁就完全惰性。
2. **HiCache × fp4 KV = 静默腐坏。** host 传输只搬打包数据缓冲、不搬分块 scale 缓冲，驱逐回载后吐出"看起来像人话"的垃圾。sglang PR #36121 提了 fail-fast，落地之前 `enable-hierarchical-cache: false` 是硬性要求。
3. **Radix cache × fp4 KV 在当前构建上是安全的** —— `_slot_move_pointer_buffers` 已经把 `k/v_scale_buffer` 和数据一起搬。553K token 驱逐压力探针验证：强制驱逐 + 重命中后 needle 依然正确。
4. **TP=1 sampler OOM（本仓库已修）。** 开 grammar 后 sampler 即使 world_size=1 也走跨 TP token 同步；首个 NCCL op 惰性分配 512 MB —— 池子 99.3% 满时致命。一行守卫，上游 #37962。
5. **MTP steps 必须按 KV 类型重新校准。** fp4 KV 下 steps=2 同时打赢 1 和 3（dequant 开销随 verify token 数线性涨；steps=3 的接受率增益填不平这个坑）。fp8 时代"steps 越多越好"的结论不迁移。
6. **Radix 会翻转 mamba 槽经济学。** Radix OFF：~1 槽/请求（conc6 用 16 就够）。Radix ON：完成请求的 mamba 状态驻留等前缀复用 → ~3 槽/请求；配 extra_buffer_lazy 降到 ~4/请求，钉 24 即够。这几个旋钮是联动的，要一起调。
7. **Triton 惰性进场加载是一类真实的 OOM。** 首次触发的 kernel 特化（GDN chunk prefill、sparse-GQA、xgrammar bitmask）会在池子打满*之后*才分配显存。真实流量进来之前把每个形状族都 warm 一遍；极端形状仍可能触发新特化 —— 用 journal 里的 `device-loaded` 做监控。
8. **NVFP4 每专家侧张量会击穿 `dim0 == E` 启发式。** 除了 `w13/w2_weight` + 分块 scale，每个专家还有 `weight_scale_2`/`input_scale` 标量与派生 `g1_alphas`/`g1_alphas_up`/`g2_alphas`；`process_weights_after_loading` 会解交织 w13、swizzle scale、并把派生参数别名进源存储。收缩必须快照 **pwal 之后**的状态（钩在 `model_runner.load_model`，不是权重加载器里），白名单张量逐个往返校验，未知项拒绝武装。`w*_blockscale_swizzled` 只在与其源 Parameter 同对象时才安全 —— 为此设守卫。
9. **被 keep-mask 屏蔽的 router 喂不饱自己的需求信号。** 冷专家是 -inf 屏蔽，选择路径的需求恒为 0，池子永久饥饿（实测：30 分钟流量 staged=0）。需求必须从**屏蔽前**原始 logits 的 alpha 带探测。同类陷阱：暂存缓冲的切片若在回填前读取必须 `.clone()` —— 一个被 `fill_` 抹掉的视图曾静默杀死探针 v1 的强需求过滤。
10. **投机解码草稿与 decoder 共享进程内 `layer_id` 号段。** NEXTN 草稿模型就是同一个类、`num_hidden_layers=1` → 它的第 0 层与 target 第 0 层撞车。任何只按 layer_id 门控的每层运行时机制（mask、remap、统计）都会静默劫持草稿 —— 要按武装时注册的 TopKConfig **对象身份**门控（v2.9；症状：草稿被 mask+remap 后 accept rate 0.21）。
11. **`extra_buffer_lazy` + int8 mamba checkpoint 下，前缀复用的天花板是 checkpoint 池，不是 KV 树。** chunked prefill 每 4096 token 捐一份 int8 checkpoint；默认每路径帽 -1 时，单条 549K 链吃掉 ≈134 / 128 槽，两条链就互相驱逐成 `cached=0`（树和 KV 页都在，匹配已死）。`--mamba-max-states-per-path 16` 是零显存修复，两方案通吃；重问悬崖（89s → 0.9s）就是证据。上游 #22935 看着像但属不同病族（`no_buffer`）—— 别去追它的 split 墓碑修复。
12. **1M 深度下冷 prefill 的瓶颈是 QSA indexer 的 CTA 几何，不是稀疏注意力 kernel。** 128MB logits 暂存预算把晚期行块钳到 128 行；`block_q=32` 时 tilelang MQA kernel 只开 4 个 CTA 对阵 148 个 SM（1.6 TF/s vs 并行化后 91 TF/s）—— 293s 冷灌里它占 ≈208s。键瓦片 split-K（grid.y=64、互斥列写入、mask kernel 不变）把 TTFT 砍半（→133s），且对 split=1 在 9 个生产形状类上逐比特相等。路上两个陷阱：微基准形状必须忠实于生产（因果掩码、连续起点、真实 launch 几何 —— 每项都能扭曲 4–30 倍，用账目算术当测谎仪）；任何在模型前向路径里 sync 的探针都会废掉 CUDA-graph capture 炸启动（`cudaErrorStreamCaptureInvalidated`）—— 必须用 `is_current_stream_capturing()` 守卫。

## 复现

```bash
# 0. 为 sm120 构建 qwen4-main-squashed 分支（PR #36497 head）的 sglang，
#    如 CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0" —— 细节见 jpezzulli/gabrielolympie 两仓库。
#    完整恢复已核验：官方 sgl-project commit 78c5024e9 即可构建；
#    钉死 tarball sha256、环境锁、ninja shim、聊天模板 → bootstrap/MANIFEST.md
export SGLANG_SRT=/opt/sglang-src/sglang/python/sglang/srt

# 1. 补丁（幂等；每次同步源码后重跑）
python3 patches/apply_nvfp4_patches.py
# 0001/0002 是裸 diff 且路径前缀不一致 —— strip 级别随 cwd 变化，
# 以下组合经 git apply --check 路径解析实测验证：
git -C /opt/sglang-src/sglang apply -p1 /path/to/repo/patches/0001-sampler-37962-tp1-sync-skip.diff   # 路径: python/sglang/…
git -C /opt/sglang-src          apply -p1 /path/to/repo/patches/0002-mamba-ckpt-ple.diff               # 路径: sglang/python/sglang/…
# （等价方案：在包根用 -p2 打 0002 —— bootstrap/MANIFEST.md 金标准演练即此法）
python3 patches/mqa_splitk.py        # 第六轮：QSA indexer split-K（跑在 patcher 之后）

# 2. 配置 —— YAML 放入 /opt/sglang-config，编辑 unit 里的 --model-path
#    （现役 unit 硬编码 CT112 模型目录，YAML 里的 chat-template: 同理需改；
#     仅 baseline unit 保留 ${MODEL_DIR} 占位符）
systemctl start sglang-dealignai-qwen4exp-fp8kv        # 第六轮现役方案
systemctl enable --now sglang-dealignai-fp8kv-warmup   # one-shot，等 /health
# 容量兜底（单卡互斥 —— 先停 fp8kv）：
#   systemctl start sglang-dealignai-qwen4exp-nvfp4kv + 其 warmup unit

# 3.（第四轮，可选）专家冷池 —— overlay 树 + 钩子 + keep 集 + env
cp -a /opt/sglang-src /opt/sglang-patch            # fork 隔离，基线 commit 钉死
/opt/sglang-env/bin/python scripts/deploy_expert_cold_pool.py   # 5 锚点，幂等
# patches/expert_cold_pool.py -> $S/layers/moe/expert_cold_pool.py
# config/expert_keep_330_final.json -> /opt/sglang-config/
# unit 需要：Environment=PYTHONPATH=/opt/sglang-patch/sglang/python
#            Environment=SGLANG_EXPERT_KEEP_MASK=… SGLANG_EXPERT_KEEP_OFFLOAD=1
#            Environment=SGLANG_EXPERT_COLD_POOL_SLOTS=16
/opt/sglang-env/bin/python tests/test_cold_pool_logic.py       # T1–T9，纯 CPU
bash scripts/switch_coldpool.sh eager    # 先在 eager 态验证搬运
bash scripts/switch_coldpool.sh graphs   # 再开 CUDA graph
bash scripts/switch_coldpool.sh mtp      # 最后开 NEXTN 投机
bash scripts/switch_coldpool.sh off      # 任何一步都可完全回滚

# 3b.（第五轮，两方案）cached0 帽 —— unit 上的 CLI flag：
#   --enable-int8-mamba-checkpoint --mamba-max-states-per-path 16
#   int8 池自动 = 2× max_mamba_cache_size；帽让 N 条长链共存
#   （每路径 16 份 vs 无帽时 549K 链 ≈134 份）

# 4. 验证
curl -s localhost:8000/v1/models          # → Qwen3.8-Flash-Next-NVFP4
curl -s localhost:8000/get_server_info | jq '.kv_cache_dtype, .disable_radix_cache'
journalctl -u sglang-dealignai-qwen4exp-nvfp4kv --no-pager | grep "COLD-POOL"  # SHRUNK 行
```

nvfp4 KV 路径完全由 `kv-cache-dtype: nvfp4` 门控 —— 换回 `fp8_e4m3`（用 fp8kv 方案文件）后，本仓库所有补丁均为惰性。要跑 fp8kv 方案：停 nvfp4kv unit，起 fp8kv 同名 unit 对（main + warmup），预期值见 §fp8kv 方案一节。

## 约束与诚实声明

- 定档调优栈（第二轮，显存项已被第三轮取代）：GDN flashinfer 双端、mamba 钉 24、extra_buffer_lazy（别名参数走 CLI）、SAM=decode。第三轮现状：ctx 1M / YaRN ×4 显式 / spec OFF / KV 池 1,179,648 / int8 ckpt ON（补丁 0002 overlay）。**第五轮定档栈（第六轮起 nvfp4kv 转为兜底待机）：冷池 keep330+16 / spec ON steps=2 / conc 12 / mamba 64 / 每路径 ckpt 帽 16（int8 池 128 槽）/ KV 池 2,359,296 / decode CG bs[1,2,4,6,8,10,12]。** 热态：C1 ≈122–129 / C12 聚合 ≈704–710 / prefill ≈9.7K（冷前缀）——预热后验收 bench（steps=2/draft3），取代早先热缓存窗口的 146–171 / 773–825。回滚 = `switch_coldpool.sh off`，或用 `config/baseline/` 文件覆盖现役文件后重启 unit。
- 定档 fp8kv 栈（第六轮，现役）：可移植项已移植（见 §fp8kv 方案）、MTP steps=2（第四轮 A/B）、HiCache 关（int8 ckpt fail-fast）、conc **8** / **YaRN×4 1M** / **KV 池 1,651,520 = (1×1M + 2×256K) × 1.05 防溢出余量**（池 ≈20.3 GB，**capture 后 fence 实测 4.00 GB** —— 回退触发线 <2.5 GB → 砍 decode CG 桶到 [1,2,4,6]）/ mamba 48 + 帽 16 / 专家冷池 / **QSA split-K**（1M 冷灌 TTFT 133s）。2×1M 双开已否决：需 25.77 GB，拆墙后 fence 剩 1.75 GB 属走钢丝。NIAH@1M 背书：**PASS**（第六轮开机验证，chat 端点 + `enable_thinking=false`）。当前现役；nvfp4kv 停机作 conc-12 兜底。回滚 = 用 `config/baseline/` 的 fp8kv 同名对覆盖现役文件。
- 单卡消费级 GPU + 44 GB pinned PLE 表：nvfp4kv 与 fp8kv 两方案互斥；冷启动约 4-5 分钟（第四轮多 ~3.5 分钟权重加载 + 收缩）。unit 故意不 enable，避免开机抢 GPU。
- 第四轮主机内存预算是刻意压到极限的：冷池 pinned 24.15 GB + PLE 表 pinned 64 GB（47.7 GB 被 2 的幂取整 —— `file` 后端 A/B 已调研、暂缓）压在 112 GB 机器上 → soak 末 MemAvailable ~20 GB。再要挂任何 pinned 消费者之前先看 `Shmem` 与 swap。
- 冷池 + 投机的 C1（≈122–129）已逼近但略低于 fp8kv（≈165）—— 第三轮"fp4 KV 更慢"的差距主要是 spec-off 税，不全是 gather-dequant。上游原生 fp4 QSA 解码池（#37798）是追平或反超的路。
- 实验 unit 用 `Restart=no` 是故意的 —— 崩溃保留现场，不循环重启。
- 所有数字来自单张 RTX PRO 6000（96 GB）、sglang `0.5.19.dev6+g78c5024e` + 本地补丁。SM121/GB10 是另一个故事（参考 gabrielolympie 关于该架构长上下文静默腐坏的记录）。
- 更换 FR-Spec 热表会一次性重置 prefix cache namespace —— 换表后预热 2–3 轮再上真实流量。

## 致谢

- [sgl-project/sglang](https://github.com/sgl-project/sglang) —— `qwen4-main-squashed` 分支（PR #36497、#36806）
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks)（MiaAI Lab，MIT）—— 本仓库移植并向 SM120 扩展的 dspark QSA×nvfp4 补丁
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) —— sm120 铺路工作
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— fp8-KV 性能天花板，本仓库 KV 权衡的参照系
- [ranxianglei/sglang `ours/main`](https://github.com/ranxianglei/sglang) + [ranxianglei/sglang-expert-profile](https://github.com/ranxianglei/sglang-expert-profile) —— keep-mask / keep-offload / cold-pool v2 机制架构（W4A16），第四轮为其 NVFP4 重新实现

## 许可证

Apache-2.0（与补丁所基于的 sglang 一致）。dspark 衍生 patcher 的文件头保留 MiaAI Lab 的 MIT 署名。见 [LICENSE](LICENSE)。
