# Qwen3.8-Flash-Next × NVFP4 KV —— 单卡 RTX PRO 6000（SM120）部署实录

[English](README.md) | **简体中文**

**在单张 RTX PRO 6000 Blackwell（96 GB，sm120）上跑通 Qwen3.8-Flash-Next（180B MoE，NVFP4 权重）的补丁、部署配置与校准数据 —— 含 QSA 稀疏注意力。两套方案都已完成调优：`nvfp4kv`（`--kv-cache-dtype nvfp4`，现役主线，二次调优版）与 `fp8kv`（`fp8_e4m3`，回滚方案，吸收了同一批调优中可移植的一半）。各方案的调优前基准快照保留在 `config/baseline/`，作回滚锚点与前后对照证据。**

上游 sglang 无法让 NVFP4 *KV cache* 与 Qwen 稀疏注意力（QSA）共存：Triton gather 路径拿到的是打包 fp4 缓冲区，直接死在 `KeyError: 'float4_e2m1fn_x2'`。本仓库给出可用的修复（移植自 [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) 的 MIT 补丁，并针对 dspark 的 SM121 环境根本走不到的 SM120 trtllm-gen 稀疏解码路径做了适配），外加整套调优经验：sampler OOM 修复（[#37962](https://github.com/sgl-project/sglang/issues/37962) 同类问题）、HiCache 硬性约束、fp4 KV 下的 radix cache 经济学、完整的投机解码 steps 校准。

前作基础：[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) 与 [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— 两家都是 fp8-KV；**本仓库补上的是他们没有的 nvfp4-KV 数据点**。

## 成果（二次调优版栈）

| | fp8kv 方案 | nvfp4kv 基准版 | **nvfp4kv 二次调优版（现役）** |
|---|---|---|---|
| KV cache 类型 | fp8_e4m3 | nvfp4（打包 e2m1 + 分块 scale） | **nvfp4** |
| 上下文 | 512K（YaRN ×2） | 768K（YaRN ×3） | **1M（YaRN ×4）** |
| KV 池 | 552,960 | 786,432 | **1,179,648**（第三轮为 int8 ckpt 池腾资后） |
| 并发 | 4 | 6 | **6** |
| 解码 C1 | ≈165 tok/s | ~122 tok/s | **≈75–90 tok/s**（1M 下 spec OFF） |
| 解码 C6 聚合 | ≈355–365 tok/s | ~409 tok/s | **≈72–91 tok/s** |
| Prefill | ~11K tok/s | ~10K tok/s | **≈9.7K tok/s**（冷前缀；radix 命中更高） |
| MTP（NEXTN）steps | 3 | 2 | **OFF**（1M 单卡下 steps≥1 即 OOM） |
| Radix 前缀复用 | 开 | 开 | **开 —— 58K 共享前缀命中 99.96%，6.3s → 0.6s** |
| HiCache L2 | **开**（fp8 下安全） | 关 | **关**（fp4 KV 硬约束，见约束一节） |

nvfp4 KV 路径的质量门禁全绿：NIAH 200K、6×107K 并发池压下的 needle 测试、驱逐后前缀重查正确性、grammar JSON ×6、工具调用、零 retract、零报错。accept len ≈2.0–2.3，accept rate ≈0.5–0.66。

## 相对基准版改了什么、为什么、效果如何

调优版相对基准快照（`config/baseline/`）恰好七个改动点。汇总效果：**C1 +11% / C6 +36%**，下表即调优证据。

| # | 改动（基准 → 调优） | 为什么 | 效果 |
|---|---|---|---|
| 1 | GDN linear-attn 后端：仅 decode flashinfer → **prefill+decode 双端 flashinfer** | SM120 上自动解析走 SM100-only 判断，prefill 会回落到 triton；显式钉死才能用上 FlashInferGDNKernel | GDN prefill 提速，也是后面几项收益的地基 |
| 2 | `max-mamba-cache-size`：32（自动 sizing）→ **钉死 24** | radix 峰值 usage 实测 0.62 ≈ 20 槽（lazy 驱逐下），多余槽是死显存 | 省出的显存转进 KV 池（+64K → 851,968），多会话前缀驻留更多，C6 容量直接受益 |
| 3 | `--mamba-radix-cache-strategy`：extra_buffer → **extra_buffer_lazy** | 完成请求的状态为前缀复用而驻留，但不需要预扩额外槽；lazy 把每请求槽开销从 ~5 降到 ~4 | **C6 +38%**（单项最大收益）；单流微亏 −3%，并发场景净赚 |
| 4 | prefill CUDA graph：auto → **强制 full 捕获**（`cuda-graph-config.prefill.backend: full` + `max_bs: 4096`） | Breakable×multimodal 自动禁用会把 prefill 批次留在 eager（#28386 一类问题）；强制 full 绕开。一次性成本 ~0.56 GB / ~30s | prefill 稳定 ≈10K tok/s，无 eager 抖动 |
| 5 | `speculative-attention-mode`：不设（prefill 路径）→ **decode** | verify/draft 批次形状就是 decode 形状，路由到 trtllm_mha/XQA 才对得上；真实 prefill 批次不受影响（仍走 triton） | C1/C6 各 +2–4%，accept 分布不变 |
| 6 | FR-Spec 投机热表：无 → **自建 64K 表**（`speculative-token-map: frspec_map_64k.pt`，coverage 1.0） | 用自有语料（obsidian 笔记 + skill 文件 → 65,536 IDs）重建草稿接受源，比通用表命中率更高；表 hash 进 cache namespace，旧前缀自动隔离 | accept len ≈2.0 → 2.0–2.3；注意：换表后 prefix cache namespace 一次性重置，需预热 2–3 轮 |
| 7 | MTP steps：3 → **2**（draft=3/topk=1） | fp4 KV 下 steps=3 的接受率增益填不平随 verify token 数线性涨的 dequant 开销；实测 steps=3：C1 掉到 ~133–135，accept rate 波动 0.29–0.62 | steps=2 是甜点；此结论在基准期已定，调优后复验依旧成立 |

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

## fp8kv 方案（回滚栈）

fp8kv 不是原始旧配置 —— 2026-09-08 它吸收了 nvfp4kv 调优中可移植的一半，两方案同卡互换不丢通用收益。

| 项 | fp8kv 状态 | 理由 |
|---|---|---|
| GDN flashinfer 双端 | 已移植 | 纯后端路由，与 KV 类型无关 |
| prefill CUDA graph 强制 full | 已移植 | 同样绕开 eager 抖动（#28386 一类） |
| SAM=decode | 已移植 | verify/draft 纯后端路由 |
| FR-Spec 64K 热表 | 已移植，共用同一 `.pt` | 表按 tokenizer 建，与 KV 类型无关 |
| extra_buffer_lazy（别名参数走 CLI） | 已移植 | 槽经济学相同 |
| MTP steps | 保持 **3**（nvfp4kv 为 2） | fp8 accept-len 实测 2.08–2.50，深度链条仍有收益 |
| KV 池 | 保持 **552,960** | nvfp4kv 的 +64K 靠 fp4 砍半 KV 字节换来，fp8 无等价余量 |
| HiCache L2 | 保持 **开** | fp8 下正常工作；nvfp4kv 必须关（#36121） |

移植后热态实测（两轮取数，取第二轮）：fp8kv C1 ≈165 / C6 聚合 ≈355–365 tok/s。更早的单轮 ~200 C1 读数在移植后未复现，以两轮值为准。注意交叉点：fp8kv（spec-on 512K）C1 ≈165 反超当前 nvfp4kv（spec-off 1M）的 ≈75–90，但 nvfp4kv 换来了两倍的上下文容量与前缀驱逐余量。单流延迟敏感选 fp8kv，长上下文容量优先选 nvfp4kv。同端口、同 served-model-name，互换 = 停一个 unit 起另一个。回滚锚点：`config/baseline/` 里的 fp8kv 同名对。

## 为什么要 nvfp4 KV

fp4 KV 池把 KV 显存砍半（打包 e2m1 + 很小的分块 scale）。在 96 GB 卡上，这是 conc 6 下 1M 上下文能成立的前提 —— 权重、~44 GB 的 PLE n-gram pinned 表、CUDA graph 把其他显存全吃光了。gather-dequant 的单流代价是固有的（每步多两次 Triton launch + 一次反量化 kernel）；1M spec-off 下热态 C1 ≈75–90 tok/s，第三轮的 int8 checkpoint 池为大量不同长前缀提供了驱逐余量。要裸解码速度选 fp8kv（512K + spec-on），要上下文容量选 nvfp4kv。

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
config/
  dealignai-qwen4exp-nvfp4kv.yaml   nvfp4 KV 方案 · 第三轮（现役）：conc6 / 1M / spec OFF /
                                    mamba24 钉死 / KV 池 1179648 / int8 ckpt ON（PLE 镜像，补丁 0002）/
                                    decode CG bs[1,2,4,6] / spec-off 下 prefill CG disabled
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV 方案 · 调优后回滚栈（conc4 / 512K / MTP3 / mamba24 / HiCache ON；2026-09-08 移植可移植项：GDN 双端 flashinfer、prefill CG-full、SAM=decode、FR-Spec 热表、ABL=lazy —— steps 保持 3，fp8 accept-len 2.08–2.50 下深度投机仍是甜点）
  baseline/                         双方案调优前基准快照，作回滚锚点与前后对照证据：
                                    nvfp4kv —— mamba32 自动 sizing、KV 池 786432、无 FR-Spec 表、
                                             SAM 不设、extra_buffer。
                                    fp8kv   —— 移植前状态（不含 2026-09-08 调优移植）：GDN prefill
                                             仍 triton、无 CG-full、无 SAM、无 FR-Spec 表、extra_buffer 非 lazy。
  systemd/                          每方案 main + warmup（同端口、同模型名，互斥 —— 停一个才能起另一个）
scripts/
  test_ckpt_ple_patch.py     补丁 0002 的 CPU 集成测试 T1–T6（镜像往返、dtype
                              保真、factory 路径 kwargs 检查）
  frspec_map_64k.pt           FR-Spec 投机热表产物（65,536 IDs；sha256 598b0dc4… 与 manifest
                              一致；随仓库分发，保证调优栈可复现）
  frspec_map_64k.manifest.json  构建溯源：tokenizer sha256、语料文件清单（逐文件 sha256）、
                              coverage 1.0、size/base_count/特殊 token 列表
  build_token_map.py            从语料 + 模型 tokenizer 重建热表
  warmup_qsa_nvfp4kv.py       在真实流量前进场加载，关掉 late-device-load OOM 窗口
                              （长 prefill、grammar bitmask、bs6 图桶）
  warmup_qsa_fp8kv.py         fp8 变体（含前缀缓存命中路径）
docs/
  RESULTS.md                  完整压测矩阵、MTP steps 1–3 校准表、mamba 槽经济学、
                              每一条死路和根因
```

## 关键发现（省流版）

1. **QSA × nvfp4 KV 必须打补丁。** 上游把 nvfp4 的访问声明为 FlashInfer-prefill / TRT-LLM 原生 decode，QSA 的 Triton gather 路径两头都不是。patcher 把 QSA 路由到 PLAIN BF16 访问，在点积之前把打包数据和 scale 缓冲（都按 uint8 视图）gather-dequant 成 BF16。CUDA-graph 安全，且只要不传 `--kv-cache-dtype nvfp4` 补丁就完全惰性。
2. **HiCache × fp4 KV = 静默腐坏。** host 传输只搬打包数据缓冲、不搬分块 scale 缓冲，驱逐回载后吐出"看起来像人话"的垃圾。sglang PR #36121 提了 fail-fast，落地之前 `enable-hierarchical-cache: false` 是硬性要求。
3. **Radix cache × fp4 KV 在当前构建上是安全的** —— `_slot_move_pointer_buffers` 已经把 `k/v_scale_buffer` 和数据一起搬。553K token 驱逐压力探针验证：强制驱逐 + 重命中后 needle 依然正确。
4. **TP=1 sampler OOM（本仓库已修）。** 开 grammar 后 sampler 即使 world_size=1 也走跨 TP token 同步；首个 NCCL op 惰性分配 512 MB —— 池子 99.3% 满时致命。一行守卫，上游 #37962。
5. **MTP steps 必须按 KV 类型重新校准。** fp4 KV 下 steps=2 同时打赢 1 和 3（dequant 开销随 verify token 数线性涨；steps=3 的接受率增益填不平这个坑）。fp8 时代"steps 越多越好"的结论不迁移。
6. **Radix 会翻转 mamba 槽经济学。** Radix OFF：~1 槽/请求（conc6 用 16 就够）。Radix ON：完成请求的 mamba 状态驻留等前缀复用 → ~3 槽/请求；配 extra_buffer_lazy 降到 ~4/请求，钉 24 即够。这几个旋钮是联动的，要一起调。
7. **Triton 惰性进场加载是一类真实的 OOM。** 首次触发的 kernel 特化（GDN chunk prefill、sparse-GQA、xgrammar bitmask）会在池子打满*之后*才分配显存。真实流量进来之前把每个形状族都 warm 一遍；极端形状仍可能触发新特化 —— 用 journal 里的 `device-loaded` 做监控。

## 复现

```bash
# 0. 为 sm120 构建 qwen4-main-squashed 分支（PR #36497 head）的 sglang，
#    如 CUDAARCHS=120 TORCH_CUDA_ARCH_LIST="12.0" —— 细节见 jpezzulli/gabrielolympie 两仓库。
export SGLANG_SRT=/opt/sglang-src/sglang/python/sglang/srt

# 1. 补丁（幂等；每次同步源码后重跑）
python3 patches/apply_nvfp4_patches.py
git -C /opt/sglang-src/sglang apply ../patches/0001-sampler-37962-tp1-sync-skip.diff

# 2. 配置 —— 编辑 unit 里的 MODEL_DIR（YAML 里的 ${MODEL_DIR} 需手动替换，
#    systemd 不展开 YAML 文件内容），文件放到位
systemctl start sglang-dealignai-qwen4exp-nvfp4kv
systemctl enable --now sglang-dealignai-nvfp4kv-warmup   # one-shot，等 /health

# 3. 验证
curl -s localhost:8000/v1/models          # → Qwen3.8-Flash-Next-NVFP4
curl -s localhost:8000/get_server_info | jq '.kv_cache_dtype, .disable_radix_cache'
```

nvfp4 KV 路径完全由 `kv-cache-dtype: nvfp4` 门控 —— 换回 `fp8_e4m3`（用 fp8kv 方案文件）后，本仓库所有补丁均为惰性。要跑 fp8kv 方案：停 nvfp4kv unit，起 fp8kv 同名 unit 对（main + warmup），预期值见 §fp8kv 方案一节。

## 约束与诚实声明

- 定档调优栈（2026-09-08 第二轮，显存项已被第三轮取代）：GDN flashinfer 双端、mamba 钉 24、extra_buffer_lazy（别名参数走 CLI）、SAM=decode。第三轮现状：ctx 1M / YaRN ×4 显式 / spec OFF / KV 池 1,179,648 / int8 ckpt ON（补丁 0002 overlay）。第三轮热态：C1 ≈75–90 / C6 ≈72–91 / prefill ≈9.7K（冷前缀）。回滚 = 用 `config/baseline/` 文件覆盖现役文件后重启 unit。
- 定档 fp8kv 栈（同日）：可移植项已移植（见 §fp8kv 方案）、steps 保持 3、HiCache 开、conc4 / YaRN×2 512K / KV 池 552,960、FR-Spec 热表共用。热态：C1≈165 / C6≈355–365 / accept len≈2.08–2.50。回滚 = 用 `config/baseline/` 的 fp8kv 同名对覆盖现役文件。
- 单卡消费级 GPU + 44 GB pinned PLE 表：nvfp4kv 与 fp8kv 两方案互斥；冷启动约 4 分钟。unit 故意不 enable，避免开机抢 GPU。
- C1 解码仍低于 fp8（≈136 vs ≈165 tok/s，gather-dequant 的固有代价）；上游原生 fp4 QSA 解码池（#37798）能消掉这个差距。
- 实验 unit 用 `Restart=no` 是故意的 —— 崩溃保留现场，不循环重启。
- 所有数字来自单张 RTX PRO 6000（96 GB）、sglang `0.5.19.dev6+g78c5024e` + 本地补丁。SM121/GB10 是另一个故事（参考 gabrielolympie 关于该架构长上下文静默腐坏的记录）。
- 更换 FR-Spec 热表会一次性重置 prefix cache namespace —— 换表后预热 2–3 轮再上真实流量。

## 致谢

- [sgl-project/sglang](https://github.com/sgl-project/sglang) —— `qwen4-main-squashed` 分支（PR #36497、#36806）
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks)（MiaAI Lab，MIT）—— 本仓库移植并向 SM120 扩展的 dspark QSA×nvfp4 补丁
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) —— sm120 铺路工作
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— fp8-KV 性能天花板，本仓库 KV 权衡的参照系

## 许可证

Apache-2.0（与补丁所基于的 sglang 一致）。dspark 衍生 patcher 的文件头保留 MiaAI Lab 的 MIT 署名。见 [LICENSE](LICENSE)。
