# Qwen3.8-Flash-Next × NVFP4 KV —— 单卡 RTX PRO 6000（SM120）部署实录

[English](README.md) | **简体中文**

**在单张 RTX PRO 6000 Blackwell（96 GB，sm120）上以 `--kv-cache-dtype nvfp4` 跑通 Qwen3.8-Flash-Next（180B MoE，NVFP4 权重）的补丁、双方案部署配置与 MTP 校准数据 —— 含 QSA 稀疏注意力。**

上游 sglang 无法让 NVFP4 *KV cache* 与 Qwen 稀疏注意力（QSA）共存：Triton gather 路径拿到的是打包 fp4 缓冲区，直接死在 `KeyError: 'float4_e2m1fn_x2'`。本仓库给出可用的修复（移植自 [dspark](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks) 的 MIT 补丁，并针对 dspark 的 SM121 环境根本走不到的 SM120 trtllm-gen 稀疏解码路径做了适配），外加整套调优经验：sampler OOM 修复（[#37962](https://github.com/sgl-project/sglang/issues/37962) 同类问题）、HiCache 硬性约束、fp4 KV 下的 radix cache 经济学、完整的投机解码 steps 校准。

前作基础：[jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) 与 [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— 两家都是 fp8-KV；**本仓库补上的是他们没有的 nvfp4-KV 数据点**。

## 成果

| | fp8kv 方案 | **nvfp4kv 方案** |
|---|---|---|
| KV cache 类型 | fp8_e4m3 | **nvfp4（打包 e2m1 + 分块 scale）** |
| 上下文 | 512K（YaRN ×2） | **768K（YaRN ×3）** |
| KV 池 @ 786432 tokens | 9.4 GB（fp8，装不下） | **~5.1 GB** |
| 并发 | 4 | **6** |
| 解码 C1 | ~200 tok/s | 122 tok/s（比 fp8 C1 低 34%） |
| 解码 C6 聚合 | — | **409 tok/s** |
| Prefill | ~11K tok/s | ~11K tok/s |
| MTP（NEXTN）steps | 3 | 2（实测校准得出，见下文） |
| Radix 前缀复用 | 开 | **开 —— 58K 共享前缀命中 99.96%，6.3s → 0.6s** |
| HiCache L2 | 开 | **必须关**（见约束） |

nvfp4 KV 路径的质量门禁全绿：NIAH 200K、6×107K 并发池压下的 needle 测试（786K 池常驻 645K）、驱逐后前缀重查正确性、grammar JSON ×6、工具调用、零 retract、零报错。

## 为什么要 nvfp4 KV

fp4 KV 池把 KV 显存砍半（打包 e2m1 + 很小的分块 scale）。在 96 GB 卡上，这就是 conc 6 下 512K 和 768K 上下文的区别 —— 权重、~44 GB 的 PLE n-gram pinned 表、MTP 图把其他显存全吃光了。代价是单流解码 −34% 的回退（gather-dequant 每步多两次 Triton launch + 一次反量化 kernel）；并发场景下这个代价被摊薄，净收益是容量。

## 目录结构

```
patches/
  apply_nvfp4_patches.py     幂等 patcher：给 sglang 打上 QSA×NVFP4-KV 支持
                             （锚点 count==1 校验 + ast.parse + py_compile）
  0001-sampler-37962-tp1-sync-skip.diff
                             TP=1 sampler 修复：跳过跨 TP token 同步 —— 它的首个
                             NCCL op 会惰性分配 512 MB，池子接近打满时直接 OOM
                             （grammar 请求触发）。现场根因定位。
config/
  dealignai-qwen4exp-nvfp4kv.yaml   nvfp4 KV 方案（conc6 / 768K / MTP2 / mamba32 / radix ON）
  dealignai-qwen4exp-fp8kv.yaml     fp8 KV 方案（conc4 / 512K / MTP3 / mamba24 / HiCache ON）
  systemd/                          4 个 unit：每方案 main + warmup（同端口、同模型名，
                                    互斥 —— 停一个才能起另一个）
scripts/
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
6. **Radix 会翻转 mamba 槽经济学。** Radix OFF：~1 槽/请求（conc6 用 16 就够）。Radix ON：完成请求的 mamba 状态要驻留等前缀复用 → ~3 槽/请求，得上 32。这两个旋钮是联动的。
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

nvfp4 KV 路径完全由 `kv-cache-dtype: nvfp4` 门控 —— 换回 `fp8_e4m3`（用 fp8kv 方案文件）后，本仓库所有补丁均为惰性。

## 约束与诚实声明

- 单卡消费级 GPU + 44 GB pinned PLE 表：两方案互斥；冷启动约 4 分钟。
- C1 解码比 fp8 低 34% 是当前 gather-dequant 方案的固有代价；上游原生 fp4 QSA 解码池（#37798）能消掉它。
- 实验 unit 用 `Restart=no` 是故意的 —— 崩溃保留现场，不循环重启。
- 所有数字来自单张 RTX PRO 6000（96 GB）、sglang `0.5.19.dev6+g78c5024e` + 本地补丁。SM121/GB10 是另一个故事（参考 gabrielolympie 关于该架构长上下文静默腐坏的记录）。

## 致谢

- [sgl-project/sglang](https://github.com/sgl-project/sglang) —— `qwen4-main-squashed` 分支（PR #36497、#36806）
- [Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/Olyno/Qwen3.8-Flash-Next-Dual-DGX-Sparks)（MiaAI Lab，MIT）—— 本仓库移植并向 SM120 扩展的 dspark QSA×nvfp4 补丁
- [jpezzulli/sglang-rtxpro6000](https://github.com/jpezzulli/sglang-rtxpro6000) —— sm120 铺路工作
- [gabrielolympie/sglang-flashnext-sm120](https://github.com/gabrielolympie/sglang-flashnext-sm120) —— fp8-KV 性能天花板，本仓库 KV 权衡的参照系

## 许可证

Apache-2.0（与补丁所基于的 sglang 一致）。dspark 衍生 patcher 的文件头保留 MiaAI Lab 的 MIT 署名。见 [LICENSE](LICENSE)。
