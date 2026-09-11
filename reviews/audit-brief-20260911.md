# 审计简报 — 2026-09-11（星野 → 星鉴）

## 范围界定
本轮审计窗口：2026-09-07 → 2026-09-11，仓库 /root/sglang-fl-nvfp4（main @ 50b9c33），共 ~30 commits、60 files、+18K 行。其中大部分是文档/探针，真正进生产运行时的代码分三块（R1 冷池、R2 split-K+mamba、R3 配置运维），R4 审证据可证性。

## Live 与仓库一致性（星野已核验，星鉴可抽查）
仓库 runtime-src/ + patches/ 快照 == CT112 /opt/sglang-patch 运行树（md5 逐一相等）：

| 文件 | md5（repo == live） |
|---|---|
| runtime-src/mqa.py = srt/layers/attention/qsa/mqa.py | d04fb0a5a90a06579b159cfde4411d68 |
| runtime-src/topk.py = srt/layers/moe/topk.py | d150f8fde0a68d681d70c341ba201603 |
| runtime-src/model_runner.py = srt/model_executor/model_runner.py | 32e773e0111c9b305e30417c2d4648a8 |
| runtime-src/schedule_batch.py = srt/managers/schedule_batch.py | 83fa9f38dee96b1105cf00957d82d413 |
| runtime-src/mamba_component.py = unified_cache/components/mamba_component.py | fadf4a13fd94824e3ba2f416b9058a39 |
| E3 探针残留 | 已清除（journalctl grep [E3] 空源文件核验 0 命中） |

结论：**审仓库快照即审生产**，无需登 CT112（也不允许登，见硬约束）。

## 上游基线
- 官方 sgl-project commit 78c5024e9（2026-08-30），sha256 钉死于 bootstrap/MANIFEST.md
- 含 #24954（extra_buffer overlap race 修复）；不含 #38625（interior grid-boundary checkpoints，#22935 的 fix 且其 track entry 对 spec/int8-ckpt 关闭）

## 这几天形成的关键结论（待星鉴独立证伪，勿背书）
1. cached0 根因 = int8 ckpt 池撑爆（1 ckpt/4096-tok chunk 捐赠风暴），修复 = mamba-max-states-per-path=16（upstream 已有参数，纯 YAML 配置，非改码）
2. split-K：QSA indexer MQA logits CTA 饥饿（grid 仅 4 CTA vs 148 SM），tilelang kernel split=64，1M 冷 TTFT 292.9s→133.3s（fp8kv 实测，nvfp4 继承同一源码树）
3. nvfp4 池 2,477,312 实炸教训：late-load Triton kernel 税 ~4.25-5.3GB 必须计入 fence 预算；终值 2,202,048（≥2,202,010 向上 64 页对齐），boot fence 6.42GB、稳态 free ~1.1-1.9GB
4. mamba 参数与 KV 池无联动需求（shape 探针实测三链共存 mamba usage 峰值 0.14）
5. sleep-on-idle 生效（与 memory_saver 无关）；prefill CUDA graph intent=full 但被 #28386 守卫运行时禁用
6. HiCache 与 int8 mamba ckpt 互斥（upstream server_args.py:6333 fail-fast）
7. 单卡互斥：fp8kv 当前 stopped，nvfp4kv live（健康 200，全部验证通过：NIAH@1M PASS、D2 双 549K 全命中、C12 541 tok/s errors 0）

## 星鉴纪律（重要）
- **只读审计**：不修改任何文件、不执行 systemctl/restart、不跑任何探针、不 SSH 到 10.10.4.12（生产在用，重启权在棣民）
- **单卡 ≤ 45 分钟**：每卡自带 --max-runtime=50m，宁可交部分报告并标注未覆盖面，不要无限深挖
- 评分制：🔴 每项 -40 / 🟡 每项 -20 / 🟢 每项 -10；<95 = 打回并列必修复项
- 报告落盘 reviews/ 目录 + kanban 评论结论，两处一致