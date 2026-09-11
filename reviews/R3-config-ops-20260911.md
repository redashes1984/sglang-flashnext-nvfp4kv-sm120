# R3 审计：配置运维面一致性与状态

日期：2026-09-11 | 执行：subagent 取证（deleg_a9c41e92，13 步核验后 600s 超时未写报告）+ 星野收口补齐终验（fp8-warmup unit 比对为星野完成）
方法：只读。repo ↔ CT112 live 仅 md5/内容比对，无 SSH 写、无 systemctl、无探针。算术全部 python3 机器计算。

## 核验结果（简报 7 条事实独立证伪）

| # | 事实 | 判定 | 证据 |
|---|---|---|---|
| 1 | nvfp4 池 2,202,048 ≥2,202,010 且 64 页对齐 | ✅ | 2202048%64=0；YAML:28 键值（注释剥离后解析）=2202048；remote md5 == local（512d825c） |
| 2 | fp8kv 池 1,651,520 =(1M+2×256K)×1.05 页对齐 | ✅ | 1572864×1.05=1651507.2→ceil64=1651520；slack=78656=5.0008%；%64=0；remote md5==local（dc5a4a2b） |
| 3 | mamba 参数双 YAML 一致（cap16；active 64/48 按方案分档） | ✅ | nvfp4=64/conc12，fp8=48/conc8，cap16 双同；track-interval/策略 extra_buffer_lazy 双同（unit 内） |
| 4 | 单卡互斥在文档/unit 表达一致 | ✅ | 两主 unit 都 CUDA_VISIBLE_DEVICES=0，无 socket 分离；README/MANIFEST 均声明 |
| 5 | sleep-on-idle: true 双 YAML 在键值层（非注释） | ✅ | 机器解析双 true |
| 6 | HiCache OFF 表达 | ⚠️ 部分 | nvfp4 是 active 行 `enable-hierarchical-cache: false`，fp8kv 是注释行——效果同为 OFF，表达不统一（见 🟢-1）；互斥注释（server_args.py:6333）双 YAML 均在 |
| 7 | switch 脚本幂等 | ✅ | subagent sandbox 实测：arm_cold 双跑 env 行数=1（ba10b56 修复有效）、hicache_off 双跑幂等；strip→re-arm 等价双 arm。唯一 sandbox 报错是拿 repo 快照喂 PLEDIR 参数不齐的 harness 不匹配，生产路径 unit 含 --ple-offload-embedding（live grep 证实），非缺陷 |
| + | 凭据泄漏扫描 | ✅ 干净 | 全仓 regex（password/secret/api_key/bearer/token 赋值形态）仅命中 topk.py:1569 变量名；无凭据类文件入库 |
| + | 部署面 live 锚点 | ⚠️ 一处漂移 | 见 🔴/🟡-1 |

## 🔴 阻断

无。

## 🟡 重要

1. **config/systemd/sglang-dealignai-fp8kv-warmup.service 快照陈旧（唯一 repo≠live 的 unit）**
   repo md5 ae2fefeb ≠ live 062b120b；repo ExecStart 仍指迁移前路径 `/opt/sglang/bin/python /etc/sglang-flashnext/warmup_qsa_fp8kv.py`，live 为 `/opt/sglang-env/bin/python /opt/sglang-config/warmup_qsa_fp8kv.py`。
   其余三个 unit（两主服务 + nvfp4 warmup）repo==live 逐字节一致（22e3791f/01685e74/759d65ed）。
   风险：bootstrap/MANIFEST 把 repo systemd/ 当恢复锚点——按快照恢复新机器，fp8kv warmup 起不来（旧路径不存在）。修法：`systemctl cat` 回灌一行 ExecStart 即可。
2. **nvfp4 YAML 头注释陈旧**（line 2：`conc6 / mamba 24 / MTP OFF / CG both eager` vs 实际 conc12/mamba64/NEXTN steps2/decode CG bs 到 12）——与 R2 ℹ️ 同项交叉引用，不重复计分；运维读数以键值为准的规矩要写进 SKILL。

## 🟢 建议

1. HiCache OFF 表达统一（显式 false vs 注释二选一，两 YAML 对齐），防止未来某次 sed 把注释行当"未配置"再开出来——ba10b56 那类堆叠 bug 的孪生风险面。
2. fp8kv `page-size` 未显式钉值（依赖 upstream 默认 64），nvfp4 同。页对齐算术全建立在默认值上——upstream 哪天改默认即静默错位。建议两 YAML 显式 `page-size: 64`。

## ℹ️ 风格

warmup 脚本 bs 注释/代码不一已由 R2 🟢-2 记录（交叉引用）。

## 评分

🔴×0 −0 | 🟡×1 −20（第 2 条与 R2 合并计）| 🟢×2 −20 | ℹ️ 交叉引用 −0 → **77/100 有条件通过**
必修复项 = 🟡-1 单行回灌。7 条既证事实全部独立复核成立；配置面主体（键值、对齐算术、互斥声明、幂等、凭据纪律）无一处失真。

## 备注（流程诚实性）

本卡执行因 reviewer 模型通道两次陷入采样退化循环（"hmm"×3 次 kill），第三轮由星野原生 subagent 代执行——subagent 完成全部取证后 600s 超时未及写报告，本报告由其 transcript 内已落盘的核验输出 + 星野补完的最后一步（warmup unit 比对）合成。取证数据均为真实工具输出，无推测项。
