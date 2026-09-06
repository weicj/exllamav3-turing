# TP2 Quality Baseline

状态：质量关通过，性能关未通过。

## 运行范围

- 主机：`.31` / `superserver`
- GPU：两张 RTX 2080 Ti 22 GiB，固定 UUID
  - `GPU-1c2b8831-227f-96d1-0c0f-92a189726072`
  - `GPU-4da87347-023e-7014-a837-6b1fb649a42e`
- 模型：`/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4`
- 并行：ExLlamaV3 NCCL TP2，`use_per_device=22,22`
- 基准：随机 4K prompt、greedy 128-token decode、无 MTP、fresh KV/recurrent state

## 合格配置

```text
CUDA_VISIBLE_DEVICES=GPU-1c2b8831-227f-96d1-0c0f-92a189726072,GPU-4da87347-023e-7014-a837-6b1fb649a42e
NCCL_ALGO=Ring
NCCL_PROTO=Simple
EXL3_NGRAM_STREAM=1
EXL3_FLASHINFER_WORKSPACE_MB=128
```

命令入口：`tools/bench_qwen38_tp2_4k128.py`。基准脚本要求
`EXL3_NGRAM_STREAM=1`，并固定两个可见 GPU 的 UUID；不要按整机 GPU index 猜卡。

## 质量证据

当前 sparse-QSA TP2 基线 token hash：

```text
c6e8218f6c53cdecbdfaedc4a124ed50f36e5447e62241c6ba77452184667e44
```

- 历史 4K/128 fresh-state 重复验收：默认 sparse-QSA `8/8` hash 一致。
- 最新功耗修复后按相同请求顺序复验：第二个正式请求（seed 3）仍为上述 hash。
- 真实检索题：输出 `cobalt blue`，并以 token `248046` 正常 EOS。
- 专家路由：两个 TP rank 的记录一致；完整 128-token decode 中专家 `0--255` /
  `256--511` 命中为 `33647 / 31633`（`51.54% / 48.46%`），逐层平均绝对差 `2.40`，
  最大差 `10`。没有证据支持静态 expert 重分片作为主要优化方向。

## 性能证据

功耗修复后，Ring/Simple、sparse-QSA 的三次请求为：

| 请求 | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1 | 692.38 | 25.78 |
| 2 | 660.75 | 25.55 |
| 3 | 658.08 | 25.35 |

稳态参考取请求 2/3：约 `659 tok/s prefill`、`25.5 tok/s decode`。同口径 seed 3
复验为 `662.16 / 25.11 tok/s`，hash 合格。功耗监控期间两卡约 `1815/1845 MHz`、
`214/228 W`，因此当前瓶颈不是功耗墙。

项目目标 `>1000 tok/s prefill` 与 `>50 tok/s decode` 尚未达到，不能将本文件视为最终
性能交付版本。

## 已淘汰路径

- `EXL3_QSA_PREFILL_DENSE=1`：约 `825 / 25 tok/s`，但 token hash `10e009...`，与
  sparse-QSA 基线漂移，只能作为 kernel 上限 A/B。
- `EXL3_TP_ASYNC_ALLREDUCE=1`：约 `694 / 25 tok/s`，hash 漂移；当前实现丢弃
  `Work`，存在消费未完成 NCCL 结果的 race，不得作为默认配置。
- native CPU reduce：约 `36 tok/s decode`，但 FP32 residual 使用 BF16 wire 且归约到达
  顺序不固定，质量不稳定。
- `EXL3_MOE_FORCE_FUSED_DECODE=1`：约 `14 tok/s decode` 且 hash 漂移。

## 验证和限制

- `tests/test_moe_deterministic_reduce_policy.py`、`tests/test_qsa_prefill_policy.py`：
  远端 venv 中 `4 passed`。
- `git diff --check`、全仓库 Python 编译检查：通过。
- `tests/test_gated_delta_rule.py`：小规模和长序列数值用例可通过；在同一进程完整
  22-case 连跑时，Triton autotune 受到 22 GiB 显存碎片影响，出现 OOM。该 OOM 是测试
  资源问题，不作为模型质量通过证据；需要分进程或清理显存后再做完整回归。

## 结果目录

- TP2 功耗复验：`/home/max/results/tp2_powerfix_retest_20260906_01/`
- 完整专家路由：`/home/max/results/tp2_moe_route_balance_20260906_005/`
- 模块/NCCL profile：`/home/max/results/tp2_profile_20260906_01/`

后续性能实验必须从本基线重新加载模型，并同时检查 token hash、真实检索题和吞吐；
未经这三项验证的 kernel/通信改动不得覆盖本版本。
