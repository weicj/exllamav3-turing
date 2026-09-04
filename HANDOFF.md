# ExLlamaV3-Turing handoff

更新时间：2026-09-04  
目的：在两张 22 GiB RTX 2080 Ti（Turing，SM75）上运行
`turboderp/Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4`，并修复 TP2 的质量漂移。

## 当前结论

这个仓库是当前实验代码的独立、可追踪基线，不代表质量问题已经解决。
模型可以在两张 2080 Ti 上通过 EXL3 TP2 加载和生成，但在相同输入、空 cache、空 recurrent state、greedy 解码的重复运行中，输出 token/hash 不稳定。
目前没有证据表明单纯的 PLE、FlashInfer、NCCL ack 或 CUDA 异步返回是唯一根因。

最可靠的定位结果是：漂移通常在早期 full-attention 路径暴露，重复 trace 中第一次明显差异出现在 layer 3；Q/K/V 投影基本稳定，`attn_o` 开始变化。跨 PP2/TP2 拓扑直接比较会在更早的 layer 0 出现数值差异，因此不能把跨拓扑差异直接当作 TP bug。

下一会话应从 `tools/` 的模块边界 trace 和确定性 probe 继续，优先确认 Flash-Next 特有的 QSA、hyper-connection、PLE/GDN 状态或 cache 写入路径，而不是再次重复已经完成的开关 A/B。

## 硬件和运行环境

- 远端实验主机：`superserver`；本仓库在本机：`/home/max/workspace/ExLlamaV3-Turing`
- GPU 0（实验逻辑卡）：`GPU-1c2b8831-227f-96d1-0c0f-92a189726072`，RTX 2080 Ti，SM75，22528 MiB
- GPU 1（实验逻辑卡）：`GPU-4da87347-023e-7014-a837-6b1fb649a42e`，RTX 2080 Ti，SM75，22528 MiB
- 远端同时有 Tesla T10（也是 SM75）；所有 2080 Ti 实验必须使用上面两个 UUID，不能按全机 index 猜卡。
- 模型目录：`/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4`
- Python：远端实验使用 `/home/max/workspace/VLLM-2080ti-0.2.1-pre/.venv/bin/python`
- PyTorch/NCCL：以远端 venv 实际版本为准；不要把本机环境直接当成可运行环境。
- 上游基线提交记录在 `UPSTREAM_COMMIT`：`b3a384d95fd4a2a6998f8a670bd761375a1acc94`。

## 仓库内容

主包在 `exllamav3/`。`tools/` 保存本次 TP2 实验使用的 benchmark、trace、NCCL 和 GDN probe。顶层 benchmark 是上游/早期实验入口；新实验优先复制到独立结果目录，不把结果文件放进 Git。

本分支相对上游实验副本的关键改动：

- `model/model_tp_fn.py`：worker CUDA 命令完成后同步；NCCL worker 不使用 post-NCCL pinned input staging；加入模块、边界、attention、PLE profile/trace。
- `model/model_tp.py`：`EXL3_TP_DEFER_FORWARD_ACKS=0` 时立即 drain forward ack，用于排除 deferred-ack 生命周期问题。
- `architecture/qwen4_exp.py`：`EXL3_DISABLE_PLE=1` 可禁用 Flash-Next PLE。
- `modules/ple.py`：PLE 的 TP 导出/导入、复制的小投影、异构 recurrent state 查找，以及 PLE trace。
- `modules/attn.py`：QSA sparse 开关、attention/cache trace，以及 module identity 记录。
- `modules/attention_fn/flashinfer.py`：SM75 上的 FlashInfer 选择和 workspace 保护逻辑。
- `exllamav3_ext/`：当前 SM75 编译源；已编译 `.so` 不入库，必须在目标环境重编译。

## 已完成实验审计

| 实验 | 结果 | 证据/结论 |
|---|---|---|
| 27B EXL3 单卡 2080 Ti | 8 次重复稳定 | `qwen38-27b-exl3-35bpw-sm75-20260904`；证明通用 GDN/量化路径可稳定 |
| 27B EXL3 实际 PP2（layer split 8/22 GiB） | 8 次重复稳定 | `qwen38-27b-exl3-35bpw-sm75-tp2-quality-20260904`；证明通用 PP/P2P/layer split 不是 Flash-Next 特有质量根因 |
| Flash-Next EXL3 TP2，4K/128，默认路径 | 加载/生成成功，但重复 hash 不稳定 | `qwen38-flash-next-exl3-205-sm75-tp2-*`；典型 prefill 约 548–563 tok/s，decode 约 19–25 tok/s |
| Flash-Next TP2，worker CUDA synchronize | 仍 `stable=false` | 只修复明显的 ack/lifetime 风险，不足以解决质量 |
| Flash-Next TP2，`EXL3_TP_DEFER_FORWARD_ACKS=0` | 仍 `stable=false`，性能下降 | 典型 prefill 约 476–488 tok/s，decode 约 14–15 tok/s；deferred ack 不是唯一根因 |
| Flash-Next TP2，`EXL3_DISABLE_PLE=1` | 仍 `stable=false` | PLE 不是唯一根因；不能据此删除 PLE 修复 |
| FlashInfer off | 仍不稳定 | 关闭 FlashInfer 没有消除漂移 |
| layer 3 强制 Torch math SDPA | 仍不稳定 | full-attention 的 FlashInfer kernel 不是唯一根因 |
| `EXL3_TP_NO_FWD_BARRIER=0` | 仍不稳定 | forward barrier 不是唯一根因 |
| TP2 NCCL/collective 微基准 | collective 能运行，无质量证明 | `qwen38-flash-next-exl3-205-sm75-tp2-nccl-tiny-20260904`；小张量约 83–123 us/call，不能据此证明模型路径正确 |
| TP2/PP2 模块 trace | 重复 trace 首次明显差异通常在 layer 3 | `...parity-rerun-20260904/module-compare.json`；Q/K/V 稳定，`attn_o` 后开始变化 |
| PP2 与 TP2 跨拓扑直接 trace | 某次在 layer 0 即有差异 | `...layertrace-20260903/compare-pp-tp-rank0.json`；这是跨拓扑数值差异，不等于 TP2 重复运行根因 |
| 独立 GDN kernel probe | 尚未形成最终可采信结论 | `tools/probe_gdn_kernel_determinism.py`；必须在干净 2080 Ti、固定 extension、无其他 GPU 进程下重跑并保存完整 JSON |

### 已排除或降级的假设

1. 不是“模型根本不能 TP2 加载”：TP2 已成功加载并生成。
2. 不是一般的 GDN/PP/P2P 失效：27B 对照模型在单卡和 PP2 稳定。
3. 不是单纯的 PLE 开关：禁用 PLE 仍漂移。
4. 不是单纯的 FlashInfer：关闭 FlashInfer、layer 3 Torch math SDPA 仍漂移。
5. 不是单纯的 deferred ack：立即 drain ack 仍漂移且更慢。
6. 不能把 T10 结果混入 2080 Ti 结论：远端有多张 T10，必须 UUID pin。

## 相关 vLLM Flash-Next 进展（不要与本仓库混淆）

此前还在 `/home/max/workspace/VLLM-2080ti-qwen38-flash-pr` 做过同一模型族的 vLLM/NVFP4 实验。这些结果用于背景和交叉验证，不是本仓库的运行时：

- 已有 Qwen3.8-27B vLLM SM75 路线在双 2080 Ti TP2 上通过质量和 CUDA Graph 验证；迁移验证中，NVFP4 text-only noMTP 约 `1421/38.89 tok/s`，TurboQuant K8V4 MTP3 约 `1411.91/102.60 tok/s`（prefill/decode）。这说明硬件、驱动和一般 GDN/FlashQLA 路径可用。
- 针对 Flash-Next NVFP4 的 PLE CPU/SSD offload，TP4xPP2 和 TP2xPP4 都能完成 safetensors mmap/PLE registration，但随后在 KV-cache Piecewise TorchInductor 编译处失败：`AssertionError: auto_functionalized was not removed`。这不是 OOM，也没有可采信吞吐结果。
- TP2xPP3 的混合 T10/2080 Ti 映射，以及六张 T10 映射，都复现同一 TorchInductor 失败；因此不能把该失败解释为 2080 Ti 选错或 P2P 单独导致。
- vLLM 的 SM75 相关修复已在其分支测试中通过 targeted regression（`124 passed, 14 warnings`），但 Flash-Next 专用路径仍未达到可发布质量标准。

本仓库的 ExLlamaV3 TP2 质量问题与 vLLM 的 TorchInductor 启动失败是两条不同故障线：前者已加载并生成但输出漂移，后者多数实验在 graph compile 阶段就停止。新会话不要把两类日志或性能数字混成一个结论。

## 重要实验文件位置

结果在远端 `superserver:/home/max/results/`，本仓库只保留复现实验脚本。重点目录：

- `qwen38-flash-next-exl3-205-sm75-tp2-quality-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-tune-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-parity-rerun-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-layertrace-20260903`
- `qwen38-flash-next-exl3-205-sm75-tp2-gdn-parity-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-block-parity-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-module-profile-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-module-profile-by-module-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-nccl-tiny-20260904`
- `qwen38-flash-next-exl3-205-sm75-tp2-nccl-ring-simple-20260904`
- `qwen38-27b-exl3-35bpw-sm75-20260904`
- `qwen38-27b-exl3-35bpw-sm75-tp2-quality-20260904`

> 这些目录含有大体积 `.pt` trace，默认不要复制进仓库。若远端目录被清理，先从其日志和本仓库脚本恢复实验，不要把“没有日志”误判成“没有做过”。

## 下一会话执行顺序

### 0. 编译这个专用分支

模型和 Python venv 不放在 Git 仓库内。目标机上先确认 CUDA/PyTorch 版本，再在本仓库编译 SM75 扩展：

```bash
cd /home/max/workspace/ExLlamaV3-Turing
MAX_JOBS=40 TORCH_CUDA_ARCH_LIST=7.5 \
  /home/max/workspace/VLLM-2080ti-0.2.1-pre/.venv/bin/python -m pip install -e .
```

若目标环境内存不足，可把 `MAX_JOBS` 降到 24 或 16；编译产物会被 `.gitignore` 排除。不要把另一台机器编译出的 `.so` 当作确定性基线。

### 1. 先审计进程和 GPU

```bash
runuser -u max -- ssh superserver 'nvidia-smi --query-gpu=index,uuid,name,compute_cap,memory.total,memory.used --format=csv,noheader'
runuser -u max -- ssh superserver 'pgrep -af "python|exllama|probe|server" || true'
```

确认两张 2080 Ti UUID 空闲；远端当前曾残留一个使用 T10 的 vLLM 进程，除非明确要求，不要杀掉或把它的结果混进 TP2 实验。

### 2. 干净重跑独立 GDN probe

```bash
CUDA_VISIBLE_DEVICES=GPU-1c2b8831-227f-96d1-0c0f-92a189726072 \
  /home/max/workspace/VLLM-2080ti-0.2.1-pre/.venv/bin/python -u \
  tools/probe_gdn_kernel_determinism.py
```

保存完整 `GDN_DETERMINISM_JSON`。若 output/state hash 自身漂移，优先查 `cuda_recurrent_gated_delta_rule` 的 atomic/reduction/状态写入；若稳定，继续查 Flash-Next 组合路径。

### 3. 用真实 TP2 做模块边界重复 trace

使用 `tools/qwen38_module_trace.py`、`tools/compare_module_trace.py`、`tools/compare_gdn_subtrace.py`，每次使用新的结果目录和新的 `CUDA_VISIBLE_DEVICES` UUID。不要复用 cache、recurrent state 或旧 trace 文件。

重点阶段：`shared_input`、PLE embedding/delta、GDN output/state、hyper-connection mixing、QSA index、Q/K/V、`attn_o`、cache write。

### 4. 只补做真正缺口的组合

若独立 GDN 稳定，再按下表补做尚未有可信结论的组合：

- PLE off + QSA sparse off
- PLE on + QSA sparse off
- PLE off + QSA sparse on

变量：`EXL3_DISABLE_PLE=1`、`EXL3_QSA_DISABLE_SPARSE=1`、`EXL3_FLASHINFER=0`。每个组合至少 3 次 fresh-load/fresh-state greedy，记录 token hash、边界 hash、prefill 和 decode。

### 5. 质量修复的验收条件

- 同一 prompt、同一权重、同一 seed、fresh cache/state，连续至少 8 次 greedy 输出 token 完全一致。
- 对照 27B 单卡/PP2 仍稳定。
- TP2 4K/128 至少记录 prefill tok/s 和 decode tok/s；质量修复不能把 decode 从约 20 tok/s 再无说明地降到约 14 tok/s。
- 无 OOM、无 silent NaN、无残留 worker；实验结束后检查 `pgrep` 和 GPU memory。

## Git 约定

当前仓库应保持“源码和小型诊断工具入 Git，模型/编译物/结果出 Git”。建议每个可验证修复一个 commit，commit message 写清触及的 kernel/路径和验证 artifact。不要把在线 PR 或生产服务改动混入此仓库。
