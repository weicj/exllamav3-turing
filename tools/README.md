# Turing experiment tools

这些脚本来自 2026-09-04 的 Qwen3.8 Flash-Next TP2 质量定位实验。

- `bench_qwen38_tp2_4k128.py`: 两张 2080 Ti、EXL3 TP2、4K prompt/128 decode 基准。
- `bench_qwen38_pp2_capture.py`: PP2 对照基准。
- `probe_gdn_kernel_determinism.py`: 独立 GDN CUDA kernel 重复性 probe。
- `qwen38_module_trace.py`: PP/TP 模块边界 trace。
- `compare_module_trace.py`, `compare_gdn_subtrace.py`, `compare_subtrace.py`: trace 比较。
- `probe_qwen38_*`: fresh-load、PP2、TP2、ngram 和 cache 相关重复性 probe。
- `nccl_tiny_allreduce.py`: 仅用于 collective 延迟和基本通信可用性，不是模型质量验证。
- `serve_qwen38_tp2_144k.py`: 单飞行 OpenAI-compatible 144K NCCL TP2 服务，支持流式思考字段和模型侧计时。
- `launch_qwen38_tp2_144k_8037.sh`: 在目标机固定两张 2080 Ti UUID 启动上述服务。

运行前必须 UUID pin 两张 RTX 2080 Ti，并将结果写到仓库外的 `/home/max/results/<run-id>`。
详见仓库根目录的 `HANDOFF.md`、`QUALITY_BASELINE.md` 和 `SERVICE_BASELINE_20260907.md`。
