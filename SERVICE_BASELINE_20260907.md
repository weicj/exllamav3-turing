# Qwen3.8 Flash-Next 144K TP2 Service Baseline

Date: 2026-09-07

## Status

This is the operational baseline for the OpenAI-compatible ExLlamaV3 service on
`superserver` (`192.168.1.31`). It is limited to one request at a time and uses
the quality-qualified NCCL TP2 sparse-QSA path. It is not a claim that the
project performance target has been met.

## Pinned Runtime

- Source entry: `tools/serve_qwen38_tp2_144k.py`
- Launcher: `tools/launch_qwen38_tp2_144k_8037.sh`
- Model: `/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4`
- Python: `/home/max/workspace/VLLM-2080ti-0.2.1-pre/.venv/bin/python`
- Parallelism: ExLlamaV3 NCCL TP2
- GPUs, in logical CUDA order:
  - `GPU-1c2b8831-227f-96d1-0c0f-92a189726072`
  - `GPU-4da87347-023e-7014-a837-6b1fb649a42e`
- Context/cache: `147456` / `147712` tokens
- Runtime environment:

```text
NCCL_ALGO=Ring
NCCL_PROTO=Simple
EXL3_NGRAM_STREAM=1
EXL3_FLASHINFER_WORKSPACE_MB=128
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`EXL3_QSA_PREFILL_DENSE` and `EXL3_QSA_DISABLE_SPARSE` are intentionally unset.
The delivered path uses sparse QSA, not the faster but output-drifting dense
prefill experiment.

## Service Contract

- Base URL: `http://192.168.1.31:8037/v1`
- Endpoint: `POST /v1/chat/completions`
- Models endpoint: `GET /v1/models`
- Health endpoint: `GET /health`
- Thinking accepts both `enable_thinking` and
  `chat_template_kwargs.enable_thinking`.
- SSE separates thought text into `delta.reasoning_content` and final text into
  `delta.content`.
- Non-streaming responses and the terminal stream event include `timings` with
  generator-measured prefill and decode rates. These are model timings, not
  browser end-to-end latency.
- A dead TP child makes `/health` return HTTP 503. A disconnected stream cancels
  its queued job at the next generator boundary and releases the single-flight
  lock.

## Verification Evidence

The service was launched at `2026-09-07T04:10:48+00:00` as PID `2592392` in its
own session (`PPID=1`). The ready snapshot showed a live TP child, health
`status=ok`, and model memory of `21829 MiB` / `21569 MiB` on the pinned GPUs.
The API reported `NCCL TP2`, context `147456`, and cache `147712`.

The restart performance replay is stored outside the repository:

```text
/home/max/results/tp2_restart_bench_20260906_02/bench.log
```

It ran `tools/bench_qwen38_tp2_4k128.py` with a fresh 4K random prompt, greedy
128-token decode, `BENCH_REPEAT_COUNT=2`, and the pinned environment above:

| Sample | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 1 | 686.55 | 24.73 |
| 2 | 703.65 | 24.92 |
| Mean | 695.10 | 24.82 |

The benchmark exited through its unload path and both GPUs returned to `8 MiB`,
so it did not leave a hidden competing model process.

This is a performance replay only: it did not capture token IDs. Quality remains
anchored to `QUALITY_BASELINE.md`: sparse-QSA 4K/128 hash
`c6e8218f6c53cdecbdfaedc4a124ed50f36e5447e62241c6ba77452184667e44`, prior
8/8 repeatability, and the 144K `cobalt blue` retrieval validation.

## Version Audit

- Parent commit: `a3bd6c6adff7f668817f038dddacd8bc5ec460c4`
- Pre-solidification tracked worktree diff SHA-256:
  `1ce16b168404da643ef4d169ae16b640b0a4fc116464d508088b3eb6ba47e26b`
- Target service source SHA-256 before local synchronization:
  `3bab698d7d58c91d7520a7bb504771b252005bc457424cc7767086a02b86e427`

The repository already contained unrelated and unstaged kernel/experiment work.
This baseline records it rather than claiming a clean-tree release. Any future
kernel, transport, QSA, cache, or service change must re-run the quality and
performance checks above before replacing this record.

## Operational Commands

On `superserver`:

```bash
cd /home/max/workspace/ExLlamaV3-Turing
tools/launch_qwen38_tp2_144k_8037.sh
curl -fsS http://127.0.0.1:8037/health
```

The launcher refuses to replace a listener or an identified live service. Stop
only the PID recorded in `serve_qwen38_tp2_144k-8037.pid`, then confirm both
pinned GPUs are released before starting another instance.
