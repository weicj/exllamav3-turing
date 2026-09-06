"""Measure the NCCL all-reduce payloads used by TP decode.

Run under ``torch.distributed.run`` with exactly the two TP GPUs visible.  The
benchmark deliberately uses a dependency chain of reductions, matching the
row-parallel TP path where the next layer consumes the previous reduction.
"""

import json
import os
import time

import torch
import torch.distributed as dist


NUMEL = int(os.environ.get("NCCL_BENCH_NUMEL", "2560"))
DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}[os.environ.get("NCCL_BENCH_DTYPE", "float32")]
WARMUP = int(os.environ.get("NCCL_BENCH_WARMUP", "200"))
ITERATIONS = int(os.environ.get("NCCL_BENCH_ITERATIONS", "5000"))


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    rank = dist.get_rank()
    x = torch.ones(NUMEL, dtype=DTYPE, device=local_rank)
    for _ in range(WARMUP):
        dist.all_reduce(x)
    torch.cuda.synchronize(local_rank)
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(ITERATIONS):
        dist.all_reduce(x)
    end_event.record()
    torch.cuda.synchronize(local_rank)
    wall_seconds = time.perf_counter() - wall_start
    device_ms = start_event.elapsed_time(end_event)

    report = {
        "rank": rank,
        "numel": NUMEL,
        "dtype": str(DTYPE),
        "bytes": NUMEL * torch.empty((), dtype=DTYPE).element_size(),
        "iterations": ITERATIONS,
        "device_us_per_all_reduce": device_ms * 1000 / ITERATIONS,
        "wall_us_per_all_reduce": wall_seconds * 1_000_000 / ITERATIONS,
        "nccl_algo": os.environ.get("NCCL_ALGO", "<default>"),
        "nccl_proto": os.environ.get("NCCL_PROTO", "<default>"),
        "nccl_p2p_level": os.environ.get("NCCL_P2P_LEVEL", "<default>"),
    }
    # Do not use gather_object for reporting: forced NCCL algorithm/protocol choices
    # apply to that implementation detail too, while some valid all-reduce choices do
    # not support the int8 all-gather gather_object uses internally.
    print("NCCL_ALLREDUCE_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
