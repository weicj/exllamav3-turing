"""Measure the tiny NCCL all-reduces used by the Qwen3.8 TP2 decode path."""

import os
import socket
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


ELEMENTS = int(os.environ.get("BENCH_ELEMENTS", "2560"))
ITERATIONS = int(os.environ.get("BENCH_ITERATIONS", "1000"))
WARMUP = int(os.environ.get("BENCH_WARMUP", "100"))
DTYPE = getattr(torch, os.environ.get("BENCH_DTYPE", "float16"))


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    tensor = torch.ones(ELEMENTS, dtype=DTYPE, device=rank)
    for _ in range(WARMUP):
        dist.all_reduce(tensor)
    torch.cuda.synchronize(rank)
    dist.barrier()

    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    wall_start = time.perf_counter()
    for _ in range(ITERATIONS):
        dist.all_reduce(tensor)
    stop_event.record()
    torch.cuda.synchronize(rank)
    wall_seconds = time.perf_counter() - wall_start
    gpu_ms = start_event.elapsed_time(stop_event)
    if rank == 0:
        payload_bytes = ELEMENTS * tensor.element_size()
        print(
            "NCCL_TINY_ALLREDUCE="
            f"elements={ELEMENTS} dtype={DTYPE} bytes={payload_bytes} iterations={ITERATIONS} "
            f"gpu_us_per_call={gpu_ms * 1000 / ITERATIONS:.3f} "
            f"wall_us_per_call={wall_seconds * 1_000_000 / ITERATIONS:.3f}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(free_port(),), nprocs=2, join=True)
