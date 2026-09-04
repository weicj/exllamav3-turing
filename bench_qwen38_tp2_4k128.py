import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 4096
OUTPUT_TOKENS = 128
CACHE_TOKENS = 4608
CHUNK_TOKENS = 512


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(input_ids=ids, max_new_tokens=output_tokens, sampler=GreedySampler()))
    finished = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if result.get("eos"):
                finished = result
    if finished is None:
        raise RuntimeError("generation completed without an EOS result")
    return finished


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for this disk-streaming benchmark")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)

    load_start = time.monotonic()
    model.load(
        tensor_p=True,
        reserve_per_device=[1.5, 1.5],
        max_chunk_size=CHUNK_TOKENS,
        max_batch_size=1,
        tp_backend="nccl",
        verbose=True,
    )
    load_s = time.monotonic() - load_start
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(
        model=model,
        cache=cache,
        tokenizer=tokenizer,
        max_batch_size=1,
        max_chunk_size=CHUNK_TOKENS,
    )

    # Kernel selection, NCCL setup, and PLE page-cache warmup are excluded from the reported lane.
    warmup = run_job(generator, 128, 8, 1)
    result = run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2)

    generated = result["new_tokens"]
    report = {
        "model": MODEL_DIR,
        "tp": 2,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "ngram_stream_from_disk": config.infer_params.ngram_stream_from_disk,
        "prompt_tokens_requested": PROMPT_TOKENS,
        "prompt_tokens_measured": result["prompt_tokens"] - result["cached_tokens"],
        "output_tokens_requested": OUTPUT_TOKENS,
        "output_tokens_measured": generated,
        "cache_tokens": CACHE_TOKENS,
        "prefill_chunk_tokens": CHUNK_TOKENS,
        "load_seconds": load_s,
        "prefill_seconds": result["time_prefill"],
        "decode_seconds": result["time_generate"],
        "prefill_tok_s": (result["prompt_tokens"] - result["cached_tokens"]) / result["time_prefill"],
        "decode_tok_s": generated / result["time_generate"],
        "warmup_output_tokens": warmup["new_tokens"],
    }
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    model.unload()


if __name__ == "__main__":
    main()
