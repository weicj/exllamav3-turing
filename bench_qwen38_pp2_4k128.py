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
CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))
USE_PER_DEVICE = [float(x) for x in os.environ.get("BENCH_USE_PER_DEVICE", "19,22").split(",")]


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
    visible_gpus = [
        {
            "logical_index": i,
            "name": torch.cuda.get_device_properties(i).name,
            "total_mib": torch.cuda.get_device_properties(i).total_memory // 2**20,
            "uuid": str(torch.cuda.get_device_properties(i).uuid),
        }
        for i in range(torch.cuda.device_count())
    ]
    print("VISIBLE_GPUS=" + json.dumps(visible_gpus, sort_keys=True), flush=True)

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)

    load_start = time.monotonic()
    model.load(
        use_per_device=USE_PER_DEVICE,
        max_chunk_size=LOAD_CHUNK_TOKENS,
        max_batch_size=1,
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

    # Warmup is intentionally excluded from the measured 4K/128 lane.
    # Job internally reserves one final token for its completion event.
    warmup = run_job(generator, 128, 9, 1)
    result = run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS + 1, 2)

    measured_prompt = result["prompt_tokens"] - result["cached_tokens"]
    report = {
        "model": MODEL_DIR,
        "parallelism": "PP2 layer split",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_gpus": visible_gpus,
        "ngram_stream_from_disk": config.infer_params.ngram_stream_from_disk,
        "prompt_tokens_requested": PROMPT_TOKENS,
        "prompt_tokens_measured": measured_prompt,
        "output_tokens_requested": OUTPUT_TOKENS,
        "output_tokens_measured": result["new_tokens"],
        "cache_tokens": CACHE_TOKENS,
        "prefill_chunk_tokens": CHUNK_TOKENS,
        "loader_chunk_tokens": LOAD_CHUNK_TOKENS,
        "use_per_device_gib": USE_PER_DEVICE,
        "load_seconds": load_s,
        "prefill_seconds": result["time_prefill"],
        "decode_seconds": result["time_generate"],
        "prefill_tok_s": measured_prompt / result["time_prefill"],
        "decode_tok_s": result["new_tokens"] / result["time_generate"],
        "warmup_output_tokens": warmup["new_tokens"],
    }
    print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    model.unload()


if __name__ == "__main__":
    main()
