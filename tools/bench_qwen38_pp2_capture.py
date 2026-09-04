"""Capture normal greedy output for the Qwen3.8 Flash-Next PP2 reference lane."""

import hashlib
import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 4096
OUTPUT_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "128"))
CACHE_TOKENS = 4608
CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(
        input_ids=ids,
        max_new_tokens=output_tokens + 1,
        sampler=GreedySampler(),
    ))
    finished = None
    token_ids = []
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            event_token_ids = result.get("token_ids")
            if event_token_ids is not None:
                token_ids.extend(event_token_ids.reshape(-1).cpu().tolist())
            if result.get("eos"):
                finished = result
    if finished is None:
        raise RuntimeError("generation completed without an EOS result")
    if len(token_ids) != finished["new_tokens"]:
        raise RuntimeError(
            f"captured {len(token_ids)} tokens, expected {finished['new_tokens']}"
        )
    return finished, token_ids


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for this disk-streaming benchmark")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")

    visible_gpus = [
        {
            "logical_index": index,
            "name": torch.cuda.get_device_properties(index).name,
            "total_mib": torch.cuda.get_device_properties(index).total_memory // 2**20,
            "uuid": str(torch.cuda.get_device_properties(index).uuid),
        }
        for index in range(torch.cuda.device_count())
    ]

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    try:
        load_started = time.monotonic()
        model.load(
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=LOAD_CHUNK_TOKENS,
            max_batch_size=1,
            verbose=True,
        )
        load_seconds = time.monotonic() - load_started
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=CHUNK_TOKENS,
        )

        warmup, _ = run_job(generator, 128, 8, 1)
        result, token_ids = run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2)
        generated = result["new_tokens"]
        report = {
            "model": MODEL_DIR,
            "parallelism": "PP2 layer split",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpus": visible_gpus,
            "ngram_stream_from_disk": config.infer_params.ngram_stream_from_disk,
            "prompt_tokens_requested": PROMPT_TOKENS,
            "prompt_tokens_measured": PROMPT_TOKENS,
            "output_tokens_requested": OUTPUT_TOKENS,
            "output_tokens_measured": generated,
            "cache_tokens": CACHE_TOKENS,
            "prefill_chunk_tokens": CHUNK_TOKENS,
            "loader_chunk_tokens": LOAD_CHUNK_TOKENS,
            "use_per_device_gib": USE_PER_DEVICE,
            "load_seconds": load_seconds,
            "prefill_seconds": result["time_prefill"],
            "decode_seconds": result["time_generate"],
            "prefill_tok_s": PROMPT_TOKENS / result["time_prefill"],
            "decode_tok_s": generated / result["time_generate"],
            "warmup_output_tokens": warmup["new_tokens"],
            "first_token": token_ids[0],
            "last_token": token_ids[-1],
            "captured_token_count": len(token_ids),
            "token_ids": token_ids,
            "token_ids_sha256": hashlib.sha256(
                b"".join(token.to_bytes(4, "little", signed=False) for token in token_ids)
            ).hexdigest(),
        }
        print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
