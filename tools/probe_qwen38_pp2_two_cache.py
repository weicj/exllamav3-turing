"""Run identical PP2 requests through separate caches in one loaded model."""

import hashlib
import json
import os

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 4096
OUTPUT_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "1"))
CACHE_TOKENS = int(os.environ.get("BENCH_CACHE_TOKENS", "4608"))
NUM_CACHES = int(os.environ.get("BENCH_NUM_CACHES", "2"))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(input_ids=ids, max_new_tokens=output_tokens + 1, sampler=GreedySampler()))
    token_ids = []
    finished = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            event_token_ids = result.get("token_ids")
            if event_token_ids is not None:
                token_ids.extend(event_token_ids.reshape(-1).cpu().tolist())
            if result.get("eos"):
                finished = result
    if finished is None or len(token_ids) != finished["new_tokens"]:
        raise RuntimeError("job did not emit the expected token count")
    return {
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            b"".join(token.to_bytes(4, "little", signed=False) for token in token_ids)
        ).hexdigest(),
        "prefill_seconds": finished["time_prefill"],
        "decode_seconds": finished["time_generate"],
        "prefill_tok_s": prompt_tokens / finished["time_prefill"],
        "decode_tok_s": finished["new_tokens"] / finished["time_generate"],
    }


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for this probe")
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    # Two one-slot caches avoid prompt-page/recurrent-state reuse without materially changing
    # the per-request resident state footprint.
    caches = [Cache(model, max_num_tokens=CACHE_TOKENS, max_batch_size=1) for _ in range(NUM_CACHES)]
    try:
        def report_load_progress(index, total):
            print(f"LOAD_PROGRESS={index}/{total}", flush=True)

        model.load(
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=256,
            max_batch_size=1,
            verbose=True,
            callback=report_load_progress,
        )
        tokenizer = Tokenizer.from_config(config)
        generators = [Generator(model=model, cache=cache, tokenizer=tokenizer,
                                max_batch_size=1, max_chunk_size=512) for cache in caches]
        jobs = [run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2) for generator in generators]
        print("PP2_TWO_CACHE_JSON=" + json.dumps({
            "parallelism": "PP2 layer split",
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "cache_tokens": CACHE_TOKENS,
            "num_caches": NUM_CACHES,
            "jobs": jobs,
            "stable": len(jobs) < 2 or jobs[0]["token_ids_sha256"] == jobs[1]["token_ids_sha256"],
        }, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
