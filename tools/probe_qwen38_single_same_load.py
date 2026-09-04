"""Repeat the Flash-Next model on one GPU with a fresh cache per request."""

import hashlib
import json
import os

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = os.environ.get(
    "BENCH_MODEL_DIR",
    "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4",
)
PROMPT_TOKENS = int(os.environ.get("BENCH_PROMPT_TOKENS", "4096"))
OUTPUT_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "8"))
REPEATS = int(os.environ.get("PROBE_REPEATS", "3"))
CACHE_TOKENS = int(os.environ.get("BENCH_CACHE_TOKENS", "4352"))
CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))


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
        raise RuntimeError("job did not emit expected tokens")
    return {
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            b"".join(token.to_bytes(4, "little", signed=False) for token in token_ids)
        ).hexdigest(),
        "prefill_tok_s": prompt_tokens / finished["time_prefill"],
        "decode_tok_s": finished["new_tokens"] / finished["time_generate"],
    }


def clear_request_cache(generator):
    assert not generator.active_jobs
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()
    for page in generator.pagetable.all_pages:
        assert page.ref_count == 0
        page.clear()


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)
    try:
        model.load(use_per_device=[22.0], max_chunk_size=LOAD_CHUNK_TOKENS, max_batch_size=1, verbose=True)
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=CHUNK_TOKENS,
        )
        run_job(generator, 128, 8, 1)
        clear_request_cache(generator)
        jobs = []
        for _ in range(REPEATS):
            jobs.append(run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2))
            clear_request_cache(generator)
        print("SINGLE_SAME_LOAD_JSON=" + json.dumps({
            "jobs": jobs,
            "stable": len({job["token_ids_sha256"] for job in jobs}) == 1,
        }, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
