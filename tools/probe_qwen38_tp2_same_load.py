"""Run identical fresh-cache Flash-Next TP2 jobs in one model load.

This is intentionally distinct from the PP2 probe: tensor_p=True distributes every
supported module shard across the two workers and exercises the TP backend.
"""

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
REPEATS = int(os.environ.get("PROBE_REPEATS", "8"))
CACHE_TOKENS = int(os.environ.get("BENCH_CACHE_TOKENS", "4352"))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]
TP_BACKEND = os.environ.get("BENCH_TP_BACKEND", "nccl")


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(
        input_ids=ids,
        max_new_tokens=output_tokens + 1,
        sampler=GreedySampler(),
    ))
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


def clear_request_cache(generator):
    """Force the next request to use a new KV and recurrent state."""
    assert not generator.active_jobs
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()
    for page in generator.pagetable.all_pages:
        assert page.ref_count == 0
        page.clear()


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for this probe")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    # Qwen4Exp TP import/export is implemented in the lab tree. Keep this experimental
    # capability opt-in until its quality checks are complete.
    model.caps["supports_tp"] = True
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_batch_size=1)
    try:
        model.load(
            tensor_p=True,
            tp_backend=TP_BACKEND,
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=256,
            max_batch_size=1,
            verbose=True,
        )
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=512,
        )
        warmup = run_job(generator, 128, 8, 1)
        clear_request_cache(generator)
        jobs = []
        for _ in range(REPEATS):
            jobs.append(run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2))
            clear_request_cache(generator)
        print("TP2_SAME_LOAD_JSON=" + json.dumps({
            "parallelism": "TP2",
            "tp_backend": TP_BACKEND,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": OUTPUT_TOKENS,
            "repeat_count": REPEATS,
            "tp_plan": model.plan,
            "warmup": warmup,
            "jobs": jobs,
            "stable": len({job["token_ids_sha256"] for job in jobs}) == 1,
        }, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
