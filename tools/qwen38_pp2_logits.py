"""Save PP2 reference logits for a fixed Qwen3.8 Flash-Next 4K prompt."""

import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 4096
CACHE_TOKENS = 4608
CHUNK_TOKENS = 512
LOAD_CHUNK_TOKENS = 256
USE_PER_DEVICE = [19.0, 22.0]
LOGITS_PATH = os.environ["BENCH_LOGITS_PATH"]


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(
        input_ids=ids,
        max_new_tokens=output_tokens + 1,
        sampler=GreedySampler(),
        return_logits=True,
    ))
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
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected exactly two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    try:
        load_start = time.monotonic()
        model.load(
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=LOAD_CHUNK_TOKENS,
            max_batch_size=1,
            verbose=True,
        )
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=CHUNK_TOKENS,
        )
        run_job(generator, 128, 8, 1)
        result = run_job(generator, PROMPT_TOKENS, 1, 2)
        logits = result.get("logits")
        if logits is None or logits.shape[1] == 0:
            raise RuntimeError("reference logits were not returned")
        token_ids = result.get("token_ids")
        torch.save(
            {
                "input_seed": 2,
                "first_logits": logits[0, 0].float().cpu(),
                "token_ids": token_ids.cpu() if token_ids is not None else None,
            },
            LOGITS_PATH,
        )
        print("PP2_LOGITS_JSON=" + json.dumps({
            "load_seconds": time.monotonic() - load_start,
            "logits_path": LOGITS_PATH,
            "first_token": int(token_ids[0, 0]) if token_ids is not None else None,
        }, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
