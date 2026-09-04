"""Trace the first TP2 decode pass after a deterministic short prefill."""

import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 32
CACHE_TOKENS = 4608
OUTPUT_TOKENS = int(os.environ.get("EXL3_TRACE_OUTPUT_TOKENS", "2"))


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    try:
        started = time.monotonic()
        model.load(
            tensor_p=True,
            tp_backend="nccl",
            use_per_device=[22.0, 22.0],
            max_chunk_size=256,
            max_batch_size=1,
            verbose=True,
        )
        input_ids = torch.randint(
            1000,
            20000,
            (1, PROMPT_TOKENS),
            generator=torch.Generator(device="cpu").manual_seed(2),
            dtype=torch.long,
        )
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=256,
        )
        generator.enqueue(Job(
            input_ids=input_ids,
            max_new_tokens=OUTPUT_TOKENS,
            sampler=GreedySampler(),
        ))
        result = None
        while generator.num_remaining_jobs():
            for event in generator.iterate():
                if event.get("eos"):
                    result = event
        if result is None:
            raise RuntimeError("generation completed without an EOS result")
        print("DECODE_TRACE_JSON=" + json.dumps({
            "elapsed_s": time.monotonic() - started,
            "output_tokens": result["new_tokens"],
        }, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
