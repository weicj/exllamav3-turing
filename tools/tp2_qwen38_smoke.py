"""Isolated Qwen3.8 Flash-Next EXL3 TP2 load and 4K sparse-attention smoke test."""

import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = 4096
CACHE_TOKENS = 4608
CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))
USE_PER_DEVICE = [22.0, 22.0]


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(input_ids=ids, max_new_tokens=output_tokens + 1, sampler=GreedySampler()))
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
        started = time.monotonic()
        model.load(
            tensor_p=True,
            tp_backend="nccl",
            use_per_device=USE_PER_DEVICE,
            max_chunk_size=LOAD_CHUNK_TOKENS,
            max_batch_size=1,
            verbose=True,
        )
        load_seconds = time.monotonic() - started
        generator = Generator(
            model=model,
            cache=cache,
            tokenizer=Tokenizer.from_config(config),
            max_batch_size=1,
            max_chunk_size=CHUNK_TOKENS,
        )
        run_job(generator, 128, 8, 1)
        result = run_job(generator, PROMPT_TOKENS, 1, 2)
        # TP Job completion events do not currently mirror the layer-split prompt/cache
        # counters. This smoke has one fresh, uncached fixed-length request.
        measured_prompt = PROMPT_TOKENS
        visible_gpus = [
            {
                "logical_index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "uuid": str(torch.cuda.get_device_properties(index).uuid),
            }
            for index in range(torch.cuda.device_count())
        ]
        report = {
            "model": MODEL_DIR,
            "parallelism": "TP2",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpus": visible_gpus,
            "ngram_stream_from_disk": config.infer_params.ngram_stream_from_disk,
            "load_seconds": load_seconds,
            "tp_plan": model.plan,
            "prefill_seconds": result["time_prefill"],
            "prefill_tok_s": measured_prompt / result["time_prefill"],
            "decode_seconds": result["time_generate"],
            "decode_tok_s": result["new_tokens"] / result["time_generate"],
            "prompt_tokens": measured_prompt,
            "output_tokens": result["new_tokens"],
        }
        print("TP_SMOKE_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if model.loaded_tp:
            model.unload()
        elif model.mp_children:
            model.destroy_tp_context()


if __name__ == "__main__":
    main()
