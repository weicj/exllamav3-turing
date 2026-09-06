"""Strict Qwen3.8 Flash-Next EXL3 TP2 benchmark for the dual RTX 2080 Ti lane."""

import hashlib
import json
import os
import time

import torch

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer
from exllamav3.generator.sampler import GreedySampler


MODEL_DIR = os.environ.get(
    "BENCH_MODEL_DIR",
    "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4",
)
PROMPT_TOKENS = 4096
OUTPUT_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "128"))
CACHE_TOKENS = 4608
CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]
TP_BACKEND = os.environ.get("BENCH_TP_BACKEND", "nccl")
LOGITS_PATH = os.environ.get("BENCH_LOGITS_PATH")
CAPTURE_TOKENS = os.environ.get("BENCH_CAPTURE_TOKENS") == "1"
CAPTURE_TOKEN_IDS = os.environ.get("BENCH_CAPTURE_TOKEN_IDS") == "1"
REPEAT_COUNT = int(os.environ.get("BENCH_REPEAT_COUNT", "1"))


def runtime_toggles():
    """Record every performance knob that can change the measured execution path."""
    names = (
        "EXL3_BC_ATTN",
        "EXL3_FLASHINFER",
        "EXL3_FLASHINFER_WORKSPACE_MB",
        "EXL3_GEMV",
        "EXL3_INT8_GEMV",
        "EXL3_MGEMM_K_THRESHOLD",
        "EXL3_MGEMM_N_THRESHOLD",
        "EXL3_QSA_DISABLE_SPARSE",
        "EXL3_QSA_PREFILL_DENSE",
        "EXL3_QSA_BC_FORCE_DENSE",
    )
    return {name: os.environ.get(name, "<default>") for name in names}


def run_job(generator, prompt_tokens, output_tokens, seed):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1000, 20000, (1, prompt_tokens), generator=rng, dtype=torch.long)
    generator.enqueue(Job(
        input_ids=ids,
        max_new_tokens=output_tokens + 1,
        sampler=GreedySampler(),
        return_logits=LOGITS_PATH is not None,
    ))
    finished = None
    captured_tokens = []
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if CAPTURE_TOKENS:
                token_ids = result.get("token_ids")
                if token_ids is None:
                    token_ids = result.get("held", {}).get("token_ids")
                if token_ids is not None:
                    captured_tokens.extend(token_ids.reshape(-1).cpu().tolist())
            if result.get("eos"):
                finished = result
    if finished is None:
        raise RuntimeError("generation completed without an EOS result")
    if CAPTURE_TOKENS:
        finished["_bench_token_ids"] = captured_tokens
    return finished


def clear_request_cache(generator):
    """Make each measured request start from fresh KV and recurrent state."""
    assert not generator.active_jobs
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()
    for page in generator.pagetable.all_pages:
        assert page.ref_count == 0
        page.clear()


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
        started = time.monotonic()
        model.load(
            tensor_p=True,
            tp_backend=TP_BACKEND,
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

        if REPEAT_COUNT <= 0:
            raise ValueError("BENCH_REPEAT_COUNT must be positive")

        warmup = run_job(generator, 128, 8, 1)
        clear_request_cache(generator)
        results = []
        for repeat in range(REPEAT_COUNT):
            results.append(run_job(generator, PROMPT_TOKENS, OUTPUT_TOKENS, 2 + repeat))
            clear_request_cache(generator)
        result = results[-1]
        measured_prompt = PROMPT_TOKENS
        generated = result["new_tokens"]
        report = {
            "model": MODEL_DIR,
            "parallelism": "TP2",
            "tp_backend": TP_BACKEND,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpus": visible_gpus,
            "ngram_stream_from_disk": config.infer_params.ngram_stream_from_disk,
            "prompt_tokens_requested": PROMPT_TOKENS,
            "prompt_tokens_measured": measured_prompt,
            "output_tokens_requested": OUTPUT_TOKENS,
            "output_tokens_measured": generated,
            "repeat_count": REPEAT_COUNT,
            "cache_tokens": CACHE_TOKENS,
            "prefill_chunk_tokens": CHUNK_TOKENS,
            "loader_chunk_tokens": LOAD_CHUNK_TOKENS,
            "use_per_device_gib": USE_PER_DEVICE,
            "load_seconds": load_seconds,
            "runtime_toggles": runtime_toggles(),
            "runs": [
                {
                    "prefill_seconds": run["time_prefill"],
                    "decode_seconds": run["time_generate"],
                    "prefill_tok_s": measured_prompt / run["time_prefill"],
                    "decode_tok_s": run["new_tokens"] / run["time_generate"],
                    "output_tokens": run["new_tokens"],
                }
                for run in results
            ],
            # Keep the historical top-level fields as the last sample for existing parsers.
            "prefill_seconds": result["time_prefill"],
            "decode_seconds": result["time_generate"],
            "prefill_tok_s": measured_prompt / result["time_prefill"],
            "decode_tok_s": generated / result["time_generate"],
            "warmup_output_tokens": warmup["new_tokens"],
            "tp_plan": model.plan,
        }
        if CAPTURE_TOKENS:
            token_ids = result.pop("_bench_token_ids", [])
            if not token_ids:
                raise RuntimeError("BENCH_CAPTURE_TOKENS requested but no generated token IDs were returned")
            report.update({
                "first_token": token_ids[0],
                "last_token": token_ids[-1],
                "captured_token_count": len(token_ids),
                "token_ids_sha256": hashlib.sha256(
                    b"".join(token.to_bytes(4, "little", signed = False) for token in token_ids)
                ).hexdigest(),
            })
            if CAPTURE_TOKEN_IDS:
                report["token_ids"] = token_ids
        if LOGITS_PATH is not None:
            logits = result.get("logits")
            if logits is None or logits.shape[1] == 0:
                raise RuntimeError("requested logits were not returned")
            token_ids = result.get("token_ids")
            torch.save(
                {
                    "input_seed": 2,
                    "first_logits": logits[0, 0].float().cpu(),
                    "token_ids": token_ids.cpu() if token_ids is not None else None,
                },
                LOGITS_PATH,
            )
            report["first_logits_path"] = LOGITS_PATH
        print("BENCHMARK_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    finally:
        if model.loaded_tp:
            model.unload()
        elif model.mp_children:
            model.destroy_tp_context()


if __name__ == "__main__":
    main()
