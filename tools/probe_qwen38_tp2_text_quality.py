"""Run a readable long-context TP2 quality probe for Qwen3.8 Flash-Next.

The numeric determinism probe intentionally uses random token IDs, which is good for
detecting drift but cannot show whether a stable sequence is coherent. This companion
probe pads a simple retrieval task to a fixed token count and records greedy text,
token IDs, and timings for fresh cache/recurrent-state runs.
"""

from __future__ import annotations

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
OUTPUT_TOKENS = int(os.environ.get("BENCH_OUTPUT_TOKENS", "128"))
REPEATS = int(os.environ.get("PROBE_REPEATS", "2"))
CACHE_TOKENS = int(os.environ.get(
    "BENCH_CACHE_TOKENS",
    str(((PROMPT_TOKENS + OUTPUT_TOKENS + 511) // 256) * 256),
))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]
LOAD_CHUNK_TOKENS = int(os.environ.get("BENCH_LOAD_CHUNK_TOKENS", "256"))
PREFILL_CHUNK_TOKENS = int(os.environ.get("BENCH_CHUNK_TOKENS", "512"))


def make_prompt(tokenizer: Tokenizer) -> torch.Tensor:
    """Build a long retrieval task without altering the final question's position."""
    note = (
        "Reference note: the marker hidden in the archive is cobalt blue. "
        "The catalog repeats this fact so it can be checked after a long context.\n"
    )
    question = (
        "Use the reference notes above. In one short sentence, what color is the "
        "marker hidden in the archive?"
    )
    notes = ""
    longest = None
    while True:
        ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": notes + "\n" + question}],
            add_generation_prompt = True,
            enable_thinking = False,
        )
        if ids.shape[1] >= PROMPT_TOKENS:
            if longest is None:
                raise ValueError("BENCH_PROMPT_TOKENS is shorter than the chat prompt")
            return longest
        longest = ids
        notes += note


def clear_request_cache(generator: Generator) -> None:
    assert not generator.active_jobs
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()
    for page in generator.pagetable.all_pages:
        assert page.ref_count == 0
        page.clear()


def run_job(
    generator: Generator,
    input_ids: torch.Tensor,
    stop_conditions: list[int],
) -> dict:
    generator.enqueue(Job(
        input_ids = input_ids,
        max_new_tokens = OUTPUT_TOKENS + 1,
        sampler = GreedySampler(),
        # A raw Job deliberately has no implicit EOS policy. Use the tokenizer's
        # complete HF list here: for Flash-Next that includes <|im_end|>, which is
        # supplied by generation_config.json rather than the text sub-config.
        stop_conditions = stop_conditions,
    ))
    text = ""
    token_ids: list[int] = []
    finished = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            text += result.get("text", "")
            ids = result.get("token_ids")
            if ids is not None:
                token_ids.extend(ids.reshape(-1).cpu().tolist())
            if result.get("eos"):
                finished = result
    if finished is None:
        raise RuntimeError("generation completed without an EOS result")
    eos_reason = finished.get("eos_reason")
    if eos_reason == "stop_token":
        # The triggering stop token remains held by Job and is intentionally not
        # emitted to the text/token stream.
        if len(token_ids) != finished["new_tokens"] - 1:
            raise RuntimeError("job did not emit the expected pre-stop token IDs")
    elif len(token_ids) != finished["new_tokens"]:
        raise RuntimeError("job did not emit the expected number of token IDs")
    return {
        "text": text,
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(
            b"".join(token.to_bytes(4, "little", signed = False) for token in token_ids)
        ).hexdigest(),
        "eos_reason": eos_reason,
        "eos_triggering_token_id": finished.get("eos_triggering_token_id"),
        "generated_tokens": finished["new_tokens"],
        "prefill_tok_s": input_ids.shape[1] / finished["time_prefill"],
        "decode_tok_s": finished["new_tokens"] / finished["time_generate"],
    }


def main() -> None:
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")
    if REPEATS < 1:
        raise ValueError("PROBE_REPEATS must be positive")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_batch_size = 1)
    try:
        model.load(
            tensor_p = True,
            tp_backend = "nccl",
            use_per_device = USE_PER_DEVICE,
            max_chunk_size = LOAD_CHUNK_TOKENS,
            max_batch_size = 1,
            verbose = True,
        )
        tokenizer = Tokenizer.from_config(config)
        input_ids = make_prompt(tokenizer)
        generator = Generator(
            model = model,
            cache = cache,
            tokenizer = tokenizer,
            max_batch_size = 1,
            max_chunk_size = PREFILL_CHUNK_TOKENS,
        )
        stop_conditions = list(tokenizer.config.eos_token_id_list)
        if not stop_conditions:
            raise RuntimeError("tokenizer did not provide any EOS token IDs")
        runs = []
        for _ in range(REPEATS):
            runs.append(run_job(generator, input_ids, stop_conditions))
            clear_request_cache(generator)
        print("TP2_TEXT_QUALITY_JSON=" + json.dumps({
            "prompt_tokens": input_ids.shape[1],
            "output_tokens": OUTPUT_TOKENS,
            "repeat_count": REPEATS,
            "stop_conditions": stop_conditions,
            "runs": runs,
            "stable": len({run["token_ids_sha256"] for run in runs}) == 1,
        }, ensure_ascii = False, sort_keys = True), flush = True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
