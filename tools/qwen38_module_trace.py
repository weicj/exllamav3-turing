"""Run a deterministic short Qwen3.8 Flash-Next request with optional module tracing."""

import json
import os
import time

import torch

from exllamav3 import Cache, Config, Model


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
MODE = os.environ["EXL3_TRACE_MODE"]
PROMPT_TOKENS = int(os.environ.get("EXL3_TRACE_PROMPT_TOKENS", "32"))
TRACE_REPEATS = int(os.environ.get("EXL3_TRACE_REPEATS", "1"))
# Match the known-good PP2/TP2 lane. The trace payload is small; cache capacity is not the
# variable under investigation and using the validated layout avoids loader-only behavior.
CACHE_TOKENS = 4608


def main():
    if MODE not in ("pp", "tp"):
        raise ValueError("EXL3_TRACE_MODE must be pp or tp")
    if TRACE_REPEATS < 1:
        raise ValueError("EXL3_TRACE_REPEATS must be positive")
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    if MODE == "tp":
        # Flash-Next TP import/export is still experimental; tracing it is deliberate.
        model.caps["supports_tp"] = True
    cache = Cache(model, max_num_tokens=CACHE_TOKENS)
    started = time.monotonic()
    try:
        load_kwargs = {
            # The autosplit dry-run supplies this as paged-attention's batch shape.
            "max_chunk_size": 256,
            "max_batch_size": 1,
            "verbose": True,
        }
        if MODE == "tp":
            load_kwargs.update({
                "tensor_p": True,
                "tp_backend": "nccl",
                "use_per_device": [22.0, 22.0],
            })
        else:
            # Match the established PP2 performance baseline rather than the older
            # asymmetric quality-only split.
            load_kwargs.update({"use_per_device": [22.0, 22.0]})
        print("MODULE_TRACE_STAGE=load_start", flush = True)
        model.load(**load_kwargs)
        print("MODULE_TRACE_STAGE=load_complete", flush = True)

        input_ids = torch.randint(
            1000,
            20000,
            (1, PROMPT_TOKENS),
            generator = torch.Generator(device = "cpu").manual_seed(2),
            dtype = torch.long,
        )
        runs = []
        for repeat in range(TRACE_REPEATS):
            print(f"MODULE_TRACE_STAGE=request_start repeat={repeat}", flush = True)
            # A new params dict makes prepare_for_recurrence allocate and clear a distinct
            # recurrent slot while preserving the identical rectangular KV layout.
            params = {
                "attn_mode": "flash_attn",
                "cache": cache,
                "batch_shape": (1, CACHE_TOKENS),
                "past_len": 0,
            }
            request_started = time.monotonic()
            model.prefill(input_ids = input_ids, params = params)
            runs.append({
                "repeat": repeat,
                "seconds": time.monotonic() - request_started,
                "recurrent_position": params["recurrent_states"][0].position,
            })
            print(f"MODULE_TRACE_STAGE=request_complete repeat={repeat}", flush = True)
        print("MODULE_TRACE_JSON=" + json.dumps({
            "mode": MODE,
            "load_and_run_s": time.monotonic() - started,
            "prompt_tokens": PROMPT_TOKENS,
            "runs": runs,
        }, sort_keys = True), flush = True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
