"""Check whether a streamed Qwen3.8 PLE table lookup is bit-stable in one loaded model."""

import hashlib
import json
import os

import torch

from exllamav3 import Config, Model
from exllamav3.modules.ple import PLELayer


MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
PROMPT_TOKENS = int(os.environ.get("PROBE_PROMPT_TOKENS", "4096"))
REPEATS = int(os.environ.get("PROBE_REPEATS", "3"))
USE_PER_DEVICE = [float(value) for value in os.environ.get("BENCH_USE_PER_DEVICE", "22,22").split(",")]


def tensor_sha256(tensor):
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def main():
    if os.environ.get("EXL3_NGRAM_STREAM") != "1":
        raise RuntimeError("EXL3_NGRAM_STREAM=1 is required for this probe")
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"expected two visible GPUs, got {torch.cuda.device_count()}")

    config = Config.from_directory(MODEL_DIR)
    model = Model.from_config(config)
    try:
        model.load(use_per_device=USE_PER_DEVICE, max_chunk_size=256, max_batch_size=1, verbose=True)
        ple_layers = [module for module in model.modules if isinstance(module, PLELayer)]
        if len(ple_layers) != 1:
            raise RuntimeError(f"expected one PLE layer, found {len(ple_layers)}")
        ngram = ple_layers[0].ple_embedding
        context = ngram.context_len
        ids = torch.Generator(device="cpu").manual_seed(2)
        history = torch.randint(1000, 20000, (1, context + PROMPT_TOKENS), generator=ids, dtype=torch.long)

        digests = []
        for _ in range(REPEATS):
            output = ngram.forward(history, {})
            torch.cuda.synchronize(ngram.device)
            digests.append(tensor_sha256(output))

        report = {
            "prompt_tokens": PROMPT_TOKENS,
            "context_tokens": context,
            "repeat_count": REPEATS,
            "stream_mode": ngram.mode,
            "device": str(ngram.device),
            "output_shape": list(output.shape),
            "digests": digests,
            "stable": len(set(digests)) == 1,
        }
        print("NGRAM_REPEAT_JSON=" + json.dumps(report, sort_keys=True), flush=True)
    finally:
        model.unload()


if __name__ == "__main__":
    main()
