"""Repeat the GDN CUDA core from the Qwen3.8 layer-0 geometry."""

import hashlib
import json

import torch

from exllamav3.ext import exllamav3_ext as ext


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def run_once(qkv, gate, beta, slots):
    state = torch.zeros((1, 1, 48, 128, 128), dtype=torch.float, device=qkv.device)
    output = torch.empty((1, qkv.shape[1], 48, 128), dtype=torch.bfloat16, device=qkv.device)
    ext.cuda_recurrent_gated_delta_rule(
        qkv, gate, beta, state, output, 16, 48, 128, 128, slots, False
    )
    torch.cuda.synchronize(qkv.device)
    return digest(output), digest(state), output.float().cpu()


def main():
    torch.manual_seed(20260904)
    device = torch.device("cuda:0")
    qkv = torch.randn((1, 512, 10240), device=device, dtype=torch.float).to(torch.bfloat16)
    gate = -torch.rand((1, 512, 48), device=device, dtype=torch.float)
    beta = torch.rand((1, 512, 48), device=device, dtype=torch.float).to(torch.bfloat16)
    slots = torch.zeros((1,), device=device, dtype=torch.int32)
    results = []
    reference = None
    for iteration in range(8):
        output_sha, state_sha, output = run_once(qkv, gate, beta, slots)
        max_abs = 0.0 if reference is None else float((output - reference).abs().max())
        if reference is None:
            reference = output
        results.append({
            "iteration": iteration,
            "output_sha256": output_sha,
            "state_sha256": state_sha,
            "max_abs_vs_first": max_abs,
        })
    print("GDN_DETERMINISM_JSON=" + json.dumps({
        "unique_output_hashes": len({row["output_sha256"] for row in results}),
        "unique_state_hashes": len({row["state_sha256"] for row in results}),
        "results": results,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
