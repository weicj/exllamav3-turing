"""Isolate the long-prefill GatedResidual mixer used by Qwen3.8-Flash-Next.

For R > 32, GatedResidual does not use ``gr_mix``.  Its execution is an extension
RMSNorm followed by two CUDA GEMMs and a torch pointwise reduction.  This probe
runs those exact stages repeatedly on a saved residual stream and reports the
first stage whose bytes differ.  It is intentionally independent of TP, caches,
and recurrent state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from exllamav3 import Config
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.hyperconnections import GatedResidual


DEFAULT_MODEL_DIR = "/mnt/nvme/models/turboderp-Qwen3.8-Flash-Next-exl3/2.05bpw_h4_ng4"
DEFAULT_INPUT = (
    "/home/max/results/qwen38-flash-next-exl3-205-sm75-tp2-prefix-error-magnitude-20260905/"
    "tensors/unknown-cuda_0-l001-c009-post_mlp.pt"
)
DEFAULT_MODULE_KEY = "model.language_model.layers.2.attn_hyper_connection"


def digest(tensor: torch.Tensor) -> str:
    """Hash the exact device result after the caller has synchronized."""
    data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def error(reference: torch.Tensor, value: torch.Tensor) -> dict[str, float]:
    diff = (value.float() - reference.float()).flatten()
    ref = reference.float().flatten()
    rms = torch.sqrt(torch.mean(diff.square())).item()
    reference_rms = torch.sqrt(torch.mean(ref.square())).item()
    return {
        "max_abs_vs_first": float(diff.abs().max().item()),
        "rmse_vs_first": float(rms),
        "relative_rmse_vs_first": float(rms / reference_rms) if reference_rms else 0.0,
    }


def load_streams(path: Path, device: torch.device) -> torch.Tensor:
    saved = torch.load(path, map_location="cpu", weights_only=True)
    streams = saved["tensor"] if isinstance(saved, dict) else saved
    if streams.dim() != 4:
        raise ValueError(f"expected residual streams with shape (B, S, H, D), got {tuple(streams.shape)}")
    if streams.dtype != torch.float:
        streams = streams.float()
    return streams.contiguous().to(device)


def torch_rms_norm(streams: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Numerically similar, deterministic-control alternative to ext.rms_norm."""
    normed = streams * torch.rsqrt(streams.square().mean(-1, keepdim=True) + eps)
    return (normed * weight.float().view(1, 1, streams.shape[2], streams.shape[3])).half()


def run_once(module: GatedResidual, streams: torch.Tensor, norm_backend: str) -> dict[str, torch.Tensor]:
    h, d = module.hc_mult, module.hidden_size
    r = streams.shape[0] * streams.shape[1]
    s3 = streams.reshape(r, h, d).contiguous()
    if norm_backend == "ext":
        normed = torch.empty((r * h, d), dtype=torch.half, device=streams.device)
        ext.rms_norm(s3.view(r * h, d), module.w_h, normed, module.rms_eps, 0.0, 1.0, False, False, h)
        normed = normed.view(r, h, d)
    else:
        normed = torch_rms_norm(s3, module.w_h, module.rms_eps)

    dm = torch.matmul(normed.view(r, h * d), module.proj_h.t())
    t = F.silu(dm[:, : module.rank] / h)
    g = torch.matmul(t, module.up_h.t())
    mixed = (torch.sigmoid(g.float()).view(r, h, d) * normed.float()).mean(dim=-2).half()
    post = 2.0 * torch.sigmoid(dm[:, module.rank :].float() / h) if module.use_combine else None
    return {
        "normed": normed,
        "dm": dm,
        "t": t,
        "g": g,
        "post": post,
        "mixed": mixed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--module-key", default=DEFAULT_MODULE_KEY)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--norm-backend", choices=("ext", "torch"), default="ext")
    args = parser.parse_args()

    if args.repeats < 2:
        raise ValueError("--repeats must be at least two")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    device = torch.device(args.device)
    config = Config.from_directory(args.model_dir)
    module = GatedResidual(
        config=config,
        key=args.module_key,
        hc_mult=config.hc_mult,
        hidden_size=config.hidden_size,
        rms_norm_eps=config.rms_norm_eps,
    )
    try:
        module.load(device)
        streams = load_streams(args.input, device)
        torch.cuda.synchronize(device)

        stage_names = ("normed", "dm", "t", "g", "post", "mixed")
        references: dict[str, torch.Tensor] = {}
        rows = []
        for iteration in range(args.repeats):
            stages = run_once(module, streams, args.norm_backend)
            torch.cuda.synchronize(device)
            row = {"iteration": iteration, "stages": {}}
            for name in stage_names:
                value = stages[name]
                if value is None:
                    continue
                if name not in references:
                    references[name] = value.clone()
                    metrics = {
                        "max_abs_vs_first": 0.0,
                        "rmse_vs_first": 0.0,
                        "relative_rmse_vs_first": 0.0,
                    }
                else:
                    metrics = error(references[name], value)
                row["stages"][name] = {"sha256": digest(value), **metrics}
            rows.append(row)

        unique_hashes = {
            name: len({row["stages"][name]["sha256"] for row in rows})
            for name in stage_names
            if name in rows[0]["stages"]
        }
        print("GATED_RESIDUAL_DETERMINISM_JSON=" + json.dumps({
            "device": str(device),
            "input": str(args.input),
            "input_sha256": digest(streams),
            "input_shape": list(streams.shape),
            "module_key": args.module_key,
            "norm_backend": args.norm_backend,
            "repeats": args.repeats,
            "unique_hashes": unique_hashes,
            "results": rows,
        }, sort_keys=True), flush=True)
    finally:
        module.unload()


if __name__ == "__main__":
    main()
