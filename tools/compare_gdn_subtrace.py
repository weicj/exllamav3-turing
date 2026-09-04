"""Reconstruct TP2 GDN head shards and compare them to the PP2 intermediate tensors."""

import json
import math
import sys
from pathlib import Path

import torch


Q, K, V = 16 * 128, 16 * 128, 48 * 128


def tensor(directory: Path, prefix: str, stage: str):
    paths = sorted(directory.glob(f"{prefix}-l000-{stage}.pt"))
    return [torch.load(p, map_location = "cpu", weights_only = True)["tensor"] for p in paths]


def metric(name, pp, tp):
    a, b = pp.float().flatten(), tp.float().flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    d = (a - b).abs()
    den = a.norm() * b.norm()
    return {
        "stage": name,
        "shape": list(pp.shape),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(d.square().mean().sqrt()),
        "cosine": float(torch.dot(a, b) / den) if den else math.nan,
    }


def main():
    directory = Path(sys.argv[1])
    rows = []
    for stage in ("qkv", "z", "b", "a", "core", "norm", "o_pre_reduce", "output"):
        pp = tensor(directory, "pp-*", stage)[0]
        tp = tensor(directory, "tp-*", stage)
        if stage == "qkv":
            # Each rank stores [local Q | local K | local V]; restore PP's [all Q | all K | all V].
            hq, hk, hv = Q // 2, K // 2, V // 2
            rebuilt = torch.cat((
                tp[0][..., :hq], tp[1][..., :hq],
                tp[0][..., hq:hq + hk], tp[1][..., hq:hq + hk],
                tp[0][..., hq + hk:hq + hk + hv], tp[1][..., hq + hk:hq + hk + hv],
            ), dim = -1)
        elif stage in ("z", "b", "a", "core"):
            rebuilt = torch.cat(tp, dim = 2 if stage in ("z", "core") else -1)
        elif stage == "norm":
            rebuilt = torch.cat(tp, dim = -1)
        elif stage == "o_pre_reduce":
            rebuilt = sum(tp)
        else:
            rebuilt = tp[0]
        rows.append(metric(stage, pp, rebuilt))
    print(json.dumps(rows, indent = 2, sort_keys = True))


if __name__ == "__main__":
    main()
