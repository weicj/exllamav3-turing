"""Compare selected TransformerBlock PP2/TP2 submodule boundary tensors."""

import json
import math
import sys
from pathlib import Path

import torch


def metric(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    a, b = a[finite], b[finite]
    d = (a - b).abs()
    denom = a.norm() * b.norm()
    return {
        "finite": int(finite.sum()),
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rmse": float(d.square().mean().sqrt()),
        "cosine": float(torch.dot(a, b) / denom) if denom else math.nan,
    }


def main():
    directory = Path(sys.argv[1])
    rows = []
    for pp_path in sorted(directory.glob("pp-*-l000-*.pt")):
        stage = torch.load(pp_path, map_location = "cpu", weights_only = True)["stage"]
        tp_paths = sorted(directory.glob(f"tp-*-l000-{stage}.pt"))
        pp = torch.load(pp_path, map_location = "cpu", weights_only = True)["tensor"]
        for tp_path in tp_paths:
            tp = torch.load(tp_path, map_location = "cpu", weights_only = True)["tensor"]
            rows.append({"stage": stage, "pp": pp_path.name, "tp": tp_path.name, **metric(pp, tp)})
    print(json.dumps(rows, indent = 2, sort_keys = True))


if __name__ == "__main__":
    main()
