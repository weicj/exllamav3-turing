"""Compare PP2 module-boundary tensors against TP2 output-rank boundaries."""

import json
import math
import sys
from pathlib import Path

import torch


def load_index(directory: Path, prefix: str):
    out = {}
    for path in sorted(directory.glob(f"{prefix}-*.pt")):
        data = torch.load(path, map_location = "cpu", weights_only = True)
        module_idx = int(path.stem.rsplit("-m", 1)[1])
        out[module_idx] = data
    return out


def main():
    pp_dir = Path(sys.argv[1])
    tp_dir = Path(sys.argv[2])
    pp = load_index(pp_dir, "pp")
    tp = load_index(tp_dir, "tp-d0")
    rows = []
    for idx in sorted(set(pp) & set(tp)):
        a = pp[idx]["tensor"].float().flatten()
        b = tp[idx]["tensor"].float().flatten()
        if a.shape != b.shape:
            rows.append({"module": idx, "key": pp[idx]["key"], "shape_pp": list(a.shape), "shape_tp": list(b.shape)})
            continue
        finite = torch.isfinite(a) & torch.isfinite(b)
        da = a[finite]
        db = b[finite]
        diff = (da - db).abs()
        denom = da.norm() * db.norm()
        cosine = float(torch.dot(da, db) / denom) if denom else math.nan
        rows.append({
            "module": idx,
            "key": pp[idx]["key"],
            "shape": list(a.shape),
            "finite": int(finite.sum()),
            "max_abs": float(diff.max()) if diff.numel() else math.nan,
            "mean_abs": float(diff.mean()) if diff.numel() else math.nan,
            "rmse": float(diff.square().mean().sqrt()) if diff.numel() else math.nan,
            "cosine": cosine,
        })
    first = next((r for r in rows if r.get("max_abs", 0.0) > 0.01 or r.get("cosine", 1.0) < 0.9999), None)
    print(json.dumps({"first_divergence": first, "rows": rows}, indent = 2, sort_keys = True))


if __name__ == "__main__":
    main()
