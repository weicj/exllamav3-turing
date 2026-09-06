"""Summarize TP expert-shard routing from an opt-in MoE subtrace."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(records: list[dict], split: int) -> dict:
    # A [1, top-k] routing tensor marks a decode pass. Prefill rows are intentionally
    # excluded: their aggregate distribution says little about the serial decode limit.
    decode = [
        row for row in records
        if row.get("stage") == "selected_experts" and row.get("shape", [0])[0] == 1
    ]
    by_call: dict[int, list[dict]] = collections.defaultdict(list)
    for row in decode:
        by_call[row["call"]].append(row)

    layer_imbalances = []
    rank0_only = 0
    rank1_only = 0
    for row in decode:
        rank0 = sum(expert < split for expert in row["values"])
        rank1 = len(row["values"]) - rank0
        layer_imbalances.append(abs(rank0 - rank1))
        rank0_only += rank1 == 0
        rank1_only += rank0 == 0

    calls = []
    for call, rows in sorted(by_call.items()):
        rank0 = sum(expert < split for row in rows for expert in row["values"])
        rank1 = sum(expert >= split for row in rows for expert in row["values"])
        calls.append({
            "call": call,
            "layers": len(rows),
            "rank0_assignments": rank0,
            "rank1_assignments": rank1,
            "imbalance": rank0 - rank1,
        })

    rank0 = sum(row["rank0_assignments"] for row in calls)
    rank1 = sum(row["rank1_assignments"] for row in calls)
    return {
        "decode_steps": len(calls),
        "assignments": rank0 + rank1,
        "rank0_assignments": rank0,
        "rank1_assignments": rank1,
        "rank0_share": rank0 / (rank0 + rank1) if rank0 + rank1 else None,
        "per_layer_assignment_imbalance": {
            "samples": len(layer_imbalances),
            "mean_abs": sum(layer_imbalances) / len(layer_imbalances) if layer_imbalances else None,
            "max_abs": max(layer_imbalances, default = None),
            "rank0_only_samples": rank0_only,
            "rank1_only_samples": rank1_only,
        },
        "steps": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="Trace prefix or one rank's JSONL file")
    parser.add_argument("--split", type=int, default=256, help="First expert ID on rank 1")
    args = parser.parse_args()

    paths = sorted(args.trace.parent.glob(f"{args.trace.name}.d*"))
    if not paths and args.trace.is_file():
        paths = [args.trace]
    if not paths:
        raise FileNotFoundError(f"no trace files match {args.trace}.d*")

    reports = {path.name: summarize(load_records(path), args.split) for path in paths}
    # Routing is computed once and broadcast in TP; disagreement would be a correctness bug.
    shares = {report["rank0_share"] for report in reports.values()}
    print("MOE_ROUTE_SUMMARY_JSON=" + json.dumps({
        "split": args.split,
        "per_rank_trace": reports,
        "routing_agrees_across_traces": len(shares) == 1,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
