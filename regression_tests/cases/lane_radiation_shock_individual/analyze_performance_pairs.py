#!/usr/bin/env python3
"""Apply the predeclared 10x acceptance rule to paired full-run timings."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


ONE_SIDED_T_95 = {
    1: 6.3138, 2: 2.9200, 3: 2.3534, 4: 2.1318, 5: 2.0150,
    6: 1.9432, 7: 1.8946, 8: 1.8595, 9: 1.8331, 10: 1.8125,
    11: 1.7959, 12: 1.7823, 13: 1.7709, 14: 1.7613, 15: 1.7531,
    16: 1.7459, 17: 1.7396, 18: 1.7341, 19: 1.7291, 20: 1.7247,
    21: 1.7207, 22: 1.7171, 23: 1.7139, 24: 1.7109, 25: 1.7081,
    26: 1.7056, 27: 1.7033, 28: 1.7011, 29: 1.6991, 30: 1.6973,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("pairs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=5)
    parser.add_argument("--minimum-speedup", type=float, default=10.0)
    parser.add_argument("--minimum-median", type=float, default=10.5)
    return parser.parse_args()


def load_pairs(path: Path) -> list[dict[str, object]]:
    result = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"pair", "baseline_seconds", "optimized_seconds"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain {sorted(required)}")
        for row in reader:
            baseline = float(row["baseline_seconds"])
            optimized = float(row["optimized_seconds"])
            if not math.isfinite(baseline) or not math.isfinite(optimized):
                raise ValueError("timings must be finite")
            if baseline <= 0 or optimized <= 0:
                raise ValueError("timings must be positive")
            result.append(
                {
                    "pair": row["pair"],
                    "baseline_seconds": baseline,
                    "optimized_seconds": optimized,
                    "speedup": baseline / optimized,
                }
            )
    return result


def lower_confidence_bound(speedups: list[float]) -> float:
    logs = [math.log(value) for value in speedups]
    mean = statistics.fmean(logs)
    if len(logs) == 1 or all(value == logs[0] for value in logs):
        return math.exp(mean)
    standard_error = statistics.stdev(logs) / math.sqrt(len(logs))
    degrees = len(logs) - 1
    # Retain the df=30 value above the table rather than becoming
    # anti-conservative for df=31.  The small excess conservatism vanishes as
    # the sample grows and cannot create a false 10x acceptance.
    critical = ONE_SIDED_T_95.get(degrees, ONE_SIDED_T_95[30])
    return math.exp(mean - critical * standard_error)


def main() -> int:
    args = parse_args()
    pairs = load_pairs(args.pairs)
    speedups = [float(pair["speedup"]) for pair in pairs]
    enough_pairs = len(pairs) >= args.minimum_pairs
    minimum = min(speedups) if speedups else 0.0
    median = statistics.median(speedups) if speedups else 0.0
    geometric_mean = (
        math.exp(statistics.fmean(math.log(value) for value in speedups))
        if speedups else 0.0
    )
    lower_bound = lower_confidence_bound(speedups) if speedups else 0.0
    checks = {
        "minimum_pair_count": enough_pairs,
        "every_pair_at_least_minimum": minimum >= args.minimum_speedup,
        "median_at_least_required": median >= args.minimum_median,
        "one_sided_95_percent_lower_bound_at_least_minimum":
            lower_bound >= args.minimum_speedup,
    }
    result = {
        "schema": "rich-autopartial-performance-acceptance-v1",
        "input": str(args.pairs.resolve()),
        "input_sha256": hashlib.sha256(args.pairs.read_bytes()).hexdigest(),
        "requirements": {
            "minimum_pairs": args.minimum_pairs,
            "minimum_speedup": args.minimum_speedup,
            "minimum_median_speedup": args.minimum_median,
            "confidence": "one-sided 95% Student-t interval on log speedup",
        },
        "pairs": pairs,
        "statistics": {
            "count": len(pairs),
            "minimum_speedup": minimum,
            "median_speedup": median,
            "geometric_mean_speedup": geometric_mean,
            "one_sided_95_percent_lower_bound": lower_bound,
        },
        "checks": checks,
        "accepted": all(checks.values()),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        sys.stdout.write(encoded)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if result["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
