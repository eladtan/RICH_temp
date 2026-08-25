#!/usr/bin/env python3
"""Validate MPI-global grey-diffusion timestep reference scales."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FIELD = re.compile(r"([A-Za-z_]+)=(\S+)")


def record(path: Path, marker: str) -> dict[str, str]:
    with path.open(errors="replace") as stream:
        for line in stream:
            if marker in line:
                return dict(FIELD.findall(line))
    raise RuntimeError(f"{path}: no {marker} record")


def require_success(path: Path) -> None:
    exit_code_path = path.parent / "exit_code.txt"
    exit_code = exit_code_path.read_text().strip()
    if exit_code != "0":
        raise RuntimeError(f"{path}: simulation exit code is {exit_code!r}")


def limit(path: Path, mode: str) -> dict[str, str]:
    result = record(path, f"GREY_TIMESTEP_LIMIT mode={mode}")
    required = {
        "current_dt", "suggested_dt", "difference", "max_Er",
        "growth_cap", "reference_scope",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"{path}: grey timestep record misses {missing}")
    return result


def number(value: dict[str, str], field: str) -> float:
    result = float(value[field])
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite {field}={result}")
    return result


def close(label: str, left: float, right: float, rtol: float) -> None:
    scale = max(abs(left), abs(right), float.fromhex("0x1p-1022"))
    error = abs(left - right) / scale
    if error > rtol:
        raise RuntimeError(
            f"{label}: {left:.17g} versus {right:.17g}; "
            f"relative error {error:.6g} > {rtol:.6g}")


def check_formula(label: str, value: dict[str, str], cap: float) -> None:
    current = number(value, "current_dt")
    difference = number(value, "difference")
    expected = current * min(cap, 0.15 / max(difference, float.fromhex("0x1p-1022")))
    close(label + " formula", number(value, "suggested_dt"), expected, 2e-6)
    close(label + " growth cap", number(value, "growth_cap"), cap, 1e-15)


def compare_limits(
    label: str,
    left: dict[str, str],
    right: dict[str, str],
    difference_rtol: float,
) -> None:
    close(label + " max_Er", number(left, "max_Er"),
          number(right, "max_Er"), 1e-10)
    close(label + " difference", number(left, "difference"),
          number(right, "difference"), difference_rtol)


def check_zero_owned(label: str, path: Path) -> None:
    owned = record(path, "GREY_MPI_OWNED_CELL_RANGE")
    minimum = int(owned["minimum"])
    maximum = int(owned["maximum"])
    empty_ranks = int(owned["empty_ranks"])
    ranks = int(owned["ranks"])
    if minimum != 0 or maximum <= 0 or not (0 < empty_ranks < ranks):
        raise RuntimeError(
            f"invalid {label} zero-owned grey coverage: "
            f"{minimum=}, {maximum=}, {empty_ranks=}, {ranks=}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-log", type=Path, required=True)
    parser.add_argument("--global-empty-owned-log", type=Path, required=True)
    parser.add_argument("--full-log", type=Path, required=True)
    parser.add_argument("--full-variable-log", type=Path, required=True)
    parser.add_argument("--partial-log", type=Path, required=True)
    parser.add_argument("--empty-owned-log", type=Path, required=True)
    args = parser.parse_args()

    for path in (
        args.global_log,
        args.global_empty_owned_log,
        args.full_log,
        args.full_variable_log,
        args.partial_log,
        args.empty_owned_log,
    ):
        require_success(path)

    global_limit = limit(args.global_log, "global")
    global_empty_limit = limit(args.global_empty_owned_log, "global")
    full_limit = limit(args.full_log, "individual")
    full_variable_limit = limit(args.full_variable_log, "individual")
    partial_limit = limit(args.partial_log, "individual")
    empty_limit = limit(args.empty_owned_log, "individual")

    check_formula("global", global_limit, 1.25)
    check_formula("global-empty-owned", global_empty_limit, 1.25)
    for label, value in (
        ("full", full_limit),
        ("full-variable", full_variable_limit),
        ("partial", partial_limit),
        ("empty-owned", empty_limit),
    ):
        check_formula(label, value, 2.0)
        if value["reference_scope"] != "canonical_owned_global":
            raise RuntimeError(
                f"{label}: unexpected reference scope {value['reference_scope']!r}")

    for label, value in (
        ("global", global_limit),
        ("global-empty-owned", global_empty_limit),
    ):
        if value["reference_scope"] != "mesh_global":
            raise RuntimeError(
                f"{label}: unexpected reference scope "
                f"{value['reference_scope']!r}")
    for path in (args.full_variable_log, args.partial_log):
        record(path, "GREY_ACTIVE_REFERENCE_MAX_TEST")

    full_fast_path = record(
        args.full_log, "MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH")
    empty_fast_path = record(
        args.empty_owned_log, "MG_INDIVIDUAL_ALL_ACTIVE_FAST_PATH")
    for label, value in (
        ("full", full_fast_path),
        ("empty-owned", empty_fast_path),
    ):
        if number(value, "active_cells") <= 0:
            raise RuntimeError(f"{label}: all-active fast path has no cells")
        if number(value, "candidate_dt") <= 0:
            raise RuntimeError(f"{label}: all-active fast path has invalid dt")

    check_zero_owned("global", args.global_empty_owned_log)
    check_zero_owned("individual", args.empty_owned_log)

    compare_limits("global versus synchronized individual",
                   global_limit, full_limit, 2e-3)
    compare_limits("FullReference versus AutoPartial sparse event",
                   full_variable_limit, partial_limit, 5e-3)

    print("PASS: grey reference scale uses current canonical state and MPI_MAX")
    print("PASS: FullReference and AutoPartial sparse grey limiters agree")
    print("PASS: grey individual solve is collective with zero-owned ranks")
    print("PASS: grey global solver and limiter handle zero-owned ranks")
    print("PASS: grey all-active global fast path handles zero-owned ranks")
    print("PASS: grey global cap=1.25; individual one-bin cap=2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
