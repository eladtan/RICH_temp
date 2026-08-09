#!/usr/bin/env python3
"""Validate the basic physical invariants of the Compton Marshak wave."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def read_vector(path: Path) -> list[float]:
    try:
        values = [float(token) for token in path.read_text().split()]
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot read numeric vector {path}: {exc}") from exc
    if not values:
        raise ValueError(f"empty vector: {path}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite value in {path}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=Path, required=True)
    parser.add_argument("--tgas", type=Path, required=True)
    parser.add_argument("--trad", type=Path, required=True)
    parser.add_argument("--fleck", type=Path)
    parser.add_argument("--expected-cells", type=int, default=1024)
    args = parser.parse_args()

    try:
        x = read_vector(args.x)
        tgas = read_vector(args.tgas)
        trad = read_vector(args.trad)
        fleck = read_vector(args.fleck) if args.fleck is not None else None

        expected = args.expected_cells
        if len(x) != expected or len(tgas) != expected or len(trad) != expected:
            raise ValueError(
                "profile size mismatch: "
                f"x={len(x)}, Tgas={len(tgas)}, Trad={len(trad)}, expected={expected}"
            )
        if any(right <= left for left, right in zip(x, x[1:])):
            raise ValueError("x profile is not strictly increasing")
        if min(tgas) <= 0.0 or min(trad) <= 0.0:
            raise ValueError("material and radiation temperatures must remain positive")
        if fleck is not None and (min(fleck) <= 0.0 or max(fleck) > 1.0 + 1e-12):
            raise ValueError(
                f"Fleck factors must satisfy 0 < f <= 1; range=[{min(fleck)}, {max(fleck)}]"
            )

        edge_cells = min(8, expected // 4)
        front_tgas = sum(tgas[:edge_cells]) / edge_cells
        tail_tgas = sum(tgas[-edge_cells:]) / edge_cells
        front_trad = sum(trad[:edge_cells]) / edge_cells
        tail_trad = sum(trad[-edge_cells:]) / edge_cells
        if front_tgas <= tail_tgas or front_trad <= tail_trad:
            raise ValueError(
                "hot-boundary wave did not heat the front above the far-field tail: "
                f"Tgas={front_tgas}/{tail_tgas}, Trad={front_trad}/{tail_trad}"
            )
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1

    print(
        "PASS: positive Compton Marshak profiles"
        + (" and Fleck factors" if fleck is not None else "")
        + "; "
        f"Tgas(front/tail)={front_tgas:.6e}/{tail_tgas:.6e}, "
        f"Trad(front/tail)={front_trad:.6e}/{tail_trad:.6e}"
        + (f", min_fleck={min(fleck):.6e}" if fleck is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
