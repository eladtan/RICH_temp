#!/usr/bin/env python3
"""Tabulate crooked-pipe probe convergence against Fig. 8(a) references.

Reads every case subdirectory of a convergence run and compares the probe
temperatures at the common stop time to the digitized IMC, DIMC and Gentile
curves.
"""

import argparse
import math
from pathlib import Path

PROBES = ["P1 (0,0.25)", "P2 (0,2.75)", "P3 (1.25,3.5)", "P4 (0,4.25)", "P5 (0,6.75)"]
REFERENCES = ("imc", "dimc", "gentile")


def load_rows(path):
    rows = []
    with path.open() as stream:
        for line in stream:
            if line.startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in line.split(",")])
    return rows


def interpolate(reference, time_ns, probe):
    if time_ns <= reference[0][0]:
        return reference[0][probe + 1]
    for lower, upper in zip(reference, reference[1:]):
        if time_ns <= upper[0]:
            fraction = ((math.log(time_ns) - math.log(lower[0])) /
                        (math.log(upper[0]) - math.log(lower[0])))
            return lower[probe + 1] + fraction * (
                upper[probe + 1] - lower[probe + 1])
    return reference[-1][probe + 1]


def case_summary(directory):
    probes = directory / "crookedpipe_probes.txt"
    if not probes.exists():
        return None
    rows = load_rows(probes)
    if len(rows) < 2:
        return None
    wall = None
    wall_file = directory / "wall_time.txt"
    if wall_file.exists():
        first = wall_file.read_text().split()
        if len(first) >= 2:
            wall = float(first[1])
    return {
        "name": directory.name,
        "cycles": int(rows[-1][1]),
        "time_ns": rows[-1][0],
        "temperatures": rows[-1][2:7],
        "wall": wall,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    reference_dir = Path(__file__).parent / "reference"
    references = {
        name: load_rows(reference_dir / f"fig8_{name}.csv")
        for name in REFERENCES
    }

    cases = []
    for directory in sorted(p for p in args.result_dir.iterdir() if p.is_dir()):
        summary = case_summary(directory)
        if summary is not None:
            cases.append(summary)
    if not cases:
        print("No completed cases found.")
        return

    print(f"Crooked pipe convergence study: {args.result_dir}")
    print()
    for probe in range(5):
        print(f"=== {PROBES[probe]} ===")
        header = (f"{'case':<22}{'cycles':>7}{'t (ns)':>9}{'T (keV)':>10}"
                  f"{'d_imc':>9}{'d_dimc':>9}{'d_gent':>9}{'wall (s)':>10}")
        print(header)
        for case in cases:
            value = case["temperatures"][probe]
            deltas = [
                value - interpolate(references[name], case["time_ns"], probe)
                for name in REFERENCES
            ]
            wall = f'{case["wall"]:.1f}' if case["wall"] is not None else "n/a"
            print(f'{case["name"]:<22}{case["cycles"]:>7}{case["time_ns"]:>9.3f}'
                  f'{value:>10.4f}'
                  f'{deltas[0]:>+9.4f}{deltas[1]:>+9.4f}{deltas[2]:>+9.4f}'
                  f'{wall:>10}')
        print()

    print("Reference values at each case stop time:")
    for case in cases:
        parts = []
        for name in REFERENCES:
            values = [interpolate(references[name], case["time_ns"], probe)
                      for probe in range(2)]
            parts.append(f"{name}: P1={values[0]:.4f} P2={values[1]:.4f}")
        print(f'  t={case["time_ns"]:.3f} ns  ' + "  ".join(parts))


if __name__ == "__main__":
    main()
