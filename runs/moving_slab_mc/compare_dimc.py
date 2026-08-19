#!/usr/bin/env python3
"""Compare the moving-slab spectrum to the McClarren & Gentile semi-analytic result.

Steinberg & Heizler 2022 (the DIMC paper) does not include this problem. The
overlay is the same 124-group original-vacuum benchmark used in STORM, which
is the published moving-slab test (McClarren & Gentile).

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py moving_slab_mc_spectrum.txt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
STORM_CHECK = os.path.join(REPO, "source", "monte", "examples", "moving_slab", "check_spectrum.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", nargs="?",
                        default=os.path.join(HERE, "moving_slab_mc_spectrum.txt"))
    parser.add_argument("--output-dir", default=HERE)
    args = parser.parse_args()

    if not os.path.exists(args.spectrum):
        print(f"No spectrum at {args.spectrum}. Run the moving-slab job first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(STORM_CHECK):
        print(f"STORM checker not found: {STORM_CHECK}", file=sys.stderr)
        sys.exit(1)

    print("Note: Steinberg & Heizler 2022 does not contain the moving-slab problem.")
    print("Comparing to the McClarren & Gentile semi-analytic spectrum via STORM.")
    cmd = [sys.executable, STORM_CHECK,
           "--spectrum", args.spectrum,
           "--plot-dir", args.output_dir]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
