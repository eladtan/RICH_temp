#!/usr/bin/env python3
"""Compare the Densmore step-opacity profile to the published MC reference.

Steinberg & Heizler 2022 (the DIMC paper) does not include this problem. The
overlay is Figure 4 of Densmore et al., JCP 231, 6924 (2012), which is the
literature result used in the IMC manuscript.

Usage:
    python3 compare_dimc.py
    python3 compare_dimc.py desmore2012_mc_profile.txt
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "data", "densmore2012_fig4_mc.csv")
KEV_K = 1.602176634e-9 / 1.380649e-16


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?",
                        default=os.path.join(HERE, "desmore2012_mc_profile.txt"))
    parser.add_argument("--output", default=os.path.join(HERE, "compare_dimc.png"))
    args = parser.parse_args()

    if not os.path.exists(args.profile):
        print(f"No RICH profile at {args.profile}. Run the Densmore job first.", file=sys.stderr)
        sys.exit(1)

    print("Note: Steinberg & Heizler 2022 does not contain the Densmore problem.")
    print("Comparing to Densmore et al. 2012 Fig. 4 (MC), the published reference.")

    rich = np.loadtxt(args.profile, comments="#")
    if rich.ndim == 1:
        rich = rich.reshape(1, -1)
    T_keV = rich[:, 1] / KEV_K
    ref = np.loadtxt(REF, delimiter=",", comments="#")

    Ti = np.interp(ref[:, 0], rich[:, 0], T_keV)
    d = Ti - ref[:, 1]
    print(f"RICH profile: {args.profile}")
    print(f"vs Densmore 2012 Fig. 4:  RMS={np.sqrt(np.mean(d**2)):.4f} keV  max|dT|={np.max(np.abs(d)):.4f} keV")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ref[:, 0], ref[:, 1], "b-", lw=1.6, label="Densmore 2012 Fig. 4 (MC)")
    ax.plot(rich[:, 0], T_keV, "ko", ms=3, markerfacecolor="none", label="RICH IMC")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("T (keV)")
    ax.set_title("Densmore 2012 heterogeneous step-opacity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
