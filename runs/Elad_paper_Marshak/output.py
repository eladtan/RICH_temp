#!/usr/bin/env python3
"""
Plot Marshak wave snapshot curves on a single graph.

Usage:
    python output.py <snapshot_file> [snapshot_file2 ...]

Each file should have a header like:
    # ct=100 Nx=64 mode=mc
    # x, T
followed by CSV data (x, T).
"""

import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt

COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load_snapshot(path):
    ct_val = None
    mode = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"ct=([\d.eE+\-]+)", line)
                if m:
                    ct_val = float(m.group(1))
                m2 = re.search(r"mode=(\S+)", line)
                if m2:
                    mode = m2.group(1)
            else:
                break
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return ct_val, mode, data[:, 0], data[:, 1]


def main():
    if len(sys.argv) < 2:
        print("Usage: python output.py <snapshot_file> [snapshot_file2 ...]")
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, path in enumerate(sys.argv[1:]):
        ct, mode, x, T = load_snapshot(path)
        basename = os.path.basename(path)
        label = basename
        if mode and ct is not None:
            label = f"{mode}  ct={ct:g}"
        elif ct is not None:
            label = f"{basename}  ct={ct:g}"

        color = COLORS[idx % len(COLORS)]
        ax.plot(x, T, "o", ms=2, alpha=0.7, color=color, label=label)

    ax.set_xlim(0, 4.0)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("x", fontsize=13)
    ax.set_ylabel("T", fontsize=13)
    ax.set_title("Marshak Wave", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
