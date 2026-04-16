#!/usr/bin/env python3
"""
Validate the Doppler scatter regression test by comparing the Monte Carlo
emergent comoving-frame spectrum with the multigroup diffusion result.

Reads doppler_scatter_spectrum.txt and computes the relative L1 error
between the normalised MC and diffusion spectra.  Generates a comparison
plot saved as doppler_scatter_comparison.png.
"""

import argparse
import os
import sys

import numpy as np


def read_spectrum(path):
    """Parse doppler_scatter_spectrum.txt."""
    meta = {}
    data_lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                parts = line.lstrip("# ").split(None, 1)
                if len(parts) == 2:
                    key, val = parts
                    if key == "columns:":
                        continue
                    try:
                        meta[key] = float(val)
                    except ValueError:
                        meta[key] = val
                continue
            data_lines.append(line)

    cols = np.loadtxt(data_lines, ndmin=2)
    return meta, cols


def rel_l1(a, b):
    """Relative L1 error over groups with nonzero reference."""
    mask = b > 1e-30 * max(b.max(), 1e-300)
    if not np.any(mask):
        return 0.0
    return np.sum(np.abs(a[mask] - b[mask])) / np.sum(b[mask])


def main():
    parser = argparse.ArgumentParser(
        description="Doppler scatter MC-vs-diffusion spectrum check")
    parser.add_argument("--spectrum", required=True,
                        help="Path to doppler_scatter_spectrum.txt")
    parser.add_argument("--max-l1", type=float, default=0.3,
                        help="Maximum allowed relative L1 error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)

    G = cols.shape[0]
    E_lo = cols[:, 1]
    E_hi = cols[:, 2]
    Eg_mc = cols[:, 3]
    Eg_diff = cols[:, 4]

    E_boundaries = np.zeros(G + 1)
    E_boundaries[:G] = E_lo
    E_boundaries[G] = E_hi[-1]

    err_mc_diff = rel_l1(Eg_mc, Eg_diff)
    err_diff_mc = rel_l1(Eg_diff, Eg_mc)
    err = min(err_mc_diff, err_diff_mc)

    print(f"DOPPLER_SCATTER_L1={err:.8e}")
    print(f"DOPPLER_SCATTER_MAX_L1={args.max_l1:.8e}")
    print(f"Doppler scatter check:")
    print(f"  tau       = {meta.get('tau', 'N/A')}")
    print(f"  kappa_s   = {meta.get('kappa_s', 'N/A')}")
    print(f"  N_pkt     = {meta.get('N_pkt', 'N/A')}")
    print(f"  right_esc = {meta.get('right_escaped', 'N/A')}")
    print(f"  L1 (MC vs diff) = {err:.6f}")
    print(f"  Threshold        = {args.max_l1}")

    # --- plot ---
    plot_dir = args.plot_dir or os.path.dirname(args.spectrum)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        E_centers = 0.5 * (E_lo + E_hi)
        widths = E_hi - E_lo
        kev = 1.602176634e-9  # erg per keV

        fig, ax = plt.subplots(figsize=(7, 4.5))

        mc_density = Eg_mc / widths
        diff_density = Eg_diff / widths

        mask = (
            (mc_density > 1e-30 * max(mc_density.max(), 1e-300))
            | (diff_density > 1e-30 * max(diff_density.max(), 1e-300))
        )

        ec = E_centers[mask] / kev

        ax.plot(ec, mc_density[mask], "o", ms=3,
                label="MC (comoving, right boundary)")
        ax.plot(ec, diff_density[mask], "-", lw=1.5,
                label="Diffusion (comoving, right cell)")

        ax.set_xlabel("Photon energy [keV]")
        ax.set_ylabel("Normalised spectral density [arb. units]")
        ax.set_title(
            f"Doppler scatter benchmark\n"
            f"L1 = {err:.4f}, "
            f"$\\tau$ = {meta.get('tau', '?'):.0f}, "
            f"N_pkt = {meta.get('N_pkt', '?'):.0f}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 100)

        y_all = np.concatenate([mc_density[mask], diff_density[mask]])
        positive = y_all[y_all > 0]
        if len(positive) > 0:
            ax.set_ylim(positive.min() * 0.1, positive.max() * 10)

        fig.tight_layout()
        fig_path = os.path.join(plot_dir, "doppler_scatter_comparison.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {fig_path}")
    except ImportError:
        print("  matplotlib not available -- skipping plots")

    if err > args.max_l1:
        print(f"FAIL: L1 error {err:.6f} exceeds threshold {args.max_l1}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
