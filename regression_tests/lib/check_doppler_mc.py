#!/usr/bin/env python3
"""
Validate the Doppler MC regression test against the analytical solution
from Eq. V.31 of arXiv:2601.05120:

    E_nu(nu, t) = E_nu(nu * exp(-K*t), 0)

where K = -div(v)/3.

For the group-integrated quantity the analytical prediction is:
    E_g(t) = exp(K*t) * integral_{nu_lo*exp(-Kt)}^{nu_hi*exp(-Kt)} E_nu(nu',0) dnu'

Generates one plot comparing the numerical and analytical spectra,
together with the actually seeded initial photon histogram.
"""

import argparse
import os
import sys

import numpy as np

try:
    from scipy import integrate
except ImportError:
    integrate = None


def planck_spectral_density(E, T_kelvin):
    """Planck spectral radiation energy density u_E [erg/cm^3/erg] in CGS."""
    k_B = 1.380649e-16   # erg/K
    h   = 6.62607015e-27  # erg*s
    c   = 2.99792458e10   # cm/s
    x = E / (k_B * T_kelvin)
    if x > 500:
        return 0.0
    return 8.0 * np.pi / (h * c) ** 3 * E ** 3 / np.expm1(x)


def planck_group_integral(E_lo, E_hi, T, E_cut_lo, E_cut_hi):
    """Integrate Planck spectral density over the intersection of
    [E_lo, E_hi] and [E_cut_lo, E_cut_hi]."""
    a = max(E_lo, E_cut_lo)
    b = min(E_hi, E_cut_hi)
    if a >= b:
        return 0.0
    if integrate is not None:
        val, _ = integrate.quad(planck_spectral_density, a, b, args=(T,),
                                limit=200)
        return val
    xs = np.linspace(a, b, 500)
    ys = np.array([planck_spectral_density(x, T) for x in xs])
    return float(np.trapz(ys, xs))


def compute_analytical(E_boundaries, K, t, T, E_cut_lo, E_cut_hi):
    """Compute analytical group energies at time t.
    E_g(t) = exp(Kt) * int_{E_lo*exp(-Kt)}^{E_hi*exp(-Kt)} B(E',T) dE'
    """
    G = len(E_boundaries) - 1
    Eg = np.zeros(G)
    shift = np.exp(-K * t)
    jac = np.exp(K * t)
    for g in range(G):
        Eg[g] = jac * planck_group_integral(
            E_boundaries[g] * shift,
            E_boundaries[g + 1] * shift,
            T, E_cut_lo, E_cut_hi)
    return Eg


def read_spectrum(path):
    """Parse the doppler_mc_spectrum.txt output."""
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


def main():
    parser = argparse.ArgumentParser(description="Doppler MC spectrum check")
    parser.add_argument("--spectrum", required=True, help="Path to doppler_mc_spectrum.txt")
    parser.add_argument("--max-l1", type=float, default=0.7,
                        help="Maximum allowed relative L1 error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)

    K        = meta["K"]
    t_final  = meta["t_final"]
    T_kelvin = meta["T_kelvin"]
    E_cut_lo = meta["E_trunc_lo"]
    E_cut_hi = meta["E_trunc_hi"]

    G = cols.shape[0]
    E_lo      = cols[:, 1]
    E_hi      = cols[:, 2]
    if cols.shape[1] >= 6:
        Eg_init_cell = cols[:, 3]
        Eg_init_photons = cols[:, 4]
        Eg_final = cols[:, 5]
    elif cols.shape[1] == 5:
        Eg_init_cell = cols[:, 3]
        Eg_init_photons = cols[:, 3]
        Eg_final = cols[:, 4]
    else:
        raise ValueError(f"Unexpected spectrum format with {cols.shape[1]} columns")

    E_boundaries = np.zeros(G + 1)
    E_boundaries[:G] = E_lo
    E_boundaries[G] = E_hi[-1]

    anal = compute_analytical(E_boundaries, K, t_final, T_kelvin, E_cut_lo, E_cut_hi)

    def rel_l1(numerical, analytical):
        mask = analytical > 1e-30 * analytical.max()
        if not np.any(mask):
            return 0.0
        return np.sum(np.abs(numerical[mask] - analytical[mask])) / np.sum(analytical[mask])

    err = rel_l1(Eg_final, anal)

    print(f"DOPPLER_MC_L1={err:.8e}")
    print(f"DOPPLER_MC_MAX_L1={args.max_l1:.8e}")
    print(f"Doppler MC check:")
    print(f"  K       = {K:.6e}")
    print(f"  K*t     = {K*t_final:.4f}")
    print(f"  L1 error = {err:.6f}")
    print(f"  Threshold = {args.max_l1}")

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
        mask = (
            (anal > 1e-30 * anal.max())
            | (Eg_final > 1e-30 * max(Eg_final.max(), 1e-300))
            | (Eg_init_cell > 1e-30 * max(Eg_init_cell.max(), 1e-300))
            | (Eg_init_photons > 1e-30 * max(Eg_init_photons.max(), 1e-300))
        )
        ec = E_centers[mask] / kev

        ax.plot(ec, Eg_final[mask] / widths[mask],
                "o", ms=3, label="MC numerical")
        ax.plot(ec, anal[mask] / widths[mask],
                "-", lw=1.5, label="Analytical (Eq. V.31)")

        init_mask = Eg_init_cell > 1e-30 * max(Eg_init_cell.max(), 1e-300)
        if np.any(init_mask):
            ax.plot(E_centers[init_mask] / kev, Eg_init_cell[init_mask] / widths[init_mask],
                    "--", lw=1, alpha=0.5, label="Initial (cell)")

        init_photon_mask = Eg_init_photons > 1e-30 * max(Eg_init_photons.max(), 1e-300)
        if np.any(init_photon_mask):
            ax.plot(E_centers[init_photon_mask] / kev, Eg_init_photons[init_photon_mask] / widths[init_photon_mask],
                    "-.", lw=1.0, alpha=0.85, label="Initial (photons)")

        ax.set_xlabel("Photon energy [keV]")
        ax.set_ylabel(r"$E_\nu$ [erg cm$^{-3}$ erg$^{-1}$]")
        ax.set_title(f"Doppler MC — single cell\nL1 = {err:.4f}, Kt = {K*t_final:.3f}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(0.1, 100)  # 0.1 keV to 100 keV
        y_series = [
            anal[mask] / widths[mask],
            Eg_final[mask] / widths[mask],
            Eg_init_cell[mask] / widths[mask],
            Eg_init_photons[mask] / widths[mask],
        ]
        positive_parts = [arr[arr > 0] for arr in y_series if np.any(arr > 0)]
        if positive_parts:
            positive = np.concatenate(positive_parts)
            ymax = positive.max()
            ax.set_ylim(ymax * 1e-6, ymax * 10)
        fig.tight_layout()
        fig_path = os.path.join(plot_dir, "doppler_mc_mid.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping plots")

    if err > args.max_l1:
        print(f"FAIL: L1 error {err:.6f} exceeds threshold {args.max_l1}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
