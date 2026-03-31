#!/usr/bin/env python3
"""
Validate the Doppler MC regression test against the analytical solution
from Eq. V.31 of arXiv:2601.05120:

    E_nu(nu, t) = E_nu(nu * exp(-K*t), 0)

where K = -div(v)/3.

For the group-integrated quantity the analytical prediction is:
    E_g(t) = exp(K*t) * integral_{nu_lo*exp(-Kt)}^{nu_hi*exp(-Kt)} E_nu(nu',0) dnu'

Generates one plot per cell (expansion / compression) comparing the
numerical and analytical spectra.
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
    # fallback: simple trapezoidal
    xs = np.linspace(a, b, 500)
    ys = np.array([planck_spectral_density(x, T) for x in xs])
    return float(np.trapz(ys, xs))


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


def main():
    parser = argparse.ArgumentParser(description="Doppler MC spectrum check")
    parser.add_argument("--spectrum", required=True, help="Path to doppler_mc_spectrum.txt")
    parser.add_argument("--max-l1", type=float, default=0.15,
                        help="Maximum allowed relative L1 error")
    parser.add_argument("--plot-dir", default=None,
                        help="Directory for plots (default: same as spectrum)")
    args = parser.parse_args()

    meta, cols = read_spectrum(args.spectrum)

    K_left   = meta["K_left"]
    K_right  = meta["K_right"]
    t_final  = meta["t_final"]
    T_kelvin = meta["T_kelvin"]
    E_cut_lo = meta["E_trunc_lo"]
    E_cut_hi = meta["E_trunc_hi"]

    G = cols.shape[0]
    E_lo  = cols[:, 1]
    E_hi  = cols[:, 2]
    init_left   = cols[:, 3]
    init_right  = cols[:, 4]
    final_left  = cols[:, 5]
    final_right = cols[:, 6]

    E_boundaries = np.zeros(G + 1)
    E_boundaries[:G] = E_lo
    E_boundaries[G] = E_hi[-1]

    anal_left  = compute_analytical(E_boundaries, K_left,  t_final, T_kelvin, E_cut_lo, E_cut_hi)
    anal_right = compute_analytical(E_boundaries, K_right, t_final, T_kelvin, E_cut_lo, E_cut_hi)

    # Relative L1 over groups where analytical prediction is non-negligible
    def rel_l1(numerical, analytical):
        mask = analytical > 1e-30 * analytical.max()
        if not np.any(mask):
            return 0.0
        return np.sum(np.abs(numerical[mask] - analytical[mask])) / np.sum(analytical[mask])

    err_left  = rel_l1(final_left,  anal_left)
    err_right = rel_l1(final_right, anal_right)

    print(f"Doppler MC check:")
    print(f"  K_left  = {K_left:.6e},  K_right = {K_right:.6e}")
    print(f"  K*t     = {K_left*t_final:.4f} (left),  {K_right*t_final:.4f} (right)")
    print(f"  L1 error (left / expansion)   = {err_left:.6f}")
    print(f"  L1 error (right / compression) = {err_right:.6f}")
    print(f"  Threshold = {args.max_l1}")

    # --- plots ---
    plot_dir = args.plot_dir or os.path.dirname(args.spectrum)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        E_centers = 0.5 * (E_lo + E_hi)
        widths = E_hi - E_lo
        kev = 1.602176634e-9  # erg per keV

        for label, final, anal, K, err in [
            ("expansion (left cell)", final_left, anal_left, K_left, err_left),
            ("compression (right cell)", final_right, anal_right, K_right, err_right),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            mask = (anal > 1e-30 * anal.max()) | (final > 1e-30 * max(final.max(), 1e-300))
            ec = E_centers[mask] / kev
            ax.plot(ec, (final[mask] / widths[mask]),
                    "o", ms=3, label="MC numerical")
            ax.plot(ec, (anal[mask] / widths[mask]),
                    "-", lw=1.5, label="Analytical (Eq. V.31)")

            init = init_left if "left" in label else init_right
            init_mask = init > 1e-30 * init.max()
            if np.any(init_mask):
                ax.plot(E_centers[init_mask] / kev, init[init_mask] / widths[init_mask],
                        "--", lw=1, alpha=0.5, label="Initial")

            ax.set_xlabel("Photon energy [keV]")
            ax.set_ylabel(r"$E_\nu$ [erg cm$^{-3}$ erg$^{-1}$]")
            ax.set_title(f"Doppler MC — {label}\nL1 = {err:.4f}, Kt = {K*t_final:.3f}")
            ax.legend(fontsize=8)
            ax.set_yscale("log")
            ax.set_xlim(0, 20)
            yvals = anal[mask] / widths[mask]
            if len(yvals) > 0 and yvals.max() > 0:
                ax.set_ylim(yvals.max() * 1e-6, yvals.max() * 10)
            fig.tight_layout()
            tag = "left" if "left" in label else "right"
            fig_path = os.path.join(plot_dir, f"doppler_mc_{tag}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  Plot: {fig_path}")
    except ImportError:
        print("  matplotlib not available — skipping plots")

    max_err = max(err_left, err_right)
    if max_err > args.max_l1:
        print(f"FAIL: max L1 error {max_err:.6f} exceeds threshold {args.max_l1}")
        return 1

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
