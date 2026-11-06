#!/usr/bin/env python3
"""Per-band polarization summary of a post-process result, by viewing direction.

Four panels against each energy group's mid frequency (arithmetic mean of the
group edges): the median debiased polarization degree, the median luminosity
per observer patch, the median polarization S/N, and the median ratio of the
thermalization radius to the photosphere radius. Each median is over the
observers (or the source-surface rays) inside a cone around one of six
directions: +-x, +-y, +-z.

Polarization uses the generation statistics of the final generations
(statistics_state): q = sum Q / sum I and u likewise; their errors from the
ratio estimator over generations (intensity covariance included);
sigma_c = sqrt((sigma_q^2 + sigma_u^2) / 2); P_db = sqrt(max(0, P^2 -
2 sigma_c^2)); S/N = P_db / sigma_c. As in the code, a bin needs a
split-aware Kish n_eff (sum I)^2 / sum k w^2 of at least 10 over the final
generations: bins below are left out of the P median and count as S/N = 0 in
the S/N median. A P point needs at least MIN_OBSERVERS such bins.

HDF5 is read through h5dump, so no h5py is needed.

usage: plot_band_polarization.py RESULT.h5 [--out PNG] [--csv CSV]
                                 [--cone DEG] [--title TEXT]
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H_PLANCK = 6.62607015e-27   # erg s
C_LIGHT = 2.99792458e10     # cm / s
ANGSTROM = 1e-8             # cm
KEV = 1.602176634e-9        # erg

# Six cones, in the fixed order of the reference categorical palette (slots
# 1-6); each also has its own marker so identity never rests on color alone.
DIRECTIONS = [
    ("+x", np.array([1.0, 0.0, 0.0]), "#2a78d6", "o"),
    ("-x", np.array([-1.0, 0.0, 0.0]), "#eb6834", "s"),
    ("+y", np.array([0.0, 1.0, 0.0]), "#1baf7a", "^"),
    ("-y", np.array([0.0, -1.0, 0.0]), "#eda100", "D"),
    ("+z", np.array([0.0, 0.0, 1.0]), "#e87ba4", "v"),
    ("-z", np.array([0.0, 0.0, -1.0]), "#008300", "P"),
]

# Wavelength bands shaded behind the data (Angstrom).
BANDS = [
    ("optical", 4000.0, 7000.0, "#f6e7d0"),
    ("UV", 2000.0, 4000.0, "#f3dcee"),
    ("EUV", C_LIGHT / (0.3 * KEV / H_PLANCK) / ANGSTROM, 2000.0, "#e3e8ef"),
    ("X-ray", C_LIGHT / (20.0 * KEV / H_PLANCK) / ANGSTROM,
     C_LIGHT / (0.3 * KEV / H_PLANCK) / ANGSTROM, "#e9e0f5"),
]

MIN_EFFECTIVE_PACKETS = 10.0
MIN_OBSERVERS = 3

DTYPES = {
    "H5T_IEEE_F64LE": "<f8", "H5T_IEEE_F64BE": "<f8",
    "H5T_IEEE_F32LE": "<f4", "H5T_IEEE_F32BE": "<f4",
    "H5T_STD_U64LE": "<u8", "H5T_STD_U64BE": "<u8",
    "H5T_STD_I64LE": "<i8", "H5T_STD_I64BE": "<i8",
    "H5T_STD_U32LE": "<u4", "H5T_STD_U32BE": "<u4",
    "H5T_STD_I32LE": "<i4", "H5T_STD_I32BE": "<i4",
    "H5T_STD_U8LE": "u1", "H5T_STD_U8BE": "u1",
    "H5T_STD_I8LE": "i1", "H5T_STD_I8BE": "i1",
}


def read(path_h5, dataset):
    """Dataset as a numpy array (little-endian), shaped by its dataspace or,
    for flat datasets, by the companion '<name>_shape' dataset when present."""
    header = subprocess.run(["h5dump", "-H", "-d", dataset, path_h5],
                            capture_output=True, text=True)
    if header.returncode != 0:
        raise RuntimeError(f"{path_h5}: cannot read {dataset}: {header.stderr.strip()}")
    match = re.search(r"DATATYPE\s+(H5T_\w+)", header.stdout)
    if not match or match.group(1) not in DTYPES:
        raise RuntimeError(f"{dataset}: unsupported datatype in {header.stdout!r}")
    dtype = DTYPES[match.group(1)]
    dims = re.search(r"DATASPACE\s+SIMPLE\s+\{\s*\(\s*([0-9,\s]+)\)", header.stdout)
    shape = tuple(int(x) for x in dims.group(1).split(",")) if dims else ()
    with tempfile.TemporaryDirectory() as tmp:
        raw = os.path.join(tmp, "data.bin")
        dump = subprocess.run(["h5dump", "-b", "LE", "-d", dataset, "-o", raw, path_h5],
                              capture_output=True, text=True)
        if dump.returncode != 0:
            raise RuntimeError(f"{path_h5}: cannot dump {dataset}: {dump.stderr.strip()}")
        values = np.fromfile(raw, dtype=dtype)
    if shape and len(shape) > 1:
        return values.reshape(shape)
    if not shape:
        return values.reshape(())
    try:
        companion = read(path_h5, dataset + "_shape")
        return values.reshape(tuple(int(x) for x in np.atleast_1d(companion)))
    except RuntimeError:
        return values


def ratio_and_error(sum_x, sum_xx, sum_ex, sum_e, sum_ee, generations):
    """R = sum X / sum E over generations and its standard error from the
    ratio estimator (delta method with the X-E covariance)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(sum_e > 0, sum_x / sum_e, 0.0)
        residual = sum_xx - 2.0 * r * sum_ex + r * r * sum_ee
        var = np.where(sum_e > 0,
                       np.maximum(residual, 0.0) * generations /
                       ((generations - 1.0) * sum_e * sum_e), 0.0)
    return r, np.sqrt(var)


def cone_members(directions, axis, half_angle_deg):
    norms = np.linalg.norm(directions, axis=1)
    cosines = directions @ axis / np.where(norms > 0, norms, 1.0)
    return cosines >= np.cos(np.radians(half_angle_deg))


def median_or_nan(values):
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("result", help="post-process result HDF5 (forward pass)")
    parser.add_argument("--out", help="output PNG (default: next to the result)")
    parser.add_argument("--csv", help="table of the plotted medians (default: next to the PNG)")
    parser.add_argument("--cone", type=float, default=30.0, help="cone half-angle in degrees")
    parser.add_argument("--title", help="figure title (default: result name and generations)")
    args = parser.parse_args()

    f = args.result
    p = "/passes/forward"
    s = p + "/statistics_state"
    generations = float(read(f, s + "/samples"))
    if generations < 2:
        sys.exit("need at least two final generations for generation errors")

    def stat(name):
        return read(f, f"{s}/{name}").astype(float)

    sum_e, sum_ee = stat("group_energy/sum"), stat("group_energy/sum_sq")
    sum_q, sum_qq = stat("group_stokes_q/sum"), stat("group_stokes_q/sum_sq")
    sum_u, sum_uu = stat("group_stokes_u/sum"), stat("group_stokes_u/sum_sq")
    sum_eq = stat("group_energy_stokes_q_cross_sum")
    sum_eu = stat("group_energy_stokes_u_cross_sum")
    q, sigma_q = ratio_and_error(sum_q, sum_qq, sum_eq, sum_e, sum_ee, generations)
    u, sigma_u = ratio_and_error(sum_u, sum_uu, sum_eu, sum_e, sum_ee, generations)
    degree = np.hypot(q, u)
    sigma_c = np.sqrt(0.5 * (sigma_q ** 2 + sigma_u ** 2))
    p_db = np.sqrt(np.maximum(0.0, degree ** 2 - 2.0 * sigma_c ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sigma_c > 0, p_db / sigma_c, 0.0)
    weight_sq = stat("group_energy_weight_sq_sum")
    with np.errstate(divide="ignore", invalid="ignore"):
        neff = np.where(weight_sq > 0, sum_e * sum_e / weight_sq, 0.0)
    measured = (sum_e > 0) & (neff >= MIN_EFFECTIVE_PACKETS)
    snr = np.where(measured, snr, 0.0)

    luminosity = read(f, p + "/tally/multigroup/group_luminosity").astype(float)
    edges = read(f, p + "/tally/multigroup/group_boundaries").astype(float) / H_PLANCK
    nu_mid = 0.5 * (edges[:-1] + edges[1:])
    n_groups = nu_mid.size

    observers = np.column_stack([read(f, f"/observer/direction_{c}").astype(float)
                                 for c in "xyz"])
    ss = "/comparison/source_surface"
    rays = np.column_stack([read(f, f"{ss}/directions/{c}").astype(float) for c in "xyz"])
    r_th = read(f, ss + "/groups/radius").astype(float).reshape(n_groups, -1)
    r_ph = read(f, ss + "/groups/photosphere_radius").astype(float).reshape(n_groups, -1)
    ok_th = read(f, ss + "/groups/resolved").reshape(n_groups, -1) != 0
    ok_ph = read(f, ss + "/groups/photosphere_resolved").reshape(n_groups, -1) != 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ok_th & ok_ph & (r_ph > 0), r_th / r_ph, np.nan)

    rows = []
    medians = {}
    for label, axis, _, _ in DIRECTIONS:
        obs = cone_members(observers, axis, args.cone)
        ray = cone_members(rays, axis, args.cone)
        m = {"P": [], "L": [], "SNR": [], "rth_rph": []}
        for g in range(n_groups):
            lit = obs & measured[:, g]
            m["P"].append(median_or_nan(p_db[lit, g]) if lit.sum() >= MIN_OBSERVERS else float("nan"))
            m["L"].append(median_or_nan(luminosity[obs, g]))
            m["SNR"].append(median_or_nan(snr[obs, g]))
            m["rth_rph"].append(median_or_nan(ratio[g, ray]))
            rows.append([label, g, nu_mid[g], int(obs.sum()), int(lit.sum()), int(ray.sum()),
                         m["P"][-1], m["L"][-1], m["SNR"][-1], m["rth_rph"][-1]])
        medians[label] = {k: np.array(v) for k, v in m.items()}

    stem = os.path.splitext(os.path.basename(f))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(f)), stem + "_band_polarization.png")
    table = args.csv or os.path.splitext(out)[0] + ".csv"
    with open(table, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["direction", "group", "nu_mid_Hz", "observers_in_cone",
                         "observers_measured", "rays_in_cone", "median_P_db",
                         "median_L_erg_s", "median_snr", "median_rth_over_rph"])
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#6b6b66",
                         "axes.labelcolor": "#1f1f1e", "xtick.color": "#52524d",
                         "ytick.color": "#52524d"})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharex=True)
    panels = [
        (axes[0, 0], "P", r"median $P_{\rm db}$", True),
        (axes[0, 1], "L", r"median $L$ per observer patch [erg/s]", True),
        (axes[1, 0], "SNR", r"median S/N for $P$", True),
        (axes[1, 1], "rth_rph", r"median $r_{\rm th}/r_{\rm ph}$", False),
    ]
    nu_lo, nu_hi = 1.0e13, 3.0e20
    for ax, key, ylabel, logy in panels:
        for name, lam_lo, lam_hi, color in BANDS:
            ax.axvspan(C_LIGHT / (lam_hi * ANGSTROM), C_LIGHT / (lam_lo * ANGSTROM),
                       color=color, alpha=0.8, lw=0, zorder=0)
        for label, _, color, marker in DIRECTIONS:
            y = medians[label][key].copy()
            if logy:
                y[~(y > 0)] = np.nan
            ax.plot(nu_mid, y, color=color, marker=marker, ms=6, lw=1.6,
                    mec="white", mew=0.8, label=label, zorder=3)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        if key == "SNR":
            ax.axhline(7.0, color="#52524d", lw=1.0, ls="--", zorder=2)
            ax.text(nu_hi / 1.5, 7.0 * 1.08, "S/N = 7", ha="right", va="bottom",
                    color="#52524d", fontsize=9)
        ax.set_xlim(nu_lo, nu_hi)
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", color="#d9d8d2", lw=0.6, zorder=1)
        top = ax.secondary_xaxis("top", functions=(lambda nu: C_LIGHT / np.maximum(nu, 1e-30) / ANGSTROM,
                                                   lambda lam: C_LIGHT / np.maximum(lam, 1e-30) / ANGSTROM))
        top.set_xlabel(r"$\lambda$ ($\AA$)")
    for ax in axes[1]:
        ax.set_xlabel("Frequency (Hz)")
    ax0 = axes[0, 0]
    for name, lam_lo, lam_hi, _ in BANDS:
        nu_c = np.sqrt(C_LIGHT / (lam_hi * ANGSTROM) * C_LIGHT / (lam_lo * ANGSTROM))
        ax0.text(nu_c, 0.97, name, transform=ax0.get_xaxis_transform(), rotation=90,
                 ha="center", va="top", fontsize=9, color="#52524d")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, 0.018))
    fig.text(0.5, 0.004,
             f"Medians over observers / source-surface rays within {args.cone:g}° of each axis. "
             "BaseTDE orbit: plane x-y, pericentre at +x, angular momentum along -z. "
             f"Bins with n_eff < {MIN_EFFECTIVE_PACKETS:g} count as S/N = 0 and are left out of P.",
             ha="center", va="bottom", fontsize=9, color="#52524d")
    title = args.title or f"{stem}, {int(generations)} GEN"
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0.05, 1, 0.98))
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\nwrote {table}")


if __name__ == "__main__":
    main()
