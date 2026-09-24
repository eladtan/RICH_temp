#!/usr/bin/env python3
"""Compare a RICH Mach-2 profile to the Lowrie & Edwards analytic solution
and to the digitized DIMC curve of Steinberg & Heizler 2022, Fig. 9(a).

The three data sets do not share a frame:

  * the RICH profile and the analytic table use simulation coordinates, with
    the downstream state at small x and the upstream state at large x;
  * the digitized Fig. 9(a) curves are mirrored, with upstream at small x, and
    carry the arbitrary x offset of the digitizer;
  * the RICH shock front drifts away from its initial position at x = 0.

Everything is therefore replotted against the signed distance from the shock
front, oriented like the published figure: upstream on the left, downstream on
the right.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEV_KELVIN = 1.16045250e7
RHO_UPSTREAM = 1.0
RHO_DOWNSTREAM = 2.29

PANELS = [
    ("T_gas", r"$T_{\mathrm{mat}}$ (keV)"),
    ("T_rad", r"$T_{\mathrm{rad}}$ (keV)"),
    ("rho", r"$\rho$ (g cm$^{-3}$)"),
    ("vx", r"$v_x$ (cm s$^{-1}$)"),
]


def load_rich(path):
    time_ns = None
    with open(path) as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            match = re.search(r"t_ns=([\d.eE+\-]+)", line)
            if match:
                time_ns = float(match.group(1))
    data = np.loadtxt(path, delimiter=",", comments="#")
    return dict(x=data[:, 0], rho=data[:, 1], T_gas=data[:, 2], T_rad=data[:, 3],
                vx=data[:, 4], time_ns=time_ns, path=path)


def load_analytic(path):
    data = np.loadtxt(path, comments="#")
    return dict(x=data[:, 0], rho=data[:, 1], T_gas=data[:, 2] / KEV_KELVIN,
                T_rad=data[:, 3] / KEV_KELVIN, vx=data[:, 4])


def load_digitized(path):
    data = np.loadtxt(path, delimiter=",", comments="#")
    result = dict(x=data[:, 0], T_gas=data[:, 1], T_rad=data[:, 2], rho=data[:, 3],
                  vx=data[:, 4])
    return result


def front_position(x, rho, downstream_on_left):
    """Locate the front where the density crosses the mean of the two states.

    Only the crossing closest to the downstream boundary is used, so the
    outflow-boundary transients at the far end of the domain are ignored.
    """
    target = 0.5 * (RHO_UPSTREAM + RHO_DOWNSTREAM)
    sign = np.sign(rho - target)
    crossings = np.where(np.diff(sign) != 0)[0]
    if crossings.size == 0:
        raise RuntimeError("no density crossing of the mean state; is the shock inside the domain?")
    index = crossings[0] if downstream_on_left else crossings[-1]
    x0, x1 = x[index], x[index + 1]
    rho0, rho1 = rho[index], rho[index + 1]
    return x0 + (target - rho0) * (x1 - x0) / (rho1 - rho0)


def transition_width(x, rho, downstream_on_left, low=1.05, high=2.0):
    """Distance over which the density rises between two fixed levels.

    Absolute levels are used rather than fractions of the jump, because a run
    that has not yet relaxed never reaches the full downstream density.
    """
    positions = []
    for target in (low, high):
        sign = np.sign(rho - target)
        crossings = np.where(np.diff(sign) != 0)[0]
        if crossings.size == 0:
            return float("nan")
        index = crossings[0] if downstream_on_left else crossings[-1]
        x0, x1 = x[index], x[index + 1]
        rho0, rho1 = rho[index], rho[index + 1]
        positions.append(x0 + (target - rho0) * (x1 - x0) / (rho1 - rho0))
    return abs(positions[1] - positions[0])


def mirror(dataset):
    """Copy of a data set with the velocity sign appropriate to a mirrored x axis."""
    mirrored = dict(dataset)
    mirrored["vx"] = -dataset["vx"]
    return mirrored


def save(figure, output):
    stem = os.path.splitext(output)[0]
    figure.tight_layout()
    for suffix in (".png", ".pdf"):
        figure.savefig(stem + suffix, dpi=200, bbox_inches="tight")
        print(f"Wrote {stem + suffix}")
    plt.close(figure)


def find_profile(explicit=None):
    if explicit:
        return explicit
    candidates = sorted(glob.glob(os.path.join(HERE, "*_final.txt")))
    return candidates[-1] if candidates else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("profile", nargs="?", help="RICH Mach-2 profile (*_final.txt)")
    parser.add_argument("--output", default=os.path.join(HERE, "mach2_comparison.png"),
                        help="figure path; PNG and PDF are both written")
    parser.add_argument("--window", type=float, default=0.09,
                        help="half-width in cm of the shock-front zoom")
    parser.add_argument("--title-suffix", default="", help="run details appended to figure titles")
    arguments = parser.parse_args()

    path = find_profile(arguments.profile)
    if path is None or not os.path.exists(path):
        print("No RICH Mach-2 profile found. Pass a *_final.txt file.", file=sys.stderr)
        return 1

    rich = load_rich(path)
    analytic = load_analytic(os.path.join(HERE, "mach2_analytic_ic2.dat"))
    dimc = load_digitized(
        os.path.join(HERE, "reference", "fig9a_dimc.csv"),
    )

    # The RICH profile and the analytic table share the simulation orientation;
    # the digitized curve is mirrored, so its front is the last crossing.
    front_rich = front_position(rich["x"], rich["rho"], downstream_on_left=True)
    front_analytic = front_position(analytic["x"], analytic["rho"], downstream_on_left=True)
    front_dimc = front_position(dimc["x"], dimc["rho"], downstream_on_left=False)

    # Reorienting a data set to the published left-to-right layout mirrors the
    # x axis, so its velocity has to change sign with it.
    rich_xi, rich_plot = front_rich - rich["x"], mirror(rich)
    analytic_xi, analytic_plot = front_analytic - analytic["x"], mirror(analytic)
    dimc_xi, dimc_plot = dimc["x"] - front_dimc, dimc

    width_rich = transition_width(rich["x"], rich["rho"], downstream_on_left=True)
    width_analytic = transition_width(analytic["x"], analytic["rho"], downstream_on_left=True)

    print(f"RICH profile        : {path}")
    print(f"time                : {rich['time_ns']} ns")
    print(f"front position      : x = {front_rich:+.5f} cm "
          f"(analytic initial front at x = {front_analytic:+.5f} cm)")
    print(f"front drift         : {front_rich - front_analytic:+.5f} cm")
    print(f"rho 1.05-2.0 width  : RICH {width_rich:.5f} cm, analytic {width_analytic:.5f} cm")

    upstream = rich["x"] > front_rich + 0.5 * arguments.window
    if np.any(upstream):
        print(f"upstream state      : rho = {np.median(rich['rho'][upstream]):.4f}, "
              f"T_mat = {np.median(rich['T_gas'][upstream]):.4f} keV "
              f"(expected {RHO_UPSTREAM:.4f}, 0.1220 keV)")
    print(f"downstream extreme  : rho = {rich['rho'].max():.4f}, "
          f"T_mat = {rich['T_gas'].max():.4f} keV "
          f"(expected {RHO_DOWNSTREAM:.4f}, 0.2530 keV)")

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    for axis, (key, ylabel) in zip(axes.flat, PANELS):
        axis.plot(analytic_xi, analytic_plot[key], "m-", lw=1.8, label="Lowrie & Edwards analytic")
        axis.plot(dimc_xi, dimc_plot[key], "g--", lw=1.5, label="DIMC, Fig. 9(a)")
        axis.plot(rich_xi, rich_plot[key], "C0-", lw=1.1, label="RICH IMC")
        axis.axvline(0.0, color="0.6", lw=0.8, ls=":")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel(r"distance from shock front (cm), downstream $>0$")
    axes[0, 0].set_xlim(-arguments.window, arguments.window)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    title = "Mach 2 radiative shock, shock-front frame"
    if rich["time_ns"] is not None:
        title += f"  ($t = {rich['time_ns']:.2f}$ ns)"
    figure.suptitle(title)
    if arguments.title_suffix:
        figure.suptitle(title + "\n" + arguments.title_suffix)
    save(figure, arguments.output)

    figure, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), sharex=True)
    for axis, (key, ylabel) in zip(axes.flat, PANELS):
        axis.plot(analytic["x"], analytic[key], "m-", lw=1.5, label="analytic initial condition")
        axis.plot(rich["x"], rich[key], "C0-", lw=1.1, label="RICH IMC")
        axis.axvline(front_rich, color="0.6", lw=0.8, ls=":")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
    for axis in axes[1]:
        axis.set_xlabel(r"$x$ (cm)")
    axes[0, 0].set_xlim(rich["x"].min(), rich["x"].max())
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper right")
    figure.suptitle("Mach 2 radiative shock, full domain in simulation coordinates")
    if arguments.title_suffix:
        figure.suptitle("Mach 2 radiative shock, full domain in simulation coordinates\n" + arguments.title_suffix)
    save(figure, os.path.splitext(arguments.output)[0] + "_domain.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
