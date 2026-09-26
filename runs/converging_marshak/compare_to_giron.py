"""Reproduce Fig. 15 of Giron, Bennett, Steinberg, McClarren & Krief.

"Converging Marshak waves in nonhomogeneous media: similarity solutions of the
second kind of the radiation diffusion equation in curved geometries."

The paper shows temperature and total energy density profiles at three times at
which the wave has travelled 20%, 60% and 100% of its final distance. This script
overlays the RICH snapshots on the similarity solution at those same times, using
the fitted W and V profiles of the paper's Table III.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

ARAD = 7.565733250033928e-15
KEV_KELVIN = 1.160451812e7
HEV_KELVIN = 0.1 * KEV_KELVIN

# Sec. IV D of the paper. `snapshots` are the times of the paper's profile figures.
PAPER = {
    2: dict(radius=5.0e-2, delta=0.51765, beta=2.0, mu=0.6, omega=-0.5, densityCoefficient=1.0,
            energyCoefficient=3.0e13, scale=HEV_KELVIN, unit="HeV",
            profileCoefficient=0.809892, profileTimeExponent=0.100238,
            snapshots=[-58.251607, -19.068532, -1.0], figure="Fig. 15"),
    4: dict(radius=10.0, delta=0.462367, beta=4.0, mu=1.0, omega=-1.0, densityCoefficient=1.0,
            energyCoefficient=1.25 * ARAD * KEV_KELVIN ** 4, scale=KEV_KELVIN, unit="keV",
            profileCoefficient=0.552154, profileTimeExponent=0.242705,
            snapshots=[-94.706889, -27.126998, -1.0], figure="Fig. 19"),
}


def fitted_w(caseNumber, xi):
    """Table III of the paper."""
    xi = np.asarray(xi, dtype=float)
    w = np.zeros_like(xi)
    heated = xi > 1.0
    x = xi[heated]
    if caseNumber == 2:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 0.3977 * (1.244 - 0.1757 * x + 0.03186 * x * x),
                             (x - 1.0) ** 0.3401 * (1.021 - 0.0007123 * x + 0.0001726 * x * x))
    else:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 1.141 * (0.2251 + 0.127 * x + 0.001626 * x * x),
                             (x - 1.0) ** 1.102 * (0.1846 + 0.1505 * x + 0.00004394 * x * x))
    return w


def similarity_solution(caseNumber, radius, paperTimeNs):
    """Temperature [Eq. 64 / 70] and total energy density [Eq. 5] of the similarity solution."""
    p = PAPER[caseNumber]
    front = 0.1 * p["radius"] * (-paperTimeNs) ** p["delta"]
    scaledTemperature = (p["profileCoefficient"] * (-paperTimeNs) ** p["profileTimeExponent"]
                         * fitted_w(caseNumber, np.asarray(radius) / front) ** (1.0 / p["beta"]))
    density = p["densityCoefficient"] * np.maximum(radius, p["radius"] * 1.0e-12) ** (-p["omega"])
    energy = p["energyCoefficient"] * scaledTemperature ** p["beta"] * density ** (1.0 - p["mu"])
    return scaledTemperature * p["scale"], energy, front


def read_csv_columns(path):
    """Whole-file numeric parse. genfromtxt takes minutes on these 8-million-row files."""
    with open(path) as handle:
        handle.readline()
        header = [name.strip() for name in handle.readline().split(",")]
        body = handle.read()
    values = np.fromstring(body.replace("\n", ","), sep=",")
    columns = len(header)
    values = values[: (values.size // columns) * columns].reshape(-1, columns)
    return {name: values[:, i] for i, name in enumerate(header)}


def load_snapshot(path, radiusLimit, decimals):
    columns = read_csv_columns(path)
    radius = columns["radius_cm"]
    keep = radius < radiusLimit
    radius = radius[keep]
    total = (columns["umat_erg_cm3"] + columns["erad_erg_cm3"])[keep]
    temperature = columns["Tmat_K"][keep]
    _, inverse = np.unique(np.round(radius, decimals), return_inverse=True)
    count = np.bincount(inverse).astype(float)
    mean = lambda v: np.bincount(inverse, weights=v) / count
    square = mean(temperature ** 2)
    shellTemperature = mean(temperature)
    return (mean(radius), count, shellTemperature,
            np.sqrt(np.maximum(square - shellTemperature ** 2, 0.0)), mean(total))


def crossing_radius(radius, temperature, level):
    """Outermost radius at which the profile falls through `level`, linearly interpolated.

    The wave converges inwards, so the heated side is at large radius and the front is
    the innermost edge. Comparing both profiles at a fixed temperature level, rather
    than at "where the temperature leaves zero", keeps the measure well posed against
    the analytic slope discontinuity.
    """
    above = temperature >= level
    if not above.any() or above.all():
        return np.nan
    first = int(np.argmax(above))
    if first == 0:
        return np.nan
    low, high = first - 1, first
    weight = (level - temperature[low]) / (temperature[high] - temperature[low])
    return radius[low] + weight * (radius[high] - radius[low])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, default=2)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--snapshots", default="1,2")
    parser.add_argument("--label", default="RICH")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    caseNumber = arguments.case
    p = PAPER[caseNumber]
    decimals = 6 if p["radius"] < 1.0 else 5
    indices = [int(v) for v in arguments.snapshots.split(",")]
    colours = ["tab:blue", "tab:orange", "tab:green"]

    figure, axes = plt.subplots(3, 1, figsize=(9.0, 11.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 3, 2]})
    print(f"Comparison with Giron et al., {p['figure']}, test {caseNumber}\n")
    for index in indices:
        path = Path(arguments.input_dir) / f"test{caseNumber}_snapshot{index}.csv"
        if not path.exists():
            print(f"  snapshot {index}: not written yet ({path.name})")
            continue
        paperTime = p["snapshots"][index - 1]
        radius, count, temperature, spread, energy = load_snapshot(path, p["radius"], decimals)
        analyticTemperature, analyticEnergy, front = similarity_solution(caseNumber, radius, paperTime)

        heated = analyticTemperature > 0.05 * analyticTemperature.max()
        temperatureResidual = temperature[heated] / analyticTemperature[heated] - 1.0
        energyResidual = energy[heated] / analyticEnergy[heated] - 1.0
        # The test 4 mesh is radially graded, so the cell width must be read off near
        # the front rather than averaged over the whole system.
        spacing = np.diff(radius)
        nearest = int(np.argmin(np.abs(radius[:-1] - front)))
        cellWidth = np.median(spacing[max(nearest - 10, 0):nearest + 10])
        # Away from the front the profile is smooth, so a residual there measures the
        # solution itself rather than a sub-cell misplacement of a near-vertical edge.
        smooth = heated & (radius > front + 10.0 * cellWidth)
        if not smooth.any():
            smooth = heated
        smoothResidual = temperature[smooth] / analyticTemperature[smooth] - 1.0

        level = 0.5 * analyticTemperature.max()
        offset = crossing_radius(radius, temperature, level) - crossing_radius(radius, analyticTemperature, level)

        fitMask = temperature > 0.2 * temperature.max()

        def mismatch(candidate):
            model, _, _ = similarity_solution(caseNumber, radius, candidate)
            return float(np.sum((model[fitMask] - temperature[fitMask]) ** 2))

        effective = minimize_scalar(mismatch, bounds=(paperTime * 30.0, paperTime * 0.3),
                                    method="bounded").x

        print(f"  snapshot {index}: t = {paperTime:.6f} ns, analytic front = {front:.6f} cm")
        print(f"      {radius.size} shells x {int(count.min())} cells, cell width {cellWidth:.3e} cm")
        print(f"      more than 10 cells behind the front: dT/T median "
              f"{100*np.median(smoothResidual):+.2f}%, |dT/T| max {100*np.abs(smoothResidual).max():.2f}%")
        print(f"      whole heated region:             |dT/T| median "
              f"{100*np.median(np.abs(temperatureResidual)):.2f}%, |du/u| median "
              f"{100*np.median(np.abs(energyResidual)):.2f}%")
        print(f"      front offset at half maximum: {1.0e4*offset:+.2f} um = "
              f"{offset/cellWidth:+.2f} cells ({100*offset/front:+.2f}% of r_F)")
        print(f"      effective time {effective:.4f} ns ({effective - paperTime:+.4f} ns)\n")

        colour = colours[index - 1]
        label = f"t = {paperTime:g} ns"
        axes[0].plot(radius, analyticTemperature / p["scale"], "-", color=colour, linewidth=2.0,
                     label=f"Giron et al., {label}")
        axes[0].plot(radius, temperature / p["scale"], "o", color=colour, markersize=3.0,
                     markerfacecolor="none", label=f"{arguments.label}, {label}")
        axes[1].plot(radius, analyticEnergy / 1.0e13, "-", color=colour, linewidth=2.0)
        axes[1].plot(radius, energy / 1.0e13, "o", color=colour, markersize=3.0, markerfacecolor="none")
        axes[2].plot(radius[heated], 100.0 * temperatureResidual, "-", color=colour, linewidth=1.2)

    axes[2].axhline(0.0, color="k", linewidth=1.0)
    axes[2].set_ylim(-20.0, 20.0)
    axes[0].set_ylabel(f"T [{p['unit']}]")
    axes[1].set_ylabel(r"u [$10^{13}$ erg/cm$^3$]")
    axes[2].set_ylabel("T residual [%]")
    axes[2].set_xlabel("r [cm]")
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(0.0, p["radius"])
    figure.suptitle(f"Test {caseNumber}: {arguments.label} vs Giron et al. similarity solution "
                    f"({p['figure']})")
    figure.tight_layout()
    figure.savefig(arguments.out, dpi=180, bbox_inches="tight")
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
