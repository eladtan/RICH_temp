"""Independent comparison of a converging-Marshak dump against the similarity solution.

Written from scratch and deliberately self-contained: it shares no code with
compare_profiles.py. The analytic profile is rebuilt from the coefficients and the
W/V fits that live in test.cpp, rather than from the interpolated author table, so
that the two paths can be cross-checked against each other.

Beyond the profile comparison it fits an "effective paper time" to the simulated
profile. If the simulated wave is simply the analytic wave evaluated at a shifted
time, the discrepancy is a timing/labelling issue; if the shape differs, it is not.
"""

import argparse
import os
import re
import xml.etree.ElementTree as ElementTree
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import vtk
from vtk.util import numpy_support as vtk_np

ARAD = 7.565733250033928e-15
KEV_KELVIN = 1.160451812e7
HEV_KELVIN = 0.1 * KEV_KELVIN

# Transcribed from GetBenchmark() in test.cpp.
BENCHMARKS = {
    1: dict(radius=1.0e-3, eosCoefficient=3.4e13, beta=1.6, mu=0.14, omega=0.0,
            densityCoefficient=19.3, delta=0.679501, scale=HEV_KELVIN,
            profileCoefficient=1.34503465, profileTimeExponent=0.0920519),
    2: dict(radius=5.0e-2, eosCoefficient=3.0e13, beta=2.0, mu=0.6, omega=-0.5,
            densityCoefficient=1.0, delta=0.51765, scale=HEV_KELVIN,
            profileCoefficient=0.809892, profileTimeExponent=0.100238),
    3: dict(radius=1.0e-3, eosCoefficient=1.0e13, beta=2.0, mu=0.25, omega=0.45,
            densityCoefficient=1.0, delta=1.1157536, scale=HEV_KELVIN,
            profileCoefficient=1.1982, profileTimeExponent=0.0276392),
    4: dict(radius=10.0, eosCoefficient=0.25 * ARAD * KEV_KELVIN ** 4, beta=4.0, mu=1.0,
            omega=-1.0, densityCoefficient=1.0, delta=0.462367, scale=KEV_KELVIN,
            profileCoefficient=0.552154, profileTimeExponent=0.242705),
}


def fitted_w(caseNumber, xi):
    """FittedW() from test.cpp: the similarity temperature variable."""
    xi = np.asarray(xi, dtype=float)
    w = np.zeros_like(xi)
    heated = xi > 1.0
    x = xi[heated]
    if caseNumber == 1:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 0.4057 * (1.521 - 0.3762 * x + 0.06558 * x * x),
                             (x - 1.0) ** 0.2955 * (1.082 - 0.02718 * x + 0.001055 * x * x))
    elif caseNumber == 2:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 0.3977 * (1.244 - 0.1757 * x + 0.03186 * x * x),
                             (x - 1.0) ** 0.3401 * (1.021 - 0.0007123 * x + 0.0001726 * x * x))
    elif caseNumber == 3:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 0.3575 * (1.979 - 0.6195 * x + 0.1106 * x * x),
                             (x - 1.0) ** 0.2101 * (1.27 - 0.04707 * x + 0.001797 * x * x))
    else:
        w[heated] = np.where(x <= 2.0,
                             (x - 1.0) ** 1.141 * (0.2251 + 0.127 * x + 0.001626 * x * x),
                             (x - 1.0) ** 1.102 * (0.1846 + 0.1505 * x + 0.00004394 * x * x))
    return w


def analytic_profile(caseNumber, radius, paperTimeNs):
    """Temperature and total (material + radiation) energy density of the similarity solution."""
    b = BENCHMARKS[caseNumber]
    front = 0.1 * b["radius"] * (-paperTimeNs) ** b["delta"]
    w = fitted_w(caseNumber, np.asarray(radius) / front)
    scaledTemperature = (b["profileCoefficient"] * (-paperTimeNs) ** b["profileTimeExponent"]
                         * w ** (1.0 / b["beta"]))
    temperature = scaledTemperature * b["scale"]
    density = b["densityCoefficient"] * np.maximum(radius, b["radius"] * 1.0e-12) ** (-b["omega"])
    material = b["eosCoefficient"] * scaledTemperature ** b["beta"] * density ** (1.0 - b["mu"])
    return temperature, material + ARAD * temperature ** 4, front


def read_piece(path):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(path)
    reader.Update()
    cellData = reader.GetOutput().GetCellData()
    coordinates = vtk_np.vtk_to_numpy(cellData.GetArray("Coordinates"))
    return (np.linalg.norm(coordinates, axis=1),
            vtk_np.vtk_to_numpy(cellData.GetArray("temperature")),
            vtk_np.vtk_to_numpy(cellData.GetArray("density")),
            np.maximum(vtk_np.vtk_to_numpy(cellData.GetArray("Erad_time_avg")), 0.0))


def read_dump(pvtuPath, workers=24):
    """Read exactly the pieces the manifest declares, never a directory glob."""
    root = os.path.dirname(os.path.abspath(pvtuPath))
    sources = [os.path.join(root, piece.get("Source"))
               for piece in ElementTree.parse(pvtuPath).iter("Piece")]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(read_piece, sources, chunksize=4))
    return len(sources), tuple(np.concatenate([p[i] for p in parts]) for i in range(4))


def shell_reduce(radius, values, decimals):
    _, inverse = np.unique(np.round(radius, decimals), return_inverse=True)
    count = np.bincount(inverse).astype(float)
    reduced = [np.bincount(inverse, weights=v) / count for v in values]
    return count, np.bincount(inverse, weights=radius) / count, reduced


def log_entry(logPath, cycle):
    pattern = re.compile(rf"^Cycle {cycle}\s+\d+%\s+paper t=(-?[\d.]+) ns.*front=([\d.e+-]+) cm")
    for line in open(logPath):
        match = pattern.match(line)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--pvtu", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dt-ns", type=float, default=0.01)
    arguments = parser.parse_args()

    caseNumber = arguments.case
    b = BENCHMARKS[caseNumber]
    radiusLimit = b["radius"]

    paperTime, logFront = log_entry(arguments.log, arguments.cycle)
    initialPaperTime = -(10.0 ** (1.0 / b["delta"]))
    reconstructedTime = initialPaperTime + arguments.cycle * arguments.dt_ns
    print(f"paper time from log        : {paperTime:.6f} ns   (front {logFront:.6f} cm)")
    print(f"paper time from cycle*dt   : {reconstructedTime:.6f} ns   "
          f"(t0 = {initialPaperTime:.6f} ns, dt = {arguments.dt_ns} ns/cycle)")
    print(f"disagreement               : {abs(paperTime - reconstructedTime)/arguments.dt_ns:.2f} cycles")

    npieces, (radius, temperature, density, erad) = read_dump(arguments.pvtu)
    inside = radius < radiusLimit
    radius, temperature, density, erad = (radius[inside], temperature[inside],
                                          density[inside], erad[inside])
    decimals = 6 if radiusLimit < 1.0 else 5
    count, shellRadius, (shellTemperature, shellDensity, shellErad) = shell_reduce(
        radius, [temperature, density, erad], decimals)
    print(f"\nread {npieces} pieces, {radius.size} cells, {shellRadius.size} shells "
          f"x {int(count.min())} cells")

    shellEnergy = (b["eosCoefficient"] * (shellTemperature / b["scale"]) ** b["beta"]
                   * shellDensity ** (1.0 - b["mu"]) + ARAD * shellTemperature ** 4)
    analyticTemperature, analyticEnergy, front = analytic_profile(caseNumber, shellRadius, paperTime)

    # Effective paper time: the time at which the analytic wave best matches the simulated one.
    fit_mask = shellTemperature > 0.2 * shellTemperature.max()
    # Residuals are only meaningful where the analytic profile is non-zero, which
    # excludes the shells lying ahead of the analytic front.
    heated = fit_mask & (analyticTemperature > 0.0)

    def mismatch(candidateTime):
        model, _, _ = analytic_profile(caseNumber, shellRadius, candidateTime)
        return float(np.sum((model[fit_mask] - shellTemperature[fit_mask]) ** 2))

    fit = minimize_scalar(mismatch, bounds=(paperTime * 2.0, paperTime * 0.4), method="bounded")
    effectiveTime = fit.x
    lagCycles = (effectiveTime - paperTime) / arguments.dt_ns
    fittedTemperature, fittedEnergy, fittedFront = analytic_profile(caseNumber, shellRadius, effectiveTime)
    print(f"\neffective paper time of the simulated wave: {effectiveTime:.6f} ns")
    print(f"  vs labelled {paperTime:.6f} ns  ->  {effectiveTime - paperTime:+.6f} ns "
          f"= {lagCycles:+.1f} cycles")
    print(f"  analytic front {front:.6f} cm  ->  best-fit front {fittedFront:.6f} cm")
    residualAtLabel = np.abs(shellTemperature[heated] / analyticTemperature[heated] - 1.0).max()
    residualAtFit = np.abs(shellTemperature[heated] / fittedTemperature[heated] - 1.0).max()
    print(f"  max |T residual| at labelled time {100*residualAtLabel:.2f}%, "
          f"at best-fit time {100*residualAtFit:.2f}%")

    scale = 1.0e4 if radiusLimit <= 1.0e-3 else 1.0
    unit = "micron" if radiusLimit <= 1.0e-3 else "cm"
    x = shellRadius * scale
    unitName = "keV" if caseNumber == 4 else "HeV"

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 11.5), sharex=True,
                                gridspec_kw={"height_ratios": [3, 3, 2]})
    axes[0].plot(x, analyticTemperature / b["scale"], "k-", linewidth=2.0,
                 label=f"Similarity solution, t = {paperTime:g} ns")
    axes[0].plot(x, fittedTemperature / b["scale"], "--", color="tab:green", linewidth=1.5,
                 label=f"Similarity solution, t = {effectiveTime:.3f} ns (best fit)")
    axes[0].plot(x, shellTemperature / b["scale"], "o", color="tab:blue", markersize=3.0,
                 alpha=0.85, label=f"RICH, cycle {arguments.cycle}")
    axes[0].set_ylabel(f"Temperature [{unitName}]")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, analyticEnergy / 1.0e13, "k-", linewidth=2.0)
    axes[1].plot(x, fittedEnergy / 1.0e13, "--", color="tab:green", linewidth=1.5)
    axes[1].plot(x, shellEnergy / 1.0e13, "o", color="tab:blue", markersize=3.0, alpha=0.85)
    axes[1].set_ylabel(r"Total energy density [$10^{13}$ erg cm$^{-3}$]")
    axes[2].axhline(0.0, color="k", linewidth=1.0)
    axes[2].plot(x[heated], 100.0 * (shellTemperature[heated] / analyticTemperature[heated] - 1.0),
                 "o-", color="tab:red", markersize=3.0, linewidth=0.8, label="vs labelled time")
    axes[2].plot(x[heated], 100.0 * (shellTemperature[heated] / fittedTemperature[heated] - 1.0),
                 "s-", color="tab:green", markersize=2.5, linewidth=0.8, label="vs best-fit time")
    axes[2].set_ylim(-30.0, 30.0)
    axes[2].set_ylabel("T residual [%]")
    axes[2].set_xlabel(f"Radius [{unit}]")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.axvline(front * scale, color="tab:red", linestyle=":", linewidth=1.0)
        axis.axvline(fittedFront * scale, color="tab:green", linestyle=":", linewidth=1.0)
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0.0)
    figure.suptitle(f"Converging Marshak test {caseNumber}, cycle {arguments.cycle} "
                    f"({npieces} pieces, rebuilt from test.cpp fits)")
    figure.tight_layout()
    figure.savefig(arguments.out, dpi=180, bbox_inches="tight")
    print(f"\nwrote {arguments.out}")


if __name__ == "__main__":
    main()
