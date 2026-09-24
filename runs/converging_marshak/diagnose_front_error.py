"""Diagnose the origin of the converging-Marshak front-position error.

Correlates the optical thickness of a radial cell, the excess energy held inside
the sphere, and the front-position offset, all against run time.
"""

import glob
import re
import sys
import numpy as np
from scipy import integrate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_profiles import CASES, author_solution
from track_front import level_radius, analytic_level_radius

TABLE = np.genfromtxt("author_table_ii.csv", delimiter=",", names=True)
CELL_WIDTH = 1.95313e-4


def log_series(logPath):
    pattern = re.compile(r"^Cycle (\d+)\s+\d+%\s+paper t=(-?[\d.]+) ns.*front=([\d.e+-]+) cm"
                         r".*bath_mfp=([\d.e+-]+) cm")
    cycles, times, fronts, mfps = [], [], [], []
    for line in open(logPath):
        match = pattern.match(line)
        if match:
            cycles.append(int(match.group(1)))
            times.append(float(match.group(2)))
            fronts.append(float(match.group(3)))
            mfps.append(float(match.group(4)))
    return (np.array(cycles), np.array(times), np.array(fronts), np.array(mfps))


if __name__ == "__main__":
    caseNumber = 2
    case = CASES[caseNumber]
    radiusLimit = case["radius"]
    cycles, times, fronts, mfps = log_series("marshak-t2-4xyz-ddmc-10148695.out")
    timeByCycle = dict(zip(cycles, zip(times, fronts)))
    level = 0.25 * case["scale_k"]

    dumpCycles, dumpTimes, excessEnergy, ratioEnergy, frontOffset = [], [], [], [], []
    for path in sorted(glob.glob("/tmp/front_cache/case2_*.npz")):
        cycle = int(path.split("_")[-1].split(".")[0])
        if cycle not in timeByCycle:
            continue
        paperTime, logFront = timeByCycle[cycle]
        data = np.load(path)
        shellRadius, temperature, density = data["shell_radius"], data["temperature"], data["density"]
        spacing = np.diff(shellRadius).mean()
        edges = np.concatenate([[max(shellRadius[0] - spacing / 2.0, 0.0)],
                                0.5 * (shellRadius[:-1] + shellRadius[1:]), [radiusLimit]])
        volume = 4.0 * np.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
        material = (case["energy_coefficient"] * (temperature / case["scale_k"]) ** case["beta"]
                    * density ** (1.0 - case["mu"]))
        simulated = np.sum((material + data["erad"]) * volume)
        fine = np.linspace(0.0, radiusLimit, 400000)
        _, authorEnergy, _ = author_solution(caseNumber, fine, paperTime, TABLE)
        analytic = integrate.simpson(authorEnergy * 4.0 * np.pi * fine ** 2, x=fine)
        dumpCycles.append(cycle)
        dumpTimes.append(paperTime)
        excessEnergy.append(simulated - analytic)
        ratioEnergy.append(simulated / analytic)
        frontOffset.append(1.0e4 * (level_radius(shellRadius, temperature, level)
                                    - analytic_level_radius(caseNumber, paperTime, level, logFront)))

    dumpTimes = np.array(dumpTimes)
    figure, axes = plt.subplots(3, 1, figsize=(8.5, 11.0), sharex=True)
    axes[0].plot(-times, CELL_WIDTH / mfps, color="tab:purple", linewidth=1.5)
    axes[0].axhline(1.0, color="k", linestyle=":", linewidth=1.0)
    axes[0].axhspan(1.0, 6.0, color="tab:red", alpha=0.10)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cell thickness [mean free paths]")
    axes[0].text(0.98, 0.92, "shaded: cell thicker than one mfp\n(IMC teleportation regime)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8)
    axes[1].plot(-dumpTimes, np.array(excessEnergy) / 1.0e7, "o-", color="tab:orange", markersize=4.0)
    axes[1].axhline(0.0, color="k", linewidth=1.0)
    axes[1].set_ylabel(r"Excess energy in sphere [$10^7$ erg]")
    axes[2].plot(-dumpTimes, frontOffset, "o-", color="tab:red", markersize=4.0)
    axes[2].axhline(0.0, color="k", linewidth=1.0)
    axes[2].set_ylabel("Front offset [micron]")
    axes[2].set_xlabel("-t [ns]   (run proceeds right to left)")
    secondary = axes[2].secondary_yaxis("right", functions=(lambda v: v / (CELL_WIDTH * 1.0e4),
                                                            lambda v: v * CELL_WIDTH * 1.0e4))
    secondary.set_ylabel("Front offset [cells]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    figure.suptitle("Test 2: the front error is set during the optically thick start-up")
    figure.tight_layout()
    out = "current_results/test2_front_error_diagnosis.png"
    figure.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
    print("\n cycle    t[ns]   cell/mfp   excess[erg]   E_ratio   offset[um]  offset[cells]")
    for cycle, paperTime, excess, ratio, offset in zip(dumpCycles, dumpTimes, excessEnergy, ratioEnergy, frontOffset):
        thickness = CELL_WIDTH / mfps[np.argmin(np.abs(cycles - cycle))]
        print(f"{cycle:6d} {paperTime:9.3f} {thickness:9.2f} {excess:13.3e} {ratio:9.4f} {offset:11.2f} {offset/(CELL_WIDTH*1e4):13.2f}")
