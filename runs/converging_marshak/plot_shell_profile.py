"""Compare a shell-reduced RICH dump against the authors' similarity solution."""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_profiles import CASES, author_solution, normalized_l1

ARAD = 7.565733250033928e-15


def eos_coefficient(caseNumber):
    """Coefficient of the MATERIAL energy density.

    For cases 1-3 the authors' energy coefficient is the material coefficient. For
    case 4 it is 1.25*a*T^4, which already includes the radiation energy, so the
    material part is 0.25*a*T^4.
    """
    case = CASES[caseNumber]
    if caseNumber == 4:
        return 0.25 * ARAD * case["scale_k"] ** 4
    return case["energy_coefficient"]


def plot(caseNumber, npzPath, paperTime, cycleLabel, outPath):
    case = CASES[caseNumber]
    data = np.load(npzPath)
    shellRadius = data["shell_radius"]
    temperature = data["temperature"]
    temperatureStd = data["temperature_std"]
    density = data["density"]
    count = data["count"]
    totalEnergy = (eos_coefficient(caseNumber) * (temperature / case["scale_k"]) ** case["beta"]
                   * density ** (1.0 - case["mu"]) + data["erad"])

    authorTemperature, authorEnergy, front = author_solution(caseNumber, shellRadius, paperTime,
                                                             np.genfromtxt("author_table_ii.csv", delimiter=",", names=True))
    radiusScale = 1.0e4 if case["radius"] <= 1.0e-3 else 1.0
    radiusUnit = "micron" if case["radius"] <= 1.0e-3 else "cm"
    x = shellRadius * radiusScale
    hot = authorTemperature > 0.05 * authorTemperature.max()
    temperatureResidual = temperature[hot] / authorTemperature[hot] - 1.0
    energyResidual = totalEnergy[hot] / authorEnergy[hot] - 1.0

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 11.0), sharex=True, gridspec_kw={"height_ratios": [3, 3, 2]})
    label = f"t={paperTime:g} ns"
    axes[0].plot(x, authorTemperature / case["scale_k"], color="k", linewidth=2.0, label=f"Authors, {label}")
    axes[0].errorbar(x, temperature / case["scale_k"], yerr=temperatureStd / case["scale_k"], color="tab:blue",
                     linestyle="none", marker="o", markersize=3.0, capsize=1.0, elinewidth=0.7, alpha=0.85,
                     label=f"RICH DDMC, {label} ({shellRadius.size} shells x {int(count.min())} cells)")
    axes[1].plot(x, authorEnergy / 1.0e13, color="k", linewidth=2.0)
    axes[1].plot(x, totalEnergy / 1.0e13, color="tab:blue", linestyle="none", marker="o", markersize=3.0, alpha=0.85)
    axes[2].axhline(0.0, color="k", linewidth=1.0)
    axes[2].plot(x[hot], 100.0 * temperatureResidual, color="tab:blue", marker="o", markersize=3.0,
                 linewidth=0.8, label="temperature")
    axes[2].plot(x[hot], 100.0 * energyResidual, color="tab:orange", marker="s", markersize=2.5,
                 linewidth=0.8, label="total energy")
    axes[2].set_ylim(-25.0, 25.0)
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.axvline(front * radiusScale, color="tab:red", linestyle="--", linewidth=1.0, alpha=0.7)
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0.0)
    axes[0].set_ylabel("Temperature [keV]" if caseNumber == 4 else "Temperature [HeV]")
    axes[1].set_ylabel(r"Total energy density [$10^{13}$ erg cm$^{-3}$]")
    axes[2].set_ylabel("Residual vs authors [%]")
    axes[2].set_xlabel(f"Generator radius [{radiusUnit}]")
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Converging Marshak test {caseNumber}: {cycleLabel}")
    figure.tight_layout()
    figure.savefig(outPath, dpi=180, bbox_inches="tight")
    plt.close(figure)

    far = shellRadius[hot] > 1.15 * front
    summary = {"paper_time_ns": paperTime, "front_cm": float(front), "shells": int(shellRadius.size),
               "cells_per_shell": int(count.min()),
               "T_L1": normalized_l1(temperature, authorTemperature, shellRadius),
               "E_L1": normalized_l1(totalEnergy, authorEnergy, shellRadius),
               "T_residual_pct_beyond_1.15_front": [float(100 * temperatureResidual[far].min()),
                                                    float(100 * temperatureResidual[far].max())] if far.any() else None,
               "E_residual_pct_beyond_1.15_front": [float(100 * energyResidual[far].min()),
                                                    float(100 * energyResidual[far].max())] if far.any() else None}
    print(f"test{caseNumber} {cycleLabel}: front={front:.5f} cm  T_L1={summary['T_L1']:.4f}  E_L1={summary['E_L1']:.4f}")
    if far.any():
        print(f"   beyond 1.15*front: T [{100*temperatureResidual[far].min():+.2f}%, {100*temperatureResidual[far].max():+.2f}%]  "
              f"E [{100*energyResidual[far].min():+.2f}%, {100*energyResidual[far].max():+.2f}%]")
    return summary


if __name__ == "__main__":
    out = Path("current_results")
    out.mkdir(exist_ok=True)
    summary = {
        "test2": plot(2, "/tmp/t2_clean_6800.npz", -17.478532, "cycle 6800 (80%)", out / "test2_current.png"),
        "test4": plot(4, "/tmp/t4_clean_1800.npz", -127.473574, "cycle 1800 (12%)", out / "test4_current.png"),
    }
    (out / "current_t2_t4.json").write_text(json.dumps(summary, indent=2) + "\n")
