"""Track the radiation front position over a run and compare it to the similarity solution.

The analytic front is a slope discontinuity, so comparing "where the temperature
leaves zero" is ill-posed. Instead both fronts are located at the same temperature
level, which is well defined for the simulation and the analytic profile alike.
"""

import re
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from compare_profiles import CASES, author_solution

TABLE = np.genfromtxt("author_table_ii.csv", delimiter=",", names=True)


def paper_times(logPath):
    pattern = re.compile(r"^Cycle (\d+)\s+\d+%\s+paper t=(-?[\d.]+) ns.*front=([\d.]+) cm")
    times = {}
    with open(logPath) as handle:
        for line in handle:
            match = pattern.match(line)
            if match:
                times[int(match.group(1))] = (float(match.group(2)), float(match.group(3)))
    return times


def level_radius(radius, temperature, level):
    """Innermost radius at which the profile crosses `level`, linearly interpolated."""
    above = temperature >= level
    if not above.any() or above.all():
        return np.nan
    first = int(np.argmax(above))
    if first == 0:
        return np.nan
    lo, hi = first - 1, first
    weight = (level - temperature[lo]) / (temperature[hi] - temperature[lo])
    return radius[lo] + weight * (radius[hi] - radius[lo])


def analytic_level_radius(caseNumber, paperTime, level, front):
    fine = np.linspace(front, 3.0 * front, 200000)
    temperature, _, _ = author_solution(caseNumber, fine, paperTime, TABLE)
    return level_radius(fine, temperature, level)


if __name__ == "__main__":
    caseNumber = int(sys.argv[1])
    dumpDir = Path(sys.argv[2])
    logPath = sys.argv[3]
    cycles = [int(c) for c in sys.argv[4].split(",")]
    level = float(sys.argv[5]) * CASES[caseNumber]["scale_k"]
    cacheDir = Path("/tmp/front_cache")
    cacheDir.mkdir(exist_ok=True)
    decimals = 6 if CASES[caseNumber]["radius"] < 1.0 else 5
    times = paper_times(logPath)

    rows = []
    for cycle in cycles:
        pvtu = dumpDir / f"cycle_{cycle:06d}.pvtu"
        if not pvtu.exists() or cycle not in times:
            continue
        cache = cacheDir / f"case{caseNumber}_{cycle:06d}.npz"
        if not cache.exists():
            subprocess.run([sys.executable, "read_cycle_shells.py", str(pvtu), str(cache),
                            str(CASES[caseNumber]["radius"]), str(decimals)], check=True,
                           stdout=subprocess.DEVNULL)
        data = np.load(cache)
        paperTime, logFront = times[cycle]
        simulated = level_radius(data["shell_radius"], data["temperature"], level)
        analytic = analytic_level_radius(caseNumber, paperTime, level, logFront)
        rows.append((cycle, paperTime, logFront, simulated, analytic))
        print(f"cycle {cycle:6d}  t={paperTime:11.5f} ns  r_analytic={analytic:.6f}  r_RICH={simulated:.6f}  "
              f"offset={1e4*(simulated-analytic):+8.2f} um  ({100*(simulated/analytic-1):+6.2f}%)", flush=True)

    cycle, paperTime, logFront, simulated, analytic = (np.array(column) for column in zip(*rows))
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(-paperTime, analytic, "k-", linewidth=2.0, label="Authors")
    axes[0].plot(-paperTime, simulated, "o-", color="tab:blue", markersize=4.0, label="RICH")
    axes[0].set_ylabel(f"Radius of T = {level/CASES[caseNumber]['scale_k']:g} "
                       f"{'keV' if caseNumber == 4 else 'HeV'} [cm]")
    axes[0].legend()
    axes[1].axhline(0.0, color="k", linewidth=1.0)
    axes[1].plot(-paperTime, 100.0 * (simulated / analytic - 1.0), "o-", color="tab:red", markersize=4.0)
    axes[1].set_ylabel("Front position error [%]")
    axes[1].set_xlabel("-t [ns]  (time runs right to left)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.invert_xaxis()
    figure.suptitle(f"Converging Marshak test {caseNumber}: front position vs authors")
    figure.tight_layout()
    out = Path("current_results") / f"test{caseNumber}_front_tracking.png"
    figure.savefig(out, dpi=180, bbox_inches="tight")
    print(f"wrote {out}")
