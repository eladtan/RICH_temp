#!/usr/bin/env python3
"""Compare RICH E3D tallies with Figure 2 of Brantley & Novellino (M&C 2025).

The E3D driver writes one ``*_tallies.csv`` per realization.  This plots the
per-step transmission and reflection fractions (``T_step``, ``R_step``) against
the paper's Benchmark and Atomic Mix curves, and reports the deviation from
Benchmark.  ``--all-models`` adds the paper's approximate algorithms.
Per-step is the paper's normalization: the two fractions sum to one in steady
state, and a completed run matches the Benchmark curve to a few times 1e-3.

With several realizations the ensemble mean is plotted, with a +-2 standard
error band.

    python3 compare.py results/e3d_r00_tallies.csv \\
        --reference reference/e3d_brantley_reference.csv \\
        --output results/comparison.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "reference" / "e3d_brantley_reference.csv"

# Benchmark is the answer to reproduce and Atomic Mix is the approximation it
# has to beat, so those two are always drawn.  The rest are the paper's
# approximate algorithms, which RICH does not implement: --all-models shows them.
ESSENTIAL_MODELS = ("Benchmark", "Atomic Mix")
MODEL_COLORS = {
    "Benchmark": "black",
    "Atomic Mix": "0.55",
    "CLS LP": "tab:orange",
    "CLS LRP": "tab:green",
    "LRP Ramp+Exp": "tab:red",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV, skipping the driver's ``#`` comment lines."""
    with path.open(newline="") as stream:
        lines = (line for line in stream if line.strip() and not line.lstrip().startswith("#"))
        return list(csv.DictReader(lines))


def load_tallies(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Time, transmission and reflection fraction from one tallies file."""
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"{path} contains no data rows")
    values = np.array([[float(row["time_s"]), float(row["T_step"]), float(row["R_step"])]
                       for row in rows])
    values = values[np.argsort(values[:, 0])]
    return values[:, 0], values[:, 1], values[:, 2]


def load_reference(path: Path) -> dict[str, np.ndarray]:
    """Paper curves keyed by model name, each an (N, 3) array of time, T, R."""
    curves: dict[str, list[list[float]]] = {}
    for row in read_rows(path):
        model = row["model"].strip()
        curves.setdefault(model, []).append(
            [float(row["time_s"]), float(row["transmission"]), float(row["reflection"])])
    if not curves:
        raise ValueError(f"{path} contains no reference rows")
    return {model: np.array(sorted(points)) for model, points in curves.items()}


def ensemble(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean and standard error of the mean over realizations, on the first one's time grid."""
    histories = [load_tallies(path) for path in paths]
    time = histories[0][0]
    transmission = np.array([np.interp(time, t, values, left=np.nan, right=np.nan)
                             for t, values, _ in histories])
    reflection = np.array([np.interp(time, t, values, left=np.nan, right=np.nan)
                           for t, _, values in histories])

    def stats(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if stack.shape[0] == 1:
            return stack[0], np.full(time.shape, np.nan)
        mean = np.nanmean(stack, axis=0)
        return mean, np.nanstd(stack, axis=0, ddof=1) / np.sqrt(stack.shape[0])

    transmission_mean, transmission_error = stats(transmission)
    reflection_mean, reflection_error = stats(reflection)
    return time, transmission_mean, transmission_error, reflection_mean, reflection_error


def deviation(time: np.ndarray, values: np.ndarray, benchmark: np.ndarray, column: int) -> tuple[float, float]:
    """RMS and maximum absolute deviation from the Benchmark curve, over the overlap."""
    covered = (benchmark[:, 0] >= time[0]) & (benchmark[:, 0] <= time[-1])
    if not np.any(covered):
        return float("nan"), float("nan")
    difference = np.interp(benchmark[covered, 0], time, values) - benchmark[covered, column]
    return float(np.sqrt(np.mean(difference ** 2))), float(np.max(np.abs(difference)))


def plot(output: Path, time: np.ndarray, transmission: np.ndarray, transmission_error: np.ndarray,
         reflection: np.ndarray, reflection_error: np.ndarray,
         reference: dict[str, np.ndarray], realizations: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(9, 8.4), sharex=True)
    benchmark = reference["Benchmark"]

    for model, curve in reference.items():
        style = "-" if model == "Benchmark" else "--"
        width = 2.4 if model == "Benchmark" else 1.3
        color = MODEL_COLORS.get(model)
        axes[0].plot(curve[:, 0], curve[:, 1], style, color=color, lw=width, label=f"Paper: {model}")
        axes[1].plot(curve[:, 0], curve[:, 2], style, color=color, lw=width, label=f"Paper: {model}")

    label = "RICH" if realizations == 1 else f"RICH mean of {realizations} realizations"
    visible = time >= 5.0e-12
    for axis, values, errors, name, column in ((axes[0], transmission, transmission_error, "transmission", 1),
                                               (axes[1], reflection, reflection_error, "reflection", 2)):
        axis.plot(time[visible], values[visible], color="tab:blue", lw=1.7, label=label)
        if np.any(np.isfinite(errors)):
            axis.fill_between(time[visible], (values - 2.0 * errors)[visible],
                              (values + 2.0 * errors)[visible], color="tab:blue", alpha=0.18,
                              label="RICH mean $\\pm 2$ SEM")
        rms, largest = deviation(time, values, benchmark, column)
        axis.text(0.99, 0.06, f"vs Benchmark: RMS {rms:.4f}, max {largest:.4f}",
                  transform=axis.transAxes, ha="right", fontsize=9, color="tab:blue")
        axis.set_ylabel(f"Photon {name} fraction")
        axis.set_xscale("log")
        axis.set_xlim(1.0e-11, 5.2e-9)
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8, loc="upper left")
    axes[0].set_ylim(0.0, 0.28)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel("Time [s]")

    figure.suptitle(f"E3D explicit-sphere IMC benchmark, run to t={time[-1]:.2e} s")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def write_stats(path: Path, time: np.ndarray, transmission: np.ndarray, transmission_error: np.ndarray,
                reflection: np.ndarray, reflection_error: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "transmission", "transmission_sem", "reflection", "reflection_sem"])
        for values in zip(time, transmission, transmission_error, reflection, reflection_error):
            writer.writerow([f"{value:.9g}" for value in values])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tallies", nargs="+", type=Path, help="RICH *_tallies.csv file(s)")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="paper CSV")
    parser.add_argument("--output", type=Path, default=HERE / "e3d_comparison.png")
    parser.add_argument("--all-models", action="store_true",
                        help="also plot the paper's approximate algorithms (CLS LP, CLS LRP, LRP Ramp+Exp)")
    args = parser.parse_args()

    time, transmission, transmission_error, reflection, reflection_error = ensemble(args.tallies)
    reference = load_reference(args.reference)
    if not args.all_models:
        reference = {model: curve for model, curve in reference.items() if model in ESSENTIAL_MODELS}
    plot(args.output, time, transmission, transmission_error, reflection, reflection_error,
         reference, len(args.tallies))
    stats_path = args.output.with_name(args.output.stem + "_stats.csv")
    write_stats(stats_path, time, transmission, transmission_error, reflection, reflection_error)

    benchmark = reference["Benchmark"]
    print(f"realizations: {len(args.tallies)}; time range {time[0]:.3g} s to {time[-1]:.3g} s")
    print(f"wrote {args.output} and {stats_path}")
    for name, values, column in (("Transmission", transmission, 1), ("Reflection  ", reflection, 2)):
        rms, largest = deviation(time, values, benchmark, column)
        print(f"{name} vs Benchmark: RMS={rms:.6g}, max_abs={largest:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
