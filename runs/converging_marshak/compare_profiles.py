#!/usr/bin/env python3
"""Compare RICH radial profiles with the exact similarity data in paper Table II."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARAD = 7.565733250033928e-15
KEV_K = 1.160451812e7

CASES = {
    1: {
        "radius": 1.0e-3, "delta": 0.679501, "scale_k": 0.1 * KEV_K,
        "profile_coefficient": 1.34503465, "time_exponent": 0.0920519,
        "beta": 1.6, "mu": 0.14, "omega": 0.0, "density_coefficient": 19.3,
        "energy_coefficient": 3.4e13,
        "times": [-22.122309, -9.4484244, -1.0],
    },
    2: {
        "radius": 5.0e-2, "delta": 0.51765, "scale_k": 0.1 * KEV_K,
        "profile_coefficient": 0.809892, "time_exponent": 0.100238,
        "beta": 2.0, "mu": 0.6, "omega": -0.5, "density_coefficient": 1.0,
        "energy_coefficient": 3.0e13,
        "times": [-58.251607, -19.068532, -1.0],
    },
    3: {
        "radius": 1.0e-3, "delta": 1.1157536, "scale_k": 0.1 * KEV_K,
        "profile_coefficient": 1.1982, "time_exponent": 0.0276392,
        "beta": 2.0, "mu": 0.25, "omega": 0.45, "density_coefficient": 1.0,
        "energy_coefficient": 1.0e13,
        "times": [-6.5918976, -3.926451, -1.0],
    },
    4: {
        "radius": 10.0, "delta": 0.462367, "scale_k": KEV_K,
        "profile_coefficient": 0.552154, "time_exponent": 0.242705,
        "beta": 4.0, "mu": 1.0, "omega": -1.0, "density_coefficient": 1.0,
        "energy_coefficient": 1.25 * ARAD * KEV_K**4,
        "times": [-94.706889, -27.126998, -1.0],
    },
}


def author_solution(case_number, radius, paper_time_ns, table):
    case = CASES[case_number]
    front = 0.1 * case["radius"] * (-paper_time_ns) ** case["delta"]
    xi = radius / front
    w = np.zeros_like(xi)
    heated = xi > 1.0
    w[heated] = np.interp(
        np.minimum(xi[heated], table["xi"][-1]),
        table["xi"],
        table[f"W{case_number}"],
        left=0.0,
    )
    scaled_temperature = (
        case["profile_coefficient"]
        * (-paper_time_ns) ** case["time_exponent"]
        * w ** (1.0 / case["beta"])
    )
    temperature = scaled_temperature * case["scale_k"]
    density = case["density_coefficient"] * np.maximum(radius, case["radius"] * 1.0e-12) ** (-case["omega"])
    energy = (
        case["energy_coefficient"]
        * scaled_temperature ** case["beta"]
        * density ** (1.0 - case["mu"])
    )
    return temperature, energy, front


def shell_average(data, radius, bins):
    edges = np.linspace(0.0, radius, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell = np.clip(np.digitize(data["radius_cm"], edges) - 1, 0, bins - 1)
    fields = ["Tmat_K", "Trad_K", "umat_erg_cm3", "erad_erg_cm3"]
    averages = {}
    spread = {}
    for field in fields:
        values = data[field]
        weights = data["volume_cm3"]
        weighted_sum = np.bincount(shell, weights=weights * values, minlength=bins)
        weight_sum = np.bincount(shell, weights=weights, minlength=bins)
        mean = np.divide(weighted_sum, weight_sum, out=np.full(bins, np.nan), where=weight_sum > 0.0)
        variance_sum = np.bincount(shell, weights=weights * (values - mean[shell]) ** 2, minlength=bins)
        variance = np.divide(variance_sum, weight_sum, out=np.full(bins, np.nan), where=weight_sum > 0.0)
        averages[field] = mean
        spread[field] = np.sqrt(np.maximum(variance, 0.0))
    valid = np.isfinite(averages["Tmat_K"])
    return centers[valid], {key: value[valid] for key, value in averages.items()}, {
        key: value[valid] for key, value in spread.items()
    }


def normalized_l1(simulation, reference, radius):
    valid = np.isfinite(simulation)
    if not np.any(valid):
        return None
    threshold = 0.02 * np.nanmax(reference)
    valid &= reference > threshold
    if np.count_nonzero(valid) < 2:
        return None
    numerator = np.trapezoid(np.abs(simulation[valid] - reference[valid]), radius[valid])
    denominator = np.trapezoid(np.abs(reference[valid]), radius[valid])
    return float(numerator / denominator) if denominator > 0.0 else None


def compare_case(case_number, input_dir, output_dir, bins, table, plot_mode):
    case = CASES[case_number]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 9.0), sharex=True)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    metrics = {"test": case_number, "snapshots": []}

    for snapshot, (paper_time, color) in enumerate(zip(case["times"], colors), start=1):
        path = input_dir / f"test{case_number}_snapshot{snapshot}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing RICH profile: {path}")
        data = np.genfromtxt(path, delimiter=",", names=True, comments="#", skip_header=1)
        radius, average, spread = shell_average(data, case["radius"], bins)
        author_temperature, author_energy, front = author_solution(case_number, radius, paper_time, table)
        simulation_energy = average["umat_erg_cm3"] + average["erad_erg_cm3"]

        radius_scale = 1.0e4 if case["radius"] <= 1.0e-3 else 1.0
        radius_unit = "micron" if case["radius"] <= 1.0e-3 else "cm"
        temperature_scale = case["scale_k"]
        radius_display = radius * radius_scale
        label = f"t={paper_time:g} ns"

        if plot_mode == "binned":
            axes[0].plot(
                radius_display,
                author_temperature / temperature_scale,
                color=color,
                linewidth=2.0,
                label=f"Authors, {label}",
            )
            axes[0].errorbar(
                radius_display,
                average["Tmat_K"] / temperature_scale,
                yerr=spread["Tmat_K"] / temperature_scale,
                color=color,
                linestyle="none",
                marker="o",
                markersize=2.5,
                capsize=1.5,
                alpha=0.85,
                label=f"RICH IMC, {label}",
            )
            axes[1].plot(radius_display, author_energy / 1.0e13, color=color, linewidth=2.0)
            axes[1].plot(
                radius_display,
                simulation_energy / 1.0e13,
                color=color,
                linestyle="none",
                marker="o",
                markersize=2.5,
                alpha=0.85,
            )
        else:
            # Smooth author reference on a fine radial grid (1D similarity solution).
            author_radius = np.linspace(0.0, case["radius"], max(bins, 256))
            author_line_temperature, author_line_energy, _ = author_solution(
                case_number, author_radius, paper_time, table
            )
            author_radius_display = author_radius * radius_scale
            axes[0].plot(
                author_radius_display,
                author_line_temperature / temperature_scale,
                color=color,
                linewidth=2.0,
                label=f"Authors, {label}",
            )
            axes[1].plot(author_radius_display, author_line_energy / 1.0e13, color=color, linewidth=2.0)

            # Scatter every interior cell: spread in y at fixed x shows 3D asymmetry.
            cell_radius_display = data["radius_cm"] * radius_scale
            cell_temperature = data["Tmat_K"] / temperature_scale
            cell_energy = (data["umat_erg_cm3"] + data["erad_erg_cm3"]) / 1.0e13
            axes[0].scatter(
                cell_radius_display,
                cell_temperature,
                c=color,
                s=3.0,
                alpha=0.35,
                linewidths=0.0,
                label=f"RICH IMC, {label}",
            )
            axes[1].scatter(
                cell_radius_display,
                cell_energy,
                c=color,
                s=3.0,
                alpha=0.35,
                linewidths=0.0,
            )

        metrics["snapshots"].append({
            "paper_time_ns": paper_time,
            "front_radius_cm": float(front),
            "temperature_normalized_l1": normalized_l1(average["Tmat_K"], author_temperature, radius),
            "total_energy_normalized_l1": normalized_l1(simulation_energy, author_energy, radius),
            "lte_temperature_normalized_l1": normalized_l1(average["Trad_K"], average["Tmat_K"], radius),
            "radial_bins_used": int(radius.size),
        })

    axes[0].set_ylabel("Temperature [keV]" if case_number == 4 else "Temperature [HeV]")
    axes[1].set_ylabel(r"Total energy density [$10^{13}$ erg cm$^{-3}$]")
    axes[1].set_xlabel(f"Generator radius [{radius_unit}]")
    axes[0].legend(fontsize=8, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0.0)
    fig.suptitle(f"Converging Marshak benchmark {case_number}: 3D RICH versus paper")
    fig.tight_layout()
    figure_suffix = "_binned" if plot_mode == "binned" else ""
    figure_path = output_dir / f"test{case_number}_comparison{figure_suffix}.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"test{case_number}_comparison{figure_suffix}.pdf", bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / f"test{case_number}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {figure_path}")
    print(f"Wrote {metrics_path}")
    return metrics


def compare_overlay_case(case_number, overlay_dirs, output_dir, bins, table):
    case = CASES[case_number]
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 9.5), sharex=True)
    time_colors = ["tab:blue", "tab:orange", "tab:green"]
    run_markers = ["o", "s", "D", "^"]
    metrics = {
        "test": case_number,
        "runs": [{"label": label, "input_dir": str(path), "snapshots": []} for label, path in overlay_dirs],
    }

    radius_scale = 1.0e4 if case["radius"] <= 1.0e-3 else 1.0
    radius_unit = "micron" if case["radius"] <= 1.0e-3 else "cm"
    temperature_scale = case["scale_k"]

    for snapshot, (paper_time, time_color) in enumerate(zip(case["times"], time_colors), start=1):
        time_label = f"t={paper_time:g} ns"
        authors_plotted = False

        for run_index, (run_label, input_dir) in enumerate(overlay_dirs):
            path = input_dir / f"test{case_number}_snapshot{snapshot}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing RICH profile: {path}")
            data = np.genfromtxt(path, delimiter=",", names=True, comments="#", skip_header=1)
            radius, average, spread = shell_average(data, case["radius"], bins)
            author_temperature, author_energy, front = author_solution(case_number, radius, paper_time, table)
            radius_display = radius * radius_scale
            if not authors_plotted:
                axes[0].plot(
                    radius_display,
                    author_temperature / temperature_scale,
                    color=time_color,
                    linewidth=2.0,
                    label=f"Authors, {time_label}",
                )
                axes[1].plot(radius_display, author_energy / 1.0e13, color=time_color, linewidth=2.0)
                authors_plotted = True
            simulation_energy = average["umat_erg_cm3"] + average["erad_erg_cm3"]
            marker = run_markers[run_index % len(run_markers)]
            rich_label = f"RICH {run_label}, {time_label}"
            axes[0].errorbar(
                radius_display,
                average["Tmat_K"] / temperature_scale,
                yerr=spread["Tmat_K"] / temperature_scale,
                color=time_color,
                linestyle="none",
                marker=marker,
                markersize=3.0,
                capsize=1.5,
                alpha=0.9,
                label=rich_label,
            )
            axes[1].errorbar(
                radius_display,
                simulation_energy / 1.0e13,
                color=time_color,
                linestyle="none",
                marker=marker,
                markersize=3.0,
                capsize=1.5,
                alpha=0.9,
            )
            metrics["runs"][run_index]["snapshots"].append({
                "paper_time_ns": paper_time,
                "front_radius_cm": float(front),
                "temperature_normalized_l1": normalized_l1(average["Tmat_K"], author_temperature, radius),
                "total_energy_normalized_l1": normalized_l1(simulation_energy, author_energy, radius),
                "lte_temperature_normalized_l1": normalized_l1(average["Trad_K"], average["Tmat_K"], radius),
                "radial_bins_used": int(radius.size),
            })

    axes[0].set_ylabel("Temperature [keV]" if case_number == 4 else "Temperature [HeV]")
    axes[1].set_ylabel(r"Total energy density [$10^{13}$ erg cm$^{-3}$]")
    axes[1].set_xlabel(f"Generator radius [{radius_unit}]")
    axes[0].legend(fontsize=7, ncol=2)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xlim(left=0.0)
    fig.suptitle(f"Converging Marshak benchmark {case_number}: resolution comparison (binned)")
    fig.tight_layout()
    figure_path = output_dir / f"test{case_number}_comparison_overlay_binned.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / f"test{case_number}_comparison_overlay_binned.pdf", bbox_inches="tight")
    plt.close(fig)

    metrics_path = output_dir / f"test{case_number}_overlay_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Wrote {figure_path}")
    print(f"Wrote {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=int, choices=range(1, 5), action="append", help="benchmark to compare; repeatable")
    parser.add_argument("--input-dir", type=Path, default=Path("results"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--table", type=Path, default=Path(__file__).with_name("author_table_ii.csv"))
    parser.add_argument("--bins", type=int, default=96)
    parser.add_argument(
        "--plot-mode",
        choices=["scatter", "binned"],
        default="scatter",
        help="scatter: every cell; binned: volume-weighted radial shell means with error bars",
    )
    parser.add_argument(
        "--overlay",
        nargs=2,
        action="append",
        metavar=("LABEL", "DIR"),
        help="overlay binned profiles from additional runs (LABEL DIR); repeatable",
    )
    args = parser.parse_args()

    table = np.genfromtxt(args.table, delimiter=",", names=True)
    tests = args.test if args.test else [1, 2, 3, 4]
    all_metrics = []
    for case_number in tests:
        if args.overlay:
            overlay_dirs = [(label, Path(directory)) for label, directory in args.overlay]
            all_metrics.append(
                compare_overlay_case(case_number, overlay_dirs, args.output_dir, args.bins, table)
            )
        else:
            all_metrics.append(
                compare_case(case_number, args.input_dir, args.output_dir, args.bins, table, args.plot_mode)
            )
    if len(all_metrics) > 1:
        summary = args.output_dir / "all_metrics.json"
        summary.write_text(json.dumps(all_metrics, indent=2) + "\n")
        print(f"Wrote {summary}")


if __name__ == "__main__":
    main()
