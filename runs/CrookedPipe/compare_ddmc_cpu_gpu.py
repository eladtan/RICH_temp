#!/usr/bin/env python3
"""Check paired CrookedPipe CPU/GPU DDMC outputs."""

import argparse
import csv
import json
import math
from pathlib import Path


def load_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(line for line in stream if not line.startswith("#")))


def load_probes(path):
    rows = []
    with path.open() as stream:
        for row in csv.reader(line for line in stream if not line.startswith("#")):
            rows.append([float(value) for value in row])
    return rows


def relative_difference(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1.0e-300)


def reference_at(reference, time_ns, probe):
    if time_ns <= reference[0][0]:
        return reference[0][probe + 1]
    for lower, upper in zip(reference, reference[1:]):
        if time_ns <= upper[0]:
            fraction = ((math.log(time_ns) - math.log(lower[0])) /
                        (math.log(upper[0]) - math.log(lower[0])))
            return lower[probe + 1] + fraction * (
                upper[probe + 1] - lower[probe + 1])
    return reference[-1][probe + 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_dir", type=Path)
    parser.add_argument("gpu_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    cpu_metrics = load_csv(args.cpu_dir / "crookedpipe_metrics.txt")
    gpu_metrics = load_csv(args.gpu_dir / "crookedpipe_metrics.txt")
    cpu_probes = load_probes(args.cpu_dir / "crookedpipe_probes.txt")
    gpu_probes = load_probes(args.gpu_dir / "crookedpipe_probes.txt")
    reference_path = Path(__file__).parent / "reference" / "fig8_imc.csv"
    reference = load_probes(reference_path)

    checks = {}
    checks["same_metric_samples"] = len(cpu_metrics) == len(gpu_metrics) > 0
    checks["same_probe_samples"] = len(cpu_probes) == len(gpu_probes) > 1
    checks["same_cycles"] = checks["same_metric_samples"] and all(
        c["cycle"] == g["cycle"] and c["time_s"] == g["time_s"]
        for c, g in zip(cpu_metrics, gpu_metrics))
    checks["cpu_no_legacy_fallback"] = all(
        int(row["ddmc_fallback"]) == 0 for row in cpu_metrics)
    checks["gpu_no_legacy_fallback"] = all(
        int(row["ddmc_fallback"]) == 0 for row in gpu_metrics)

    cpu_steps = sum(int(row["ddmc_steps"]) for row in cpu_metrics)
    gpu_steps = sum(int(row["ddmc_steps"]) for row in gpu_metrics)
    cpu_leaks = sum(int(row["ddmc_leaks"]) for row in cpu_metrics)
    gpu_leaks = sum(int(row["ddmc_leaks"]) for row in gpu_metrics)
    checks["ddmc_active"] = min(cpu_steps, gpu_steps, cpu_leaks, gpu_leaks) > 0

    final_cpu = cpu_metrics[-1]
    final_gpu = gpu_metrics[-1]
    cpu_step_seconds = sum(float(row["step_wall_s"]) for row in cpu_metrics)
    gpu_step_seconds = sum(float(row["step_wall_s"]) for row in gpu_metrics)
    census_weight_rel = relative_difference(
        float(final_cpu["census_weight"]), float(final_gpu["census_weight"]))
    step_rel = relative_difference(cpu_steps, gpu_steps)
    leak_rel = relative_difference(cpu_leaks, gpu_leaks)
    checks["census_weight_within_10pct"] = census_weight_rel <= 0.10
    checks["ddmc_steps_within_5pct"] = step_rel <= 0.05
    checks["ddmc_leaks_within_5pct"] = leak_rel <= 0.05

    paired_probe_differences = [
        abs(c_value - g_value)
        for cpu, gpu in zip(cpu_probes, gpu_probes)
        for c_value, g_value in zip(cpu[2:], gpu[2:])
    ]
    max_probe_difference = max(paired_probe_differences, default=math.inf)
    checks["probes_within_0p02_keV"] = max_probe_difference <= 0.02

    reference_mae = {}
    reference_mae_by_probe = {}
    for name, probes in (("cpu", cpu_probes), ("gpu", gpu_probes)):
        errors_by_probe = [[] for _ in range(5)]
        for row in probes[1:]:
            for probe in range(5):
                errors_by_probe[probe].append(
                    abs(row[probe + 2] -
                        reference_at(reference, row[0], probe)))
        probe_mae = [
            sum(errors) / len(errors) if errors else math.inf
            for errors in errors_by_probe
        ]
        errors = [error for values in errors_by_probe for error in values]
        reference_mae[name] = sum(errors) / len(errors) if errors else math.inf
        reference_mae_by_probe[name] = probe_mae
        checks[f"{name}_each_probe_reference_mae_within_0p05_keV"] = (
            max(probe_mae) <= 0.05)

    result = {
        "pass": all(checks.values()),
        "checks": checks,
        "cpu_ddmc_steps": cpu_steps,
        "gpu_ddmc_steps": gpu_steps,
        "cpu_ddmc_leaks": cpu_leaks,
        "gpu_ddmc_leaks": gpu_leaks,
        "ddmc_step_relative_difference": step_rel,
        "ddmc_leak_relative_difference": leak_rel,
        "final_census_weight_relative_difference": census_weight_rel,
        "max_probe_cpu_gpu_difference_keV": max_probe_difference,
        "cpu_transport_step_seconds": cpu_step_seconds,
        "gpu_transport_step_seconds": gpu_step_seconds,
        "transport_speedup": cpu_step_seconds / gpu_step_seconds,
        "reference_mean_absolute_error_keV": reference_mae,
        "reference_mean_absolute_error_by_probe_keV":
            reference_mae_by_probe,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
