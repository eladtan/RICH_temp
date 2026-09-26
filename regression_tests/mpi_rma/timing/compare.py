#!/usr/bin/env python3
"""Repeated, interleaved backend comparison with numerical checks."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--repetitions", type=int, default=5)
parser.add_argument("--timeout", type=int, default=180)
args = parser.parse_args()
config = json.loads(args.config.read_text())
cases = config["cases"]
if args.repetitions <= 0 or args.repetitions % len(cases) != 0:
    parser.error("Repetitions must be a positive multiple of the number of cases for balanced order")
output = args.output.resolve()
output.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
env.update(config.get("environment", {}))
records, reference = [], None


def run(case, phase, repetition, position):
    global reference
    prefix = f"{phase}_{repetition}_{case['id']}"
    probes = output / (prefix + "_probes.csv")
    command = (case["launcher"] + [case["binary"]] + config["benchmark_args"]
               + case["backend_args"] + ["--output-probes", str(probes)])
    log = output / (prefix + ".log")
    started = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=output, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=args.timeout)
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"{case['id']} failed: exit {result.returncode}; {log}")
    text = log.read_text()
    finish = re.search(r"Finished at t=(\S+) s after (\d+) cycles in (\S+) s", text)
    if not finish or not math.isclose(float(finish[1]), 1e-6, rel_tol=1e-6) or int(finish[2]) != 1039:
        raise RuntimeError(f"{case['id']}: incorrect physical endpoint")
    balances = [float(x) for x in re.findall(r"balance\(inj-rem-stored\)/inj=(\S+)", text)]
    if len(balances) != 1039 or any(not math.isfinite(x) or abs(x) > 1e-8 for x in balances):
        raise RuntimeError(f"{case['id']}: energy check failed")
    with probes.open() as stream:
        history = [[float(x) for x in row] for row in csv.reader(
            line for line in stream if not line.startswith("#"))]
    if len(history) != 1040 or any(len(row) != 7 or not all(map(math.isfinite, row))
                                  or min(row[2:]) <= 0 for row in history):
        raise RuntimeError(f"{case['id']}: invalid probe history")
    if reference is None:
        reference = history
    if any(row[:2] != ref[:2] for row, ref in zip(history, reference)):
        raise RuntimeError(f"{case['id']}: cycle/time mismatch")
    delta = max(abs(a-b) for row, ref in zip(history, reference)
                for a, b in zip(row[2:], ref[2:]))
    if delta > 5e-4:
        raise RuntimeError(f"{case['id']}: temperature difference {delta} keV")
    provider = re.search(r"\[OFI\] component: (.+)", text)
    if case["id"].endswith("_ofi") and (not provider or "provider: verbs" not in provider[1]):
        raise RuntimeError("OFI did not select the expected native verbs provider")
    if "Using P2P" in text and "p2p" not in case["id"]:
        raise RuntimeError("Unexpected fallback to P2P")
    if "pending 2 references" in text:
        raise RuntimeError("Unfinished MPI requests at shutdown")
    record = {
        "case": case["id"], "phase": phase, "repetition": repetition,
        "position": position, "simulation_seconds": float(finish[3]),
        "launch_to_exit_seconds": elapsed, "cycles": int(finish[2]),
        "max_relative_energy_residual": max(map(abs, balances)),
        "max_probe_difference_keV": delta,
        "provider": provider[1] if provider else None,
        "command": command, "log": str(log),
    }
    records.append(record)
    (output / "runs.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{phase} {repetition}: {case['id']}: simulation={float(finish[3]):.3f}s "
          f"launch-to-exit={elapsed:.3f}s, checks PASS", flush=True)


# The first control supplies the common numerical reference. Each case gets
# one full unmeasured warm-up, followed by a balanced cyclic order: every case
# occupies each position equally often. No runs overlap.
warmup_order = sorted(cases, key=lambda case: case["id"] != "openmpi_p2p")
for position, case in enumerate(warmup_order):
    run(case, "warmup", 0, position)
order = list(cases)
random.Random(20260916).shuffle(order)
for repetition in range(args.repetitions):
    for position in range(len(order)):
        run(order[(position + repetition) % len(order)], "measured", repetition + 1, position)

summary = {}
for case in cases:
    selected = [row for row in records if row["case"] == case["id"] and row["phase"] == "measured"]
    summary[case["id"]] = {"samples": len(selected)}
    for key in ["simulation_seconds", "launch_to_exit_seconds"]:
        values = [row[key] for row in selected]
        summary[case["id"]][key] = {
            "median": statistics.median(values), "min": min(values),
            "max": max(values), "mean": statistics.mean(values),
            "stdev": statistics.stdev(values), "values": values,
        }
    summary[case["id"]]["max_probe_difference_keV"] = max(row["max_probe_difference_keV"] for row in selected)
(output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
