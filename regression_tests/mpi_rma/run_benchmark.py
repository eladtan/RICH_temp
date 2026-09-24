#!/usr/bin/env python3
"""Run Crooked Pipe to 1000 ns with P2P and two MPI RMA queue modes."""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shlex
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--binary", type=Path, required=True)
parser.add_argument("--launcher", required=True, help="MPI launcher and rank/host options")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--timeout", type=int, default=180)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
args.output = args.output.resolve()

common = [str(args.binary.resolve()), "400", "2", "10", "--wall-layers", "1",
          "--wall-points", "400", "--random-walls", "--boundary-photons", "20",
          "--final-time", "1e-6", "--energy-ledger"]
modes = {
    "p2p": ["--manager", "p2p"],
    "mpi_dynamic": ["--manager", "rdma", "--rdma-engine", "mpi", "--rdma-ring-size", "50"],
    "mpi_fixed": ["--manager", "rdma", "--rdma-engine", "mpi", "--optimized-rdma",
                  "--rdma-ring-size", "50", "--rdma-ring-limit", "50"],
}
histories, summary = {}, {}
for mode, options in modes.items():
    probes = args.output / (mode + "_probes.csv")
    command = shlex.split(args.launcher) + common + options + ["--output-probes", str(probes)]
    log = args.output / (mode + ".log")
    with log.open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                cwd=args.output, timeout=args.timeout)
    if result.returncode:
        raise RuntimeError(f"{mode}: exit {result.returncode}; see {log}")
    text = log.read_text()
    finish = re.search(r"Finished at t=(\S+) s after (\d+) cycles", text)
    if not finish or not math.isclose(float(finish[1]), 1e-6, rel_tol=1e-6):
        raise RuntimeError(f"{mode}: did not reach the physical endpoint")
    balances = [float(x) for x in re.findall(r"balance\(inj-rem-stored\)/inj=(\S+)", text)]
    if len(balances) != int(finish[2]) or any(not math.isfinite(x) or abs(x) > 1e-8 for x in balances):
        raise RuntimeError(f"{mode}: incomplete or nonconserving energy ledger")
    audit = re.search(r"MPI RMA audit: puts=(\d+) bytes=(\d+) gets=(\d+) windows=(\d+)", text)
    if not audit or (mode != "p2p" and any(int(x) == 0 for x in audit.groups())):
        raise RuntimeError(f"{mode}: missing MPI RMA traffic")
    with probes.open() as stream:
        rows = [[float(x) for x in row] for row in csv.reader(
            line for line in stream if not line.startswith("#"))]
    if len(rows) != int(finish[2]) + 1 or any(
        len(row) != 7 or not all(math.isfinite(x) for x in row) or min(row[2:]) <= 0 for row in rows
    ):
        raise RuntimeError(f"{mode}: invalid temperature history")
    histories[mode] = rows
    summary[mode] = {
        "command": command, "cycles": int(finish[2]), "end_time_seconds": float(finish[1]),
        "max_relative_energy_residual": max(map(abs, balances)),
        "mpi_put_calls": int(audit[1]), "mpi_put_bytes": int(audit[2]),
        "mpi_get_calls": int(audit[3]), "mpi_window_creations": int(audit[4]),
    }
    print(f"PASS {mode}: {finish[2]} cycles, energy residual {max(map(abs, balances)):.3g}", flush=True)

reference = histories["p2p"]
for mode in ("mpi_dynamic", "mpi_fixed"):
    rows = histories[mode]
    if len(rows) != len(reference) or any(row[:2] != ref[:2] for row, ref in zip(rows, reference)):
        raise RuntimeError(f"{mode}: cycle/time history differs from P2P")
    difference = max(abs(a - b) for row, ref in zip(rows, reference) for a, b in zip(row[2:], ref[2:]))
    # 0.1% of the 0.5 keV drive; this is a transport regression on a coarse
    # stochastic benchmark, not a mesh-convergence or published-physics test.
    if difference > 5e-4:
        raise RuntimeError(f"{mode}: probe deviation from P2P is {difference} keV")
    summary[mode]["max_probe_difference_from_p2p_keV"] = difference
    print(f"PASS {mode}: max probe difference from P2P {difference:.3g} keV", flush=True)
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
