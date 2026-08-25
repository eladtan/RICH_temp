#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parent / "lane_radiation_shock_individual"))

from compare_campaign import (  # noqa: E402
    check_pair,
    number,
    read_kv,
    read_table,
    symmetric_relative,
    Verdict,
)


def diagnostic_values(path: Path) -> tuple[int, float, list[int], list[int]]:
    text = path.read_text()
    iterations = [int(value) for value in re.findall(
        r"MG_BICGSTAB_CONVERGENCE[^\n]*\biterations=(\d+)", text)]
    solve_seconds = [float(value) for value in re.findall(
        r"MG_BICGSTAB_TIMING[^\n]*\btotal_seconds_max=([^\s]+)", text)]
    fallbacks = [int(value) for value in re.findall(
        r"MG_PRECONDITIONER_SETUP[^\n]*\bfallback_blocks=(\d+)", text)]
    block_sizes = [int(value) for value in re.findall(
        r"MG_PRECONDITIONER_SETUP[^\n]*\bblock_size=(\d+)", text)]
    if not iterations or not solve_seconds or not block_sizes:
        raise ValueError(f"missing solver diagnostics in {path}")
    return sum(iterations), sum(solve_seconds), fallbacks, block_sizes


def final_history(path: Path) -> list[float]:
    rows = read_table(path)
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows[-1]


def require_endpoint_snapshots(verdict: Verdict, label: str, directory: Path,
                               expected_ranks: int) -> None:
    for stem in ("initial_state", "final_state"):
        for suffix in ("h5", "pvtu"):
            master = directory / f"{stem}.{suffix}"
            verdict.require(master.is_file() and master.stat().st_size > 0,
                            f"{label} has empty {master.name}")
        pieces = directory / stem
        verdict.require(pieces.is_dir(),
                        f"{label} missing {stem}/ rank-piece directory")
        if not pieces.is_dir():
            continue
        h5_count = sum(path.is_file() and path.stat().st_size > 0
                       for path in pieces.glob("*.h5"))
        vtu_count = sum(path.is_file() and path.stat().st_size > 0
                        for path in pieces.glob("*.vtu"))
        verdict.require(h5_count == expected_ranks,
                        f"{label} {stem} has {h5_count} HDF5 rank pieces, "
                        f"expected {expected_ranks}")
        verdict.require(vtu_count == expected_ranks,
                        f"{label} {stem} has {vtu_count} VTU rank pieces, "
                        f"expected {expected_ranks}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-dir", type=Path, required=True)
    parser.add_argument("--block-dir", type=Path, required=True)
    parser.add_argument("--expected-ranks", type=int, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    verdict = Verdict()
    notes: list[str] = []
    try:
        required = (
            "exit_code.txt", "manifest.txt", "metrics.txt", "timing.txt",
            "history.txt", "radial_profile.txt", "spectrum.txt",
            "mg_solver_diagnostics.log", "initial_state.h5",
            "initial_state.pvtu", "final_state.h5", "final_state.pvtu",
        )
        for label, directory in (("scalar", args.scalar_dir),
                                 ("block", args.block_dir)):
            for name in required:
                verdict.require((directory / name).exists(),
                                f"{label} missing {name}")
            require_endpoint_snapshots(
                verdict, label, directory, args.expected_ranks)
        if verdict.failures:
            raise FileNotFoundError("required A/B artifacts are missing")

        scalar_manifest = read_kv(args.scalar_dir / "manifest.txt")
        block_manifest = read_kv(args.block_dir / "manifest.txt")
        for label, manifest in (("scalar", scalar_manifest),
                                ("block", block_manifest)):
            verdict.require(int(manifest["mpi_ranks"]) == args.expected_ranks,
                            f"{label} rank count mismatch")
            verdict.require(int(manifest["requested_initial_cells"]) ==
                            args.expected_cells,
                            f"{label} initial-cell count mismatch")
            verdict.require(int(manifest["energy_groups"]) > 0,
                            f"{label} has no energy groups")
        verdict.require(scalar_manifest["mg_preconditioner"] == "scalar_jacobi",
                        "scalar lane did not use scalar Jacobi")
        verdict.require(block_manifest["mg_preconditioner"] ==
                        "cell_block_jacobi",
                        "block lane did not use cell-block Jacobi")
        for key in ("initial_checksum_xor", "initial_checksum_sum",
                    "requested_initial_cells", "energy_groups", "initial_dt"):
            verdict.require(scalar_manifest[key] == block_manifest[key],
                            f"manifest mismatch for {key}")
        for label, directory in (("scalar", args.scalar_dir),
                                 ("block", args.block_dir)):
            verdict.require(int((directory / "exit_code.txt").read_text()) == 0,
                            f"{label} executable failed")

        scalar_iterations, scalar_solve_seconds, _, scalar_blocks = \
            diagnostic_values(args.scalar_dir / "mg_solver_diagnostics.log")
        block_iterations, block_solve_seconds, block_fallbacks, block_blocks = \
            diagnostic_values(args.block_dir / "mg_solver_diagnostics.log")
        group_count = int(block_manifest["energy_groups"])
        verdict.require(all(size == group_count for size in scalar_blocks),
                        "scalar diagnostics have wrong runtime block size")
        verdict.require(all(size == group_count for size in block_blocks),
                        "block diagnostics have wrong runtime block size")
        verdict.require(scalar_iterations > 1000,
                        "scalar lane did not exercise the >1000 iteration gate")
        verdict.require(3 * block_iterations <= scalar_iterations,
                        "cell-block iteration reduction is less than 3x")
        verdict.require(block_solve_seconds < scalar_solve_seconds,
                        "cell-block solver wall time is not lower")
        verdict.require(all(value == 0 for value in block_fallbacks),
                        "representative block case used scalar fallback")

        check_pair(verdict, "scalar_vs_block", args.scalar_dir, args.block_dir,
                   0.005, 0.005, 0.01, 0.005)

        scalar_metrics = read_kv(args.scalar_dir / "metrics.txt")
        block_metrics = read_kv(args.block_dir / "metrics.txt")
        for label, metrics in (("scalar", scalar_metrics),
                               ("block", block_metrics)):
            verdict.require(number(metrics, "minimum_material_energy") > 0,
                            f"{label} has non-positive material energy")
            verdict.require(number(metrics, "minimum_group_energy") >= 0,
                            f"{label} has negative group energy")
            verdict.require(0 < number(metrics, "minimum_fleck_factor") and
                            number(metrics, "maximum_fleck_factor") <= 1,
                            f"{label} has invalid Fleck factors")

        scalar_history = final_history(args.scalar_dir / "history.txt")
        block_history = final_history(args.block_dir / "history.txt")
        material_error = symmetric_relative(scalar_history[8], block_history[8])
        radiation_error = symmetric_relative(scalar_history[9], block_history[9])
        total_error = symmetric_relative(
            scalar_history[8] + scalar_history[9],
            block_history[8] + block_history[9])
        verdict.require(max(material_error, radiation_error, total_error) <= 0.005,
                        "integrated energy difference exceeds 0.5%")

        scalar_timing = read_kv(args.scalar_dir / "timing.txt")
        block_timing = read_kv(args.block_dir / "timing.txt")
        scalar_evolution = number(scalar_timing, "evolution_wall_max")
        block_evolution = number(block_timing, "evolution_wall_max")
        notes.extend((
            f"scalar_iterations {scalar_iterations}",
            f"block_iterations {block_iterations}",
            f"iteration_reduction {scalar_iterations / block_iterations:.17g}",
            f"scalar_solver_seconds {scalar_solve_seconds:.17g}",
            f"block_solver_seconds {block_solve_seconds:.17g}",
            f"solver_speedup {scalar_solve_seconds / block_solve_seconds:.17g}",
            f"scalar_evolution_seconds {scalar_evolution:.17g}",
            f"block_evolution_seconds {block_evolution:.17g}",
            f"evolution_speedup {scalar_evolution / block_evolution:.17g}",
            f"material_energy_error {material_error:.17g}",
            f"radiation_energy_error {radiation_error:.17g}",
            f"total_energy_error {total_error:.17g}",
        ))
    except Exception as error:
        verdict.failures.append(f"comparison exception: {error}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        stream.write("PASS\n" if not verdict.failures else "FAIL\n")
        for note in verdict.notes + notes:
            stream.write(f"{note}\n")
        for failure in verdict.failures:
            stream.write(f"failure {failure}\n")
    return 0 if not verdict.failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
