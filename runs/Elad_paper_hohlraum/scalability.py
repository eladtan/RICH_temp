#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot scalability of hohlraum runs across different processor counts.")
    parser.add_argument("--ignore", type=str, default="",
                        help="Comma-separated list of processor counts to ignore (e.g. --ignore=128,256)")
    parser.add_argument("--cycles", type=str, default="",
                        help="Comma-separated list of specific cycles to plot (e.g. --cycles=50,100,150)")
    parser.add_argument("--optimal", action="store_true",
                        help="Plot only the max selected cycle plus an ideal strong-scaling reference line")
    parser.add_argument("--sum", action="store_true",
                        help="Sum all cycle times (only for procs that reached the same max cycle) and plot with strong scaling")
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for .out files")
    return parser.parse_args()


def find_latest_files(directory):
    """Find the latest .out file for each processor count (highest job number)."""
    pattern = os.path.join(directory, "hohlraum_*_n*.out")
    files = glob.glob(pattern)

    rank_files = defaultdict(list)
    file_re = re.compile(r"hohlraum_(\d+)_n(\d+)\.out$")

    for f in files:
        m = file_re.search(f)
        if m:
            job_id = int(m.group(1))
            nprocs = int(m.group(2))
            rank_files[nprocs].append((job_id, f))

    latest = {}
    for nprocs, entries in rank_files.items():
        entries.sort(key=lambda x: x[0], reverse=True)
        latest[nprocs] = entries[0][1]

    return latest


def parse_cycle_times(filepath):
    """Parse cycle lines and return dict mapping cycle_number -> step_time_seconds."""
    cycle_re = re.compile(
        r"^Cycle\s+(\d+)\s+.*?step=([\d.]+)s"
    )
    cycle_times = {}
    with open(filepath, "r") as f:
        for line in f:
            m = cycle_re.match(line)
            if m:
                cycle_num = int(m.group(1))
                step_time = float(m.group(2))
                cycle_times[cycle_num] = step_time
    return cycle_times


def pick_default_cycles(max_common_cycle, num_curves=5):
    """Pick num_curves evenly spaced cycles up to max_common_cycle,
    avoiding cycles divisible by 10 (decrease by 1 in that case)."""
    cycles = []
    for i in range(1, num_curves + 1):
        c = round(max_common_cycle * i / num_curves)
        if c % 10 == 0:
            c -= 1
        if c < 1:
            c = 1
        cycles.append(c)
    return sorted(set(cycles))


def adjust_cycle(c):
    """If cycle is divisible by 10, decrease by 1."""
    if c % 10 == 0:
        c -= 1
    return max(c, 1)


def print_table(file_info):
    """Print a nicely formatted table of the selected runs."""
    col_procs = "Processors"
    col_file = "File"
    col_cycles = "Cycles"

    rows = []
    for nprocs in sorted(file_info.keys()):
        fname, ncycles = file_info[nprocs]
        rows.append((str(nprocs), os.path.basename(fname), str(ncycles)))

    w_procs = max(len(col_procs), max(len(r[0]) for r in rows))
    w_file = max(len(col_file), max(len(r[1]) for r in rows))
    w_cycles = max(len(col_cycles), max(len(r[2]) for r in rows))

    sep = f"+-{'-' * w_procs}-+-{'-' * w_file}-+-{'-' * w_cycles}-+"
    header = f"| {col_procs:>{w_procs}} | {col_file:<{w_file}} | {col_cycles:>{w_cycles}} |"

    print(sep)
    print(header)
    print(sep)
    for procs, fname, ncycles in rows:
        print(f"| {procs:>{w_procs}} | {fname:<{w_file}} | {ncycles:>{w_cycles}} |")
    print(sep)


def main():
    args = parse_args()

    ignore_set = set()
    if args.ignore:
        ignore_set = {int(x.strip()) for x in args.ignore.split(",")}

    latest_files = find_latest_files(args.dir)
    if not latest_files:
        print("No matching .out files found.", file=sys.stderr)
        sys.exit(1)

    for ign in ignore_set:
        latest_files.pop(ign, None)

    if not latest_files:
        print("All processor counts were ignored.", file=sys.stderr)
        sys.exit(1)

    all_cycle_data = {}
    file_info = {}
    for nprocs in sorted(latest_files.keys()):
        filepath = latest_files[nprocs]
        cycle_times = parse_cycle_times(filepath)
        if not cycle_times:
            print(f"Warning: no cycle data found in {filepath}, skipping.", file=sys.stderr)
            continue
        all_cycle_data[nprocs] = cycle_times
        file_info[nprocs] = (filepath, max(cycle_times.keys()))

    if not all_cycle_data:
        print("No cycle data found in any file.", file=sys.stderr)
        sys.exit(1)

    print_table(file_info)
    print()

    max_cycles_per_rank = {nprocs: max(ct.keys()) for nprocs, ct in all_cycle_data.items()}
    max_common_cycle = min(max_cycles_per_rank.values())

    print(f"Max common cycle: {max_common_cycle}")

    nprocs_list = sorted(all_cycle_data.keys())

    if args.sum:
        overall_max = max(max_cycles_per_rank.values())
        qualifying = sorted(
            n for n, mc in max_cycles_per_rank.items() if mc == overall_max
        )
        if not qualifying:
            print("No processors reached the overall max cycle.", file=sys.stderr)
            sys.exit(1)

        print(f"Summing cycles 1..{overall_max} for processors that reached cycle {overall_max}: {qualifying}")
        print()

        sum_times = []
        for nprocs in qualifying:
            ct = all_cycle_data[nprocs]
            total = sum(ct[c] for c in range(1, overall_max + 1) if c in ct)
            sum_times.append(total)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qualifying, sum_times, "o-", label=f"Total time (cycles 1\u2013{overall_max})", markersize=5)

        min_nprocs = qualifying[0]
        A = min_nprocs * sum_times[0]
        x = np.linspace(qualifying[0], qualifying[-1], 200)
        ax.plot(x, A / x, "--", color="gray", label=f"Strong scaling (A={A:.0f})")

        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Total time (s)")
        ax.set_title(f"Scalability: total time (cycles 1\u2013{overall_max}) vs. processor count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(qualifying)
        ax.set_xticklabels([str(n) for n in qualifying])

        plt.tight_layout()
        plt.show()
        return

    if args.cycles:
        selected_cycles = sorted(adjust_cycle(int(x.strip())) for x in args.cycles.split(","))
    else:
        selected_cycles = pick_default_cycles(max_common_cycle)

    selected_cycles = [c for c in selected_cycles if c <= max_common_cycle]
    if not selected_cycles:
        print(f"No valid cycles to plot (max common cycle is {max_common_cycle}).", file=sys.stderr)
        sys.exit(1)

    print(f"Plotting cycles: {selected_cycles}")
    print()

    if args.optimal:
        selected_cycles = [max(selected_cycles)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for cycle in selected_cycles:
        times = []
        valid_nprocs = []
        for nprocs in nprocs_list:
            ct = all_cycle_data[nprocs]
            if cycle in ct:
                times.append(ct[cycle])
                valid_nprocs.append(nprocs)
        if valid_nprocs:
            ax.plot(valid_nprocs, times, "o-", label=f"Cycle {cycle}", markersize=5)

    if args.optimal:
        cycle = selected_cycles[0]
        min_nprocs = nprocs_list[0]
        min_time = all_cycle_data[min_nprocs].get(cycle)
        if min_time is not None:
            A = min_nprocs * min_time
            x = np.linspace(nprocs_list[0], nprocs_list[-1], 200)
            ax.plot(x, A / x, "--", color="gray", label=f"Strong scaling (A={A:.0f})")

    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Step time (s)")
    ax.set_title("Scalability: step time vs. processor count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xticks(nprocs_list)
    ax.set_xticklabels([str(n) for n in nprocs_list])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
