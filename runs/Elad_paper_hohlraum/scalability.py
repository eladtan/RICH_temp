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
    parser.add_argument("--sum", type=int, nargs="?", const=0, default=None,
                        help="Sum cycle times and plot with strong scaling. "
                             "No value: sum all cycles. --sum=X: sum only the last X steps.")
    parser.add_argument("--no-rebalances", action="store_true",
                        help="With --sum, exclude rebalance cycles (11, 21, ..., 101, ..., 1001, ...)")
    parser.add_argument("--fit-A", action="store_true",
                        help="With --sum or --optimal, fit A by least squares instead of using the first data point")
    parser.add_argument("--speedup", action="store_true",
                        help="Plot speedup (T_min_procs / T_N) vs. processor count with ideal scaling line")
    parser.add_argument("--loop-times", action="store_true",
                        help="Measure MC loop elapsed times ('Elapsed: X seconds') instead of step times")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot (e.g. --output=my_plot.png)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
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


def parse_cycle_times(filepath, loop_times=False):
    """Parse cycle lines and return dict mapping cycle_number -> time_seconds.
    If loop_times is True, extract MC loop elapsed times from 'Elapsed: X seconds' lines
    instead of step times from 'Cycle N ... step=Xs' lines."""
    if loop_times:
        return _parse_loop_times(filepath)
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


def _parse_loop_times(filepath):
    """Parse MC loop elapsed times, returning dict mapping cycle_number -> elapsed_seconds.
    Associates each 'Elapsed: X seconds' line with the most recent 'Cycle N at time' line."""
    cycle_at_re = re.compile(r"^Cycle\s+(\d+)\s+at\s+time\s+")
    elapsed_re = re.compile(r"^Elapsed:\s+([\d.]+)\s+seconds")
    cycle_times = {}
    current_cycle = None
    with open(filepath, "r") as f:
        for line in f:
            m = cycle_at_re.match(line)
            if m:
                current_cycle = int(m.group(1))
                continue
            m = elapsed_re.match(line)
            if m and current_cycle is not None:
                cycle_times[current_cycle] = float(m.group(1))
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


def fit_strong_scaling_A(nprocs_arr, times_arr):
    """Fit T = A/N in log-log space (equal weight to all points on the log-log plot).
    log(T) = log(A) - log(N), so log(A) = mean(log(T*N))."""
    n = np.asarray(nprocs_arr, dtype=float)
    t = np.asarray(times_arr, dtype=float)
    return np.exp(np.mean(np.log(t * n)))


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
        cycle_times = parse_cycle_times(filepath, loop_times=args.loop_times)
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

    if args.sum is not None:
        overall_max = max(max_cycles_per_rank.values())
        qualifying = sorted(
            n for n, mc in max_cycles_per_rank.items() if mc == overall_max
        )
        if not qualifying:
            print("No processors reached the overall max cycle.", file=sys.stderr)
            sys.exit(1)

        last_n = args.sum
        if last_n > 0:
            first_cycle = overall_max - last_n + 1
            if first_cycle < 1:
                first_cycle = 1
                last_n = overall_max
        else:
            first_cycle = 1
            last_n = overall_max

        is_rebalance = lambda c: c > 1 and (c - 1) % 10 == 0

        skip_label = ""
        if args.no_rebalances:
            skip_label = ", excl. rebalances"

        print(f"Summing cycles {first_cycle}..{overall_max}{skip_label}")
        print(f"Processors that reached cycle {overall_max}: {qualifying}")
        print()

        sum_times = []
        for nprocs in qualifying:
            ct = all_cycle_data[nprocs]
            total = sum(
                ct[c] for c in range(first_cycle, overall_max + 1)
                if c in ct and not (args.no_rebalances and is_rebalance(c))
            )
            sum_times.append(total)

        range_label = f"cycles {first_cycle}\u2013{overall_max}"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qualifying, sum_times, "o-",
                label=f"Total time ({range_label}{skip_label})", markersize=5)

        if args.fit_A:
            A = fit_strong_scaling_A(qualifying, sum_times)
            print(f"Fitted A = {A:.1f}")
        else:
            A = qualifying[0] * sum_times[0]
        x = np.linspace(qualifying[0], qualifying[-1], 200)
        fit_label = "fit" if args.fit_A else "ref"
        ax.plot(x, A / x, "--", color="gray", label=f"Strong scaling {fit_label} (A={A:.0f})")

        sum_label = "Total MC loop time" if args.loop_times else "Total time"
        ax.set_xlabel("Number of processors")
        ax.set_ylabel(f"{sum_label} (s)")
        ax.set_title(f"Scalability: {sum_label.lower()} ({range_label}{skip_label}) vs. processor count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(qualifying)
        ax.set_xticklabels([str(n) for n in qualifying], rotation=90)

        plt.tight_layout()
        out = args.output or "scalability.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        if args.show:
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

    if args.optimal or args.speedup:
        selected_cycles = [max(selected_cycles)]

    if args.speedup:
        cycle = selected_cycles[0]
        sp_nprocs = []
        sp_times = []
        for nprocs in nprocs_list:
            t = all_cycle_data[nprocs].get(cycle)
            if t is not None:
                sp_nprocs.append(nprocs)
                sp_times.append(t)
        if not sp_nprocs:
            print("No data for speedup plot.", file=sys.stderr)
            sys.exit(1)

        t_base = sp_times[0]
        n_base = sp_nprocs[0]
        speedups = [t_base / t for t in sp_times]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sp_nprocs, speedups, "o-", label=f"Cycle {cycle}", markersize=5)

        x = np.linspace(sp_nprocs[0], sp_nprocs[-1], 200)
        ax.plot(x, x / n_base, "--", color="gray", label="Ideal scaling")

        time_label = "MC loop time" if args.loop_times else "Step time"
        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Speedup")
        ax.set_title(f"Speedup ({time_label.lower()}, cycle {cycle}) vs. processor count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(sp_nprocs)
        ax.set_xticklabels([str(n) for n in sp_nprocs], rotation=90)

        plt.tight_layout()
        out = args.output or "scalability_speedup.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        if args.show:
            plt.show()
        return

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
        opt_nprocs = []
        opt_times = []
        for nprocs in nprocs_list:
            t = all_cycle_data[nprocs].get(cycle)
            if t is not None:
                opt_nprocs.append(nprocs)
                opt_times.append(t)
        if opt_nprocs:
            if args.fit_A:
                A = fit_strong_scaling_A(opt_nprocs, opt_times)
                print(f"Fitted A = {A:.1f}")
            else:
                A = opt_nprocs[0] * opt_times[0]
            x = np.linspace(opt_nprocs[0], opt_nprocs[-1], 200)
            fit_label = "fit" if args.fit_A else "ref"
            ax.plot(x, A / x, "--", color="gray", label=f"Strong scaling {fit_label} (A={A:.0f})")

    time_label = "MC loop time" if args.loop_times else "Step time"
    ax.set_xlabel("Number of processors")
    ax.set_ylabel(f"{time_label} (s)")
    ax.set_title(f"Scalability: {time_label.lower()} vs. processor count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    ax.set_xticks(nprocs_list)
    ax.set_xticklabels([str(n) for n in nprocs_list], rotation=90)

    plt.tight_layout()
    out = args.output or ("scalability_optimal.png" if args.optimal else "scalability.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
