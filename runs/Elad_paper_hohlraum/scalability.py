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
                             "No value: sum all cycles (only runs reaching max). "
                             "--sum=X: sum the last X common cycles (all runs).")
    parser.add_argument("--no-rebalances", action="store_true",
                        help="With --sum, exclude rebalance cycles (11, 21, ..., 101, ..., 1001, ...)")
    parser.add_argument("--fit-A", action="store_true",
                        help="With --sum or --optimal, fit A by least squares instead of using the first data point")
    parser.add_argument("--speedup", action="store_true",
                        help="Plot speedup (T_min_procs / T_N) vs. processor count with ideal scaling line")
    parser.add_argument("--p2p", action="store_true",
                        help="Include P2P files (hohlraum_P2P_*) as additional curves")
    parser.add_argument("--loop-times", action="store_true",
                        help="Measure MC loop elapsed times ('Elapsed: X seconds') instead of step times")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot (e.g. --output=my_plot.png)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
    parser.add_argument("--plot-diff", action="store_true",
                        help="With --optimal, plot deviation from ideal scaling as a separate graph")
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for .out files")
    return parser.parse_args()


def find_latest_files(directory, include_p2p=False):
    """Find the latest .out file for each processor count (highest job number).
    Returns (regular_files, p2p_files) where p2p_files is empty if include_p2p is False."""
    pattern = os.path.join(directory, "hohlraum_*_n*.out")
    files = glob.glob(pattern)

    regular_re = re.compile(r"hohlraum_(\d+)_n(\d+)\.out$")
    p2p_re = re.compile(r"hohlraum_P2P_(\d+)_n(\d+)\.out$")

    rank_files = defaultdict(list)
    p2p_rank_files = defaultdict(list)

    for f in files:
        m = p2p_re.search(f)
        if m:
            if include_p2p:
                job_id = int(m.group(1))
                nprocs = int(m.group(2))
                p2p_rank_files[nprocs].append((job_id, f))
            continue
        m = regular_re.search(f)
        if m:
            job_id = int(m.group(1))
            nprocs = int(m.group(2))
            rank_files[nprocs].append((job_id, f))

    def pick_latest(rf):
        latest = {}
        for nprocs, entries in rf.items():
            entries.sort(key=lambda x: x[0], reverse=True)
            latest[nprocs] = entries[0][1]
        return latest

    return pick_latest(rank_files), pick_latest(p2p_rank_files)


def parse_cycle_times(filepath, loop_times=False):
    """Parse cycle lines and return dict mapping cycle_number -> time_seconds.
    If loop_times is True, extract MC loop elapsed times from 'Elapsed: X seconds' lines
    instead of step times from the progress line 'Cycle N ... step=Xs'."""
    if loop_times:
        return _parse_loop_times(filepath)
    progress_re = re.compile(r"^Cycle\s+(\d+)\s+t=.*\bstep=([\d.]+)s")
    cycle_times = {}
    with open(filepath, "r") as f:
        for line in f:
            m = progress_re.match(line)
            if m:
                cycle_times[int(m.group(1))] = float(m.group(2))
    return cycle_times


def _parse_loop_times(filepath):
    """Parse MC loop elapsed times, returning dict mapping cycle_number -> elapsed_seconds.
    The 'Elapsed: X seconds' line after 'Cycle N at time' belongs to the step reported
    on the 'Cycle N+1' progress line, so we attribute it to cycle N+1."""
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
                cycle_times[current_cycle + 1] = float(m.group(1))
    return cycle_times


def pick_default_cycles(max_common_cycle, num_curves=5):
    """Pick num_curves evenly spaced cycles up to max_common_cycle,
    avoiding rebalance cycles (c%10==1, c>1; decrease by 1 in that case)."""
    cycles = []
    for i in range(1, num_curves + 1):
        c = round(max_common_cycle * i / num_curves)
        if c > 1 and c % 10 == 1:
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
    """If cycle is a rebalance cycle (c%10==1, c>1), decrease by 1."""
    if c > 1 and c % 10 == 1:
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

    latest_files, p2p_files = find_latest_files(args.dir, include_p2p=args.p2p)
    if not latest_files:
        print("No matching .out files found.", file=sys.stderr)
        sys.exit(1)

    for ign in ignore_set:
        latest_files.pop(ign, None)
        p2p_files.pop(ign, None)

    if not latest_files:
        print("All processor counts were ignored.", file=sys.stderr)
        sys.exit(1)

    def load_cycle_data(files_dict, label=""):
        cycle_data = {}
        finfo = {}
        for nprocs in sorted(files_dict.keys()):
            filepath = files_dict[nprocs]
            ct = parse_cycle_times(filepath, loop_times=args.loop_times)
            if not ct:
                print(f"Warning: no cycle data in {filepath}, skipping.", file=sys.stderr)
                continue
            cycle_data[nprocs] = ct
            finfo[nprocs] = (filepath, max(ct.keys()))
        return cycle_data, finfo

    all_cycle_data, file_info = load_cycle_data(latest_files)
    if not all_cycle_data:
        print("No cycle data found in any file.", file=sys.stderr)
        sys.exit(1)

    p2p_cycle_data, p2p_file_info = load_cycle_data(p2p_files)

    print_table(file_info)
    if p2p_file_info:
        print("\nP2P runs:")
        print_table(p2p_file_info)
    print()

    max_cycles_per_rank = {nprocs: max(ct.keys()) for nprocs, ct in all_cycle_data.items()}
    all_max_values = list(max_cycles_per_rank.values())
    if p2p_cycle_data:
        all_max_values += [max(ct.keys()) for ct in p2p_cycle_data.values()]
    max_common_cycle = min(all_max_values)

    print(f"Max common cycle: {max_common_cycle}")

    nprocs_list = sorted(all_cycle_data.keys())

    if args.sum is not None:
        last_n = args.sum
        if last_n > 0:
            sum_max = max_common_cycle
            qualifying = sorted(all_cycle_data.keys())
            first_cycle = sum_max - last_n + 1
            if first_cycle < 1:
                first_cycle = 1
                last_n = sum_max
        else:
            sum_max = max(max_cycles_per_rank.values())
            qualifying = sorted(
                n for n, mc in max_cycles_per_rank.items() if mc == sum_max
            )
            if not qualifying:
                print("No processors reached the overall max cycle.", file=sys.stderr)
                sys.exit(1)
            first_cycle = 1
            last_n = sum_max

        is_rebalance = lambda c: c > 1 and c % 10 == 1

        skip_label = ""
        if args.no_rebalances:
            skip_label = ", excl. rebalances"

        print(f"Summing cycles {first_cycle}..{sum_max}{skip_label}")
        print(f"Qualifying processors: {qualifying}")
        print()

        sum_times = []
        for nprocs in qualifying:
            ct = all_cycle_data[nprocs]
            total = sum(
                ct[c] for c in range(first_cycle, sum_max + 1)
                if c in ct and not (args.no_rebalances and is_rebalance(c))
            )
            sum_times.append(total)

        range_label = f"cycles {first_cycle}\u2013{sum_max}"
        rdma_label = "RDMA" if p2p_cycle_data else None
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qualifying, sum_times, "o-",
                label=f"{rdma_label + ' total' if rdma_label else 'Total'} time ({range_label}{skip_label})",
                markersize=5)

        if args.fit_A:
            A = fit_strong_scaling_A(qualifying, sum_times)
            print(f"Fitted A = {A:.1f}")
        else:
            A = qualifying[0] * sum_times[0]
        x = np.linspace(qualifying[0], qualifying[-1], 200)
        fit_label = "fit" if args.fit_A else "ref"
        ax.plot(x, A / x, "--", color="gray",
                label=f"{'RDMA strong' if rdma_label else 'Strong'} scaling {fit_label} (A={A:.0f})")

        p2p_qualifying = []
        if p2p_cycle_data:
            p2p_max_cycles = {n: max(ct.keys()) for n, ct in p2p_cycle_data.items()}
            p2p_qualifying = sorted(
                n for n, mc in p2p_max_cycles.items() if mc >= sum_max
            )
            if p2p_qualifying:
                p2p_sum = []
                for nprocs in p2p_qualifying:
                    ct = p2p_cycle_data[nprocs]
                    total = sum(
                        ct[c] for c in range(first_cycle, sum_max + 1)
                        if c in ct and not (args.no_rebalances and is_rebalance(c))
                    )
                    p2p_sum.append(total)
                ax.plot(p2p_qualifying, p2p_sum, "s--",
                        label=f"P2P total time ({range_label}{skip_label})", markersize=5)
                if args.fit_A:
                    B = fit_strong_scaling_A(p2p_qualifying, p2p_sum)
                    print(f"P2P fitted B = {B:.1f}")
                else:
                    B = p2p_qualifying[0] * p2p_sum[0]
                xp = np.linspace(p2p_qualifying[0], p2p_qualifying[-1], 200)
                ax.plot(xp, B / xp, "--", color="orange",
                        label=f"P2P strong scaling {fit_label} (B={B:.0f})")

        def print_scaling_diff(label, nprocs_list, times_list, A_val):
            n_ref = nprocs_list[0]
            t_ref = times_list[0]
            print(f"\n{label} — deviation from ideal scaling (A={A_val:.0f}):")
            print(f"  {'Procs':>8}  {'Actual':>10}  {'Ideal':>10}  {'Diff':>10}  {'Diff%':>8}  {'Speedup':>8}  {'Expected':>8}  {'Effic%':>7}")
            for n, t in zip(nprocs_list, times_list):
                ideal = A_val / n
                diff = t - ideal
                pct = diff / ideal * 100
                speedup = t_ref / t
                expected = n / n_ref
                efficiency = speedup / expected * 100
                print(f"  {n:>8}  {t:>10.3f}  {ideal:>10.3f}  {diff:>+10.3f}  {pct:>+7.1f}%  {speedup:>8.2f}  {expected:>8.2f}  {efficiency:>6.1f}%")

        if args.optimal:
            lbl = "RDMA" if p2p_cycle_data else "Strong scaling"
            print_scaling_diff(lbl, qualifying, sum_times, A)
            if p2p_cycle_data and p2p_qualifying:
                print_scaling_diff("P2P", p2p_qualifying, p2p_sum, B)

        sum_label = "Total MC loop time" if args.loop_times else "Total time"
        ax.set_xlabel("Number of processors")
        ax.set_ylabel(f"{sum_label} (s)")
        ax.set_title(f"Scalability: {sum_label.lower()} ({range_label}{skip_label}) vs. processor count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        all_ticks = sorted(set(qualifying) | set(p2p_qualifying))
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([str(n) for n in all_ticks], rotation=90)

        plt.tight_layout()
        out = args.output or "scalability.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        if args.show:
            plt.show()

        if args.optimal and args.plot_diff:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            rdma_diff_pct = [(t - A / n) / (A / n) * 100 for n, t in zip(qualifying, sum_times)]
            ax2.plot(qualifying, rdma_diff_pct, "o-",
                     label=f"{'RDMA' if p2p_cycle_data else 'Measured'} ({range_label}{skip_label})",
                     markersize=5)
            if p2p_cycle_data and p2p_qualifying:
                p2p_diff_pct = [(t - B / n) / (B / n) * 100 for n, t in zip(p2p_qualifying, p2p_sum)]
                ax2.plot(p2p_qualifying, p2p_diff_pct, "s--",
                         label=f"P2P ({range_label}{skip_label})", markersize=5)
            ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax2.set_xlabel("Number of processors")
            ax2.set_ylabel("Deviation from ideal (%)")
            ax2.set_title(f"Deviation from ideal strong scaling ({range_label}{skip_label})")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale("log", base=2)
            all_ticks = sorted(set(qualifying) | set(p2p_qualifying))
            ax2.set_xticks(all_ticks)
            ax2.set_xticklabels([str(n) for n in all_ticks], rotation=90)
            plt.tight_layout()
            diff_out = os.path.splitext(out)[0] + "_diff" + os.path.splitext(out)[1]
            plt.savefig(diff_out, dpi=150)
            print(f"Saved {diff_out}")
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
        rdma_label = "RDMA" if p2p_cycle_data else None

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(sp_nprocs, speedups, "o-",
                label=f"{rdma_label + ' cycle' if rdma_label else 'Cycle'} {cycle}", markersize=5)

        x = np.linspace(sp_nprocs[0], sp_nprocs[-1], 200)
        ax.plot(x, x / n_base, "--", color="gray", label="Ideal scaling")

        all_sp_nprocs = set(sp_nprocs)
        if p2p_cycle_data:
            p2p_nprocs_list = sorted(p2p_cycle_data.keys())
            p2p_max_common = min(max(ct.keys()) for ct in p2p_cycle_data.values())
            p2p_cycle = adjust_cycle(p2p_max_common)
            p2p_sp_nprocs = []
            p2p_sp_times = []
            for nprocs in p2p_nprocs_list:
                t = p2p_cycle_data[nprocs].get(p2p_cycle)
                if t is not None:
                    p2p_sp_nprocs.append(nprocs)
                    p2p_sp_times.append(t)
            if p2p_sp_nprocs:
                p2p_t_base = p2p_sp_times[0]
                p2p_n_base = p2p_sp_nprocs[0]
                p2p_speedups = [p2p_t_base / t for t in p2p_sp_times]
                ax.plot(p2p_sp_nprocs, p2p_speedups, "s--",
                        label=f"P2P cycle {p2p_cycle}", markersize=5)
                xp = np.linspace(p2p_sp_nprocs[0], p2p_sp_nprocs[-1], 200)
                ax.plot(xp, xp / p2p_n_base, ":", color="orange", label="P2P ideal scaling")
                all_sp_nprocs |= set(p2p_sp_nprocs)

        time_label = "MC loop time" if args.loop_times else "Step time"
        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Speedup")
        ax.set_title(f"Speedup ({time_label.lower()}, cycle {cycle}) vs. processor count")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        all_sp_ticks = sorted(all_sp_nprocs)
        ax.set_xticks(all_sp_ticks)
        ax.set_xticklabels([str(n) for n in all_sp_ticks], rotation=90)

        plt.tight_layout()
        out = args.output or "scalability_speedup.png"
        plt.savefig(out, dpi=150)
        print(f"Saved {out}")
        if args.show:
            plt.show()
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    rdma_label = "RDMA" if p2p_cycle_data else None

    cycle_colors = {}
    for cycle in selected_cycles:
        times = []
        valid_nprocs = []
        for nprocs in nprocs_list:
            ct = all_cycle_data[nprocs]
            if cycle in ct:
                times.append(ct[cycle])
                valid_nprocs.append(nprocs)
        if valid_nprocs:
            label = f"{rdma_label + ' c' if rdma_label else 'C'}ycle {cycle}"
            line, = ax.plot(valid_nprocs, times, "o-", label=label, markersize=5)
            cycle_colors[cycle] = line.get_color()

    if p2p_cycle_data:
        p2p_nprocs_list = sorted(p2p_cycle_data.keys())
        if args.optimal:
            p2p_max_common = min(max(ct.keys()) for ct in p2p_cycle_data.values())
            p2p_selected = [adjust_cycle(p2p_max_common)]
        else:
            p2p_selected = selected_cycles
        for cyc in p2p_selected:
            times = []
            valid_nprocs = []
            for nprocs in p2p_nprocs_list:
                ct = p2p_cycle_data[nprocs]
                if cyc in ct:
                    times.append(ct[cyc])
                    valid_nprocs.append(nprocs)
            if valid_nprocs:
                color = cycle_colors.get(cyc)
                ax.plot(valid_nprocs, times, "s--", label=f"P2P cycle {cyc}",
                        markersize=5, color=color)

    if args.optimal:
        cycle = selected_cycles[0]
        fit_label = "fit" if args.fit_A else "ref"
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
            ax.plot(x, A / x, "--", color="gray",
                    label=f"{'RDMA strong' if rdma_label else 'Strong'} scaling {fit_label} (A={A:.0f})")

        if p2p_cycle_data:
            p2p_nprocs_list = sorted(p2p_cycle_data.keys())
            p2p_cycle = p2p_selected[0]
            p2p_opt_nprocs = []
            p2p_opt_times = []
            for nprocs in p2p_nprocs_list:
                t = p2p_cycle_data[nprocs].get(p2p_cycle)
                if t is not None:
                    p2p_opt_nprocs.append(nprocs)
                    p2p_opt_times.append(t)
            if p2p_opt_nprocs:
                if args.fit_A:
                    B = fit_strong_scaling_A(p2p_opt_nprocs, p2p_opt_times)
                    print(f"P2P fitted B = {B:.1f}")
                else:
                    B = p2p_opt_nprocs[0] * p2p_opt_times[0]
                xp = np.linspace(p2p_opt_nprocs[0], p2p_opt_nprocs[-1], 200)
                ax.plot(xp, B / xp, ":", color="orange",
                        label=f"P2P strong scaling {fit_label} (B={B:.0f})")

        def print_scaling_diff(label, nprocs_list, times_list, A_val):
            n_ref = nprocs_list[0]
            t_ref = times_list[0]
            print(f"\n{label} — deviation from ideal scaling (A={A_val:.0f}):")
            print(f"  {'Procs':>8}  {'Actual':>10}  {'Ideal':>10}  {'Diff':>10}  {'Diff%':>8}  {'Speedup':>8}  {'Expected':>8}  {'Effic%':>7}")
            for n, t in zip(nprocs_list, times_list):
                ideal = A_val / n
                diff = t - ideal
                pct = diff / ideal * 100
                speedup = t_ref / t
                expected = n / n_ref
                efficiency = speedup / expected * 100
                print(f"  {n:>8}  {t:>10.3f}  {ideal:>10.3f}  {diff:>+10.3f}  {pct:>+7.1f}%  {speedup:>8.2f}  {expected:>8.2f}  {efficiency:>6.1f}%")

        if opt_nprocs:
            lbl = "RDMA" if p2p_cycle_data else "Strong scaling"
            print_scaling_diff(lbl, opt_nprocs, opt_times, A)
        if p2p_cycle_data and p2p_opt_nprocs:
            print_scaling_diff("P2P", p2p_opt_nprocs, p2p_opt_times, B)

    time_label = "MC loop time" if args.loop_times else "Step time"
    ax.set_xlabel("Number of processors")
    ax.set_ylabel(f"{time_label} (s)")
    ax.set_title(f"Scalability: {time_label.lower()} vs. processor count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    all_nprocs = sorted(set(nprocs_list) | (set(p2p_cycle_data.keys()) if p2p_cycle_data else set()))
    ax.set_xticks(all_nprocs)
    ax.set_xticklabels([str(n) for n in all_nprocs], rotation=90)

    plt.tight_layout()
    out = args.output or ("scalability_optimal.png" if args.optimal else "scalability.png")
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()

    if args.optimal and args.plot_diff and opt_nprocs:
        cycle = selected_cycles[0]
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rdma_diff_pct = [(t - A / n) / (A / n) * 100 for n, t in zip(opt_nprocs, opt_times)]
        ax2.plot(opt_nprocs, rdma_diff_pct, "o-",
                 label=f"{'RDMA' if p2p_cycle_data else 'Measured'} cycle {cycle}",
                 markersize=5)
        if p2p_cycle_data and p2p_opt_nprocs:
            p2p_diff_pct = [(t - B / n) / (B / n) * 100 for n, t in zip(p2p_opt_nprocs, p2p_opt_times)]
            ax2.plot(p2p_opt_nprocs, p2p_diff_pct, "s--",
                     label=f"P2P cycle {p2p_cycle}", markersize=5)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Number of processors")
        ax2.set_ylabel("Deviation from ideal (%)")
        ax2.set_title(f"Deviation from ideal strong scaling (cycle {cycle})")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale("log", base=2)
        all_nprocs = sorted(set(opt_nprocs) | (set(p2p_opt_nprocs) if p2p_cycle_data else set()))
        ax2.set_xticks(all_nprocs)
        ax2.set_xticklabels([str(n) for n in all_nprocs], rotation=90)
        plt.tight_layout()
        diff_out = os.path.splitext(out)[0] + "_diff" + os.path.splitext(out)[1]
        plt.savefig(diff_out, dpi=150)
        print(f"Saved {diff_out}")
        if args.show:
            plt.show()


if __name__ == "__main__":
    main()
