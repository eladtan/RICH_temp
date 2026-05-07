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
    parser = argparse.ArgumentParser(
        description="Plot weak scalability of hohlraum runs. "
        "Automatically detects runs where N_base scales with node count. "
        "Compares runs by simulation time (not cycle number)."
    )
    parser.add_argument("--ignore", type=str, default="",
                        help="Comma-separated list of processor counts to ignore")
    parser.add_argument("--cycles", type=str, default="",
                        help="Comma-separated list of specific cycles (in the reference run) to plot")
    parser.add_argument("--optimal", action="store_true",
                        help="Plot only the max common simulation time plus an ideal weak-scaling reference line")
    parser.add_argument("--sum", type=int, nargs="?", const=0, default=None,
                        help="Sum cycle times up to max common simulation time. "
                             "No value: sum all cycles. --sum=X: use the last X "
                             "reference-run time points to define the summation window.")
    parser.add_argument("--sum-T", type=float, default=None, dest="sum_T",
                        help="Average cycle times over the last X seconds of simulation "
                             "time for each file independently (e.g. --sum-T=1e-10 for "
                             "the last 0.1 ns). Shows a table with average time per "
                             "cycle, particles, ideal time, and efficiency.")
    parser.add_argument("--no-rebalances", action="store_true",
                        help="With --sum/--sum-T, exclude rebalance cycles (11, 21, ..., 101, ..., 1001, ...)")
    parser.add_argument("--no-peaks", action="store_true",
                        help="With --sum/--sum-T, discard cycles whose time exceeds "
                             "1.4x the initial average (remove outlier peaks)")
    parser.add_argument("--show-times", action="store_true",
                        help="With --sum/--sum-T, show per-cycle running times in the table")
    parser.add_argument("--efficiency", action="store_true",
                        help="Plot weak scaling efficiency T_ref / T(N) vs. processor count")
    parser.add_argument("--per-particle", action="store_true",
                        help="Divide each cycle's time by particles per rank (total particles / nprocs)")
    parser.add_argument("--p2p", action="store_true",
                        help="Include P2P files (hohlraum_P2P_*) as additional data")
    parser.add_argument("--loop-times", action="store_true",
                        help="Use MC loop elapsed times ('Elapsed: X seconds') instead of step times")
    parser.add_argument("--base", type=int, default=None,
                        help="Use X ranks as the baseline reference instead of the minimum")
    parser.add_argument("--tasks-per-node", type=int, default=112,
                        help="Tasks per node (default: 112, for Leonardo dcgp)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for .out files")
    return parser.parse_args()


def is_finished(filepath):
    """Check if a simulation output file completed (has 'Total wall time:' line)."""
    with open(filepath, "rb") as f:
        try:
            f.seek(-4096, 2)
        except OSError:
            f.seek(0)
        tail = f.read().decode("utf-8", errors="replace")
    return "Total wall time:" in tail


def find_latest_files(directory, include_p2p=False):
    """Find the latest *finished* .out file for each processor count (highest job number).
    Returns (regular_dict, p2p_dict) where each is {nprocs: filepath}.
    p2p_dict is empty if include_p2p is False."""
    pattern = os.path.join(directory, "hohlraum_*_n*.out")
    files = glob.glob(pattern)

    regular_re = re.compile(r"hohlraum_(\d+)_n(\d+)\.out$")
    p2p_re = re.compile(r"hohlraum_P2P_(\d+)_n(\d+)\.out$")

    rank_files = defaultdict(list)
    p2p_rank_files = defaultdict(list)
    for f in files:
        basename = os.path.basename(f)
        m = p2p_re.search(basename)
        if m:
            if include_p2p:
                job_id = int(m.group(1))
                nprocs = int(m.group(2))
                p2p_rank_files[nprocs].append((job_id, f))
            continue
        m = regular_re.search(basename)
        if m:
            job_id = int(m.group(1))
            nprocs = int(m.group(2))
            rank_files[nprocs].append((job_id, f))

    def pick_latest_finished(rf):
        latest = {}
        for nprocs, entries in rf.items():
            entries.sort(key=lambda x: x[0], reverse=True)
            for _, filepath in entries:
                if is_finished(filepath):
                    latest[nprocs] = filepath
                    break
        return latest

    return pick_latest_finished(rank_files), pick_latest_finished(p2p_rank_files)


def parse_nbase(filepath):
    """Extract N_base from the 'Generated ... (N_base=X)' line."""
    nbase_re = re.compile(r"N_base=(\d+)")
    with open(filepath, "r") as f:
        for line in f:
            m = nbase_re.search(line)
            if m:
                return int(m.group(1))
    return None


def parse_cycle_times(filepath, loop_times=False):
    """Parse cycle data from output file.
    Returns {cycle: (sim_time_ns, wall_time_s)}.
    sim_time_ns is the simulation time in nanoseconds at that cycle.
    wall_time_s is either the step time or MC loop elapsed time."""
    progress_re = re.compile(
        r"^Cycle\s+(\d+)\s+t=([\d.eE+-]+)\s+ns.*\bstep=([\d.eE+-]+)s")
    cycle_at_re = re.compile(r"^Cycle\s+(\d+)\s+at\s+time\s+([\d.eE+-]+)")
    elapsed_re = re.compile(r"^(?:Elapsed|Loop time):\s+([\d.]+)\s+seconds")

    sim_times = {}
    step_times = {}
    loop_elapsed = {}
    current_cycle = None

    with open(filepath, "r") as f:
        for line in f:
            m = progress_re.match(line)
            if m:
                cycle = int(m.group(1))
                sim_times[cycle] = float(m.group(2))
                step_times[cycle] = float(m.group(3))
                continue

            if loop_times:
                m = cycle_at_re.match(line)
                if m:
                    current_cycle = int(m.group(1))
                    continue
                m = elapsed_re.match(line)
                if m and current_cycle is not None:
                    loop_elapsed[current_cycle + 1] = float(m.group(1))

    if loop_times:
        return {c: (sim_times[c], loop_elapsed[c])
                for c in loop_elapsed if c in sim_times}
    else:
        return {c: (sim_times[c], step_times[c])
                for c in step_times if c in sim_times}


def parse_particle_counts(filepath):
    """Parse total particle counts per cycle from 'Starting with N' lines.
    Returns {cycle: starting_particle_count}."""
    cycle_at_re = re.compile(r"^Cycle\s+(\d+)\s+at\s+time\s+")
    starting_re = re.compile(r"^Start(?:ing|ed) with (\d+)\.")
    counts = {}
    current_cycle = None
    with open(filepath, "r") as f:
        for line in f:
            m = cycle_at_re.match(line)
            if m:
                current_cycle = int(m.group(1))
                continue
            m = starting_re.match(line)
            if m and current_cycle is not None:
                counts[current_cycle + 1] = int(m.group(1))
    return counts


def find_cycle_at_time(cycle_data, target_time, skip_rebalances=False):
    """Find the first cycle in cycle_data whose sim_time >= target_time.
    Skips the very last cycle in the run (may be incomplete).
    If skip_rebalances is True and the matched cycle is a rebalance cycle
    (cycle > 1 and cycle % 10 == 1), advance to the next cycle.
    Returns (cycle, sim_time, step_time) or None."""
    sorted_cycles = sorted(cycle_data.keys())
    last_idx = len(sorted_cycles) - 1
    for i, cycle in enumerate(sorted_cycles):
        sim_time, step_time = cycle_data[cycle]
        if sim_time >= target_time - 1e-12:
            if i == last_idx:
                if i == 0:
                    return None
                prev = sorted_cycles[i - 1]
                ps, pt = cycle_data[prev]
                return prev, ps, pt
            if skip_rebalances and cycle > 1 and cycle % 10 == 1:
                if i + 1 < last_idx:
                    next_cycle = sorted_cycles[i + 1]
                    ns, nt = cycle_data[next_cycle]
                    return next_cycle, ns, nt
            return cycle, sim_time, step_time
    return None


def pick_default_times(ref_data, max_common_time, num_curves=5):
    """Pick evenly spaced simulation times from the reference run,
    avoiding rebalance-cycle times."""
    is_rebalance = lambda c: c > 1 and c % 10 == 1
    ref_sorted = sorted(
        ((cycle, sim_time) for cycle, (sim_time, _) in ref_data.items()
         if sim_time <= max_common_time + 1e-12 and not is_rebalance(cycle)),
        key=lambda x: x[1]
    )
    if not ref_sorted:
        return []
    ref_times = [t for _, t in ref_sorted]
    n = len(ref_times)
    indices = [round(n * i / num_curves) - 1 for i in range(1, num_curves + 1)]
    indices = sorted(set(max(0, min(idx, n - 1)) for idx in indices))
    return [ref_times[i] for i in indices]


def detect_weak_scaling_runs(all_files, tasks_per_node):
    """From all available files, identify weak scaling runs where
    N_base / nodes is approximately constant. Returns the subset
    dict {nprocs: filepath} that forms a weak scaling series,
    plus the detected nbase_per_node value."""
    file_info = {}
    for nprocs, filepath in all_files.items():
        nbase = parse_nbase(filepath)
        if nbase is None:
            continue
        nodes = nprocs / tasks_per_node
        if nodes < 1:
            continue
        ratio = nbase / nodes
        file_info[nprocs] = (filepath, nbase, nodes, ratio)

    if not file_info:
        return {}, 0

    ratios = [info[3] for info in file_info.values()]
    ratio_counts = defaultdict(list)
    for nprocs, (filepath, nbase, nodes, ratio) in file_info.items():
        rounded = round(ratio / 100) * 100
        ratio_counts[rounded].append(nprocs)

    best_ratio = max(ratio_counts, key=lambda r: len(ratio_counts[r]))
    weak_nprocs = set(ratio_counts[best_ratio])

    weak_files = {n: all_files[n] for n in weak_nprocs}
    return weak_files, best_ratio


def print_table(file_info):
    headers = ["Procs", "Nodes", "N_base", "File", "Cycles",
               "Max t (ns)", "Max Particles"]
    right_align = {0, 1, 2, 4, 5, 6}

    rows = []
    for nprocs in sorted(file_info.keys()):
        fname, ncycles, nbase, nodes, max_time, max_particles = file_info[nprocs]
        rows.append((
            str(nprocs), f"{nodes:.0f}", str(nbase),
            os.path.basename(fname), str(ncycles),
            f"{max_time:.4f}",
            f"{max_particles:,}" if max_particles else "N/A"
        ))

    w = [max(len(h), max(len(r[i]) for r in rows))
         for i, h in enumerate(headers)]

    sep = "+-" + "-+-".join("-" * wi for wi in w) + "-+"
    header = "| " + " | ".join(
        h.rjust(wi) if i in right_align else h.ljust(wi)
        for i, (h, wi) in enumerate(zip(headers, w))
    ) + " |"

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        print("| " + " | ".join(
            r[i].rjust(w[i]) if i in right_align else r[i].ljust(w[i])
            for i in range(len(headers))
        ) + " |")
    print(sep)


def print_times_table(selected_times, nprocs_list, cycle_data,
                      skip_rebalances=False, particle_data=None,
                      per_particle=False, ref_nprocs=None):
    """Print a table: rows = simulation times, columns = proc counts,
    cells = matched cycle number and step time."""
    if ref_nprocs is None:
        ref_nprocs = nprocs_list[0]
    col_time = "Sim time (ns)"
    proc_headers = [str(n) for n in nprocs_list]

    rows = []
    for target_time in selected_times:
        row = [f"{target_time:.4f}"]
        ref_match = find_cycle_at_time(cycle_data[ref_nprocs], target_time,
                                       skip_rebalances)
        ref_val = None
        if ref_match:
            ref_val = ref_match[2]
            if per_particle and particle_data:
                pc = particle_data.get(ref_nprocs, {}).get(ref_match[0])
                if pc:
                    ref_val /= (pc / ref_nprocs)
        for nprocs in nprocs_list:
            match = find_cycle_at_time(cycle_data[nprocs], target_time,
                                       skip_rebalances)
            if match:
                cycle, sim_time, step_time = match
                val = step_time
                if per_particle and particle_data:
                    pc = particle_data.get(nprocs, {}).get(cycle)
                    if pc:
                        val /= (pc / nprocs)
                if ref_val is not None and val > 0:
                    eff = ref_val / val * 100
                    row.append(f"{val:.3e}s c{cycle} {eff:.0f}%"
                               if per_particle
                               else f"{val:.3f}s c{cycle} {eff:.0f}%")
                else:
                    row.append(f"{val:.3e}s c{cycle}"
                               if per_particle
                               else f"{val:.3f}s c{cycle}")
            else:
                row.append("N/A")
        rows.append(row)

    headers = [col_time] + proc_headers
    w = [max(len(h), max(len(r[i]) for r in rows))
         for i, h in enumerate(headers)]

    sep = "+-" + "-+-".join("-" * wi for wi in w) + "-+"
    header_line = "| " + " | ".join(
        h.rjust(wi) for i, (h, wi) in enumerate(zip(headers, w))
    ) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for r in rows:
        print("| " + " | ".join(
            r[i].rjust(w[i]) for i in range(len(headers))
        ) + " |")
    print(sep)


def analyze_sum_T(nprocs_list, cycle_data, files_dict, particle_data,
                  delta_t, no_rebalances, tasks_per_node, no_peaks=False):
    """Compute per-run averages over the last delta_t ns of simulation time.
    Returns a list of result dicts, one per nprocs."""
    is_rebalance = lambda c: c > 1 and c % 10 == 1
    results = []
    for nprocs in nprocs_list:
        ct = cycle_data[nprocs]
        filepath = files_dict[nprocs]
        pc = particle_data.get(nprocs, {})
        nodes = nprocs / tasks_per_node

        sorted_cycles = sorted(ct.keys())
        if len(sorted_cycles) > 1:
            sorted_cycles = sorted_cycles[:-1]
        else:
            continue

        t_max = max(ct[c][0] for c in sorted_cycles)
        t_start = t_max - delta_t

        selected = []
        for c in sorted_cycles:
            sim_time, wall_time = ct[c]
            if sim_time >= t_start - 1e-12:
                if no_rebalances and is_rebalance(c):
                    continue
                selected.append((c, wall_time))

        if not selected:
            continue

        if no_peaks:
            preliminary_avg = np.mean([wt for _, wt in selected])
            threshold = 1.4 * preliminary_avg
            selected = [(c, wt) for c, wt in selected if wt <= threshold]
            if not selected:
                continue

        avg_time = np.mean([wt for _, wt in selected])

        part_counts = [pc.get(c, 0) for c, _ in selected]
        valid_parts = [p for p in part_counts if p > 0]
        avg_particles = np.mean(valid_parts) if valid_parts else 0

        cycle_range = f"{selected[0][0]}..{selected[-1][0]}"

        results.append({
            'nprocs': nprocs,
            'nodes': nodes,
            'filename': os.path.basename(filepath),
            'cycle_range': cycle_range,
            'count': len(selected),
            'avg_time': avg_time,
            'avg_particles': avg_particles,
            'cycle_times': [(c, wt) for c, wt in selected],
        })
    return results


def print_sum_T_table(results, ref_avg_time, show_times=False):
    """Print the --sum-T results table with ideal time, deviation, efficiency."""
    headers = ["Ranks", "Nodes", "File", "Cycles", "#",
               "Avg time (s)", "Avg particles", "Ideal (s)",
               "Deviation (s)", "Efficiency (%)"]
    right_align = {0, 1, 3, 4, 5, 6, 7, 8, 9}
    if show_times:
        headers.append("Cycle times (s)")
        right_align.add(10)

    rows = []
    for r in results:
        deviation = r['avg_time'] - ref_avg_time
        efficiency = (ref_avg_time / r['avg_time'] * 100
                      if r['avg_time'] > 0 else 0)
        row = (
            str(r['nprocs']),
            f"{r['nodes']:.0f}",
            r['filename'],
            r['cycle_range'],
            str(r['count']),
            f"{r['avg_time']:.3f}",
            f"{r['avg_particles']:.0f}",
            f"{ref_avg_time:.3f}",
            f"{deviation:+.3f}",
            f"{efficiency:.1f}",
        )
        if show_times:
            times_str = " ".join(f"{wt:.2f}" for _, wt in r['cycle_times'])
            row = row + (times_str,)
        rows.append(row)

    w = [max(len(h), max(len(r[i]) for r in rows))
         for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * wi for wi in w) + "-+"
    header_line = "| " + " | ".join(
        h.rjust(wi) if i in right_align else h.ljust(wi)
        for i, (h, wi) in enumerate(zip(headers, w))
    ) + " |"

    print(sep)
    print(header_line)
    print(sep)
    for r in rows:
        print("| " + " | ".join(
            r[i].rjust(w[i]) if i in right_align else r[i].ljust(w[i])
            for i in range(len(headers))
        ) + " |")
    print(sep)


def print_weak_scaling_table(nprocs_list, times_list, t_ref, tasks_per_node):
    print(f"\nWeak scaling analysis (T_ref = {t_ref:.3f}s at {nprocs_list[0]} procs):")
    print(f"  {'Procs':>8}  {'Nodes':>6}  {'Time':>10}  {'Ideal':>10}  "
          f"{'Overhead':>10}  {'Overhead%':>10}  {'Effic%':>7}")
    for n, t in zip(nprocs_list, times_list):
        nodes = n / tasks_per_node
        overhead = t - t_ref
        pct = overhead / t_ref * 100
        efficiency = t_ref / t * 100
        print(f"  {n:>8}  {nodes:>6.0f}  {t:>10.3f}  {t_ref:>10.3f}  "
              f"{overhead:>+10.3f}  {pct:>+9.1f}%  {efficiency:>6.1f}%")


def main():
    args = parse_args()

    ignore_set = set()
    if args.ignore:
        ignore_set = {int(x.strip()) for x in args.ignore.split(",")}

    all_files, p2p_all_files = find_latest_files(args.dir, include_p2p=args.p2p)
    if not all_files:
        print("No matching .out files found.", file=sys.stderr)
        sys.exit(1)

    for ign in ignore_set:
        all_files.pop(ign, None)
        p2p_all_files.pop(ign, None)

    weak_files, nbase_per_node = detect_weak_scaling_runs(all_files, args.tasks_per_node)
    if not weak_files or len(weak_files) < 2:
        print("Could not detect a weak scaling series (need at least 2 runs with "
              "N_base proportional to node count).", file=sys.stderr)
        if weak_files:
            print(f"  Only found 1 run: {list(weak_files.keys())[0]} procs", file=sys.stderr)
        sys.exit(1)

    def load_weak_scaling_data(files_dict):
        cd, pd, fi = {}, {}, {}
        for nprocs in sorted(files_dict.keys()):
            filepath = files_dict[nprocs]
            ct = parse_cycle_times(filepath, loop_times=args.loop_times)
            nbase = parse_nbase(filepath)
            nodes = nprocs / args.tasks_per_node
            pc = parse_particle_counts(filepath)
            if not ct:
                print(f"Warning: no cycle data in {filepath}, skipping.", file=sys.stderr)
                continue
            cd[nprocs] = ct
            pd[nprocs] = pc
            max_cycle = max(ct.keys())
            max_time = max(st for st, _ in ct.values())
            max_particles = max(pc.values()) if pc else 0
            fi[nprocs] = (filepath, max_cycle, nbase, nodes, max_time, max_particles)
        return cd, pd, fi

    cycle_data, particle_data, file_info = load_weak_scaling_data(weak_files)

    if not cycle_data:
        print("No cycle data found in any weak-scaling file.", file=sys.stderr)
        sys.exit(1)

    print(f"Detected weak scaling series: N_base/node \u2248 {nbase_per_node:.0f}")
    print()
    print_table(file_info)
    print()

    p2p_weak_files, p2p_cycle_data, p2p_particle_data = {}, {}, {}
    p2p_nprocs_list = []
    if p2p_all_files:
        p2p_weak_files, _ = detect_weak_scaling_runs(p2p_all_files, args.tasks_per_node)
        if p2p_weak_files:
            p2p_cycle_data, p2p_particle_data, p2p_file_info = \
                load_weak_scaling_data(p2p_weak_files)
            if p2p_cycle_data:
                p2p_nprocs_list = sorted(p2p_cycle_data.keys())
                print("P2P runs:")
                print_table(p2p_file_info)
                print()

    nprocs_list = sorted(cycle_data.keys())
    if args.base is not None:
        if args.base not in cycle_data:
            print(f"Error: --base={args.base} not found in available runs: {nprocs_list}",
                  file=sys.stderr)
            sys.exit(1)
        ref_nprocs = args.base
    else:
        ref_nprocs = nprocs_list[0]
    ref_data = cycle_data[ref_nprocs]

    max_time_per_run = {n: max(st for st, _ in ct.values())
                        for n, ct in cycle_data.items()}
    max_common_time = min(max_time_per_run.values())
    print(f"Reference run: {ref_nprocs} procs")
    print(f"Max common simulation time: {max_common_time:.4f} ns")

    is_rebalance = lambda c: c > 1 and c % 10 == 1

    # --sum-T mode: average over last X seconds of each file's simulation time
    if args.sum_T is not None:
        delta_t = args.sum_T * 1e9  # convert seconds to nanoseconds
        skip_label = ", excl. rebalances" if args.no_rebalances else ""
        time_label = "MC loop time" if args.loop_times else "Step time"
        rdma_label = "RDMA " if p2p_nprocs_list else ""

        results = analyze_sum_T(nprocs_list, cycle_data, weak_files,
                                particle_data, delta_t,
                                args.no_rebalances, args.tasks_per_node,
                                no_peaks=args.no_peaks)
        if not results:
            print("No valid data for --sum-T analysis.", file=sys.stderr)
            sys.exit(1)

        ref_result = next((r for r in results if r['nprocs'] == ref_nprocs),
                          results[0])
        ref_avg_time = ref_result['avg_time']

        print(f"\nAveraging {rdma_label}{time_label.lower()} over the last "
              f"{args.sum_T:.2e} s ({delta_t:.4f} ns) of each run{skip_label}")
        print(f"Reference: {ref_result['nprocs']} procs, "
              f"avg {time_label.lower()} = {ref_avg_time:.3f}s")
        print()
        print_sum_T_table(results, ref_avg_time, show_times=args.show_times)

        qualifying = [r['nprocs'] for r in results]
        efficiencies = [ref_avg_time / r['avg_time'] * 100
                        if r['avg_time'] > 0 else 0 for r in results]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qualifying, efficiencies, "o-",
                label=f"{rdma_label}Efficiency (last {args.sum_T:.2e} s{skip_label})",
                markersize=6)
        for x, y, r in zip(qualifying, efficiencies, results):
            ax.annotate(f"{r['avg_time']:.2f}s", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=7)

        p2p_results = []
        if p2p_nprocs_list and p2p_weak_files:
            p2p_results = analyze_sum_T(
                p2p_nprocs_list, p2p_cycle_data, p2p_weak_files,
                p2p_particle_data, delta_t,
                args.no_rebalances, args.tasks_per_node,
                no_peaks=args.no_peaks)
            if p2p_results:
                p2p_ref_result = next(
                    (r for r in p2p_results if r['nprocs'] == ref_nprocs),
                    p2p_results[0])
                p2p_ref_avg = p2p_ref_result['avg_time']
                print(f"\nP2P — averaging {time_label.lower()} over the last "
                      f"{args.sum_T:.2e} s ({delta_t:.4f} ns){skip_label}")
                print(f"Reference: {p2p_ref_result['nprocs']} procs, "
                      f"avg {time_label.lower()} = {p2p_ref_avg:.3f}s")
                print()
                print_sum_T_table(p2p_results, p2p_ref_avg,
                                  show_times=args.show_times)

                p2p_qualifying = [r['nprocs'] for r in p2p_results]
                p2p_efficiencies = [
                    p2p_ref_avg / r['avg_time'] * 100
                    if r['avg_time'] > 0 else 0 for r in p2p_results]
                ax.plot(p2p_qualifying, p2p_efficiencies, "s--", color="red",
                        label=f"P2P Efficiency (last {args.sum_T:.2e} s{skip_label})",
                        markersize=6)
                for x, y, r in zip(p2p_qualifying, p2p_efficiencies, p2p_results):
                    ax.annotate(f"{r['avg_time']:.2f}s", (x, y),
                                textcoords="offset points", xytext=(0, -12),
                                ha="center", va="top", fontsize=7, color="red")

        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8,
                   label="Ideal (100%)")
        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Weak scaling efficiency (%)")
        ax.set_ylim(bottom=50)
        ax.set_title(f"Weak scaling efficiency (last {args.sum_T:.2e} s{skip_label})")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        all_ticks = sorted(set(qualifying) |
                           set(r['nprocs'] for r in p2p_results))
        ax.set_xticks(all_ticks)
        ax.set_xticklabels([str(n) for n in all_ticks], rotation=90)
        secax = ax.secondary_xaxis("top",
                                   functions=(lambda x: x / args.tasks_per_node,
                                              lambda x: x * args.tasks_per_node))
        secax.set_xlabel("Number of nodes")

        plt.tight_layout()
        out = args.output or "weak_scalability_sum_T.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        if args.show:
            plt.show()
        return

    # --sum mode
    if args.sum is not None:
        last_n = args.sum

        skip_label = ""
        if args.no_rebalances:
            skip_label = ", excl. rebalances"

        qualifying = nprocs_list
        end_cycles = {}

        for nprocs in qualifying:
            ct = cycle_data[nprocs]
            match = find_cycle_at_time(ct, max_common_time)
            if match is not None:
                end_cycles[nprocs] = match[0]

        if last_n > 0:
            available = [ec for ec in end_cycles.values()]
            if available:
                effective_n = min(last_n, min(available))
                if effective_n < last_n:
                    print(f"Note: capping --sum={last_n} to {effective_n} "
                          f"(smallest end cycle across runs)")
                last_n = effective_n

        sum_times = []
        sum_ncycles = []
        sum_cycle_times = []

        for nprocs in qualifying:
            ct = cycle_data[nprocs]
            if nprocs not in end_cycles:
                sum_times.append(0.0)
                sum_ncycles.append(0)
                sum_cycle_times.append([])
                continue
            end_cycle = end_cycles[nprocs]

            if last_n > 0:
                start_cycle = end_cycle - last_n + 1
                matching = [
                    (cycle, step_time)
                    for cycle, (sim_time, step_time) in ct.items()
                    if start_cycle <= cycle <= end_cycle
                    and not (args.no_rebalances and is_rebalance(cycle))
                ]
            else:
                matching = [
                    (cycle, step_time)
                    for cycle, (sim_time, step_time) in ct.items()
                    if cycle <= end_cycle
                    and not (args.no_rebalances and is_rebalance(cycle))
                ]

            if args.no_peaks and matching:
                preliminary_avg = np.mean([st for _, st in matching])
                threshold = 1.4 * preliminary_avg
                matching = [(c, st) for c, st in matching if st <= threshold]

            if args.per_particle:
                pc_data = particle_data.get(nprocs, {})
                total = sum(st / (pc_data[c] / nprocs) for c, st in matching
                            if pc_data.get(c))
            else:
                total = sum(st for _, st in matching)
            sum_times.append(total)
            sum_ncycles.append(len(matching))
            sum_cycle_times.append(matching)

        if last_n > 0:
            print(f"Summing last {last_n} cycles before t={max_common_time:.4f} ns"
                  f"{skip_label}")
        else:
            print(f"Summing all cycles up to t={max_common_time:.4f} ns"
                  f"{skip_label}")
        print()

        sum_headers = ["Procs", "Nodes", "Summed cycles", "Count", "Total (s)"]
        if args.show_times:
            sum_headers.append("Cycle times (s)")
        sum_rows = []
        for nprocs, total, nc, ctimes in zip(qualifying, sum_times,
                                             sum_ncycles, sum_cycle_times):
            nodes = nprocs / args.tasks_per_node
            ec = end_cycles.get(nprocs, "?")
            if last_n > 0:
                sc = ec - last_n + 1 if isinstance(ec, int) else "?"
                cr = f"{sc}..{ec}"
            else:
                cr = f"1..{ec}"
            row = (str(nprocs), f"{nodes:.0f}", cr,
                   str(nc), f"{total:.3f}")
            if args.show_times:
                times_str = " ".join(f"{st:.2f}" for _, st in ctimes)
                row = row + (times_str,)
            sum_rows.append(row)
        sw = [max(len(h), max(len(r[i]) for r in sum_rows))
              for i, h in enumerate(sum_headers)]
        ssep = "+-" + "-+-".join("-" * wi for wi in sw) + "-+"
        sheader = "| " + " | ".join(
            h.rjust(wi) for h, wi in zip(sum_headers, sw)) + " |"
        print(ssep)
        print(sheader)
        print(ssep)
        for r in sum_rows:
            print("| " + " | ".join(
                r[i].rjust(sw[i]) for i in range(len(sum_headers))
            ) + " |")
        print(ssep)

        range_label = (f"last {last_n} cycles" if last_n > 0
                       else f"all cycles up to t={max_common_time:.2f} ns")
        ref_idx = qualifying.index(ref_nprocs) if ref_nprocs in qualifying else 0
        t_ref = sum_times[ref_idx]

        efficiencies = [t_ref / t * 100 for t in sum_times]
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(qualifying, efficiencies, "o-",
                label=f"Efficiency ({range_label}{skip_label})", markersize=6)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8,
                    label="Ideal (100%)")
        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Weak scaling efficiency (%)")
        ax.set_ylim(bottom=50)
        ax.set_title(f"Weak scaling efficiency ({range_label}{skip_label})")

        print_weak_scaling_table(qualifying, sum_times, t_ref, args.tasks_per_node)

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(qualifying)
        ax.set_xticklabels([str(n) for n in qualifying], rotation=90)
        secax = ax.secondary_xaxis("top",
                                    functions=(lambda x: x / args.tasks_per_node,
                                               lambda x: x * args.tasks_per_node))
        secax.set_xlabel("Number of nodes")

        plt.tight_layout()
        out = args.output or "weak_scalability_sum.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        if args.show:
            plt.show()
        return

    # Select simulation times for comparison
    if args.cycles:
        ref_cycles = sorted(int(x.strip()) for x in args.cycles.split(","))
        selected_times = []
        for c in ref_cycles:
            if c in ref_data:
                selected_times.append(ref_data[c][0])
            else:
                print(f"Warning: cycle {c} not found in reference run "
                      f"({ref_nprocs} procs), skipping.", file=sys.stderr)
        selected_times = sorted(set(selected_times))
    else:
        selected_times = pick_default_times(ref_data, max_common_time)

    selected_times = [t for t in selected_times if t <= max_common_time + 1e-12]
    if not selected_times:
        print(f"No valid simulation times to plot "
              f"(max common time is {max_common_time:.4f} ns).", file=sys.stderr)
        sys.exit(1)

    if args.optimal or args.efficiency:
        selected_times = [max(selected_times)]

    print(f"Plotting at simulation times (ns): "
          f"{[f'{t:.4f}' for t in selected_times]}")
    print()
    print_times_table(selected_times, nprocs_list, cycle_data,
                      skip_rebalances=args.no_rebalances,
                      particle_data=particle_data,
                      per_particle=args.per_particle,
                      ref_nprocs=ref_nprocs)
    print()

    if args.efficiency:
        target_time = selected_times[0]
        eff_nprocs = []
        eff_times = []
        eff_cycles = []
        for nprocs in nprocs_list:
            match = find_cycle_at_time(cycle_data[nprocs], target_time,
                                       args.no_rebalances)
            if match:
                cycle, _, step_time = match
                if args.per_particle:
                    pc = particle_data.get(nprocs, {}).get(cycle)
                    if pc:
                        step_time /= (pc / nprocs)
                eff_nprocs.append(nprocs)
                eff_times.append(step_time)
                eff_cycles.append(cycle)
        if not eff_nprocs:
            print("No data for efficiency plot.", file=sys.stderr)
            sys.exit(1)

        ref_idx = eff_nprocs.index(ref_nprocs) if ref_nprocs in eff_nprocs else 0
        t_ref = eff_times[ref_idx]
        efficiencies = [t_ref / t * 100 for t in eff_times]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(eff_nprocs, efficiencies, "o-",
                label=f"Efficiency (t={target_time:.4f} ns)", markersize=6)
        for x, y, c in zip(eff_nprocs, efficiencies, eff_cycles):
            ax.annotate(str(c), (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", va="bottom", fontsize=7)
        ax.axhline(100, color="gray", linestyle="--", linewidth=0.8,
                    label="Ideal (100%)")

        print_weak_scaling_table(eff_nprocs, eff_times, t_ref, args.tasks_per_node)

        ax.set_xlabel("Number of processors")
        ax.set_ylabel("Weak scaling efficiency (%)")
        ax.set_title(f"Weak scaling efficiency (t={target_time:.4f} ns)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(eff_nprocs)
        ax.set_xticklabels([str(n) for n in eff_nprocs], rotation=90)
        secax = ax.secondary_xaxis("top",
                                    functions=(lambda x: x / args.tasks_per_node,
                                               lambda x: x * args.tasks_per_node))
        secax.set_xlabel("Number of nodes")

        plt.tight_layout()
        out = args.output or "weak_scalability_efficiency.png"
        plt.savefig(out, dpi=150)
        print(f"\nSaved {out}")
        if args.show:
            plt.show()
        return

    # Default / --optimal: plot step time vs. nprocs for selected times
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, target_time in enumerate(selected_times):
        times = []
        valid_nprocs = []
        cycles = []
        for nprocs in nprocs_list:
            match = find_cycle_at_time(cycle_data[nprocs], target_time,
                                       args.no_rebalances)
            if match:
                cycle, _, step_time = match
                if args.per_particle:
                    pc = particle_data.get(nprocs, {}).get(cycle)
                    if pc:
                        step_time /= (pc / nprocs)
                times.append(step_time)
                valid_nprocs.append(nprocs)
                cycles.append(cycle)
        if valid_nprocs:
            ax.plot(valid_nprocs, times, "o-",
                    label=f"t={target_time:.4f} ns", markersize=6)
            va = "bottom" if idx % 2 == 0 else "top"
            for x, y, c in zip(valid_nprocs, times, cycles):
                ax.annotate(str(c), (x, y), textcoords="offset points",
                            xytext=(0, 6 if va == "bottom" else -6),
                            ha="center", va=va, fontsize=7)

    if args.optimal:
        target_time = selected_times[0]
        opt_times = []
        opt_nprocs = []
        for nprocs in nprocs_list:
            match = find_cycle_at_time(cycle_data[nprocs], target_time,
                                       args.no_rebalances)
            if match:
                cycle, _, step_time = match
                if args.per_particle:
                    pc = particle_data.get(nprocs, {}).get(cycle)
                    if pc:
                        step_time /= (pc / nprocs)
                opt_nprocs.append(nprocs)
                opt_times.append(step_time)
        if opt_nprocs:
            ref_idx = (opt_nprocs.index(ref_nprocs)
                       if ref_nprocs in opt_nprocs else 0)
            t_ref = opt_times[ref_idx]
            ax.axhline(t_ref, color="gray", linestyle="--", linewidth=0.8,
                        label=f"Ideal (T = {t_ref:.2f}s)")
            print_weak_scaling_table(opt_nprocs, opt_times, t_ref,
                                     args.tasks_per_node)

    time_label = "MC loop time" if args.loop_times else "Step time"
    ax.set_xlabel("Number of processors")
    ax.set_ylabel(f"{time_label} (s)")
    ax.set_title(f"Weak scalability: {time_label.lower()} vs. processor count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(nprocs_list)
    ax.set_xticklabels([str(n) for n in nprocs_list], rotation=90)
    secax = ax.secondary_xaxis("top",
                                functions=(lambda x: x / args.tasks_per_node,
                                           lambda x: x * args.tasks_per_node))
    secax.set_xlabel("Number of nodes")

    plt.tight_layout()
    out = args.output or ("weak_scalability_optimal.png"
                          if args.optimal else "weak_scalability.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
