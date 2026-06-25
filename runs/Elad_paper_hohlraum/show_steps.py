#!/usr/bin/env python3

import argparse
import glob
import os
import re
import sys

import matplotlib.pyplot as plt


def find_latest_file(directory, nprocs, p2p=False):
    """Find the latest .out file for the given processor count (highest job ID)."""
    if p2p:
        pattern = os.path.join(directory, f"hohlraum_P2P_*_n{nprocs}.out")
    else:
        pattern = os.path.join(directory, f"hohlraum_*_n{nprocs}.out")

    files = glob.glob(pattern)
    if p2p:
        regex = re.compile(rf"hohlraum_P2P_(\d+)_n{nprocs}\.out$")
    else:
        regex = re.compile(rf"hohlraum_(\d+)_n{nprocs}\.out$")
        files = [f for f in files if "P2P" not in os.path.basename(f)]

    best_job = -1
    best_file = None
    for f in files:
        m = regex.search(f)
        if m:
            job_id = int(m.group(1))
            if job_id > best_job:
                best_job = job_id
                best_file = f
    return best_file


def parse_cycle_times(filepath, subtract_shrink=False):
    """Parse MC physics times from 'Physics radiation-mc time: X' lines.
    Each time is associated with the most recent 'Cycle N at time' line.
    If subtract_shrink is True, subtract the 'init=' time from the preceding
    'Manager breakdown' line from each step time."""
    cycle_at_re = re.compile(r"^Cycle\s+(\d+)\s+at\s+time\s+")
    physics_re = re.compile(r"^Physics radiation-mc time:\s+([\d.]+)")
    init_re = re.compile(r"Manager breakdown \(max\): init=([\d.]+)s")
    cycles = []
    times = []
    current_cycle = None
    pending_init = 0.0
    with open(filepath) as f:
        for line in f:
            m_cycle = cycle_at_re.match(line)
            if m_cycle:
                current_cycle = int(m_cycle.group(1))
                pending_init = 0.0
                continue
            m_init = init_re.search(line)
            if m_init:
                pending_init = float(m_init.group(1))
                continue
            m_phys = physics_re.match(line)
            if m_phys and current_cycle is not None:
                step_time = float(m_phys.group(1))
                if subtract_shrink:
                    step_time = max(step_time - pending_init, 0.0)
                cycles.append(current_cycle)
                times.append(step_time)
                current_cycle = None
                pending_init = 0.0
    return cycles, times


def main():
    parser = argparse.ArgumentParser(
        description="Compare P2P vs RDMA step times for a given processor count.")
    parser.add_argument("nprocs", type=int,
                        help="Number of processors to look up")
    parser.add_argument("--no-rebalances", action="store_true",
                        help="Exclude rebalance cycles (cycle%%10 == 1, e.g. 11, 21, 401, ...)")
    parser.add_argument("--no-shrink", action="store_true",
                        help="Subtract buffer shrink time (init=) from RDMA step times")
    parser.add_argument("--dir", type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for .out files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively")
    args = parser.parse_args()

    rdma_file = find_latest_file(args.dir, args.nprocs, p2p=False)
    p2p_file = find_latest_file(args.dir, args.nprocs, p2p=True)

    if not rdma_file and not p2p_file:
        print(f"No .out files found for {args.nprocs} processors.", file=sys.stderr)
        sys.exit(1)

    if rdma_file:
        print(f"RDMA: {os.path.basename(rdma_file)}")
        rdma_cycles, rdma_times = parse_cycle_times(rdma_file,
                                                    subtract_shrink=args.no_shrink)
    else:
        print("No RDMA file found.", file=sys.stderr)
        rdma_cycles, rdma_times = [], []

    if p2p_file:
        print(f"P2P:  {os.path.basename(p2p_file)}")
        p2p_cycles, p2p_times = parse_cycle_times(p2p_file)
    else:
        print("No P2P file found.", file=sys.stderr)
        p2p_cycles, p2p_times = [], []

    is_rebalance = lambda c: c > 1 and c % 10 == 1

    if args.no_rebalances:
        if rdma_cycles:
            rdma_cycles, rdma_times = zip(
                *[(c, t) for c, t in zip(rdma_cycles, rdma_times)
                  if not is_rebalance(c)])
            rdma_cycles, rdma_times = list(rdma_cycles), list(rdma_times)
        if p2p_cycles:
            p2p_cycles, p2p_times = zip(
                *[(c, t) for c, t in zip(p2p_cycles, p2p_times)
                  if not is_rebalance(c)])
            p2p_cycles, p2p_times = list(p2p_cycles), list(p2p_times)

    if rdma_cycles and p2p_cycles:
        max_common = min(rdma_cycles[-1], p2p_cycles[-1])
        rdma_total = sum(t for c, t in zip(rdma_cycles, rdma_times) if c <= max_common)
        p2p_total = sum(t for c, t in zip(p2p_cycles, p2p_times) if c <= max_common)
        print(f"\nTotal time through cycle {max_common}:")
        print(f"  RDMA: {rdma_total:.1f}s ({rdma_total / 3600:.2f}h)")
        print(f"  P2P:  {p2p_total:.1f}s ({p2p_total / 3600:.2f}h)")
        if p2p_total > 0:
            print(f"  Ratio (RDMA/P2P): {rdma_total / p2p_total:.3f}")

    fig, ax = plt.subplots(figsize=(12, 6))
    if p2p_cycles:
        ax.plot(p2p_cycles, p2p_times, label="P2P", linewidth=1)
    if rdma_cycles:
        rdma_label = "RDMA (no shrink)" if args.no_shrink else "RDMA"
        ax.plot(rdma_cycles, rdma_times, label=rdma_label, linewidth=1)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("MC physics time (seconds)")
    title = f"Hohlraum: MC Physics Time — P2P vs RDMA ({args.nprocs} procs)"
    suffixes = []
    if args.no_rebalances:
        suffixes.append("no rebalances")
    if args.no_shrink:
        suffixes.append("no shrink")
    if suffixes:
        title += " (" + ", ".join(suffixes) + ")"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.output or os.path.join(args.dir, f"step_times_n{args.nprocs}.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
