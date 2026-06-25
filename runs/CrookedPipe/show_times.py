import re
import glob
import os
import argparse
import matplotlib.pyplot as plt

def find_latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    return max(files, key=os.path.getmtime)

def parse_cycle_times(filepath):
    cycles = []
    times = []
    rebalance_cycles = set()
    re_cycle = re.compile(r"Ended Cycle (\d+),.*time taken: ([\d.]+) seconds")
    saw_rebalance = False
    with open(filepath) as f:
        for line in f:
            if "allowRebalance=true" in line or "Doing rebalance!" in line:
                saw_rebalance = True
            m = re_cycle.search(line)
            if m:
                cycle = int(m.group(1))
                cycles.append(cycle)
                times.append(float(m.group(2)))
                if saw_rebalance:
                    rebalance_cycles.add(cycle)
                    saw_rebalance = False
    return cycles, times, rebalance_cycles

parser = argparse.ArgumentParser()
parser.add_argument("--no-rebalances", action="store_true",
                    help="Exclude cycles that contain a rebalance")
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

p2p_file = find_latest_file(os.path.join(script_dir, "CrookedPipeP2P_*.out"))
rdma_file = find_latest_file(os.path.join(script_dir, "CrookedPipe_*.out"))

print(f"P2P:  {os.path.basename(p2p_file)}")
print(f"RDMA: {os.path.basename(rdma_file)}")

p2p_cycles, p2p_times, p2p_rebal = parse_cycle_times(p2p_file)
rdma_cycles, rdma_times, rdma_rebal = parse_cycle_times(rdma_file)

if args.no_rebalances:
    skip = p2p_rebal | rdma_rebal
    p2p_cycles, p2p_times = zip(*[(c, t) for c, t in zip(p2p_cycles, p2p_times) if c not in skip]) if p2p_cycles else ([], [])
    rdma_cycles, rdma_times = zip(*[(c, t) for c, t in zip(rdma_cycles, rdma_times) if c not in skip]) if rdma_cycles else ([], [])
    p2p_cycles, p2p_times = list(p2p_cycles), list(p2p_times)
    rdma_cycles, rdma_times = list(rdma_cycles), list(rdma_times)
    print(f"Excluding {len(skip)} rebalance cycles: {sorted(skip)}")

max_common_cycle = min(p2p_cycles[-1], rdma_cycles[-1]) if p2p_cycles and rdma_cycles else 0
p2p_total = sum(t for c, t in zip(p2p_cycles, p2p_times) if c <= max_common_cycle)
rdma_total = sum(t for c, t in zip(rdma_cycles, rdma_times) if c <= max_common_cycle)
print(f"\nTotal time through cycle {max_common_cycle}:")
print(f"  P2P:  {p2p_total:.1f}s ({p2p_total/3600:.2f}h)")
print(f"  RDMA: {rdma_total:.1f}s ({rdma_total/3600:.2f}h)")
print(f"  Ratio (RDMA/P2P): {rdma_total/p2p_total:.3f}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(p2p_cycles, p2p_times, label="P2P", linewidth=1)
ax.plot(rdma_cycles, rdma_times, label="RDMA", linewidth=1)
ax.set_xlabel("Cycle")
ax.set_ylabel("Time (seconds)")
title = "CrookedPipe: Cycle Time — P2P vs RDMA"
if args.no_rebalances:
    title += " (no rebalances)"
ax.set_title(title)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "cycle_times.png"), dpi=150)
plt.show()
