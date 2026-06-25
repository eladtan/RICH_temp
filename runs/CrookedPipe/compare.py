#!/usr/bin/env python3
import re
import sys


def parse_file(filepath):
    cycles = {}
    current_rw = None
    current_total_steps = None

    re_rw = re.compile(r"^RW steps:\s+(\d+)")
    re_total = re.compile(r"Total steps:\s+(\d+)")
    re_cycle = re.compile(r"Ended Cycle (\d+),.*time taken:\s+([\d.]+)\s+seconds")

    with open(filepath) as f:
        for line in f:
            m = re_rw.match(line)
            if m:
                current_rw = int(m.group(1))
                continue

            m = re_total.search(line)
            if m:
                current_total_steps = int(m.group(1))
                continue

            m = re_cycle.search(line)
            if m:
                cycle = int(m.group(1))
                time_taken = float(m.group(2))
                cycles[cycle] = {
                    "time": time_taken,
                    "rw_steps": current_rw,
                    "total_steps": current_total_steps,
                }
                current_rw = None
                current_total_steps = None

    return cycles


def is_rdma(filepath):
    with open(filepath) as f:
        for line in f:
            if "in reallocation" in line:
                return True
            if "Total send communications:" in line:
                return False
    return False


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <file1> <file2>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]

    if is_rdma(file1):
        rdma_file, p2p_file = file1, file2
    else:
        rdma_file, p2p_file = file2, file1

    print(f"RDMA: {rdma_file}")
    print(f"P2P:  {p2p_file}")
    print()

    rdma = parse_file(rdma_file)
    p2p = parse_file(p2p_file)

    all_cycles = sorted(set(rdma) | set(p2p))

    hdr = f"{'Cycle':>6}  {'RDMA time':>10}  {'P2P time':>10}  {'RW RDMA':>14}  {'RW P2P':>14}  {'Total RDMA':>16}  {'Total P2P':>16}"
    print(hdr)
    print("-" * len(hdr))

    for c in all_cycles:
        r = rdma.get(c, {})
        p = p2p.get(c, {})

        rt = f"{r['time']:.2f}" if "time" in r else ""
        pt = f"{p['time']:.2f}" if "time" in p else ""
        rw_r = str(r.get("rw_steps", "")) if r.get("rw_steps") is not None else ""
        rw_p = str(p.get("rw_steps", "")) if p.get("rw_steps") is not None else ""
        ts_r = str(r.get("total_steps", "")) if r.get("total_steps") is not None else ""
        ts_p = str(p.get("total_steps", "")) if p.get("total_steps") is not None else ""

        print(f"{c:>6}  {rt:>10}  {pt:>10}  {rw_r:>14}  {rw_p:>14}  {ts_r:>16}  {ts_p:>16}")


if __name__ == "__main__":
    main()
