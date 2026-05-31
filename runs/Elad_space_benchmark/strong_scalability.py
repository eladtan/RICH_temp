#!/usr/bin/env python3

import argparse
import glob
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime


RDMA_COLOR = "tab:blue"
P2P_COLOR = "tab:red"
AXIS_LABEL_SIZE = 14
TICK_LABEL_SIZE = 12
TITLE_SIZE = 16
LEGEND_SIZE = 12
IDEAL_LINE_ALPHA = 0.6


class Run:
    def __init__(self, nprocs, job_id, filepath, total_time, is_p2p,
                 manager="", steps=0, dt=0.0, emitted_per_cycle=0,
                 cycle_count=0, time_source="total", run_date=""):
        self.nprocs = nprocs
        self.job_id = job_id
        self.filepath = filepath
        self.total_time = total_time
        self.is_p2p = is_p2p
        self.manager = manager
        self.steps = steps
        self.dt = dt
        self.emitted_per_cycle = emitted_per_cycle
        self.cycle_count = cycle_count
        self.time_source = time_source
        self.run_date = run_date


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot strong scaling for space benchmark runs."
    )
    parser.add_argument("--ignore", type=str, default="",
                        help="Comma-separated list of processor counts to ignore.")
    parser.add_argument("--optimal", action="store_true",
                        help="Add ideal strong-scaling reference lines and print deviations.")
    parser.add_argument("--fit-A", action="store_true",
                        help="With --optimal, fit A by least squares instead of using the first point.")
    parser.add_argument("--speedup", action="store_true",
                        help="Plot speedup (T_min_procs / T_N) with ideal scaling line.")
    parser.add_argument("--p2p", action="store_true",
                        help="Include P2P files (space_SS_P2P_*) as an additional curve.")
    parser.add_argument("--sum", type=int, default=None, metavar="X",
                        help="Use the sum of the last X per-cycle step_wall(max) timings "
                             "instead of the final total benchmark time.")
    parser.add_argument("--select", choices=("latest", "best", "median", "mean", "all"),
                        default="latest",
                        help="How to choose repeated runs at the same processor count.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot.")
    parser.add_argument("--ymin", type=float, default=None,
                        help="Minimum y-axis value for total-time plots.")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Maximum y-axis value for total-time plots.")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving.")
    parser.add_argument("--plot-diff", action="store_true",
                        help="With --optimal, plot deviation from ideal scaling separately.")
    parser.add_argument("--table-only", action="store_true",
                        help="Only parse files and print the selected run table; do not plot.")
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for space_SS_*.out files.")
    args = parser.parse_args()
    if args.sum is not None and args.sum <= 0:
        parser.error("--sum must be a positive integer")
    if args.ymin is not None and args.ymin <= 0:
        parser.error("--ymin must be positive because the time plot uses a log y-axis")
    if args.ymax is not None and args.ymax <= 0:
        parser.error("--ymax must be positive because the time plot uses a log y-axis")
    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        parser.error("--ymin must be smaller than --ymax")
    return args


def parse_run_time(filepath, sum_cycles=None):
    total_re = re.compile(
        r"Total benchmark step wall time\(max-summed\):\s*([\d.eE+\-]+)s?"
    )
    cycle_re = re.compile(
        r"^Cycle\s+(\d+)\s+.*\bstep_wall\(max\)=([\d.eE+\-]+)s?"
    )
    meta_re = re.compile(r"^Space emission benchmark:\s*(.*)$")

    total_time = None
    meta = {}
    cycle_times = []
    with open(filepath, "r", errors="replace") as f:
        for line in f:
            m = total_re.search(line)
            if m:
                total_time = float(m.group(1))
                continue

            m = cycle_re.search(line)
            if m:
                cycle_times.append((int(m.group(1)), float(m.group(2))))
                continue

            m = meta_re.match(line)
            if m:
                for part in m.group(1).split(","):
                    if "=" not in part:
                        continue
                    key, value = part.strip().split("=", 1)
                    meta[key.strip()] = value.strip()

    if sum_cycles is not None:
        if len(cycle_times) < sum_cycles:
            return None, (
                f"only {len(cycle_times)} cycle timings found; "
                f"need {sum_cycles} for --sum={sum_cycles}"
            )
        cycle_times.sort(key=lambda x: x[0])
        selected = cycle_times[-sum_cycles:]
        return (
            sum(t for _, t in selected),
            meta,
            len(cycle_times),
            f"last {sum_cycles} cycles",
        ), None

    if total_time is None:
        return None, "no completed total time"

    return (total_time, meta, len(cycle_times), "total"), None


def parse_int(meta, key, default=0):
    try:
        return int(meta.get(key, default))
    except ValueError:
        return default


def parse_float(meta, key, default=0.0):
    try:
        return float(meta.get(key, default))
    except ValueError:
        return default


def format_file_date(filepath):
    try:
        return datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""


def format_group_dates(runs):
    dates = sorted({run.run_date for run in runs if run.run_date})
    if not dates:
        return ""
    if dates[0] == dates[-1]:
        return dates[0]
    return f"{dates[0]}..{dates[-1]}"


def find_runs(directory, include_p2p=False, sum_cycles=None):
    pattern = os.path.join(directory, "space_SS_*.out")
    files = glob.glob(pattern)

    regular_re = re.compile(r"space_SS_(\d+)_n(\d+)\.out$")
    p2p_re = re.compile(r"space_SS_P2P_(\d+)_n(\d+)\.out$")

    rdma_runs = []
    p2p_runs = []
    skipped = []

    for filepath in sorted(files):
        base = os.path.basename(filepath)
        is_p2p = False
        m = p2p_re.match(base)
        if m:
            is_p2p = True
        else:
            m = regular_re.match(base)
        if not m:
            continue

        if is_p2p and not include_p2p:
            continue

        parsed, skip_reason = parse_run_time(filepath, sum_cycles=sum_cycles)
        if parsed is None:
            skipped.append((filepath, skip_reason))
            continue

        total_time, meta, cycle_count, time_source = parsed
        run = Run(
            nprocs=int(m.group(2)),
            job_id=int(m.group(1)),
            filepath=filepath,
            total_time=total_time,
            is_p2p=is_p2p,
            manager=meta.get("manager", "p2p" if is_p2p else "rdma"),
            steps=parse_int(meta, "steps"),
            dt=parse_float(meta, "dt"),
            emitted_per_cycle=parse_int(meta, "emitted/cycle"),
            cycle_count=cycle_count,
            time_source=time_source,
            run_date=format_file_date(filepath),
        )
        if is_p2p:
            p2p_runs.append(run)
        else:
            rdma_runs.append(run)

    for filepath, reason in skipped:
        print(f"Warning: {reason} in {filepath}, skipping.", file=sys.stderr)

    return rdma_runs, p2p_runs


def aggregate_runs(runs, selector):
    by_nprocs = defaultdict(list)
    for run in runs:
        by_nprocs[run.nprocs].append(run)

    selected = []
    if selector == "all":
        return sorted(runs, key=lambda r: (r.nprocs, r.job_id))

    for nprocs in sorted(by_nprocs):
        group = sorted(by_nprocs[nprocs], key=lambda r: r.job_id)
        if selector == "latest":
            selected.append(group[-1])
        elif selector == "best":
            selected.append(min(group, key=lambda r: r.total_time))
        elif selector in ("median", "mean"):
            times = [r.total_time for r in group]
            value = statistics.median(times) if selector == "median" else statistics.mean(times)
            template = group[-1]
            selected.append(Run(
                nprocs=nprocs,
                job_id=template.job_id,
                filepath=f"{selector} of {len(group)} runs",
                total_time=value,
                is_p2p=template.is_p2p,
                manager=template.manager,
                steps=template.steps,
                dt=template.dt,
                emitted_per_cycle=template.emitted_per_cycle,
                cycle_count=template.cycle_count,
                time_source=template.time_source,
                run_date=format_group_dates(group),
            ))
    return selected


def aggregate_for_line(runs, selector):
    if selector == "all":
        return aggregate_runs(runs, "median")
    return aggregate_runs(runs, selector)


def fit_strong_scaling_A(nprocs_arr, times_arr):
    return math.exp(
        sum(math.log(float(t) * float(n)) for n, t in zip(nprocs_arr, times_arr))
        / len(nprocs_arr)
    )


def linspace(start, stop, num=200):
    if num <= 1:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + i * step for i in range(num)]


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker  # noqa: F401
    except ImportError:
        print("matplotlib is required for plotting. Use --table-only to just parse results.",
              file=sys.stderr)
        sys.exit(1)
    return plt


def format_log_seconds(value, _pos=None):
    if value >= 10:
        return f"{value:g}"
    return f"{value:.1g}"


def style_axis_text(ax, title=None):
    ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_SIZE)
    ax.tick_params(axis="both", which="minor", labelsize=TICK_LABEL_SIZE)
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE)


def style_legend(ax):
    legend = ax.legend(fontsize=LEGEND_SIZE)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontsize(LEGEND_SIZE)


def p2p_acceleration_by_nprocs(p2p_runs):
    return {run.nprocs: run.total_time for run in p2p_runs}


def format_p2p_acceleration(run, p2p_times):
    if not p2p_times:
        return ""
    p2p_time = p2p_times.get(run.nprocs)
    if p2p_time is None or run.total_time <= 0:
        return ""
    return f"{p2p_time / run.total_time:.3g}x"


def print_table(title, runs, p2p_times=None):
    if not runs:
        return

    print(title)
    columns = [
        "Processors", "Job", "Run date", "Time(s)", "Measure", "Manager",
        "Steps", "dt", "Emitted/cyc",
    ]
    if p2p_times:
        columns.append("P2P/RDMA")
    columns.append("File")
    rows = []
    for run in sorted(runs, key=lambda r: (r.nprocs, r.job_id)):
        row = [
            str(run.nprocs),
            str(run.job_id),
            run.run_date,
            f"{run.total_time:.6g}",
            run.time_source,
            run.manager,
            str(run.steps) if run.steps else "",
            f"{run.dt:.3g}" if run.dt else "",
            str(run.emitted_per_cycle) if run.emitted_per_cycle else "",
        ]
        if p2p_times:
            row.append(format_p2p_acceleration(run, p2p_times))
        row.append(
            os.path.basename(run.filepath),
        )
        rows.append(tuple(row))

    widths = [len(c) for c in columns]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(sep)
    print("| " + " | ".join(f"{c:{w}}" for c, w in zip(columns, widths)) + " |")
    print(sep)
    for row in rows:
        print("| " + " | ".join(f"{v:{w}}" for v, w in zip(row, widths)) + " |")
    print(sep)
    print()


def parameter_signature(run):
    return (run.steps, run.dt, run.emitted_per_cycle)


def warn_mixed_parameters(label, runs):
    signatures = {
        parameter_signature(run)
        for run in runs
        if run.steps or run.dt or run.emitted_per_cycle
    }
    if len(signatures) > 1:
        formatted = ", ".join(
            f"steps={steps}, dt={dt:g}, emitted/cycle={emitted}"
            for steps, dt, emitted in sorted(signatures)
        )
        print(f"Warning: {label} selected runs have mixed benchmark parameters: {formatted}",
              file=sys.stderr)


def warn_cross_series_parameters(rdma_runs, p2p_runs):
    if not rdma_runs or not p2p_runs:
        return
    rdma_sig = {parameter_signature(run) for run in rdma_runs}
    p2p_sig = {parameter_signature(run) for run in p2p_runs}
    if rdma_sig != p2p_sig:
        print("Warning: RDMA and P2P selected curves do not have identical "
              "(steps, dt, emitted/cycle) metadata.", file=sys.stderr)


def print_scaling_diff(label, runs, A):
    n_ref = runs[0].nprocs
    t_ref = runs[0].total_time
    print(f"{label} — deviation from ideal scaling (A={A:.0f}):")
    print(f"  {'Procs':>8}  {'Actual':>10}  {'Ideal':>10}  {'Diff':>10}  {'Diff%':>8}  {'Speedup':>8}  {'Expected':>8}  {'Effic%':>7}")
    for run in runs:
        ideal = A / run.nprocs
        diff = run.total_time - ideal
        pct = diff / ideal * 100
        speedup = t_ref / run.total_time
        expected = run.nprocs / n_ref
        efficiency = speedup / expected * 100
        print(f"  {run.nprocs:>8}  {run.total_time:>10.3f}  {ideal:>10.3f}  {diff:>+10.3f}  {pct:>+7.1f}%  {speedup:>8.2f}  {expected:>8.2f}  {efficiency:>6.1f}%")
    print()


def plot_total_times(ax, runs, label, marker, linestyle, color=None):
    nprocs = [r.nprocs for r in runs]
    times = [r.total_time for r in runs]
    ax.plot(nprocs, times, marker=marker, linestyle=linestyle, label=label,
            markersize=5, color=color)


def measurement_label(args):
    if args.sum is not None:
        return f"Wallclock time of last {args.sum} cycles (s)"
    return "Wallclock time (s)"


def add_ideal_line(ax, runs, label, fit_A=False, color="gray", linestyle="--"):
    if len(runs) < 2:
        return None
    nprocs = [r.nprocs for r in runs]
    times = [r.total_time for r in runs]
    A = fit_strong_scaling_A(nprocs, times) if fit_A else nprocs[0] * times[0]
    x = linspace(min(nprocs), max(nprocs), 200)
    fit_label = "fit" if fit_A else "ref"
    ax.plot(x, [A / xi for xi in x], linestyle=linestyle, color=color,
            alpha=IDEAL_LINE_ALPHA,
            label=f"{label} ideal strong scaling {fit_label} (A={A:.0f})")
    return A


def make_speedup_plot(args, rdma_runs, p2p_runs):
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    all_ticks = set()

    def add_series(runs, label, marker, linestyle, color):
        if len(runs) < 1:
            return
        base = runs[0]
        nprocs = [r.nprocs for r in runs]
        speedups = [base.total_time / r.total_time for r in runs]
        ax.plot(nprocs, speedups, marker=marker, linestyle=linestyle,
                label=label, markersize=5, color=color)
        x = linspace(nprocs[0], nprocs[-1], 200)
        ax.plot(x, [xi / base.nprocs for xi in x], "--", color=color,
                alpha=IDEAL_LINE_ALPHA)
        all_ticks.update(nprocs)

    add_series(rdma_runs, "RDMA speedup", "o", "-", RDMA_COLOR)
    add_series(p2p_runs, "P2P speedup", "s", "-", P2P_COLOR)

    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Speedup")
    style_axis_text(ax, "Space benchmark strong-scaling speedup")
    ax.grid(True, alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    style_legend(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    import matplotlib.ticker as ticker
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=2, subs="auto", numticks=20))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    if all_ticks:
        ax.set_xticks(sorted(all_ticks))
        ax.set_xticklabels([str(n) for n in sorted(all_ticks)], rotation=90)

    plt.tight_layout()
    out = args.output or "space_strong_scalability_speedup.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()


def make_total_time_plot(args, rdma_runs, p2p_runs):
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))
    all_ticks = set()

    rdma_label = f"RDMA last {args.sum} cycles" if args.sum is not None else "RDMA total time"
    p2p_label = f"P2P last {args.sum} cycles" if args.sum is not None else "P2P total time"

    plot_total_times(ax, rdma_runs, rdma_label, "o", "-", color=RDMA_COLOR)
    all_ticks.update(r.nprocs for r in rdma_runs)

    if p2p_runs:
        plot_total_times(ax, p2p_runs, p2p_label, "s", "-", color=P2P_COLOR)
        all_ticks.update(r.nprocs for r in p2p_runs)

    A = B = None
    if args.optimal:
        A = add_ideal_line(ax, rdma_runs, "RDMA", fit_A=args.fit_A, color=RDMA_COLOR, linestyle="--")
        if p2p_runs:
            B = add_ideal_line(ax, p2p_runs, "P2P", fit_A=args.fit_A, color=P2P_COLOR, linestyle="--")

    ax.set_xlabel("Number of processors")
    ax.set_ylabel(measurement_label(args))
    style_axis_text(ax, "Space benchmark strong scaling")
    ax.grid(True, alpha=0.3)
    ax.grid(True, which="minor", alpha=0.15)
    style_legend(ax)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(
        ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=30)
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_log_seconds))
    ax.yaxis.set_minor_locator(
        ticker.LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
                          numticks=30)
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=4)
    shown_times = [r.total_time for r in rdma_runs]
    shown_times.extend(r.total_time for r in p2p_runs)
    if shown_times:
        min_time = min(shown_times)
        max_time = max(shown_times)
        ymin = args.ymin if args.ymin is not None else min_time * 0.75
        ymax = args.ymax if args.ymax is not None else max_time * 1.35
        ax.set_ylim(ymin, ymax)
    if all_ticks:
        ax.set_xticks(sorted(all_ticks))
        ax.set_xticklabels([str(n) for n in sorted(all_ticks)], rotation=90)

    plt.tight_layout()
    out = args.output or "space_strong_scalability.png"
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()

    if args.optimal:
        if A is not None:
            print_scaling_diff("RDMA", rdma_runs, A)
        if B is not None:
            print_scaling_diff("P2P", p2p_runs, B)

    if args.optimal and args.plot_diff and A is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rdma_diff = [
            (r.total_time - A / r.nprocs) / (A / r.nprocs) * 100
            for r in rdma_runs
        ]
        ax2.plot([r.nprocs for r in rdma_runs], rdma_diff, "o-",
                 label="RDMA", markersize=5, color=RDMA_COLOR)

        if p2p_runs and B is not None:
            p2p_diff = [
                (r.total_time - B / r.nprocs) / (B / r.nprocs) * 100
                for r in p2p_runs
            ]
            ax2.plot([r.nprocs for r in p2p_runs], p2p_diff, "s-",
                     label="P2P", markersize=5, color=P2P_COLOR)

        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Number of processors")
        ax2.set_ylabel("Deviation from ideal (%)")
        style_axis_text(ax2, "Space benchmark deviation from ideal strong scaling")
        ax2.grid(True, alpha=0.3)
        style_legend(ax2)
        ax2.set_xscale("log", base=2)
        if all_ticks:
            ax2.set_xticks(sorted(all_ticks))
            ax2.set_xticklabels([str(n) for n in sorted(all_ticks)], rotation=90)
        plt.tight_layout()
        diff_out = os.path.splitext(out)[0] + "_diff" + os.path.splitext(out)[1]
        plt.savefig(diff_out, dpi=150)
        print(f"Saved {diff_out}")
        if args.show:
            plt.show()


def main():
    args = parse_args()

    ignore_set = set()
    if args.ignore:
        ignore_set = {int(x.strip()) for x in args.ignore.split(",") if x.strip()}

    rdma_all, p2p_all = find_runs(args.dir, include_p2p=args.p2p, sum_cycles=args.sum)
    rdma_all = [r for r in rdma_all if r.nprocs not in ignore_set]
    p2p_all = [r for r in p2p_all if r.nprocs not in ignore_set]

    if not rdma_all:
        print("No completed space_SS_*.out files found.", file=sys.stderr)
        sys.exit(1)

    rdma_selected = aggregate_runs(rdma_all, args.select)
    p2p_selected = aggregate_runs(p2p_all, args.select) if args.p2p else []
    rdma_line = aggregate_for_line(rdma_all, args.select)
    p2p_line = aggregate_for_line(p2p_all, args.select) if args.p2p else []

    if args.select == "all":
        p2p_times = p2p_acceleration_by_nprocs(p2p_line) if args.p2p else None
        print_table("RDMA parsed runs:", rdma_selected, p2p_times)
        if p2p_selected:
            print_table("P2P parsed runs:", p2p_selected)
        print_table("RDMA median curve:", rdma_line, p2p_times)
        if p2p_line:
            print_table("P2P median curve:", p2p_line)
    else:
        p2p_times = p2p_acceleration_by_nprocs(p2p_selected) if args.p2p else None
        print_table(f"RDMA selected runs ({args.select}):", rdma_selected, p2p_times)
        if p2p_selected:
            print_table(f"P2P selected runs ({args.select}):", p2p_selected)

    warn_mixed_parameters("RDMA", rdma_line)
    warn_mixed_parameters("P2P", p2p_line)
    warn_cross_series_parameters(rdma_line, p2p_line)

    if args.table_only:
        return

    if args.speedup:
        make_speedup_plot(args, rdma_line, p2p_line)
    else:
        make_total_time_plot(args, rdma_line, p2p_line)


if __name__ == "__main__":
    main()
