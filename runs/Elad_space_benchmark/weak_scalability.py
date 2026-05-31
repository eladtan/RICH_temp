#!/usr/bin/env python3

import argparse
import glob
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
WEAK_TIME_YMIN = 20.0
WEAK_TIME_YMAX = 40.0


class Run:
    def __init__(self, nprocs, job_id, filepath, total_time, is_p2p,
                 manager="", steps=0, dt=0.0, nbase=0, nball=0,
                 emitted_per_cycle=0, photons_per_emitter=0,
                 radius=0.0, target_emitters=0, cycle_count=0,
                 time_source="total", direction="", last_avg_particle_steps=0.0,
                 run_date=""):
        self.nprocs = nprocs
        self.job_id = job_id
        self.filepath = filepath
        self.total_time = total_time
        self.is_p2p = is_p2p
        self.manager = manager
        self.steps = steps
        self.dt = dt
        self.nbase = nbase
        self.nball = nball
        self.emitted_per_cycle = emitted_per_cycle
        self.photons_per_emitter = photons_per_emitter
        self.radius = radius
        self.target_emitters = target_emitters
        self.cycle_count = cycle_count
        self.time_source = time_source
        self.direction = direction
        self.last_avg_particle_steps = last_avg_particle_steps
        self.run_date = run_date


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot weak scaling for uniform benchmark runs."
    )
    parser.add_argument("--ignore", type=str, default="",
                        help="Comma-separated list of processor counts to ignore.")
    parser.add_argument("--optimal", action="store_true",
                        help="Add ideal weak-scaling reference lines and print deviations.")
    parser.add_argument("--efficiency", action="store_true",
                        help="Plot weak-scaling efficiency T_ref / T(N).")
    parser.add_argument("--p2p", action="store_true",
                        help="Include P2P files (uniform_WS_P2P_*) as an additional curve.")
    parser.add_argument("--sum", type=int, nargs="?", const=0, default=None, metavar="X",
                        help="Use per-cycle step_wall(max) timings instead of the final "
                             "total benchmark time. No value: sum all available cycles. "
                             "--sum=X: sum only the last X cycles.")
    parser.add_argument("--select", choices=("latest", "best", "median", "mean", "all"),
                        default="latest",
                        help="How to choose repeated runs at the same processor count.")
    parser.add_argument("--base", type=int, default=None,
                        help="Use X ranks as the weak-scaling baseline instead of the minimum.")
    parser.add_argument("--tasks-per-node", type=int, default=112,
                        help="Tasks per node for the top x-axis and weak-series checks.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output filename for the plot.")
    parser.add_argument("--ymin", type=float, default=None,
                        help="Minimum y-axis value.")
    parser.add_argument("--ymax", type=float, default=None,
                        help="Maximum y-axis value.")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving.")
    parser.add_argument("--plot-diff", action="store_true",
                        help="With --optimal, plot deviation from ideal weak scaling separately.")
    parser.add_argument("--table-only", action="store_true",
                        help="Only parse files and print the selected run table; do not plot.")
    parser.add_argument("--dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to search for uniform_WS_*.out files.")
    args = parser.parse_args()

    if args.sum is not None and args.sum < 0:
        parser.error("--sum must be non-negative")
    if args.tasks_per_node <= 0:
        parser.error("--tasks-per-node must be positive")
    if args.ymin is not None and args.ymax is not None and args.ymin >= args.ymax:
        parser.error("--ymin must be smaller than --ymax")
    return args


def parse_int(meta, key, default=0):
    try:
        return int(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_float(meta, key, default=0.0):
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def parse_key_value_meta(text):
    meta = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.strip().split("=", 1)
        meta[key.strip()] = value.strip()
    return meta


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


def parse_run_time(filepath, sum_cycles=None):
    total_re = re.compile(
        r"Total benchmark step wall time\(max-summed\):\s*([\d.eE+\-]+)s?"
    )
    cycle_re = re.compile(
        r"^Cycle\s+(\d+)\s+.*\bstep_wall\(max\)=([\d.eE+\-]+)s?"
    )
    avg_steps_re = re.compile(r"\bavg steps:\s*([\d.eE+\-]+)")
    ball_meta_re = re.compile(r"^Uniform emission benchmark:\s*(.*)$")
    weak_param_re = re.compile(
        r"^\s*(NBASE|NBALL|DT|SOURCE_RADIUS|TARGET_EMITTERS|EMISSION_DIRECTION)=([\w.dE+\-]+)\s*$"
    )
    nodes_re = re.compile(r"^\s*nodes=(\d+)\s+ranks=(\d+)\s*$")
    generated_re = re.compile(
        r"Generated\s+\d+\s+mesh points\s+\(background=(\d+)\)"
    )
    command_re = re.compile(
        r"\s(?:\./)?rich\s+(\d+)\s+(\d+)\s+(\d+)\s+"
        r".*--dt\s+([\d.eE+\-]+)"
    )

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

            m = avg_steps_re.search(line)
            if m:
                meta["last avg particle steps"] = m.group(1)
                continue

            m = ball_meta_re.match(line)
            if m:
                meta.update(parse_key_value_meta(m.group(1)))
                continue

            m = weak_param_re.match(line)
            if m:
                key, value = m.group(1), m.group(2)
                if key == "SOURCE_RADIUS":
                    meta.setdefault("radius", value)
                elif key == "TARGET_EMITTERS":
                    meta.setdefault("target emitters", value)
                elif key == "EMISSION_DIRECTION":
                    meta.setdefault("direction", value)
                else:
                    meta[key] = value
                continue

            m = nodes_re.match(line)
            if m:
                meta["nodes"] = m.group(1)
                meta["ranks"] = m.group(2)
                continue

            m = generated_re.search(line)
            if m:
                meta["NBASE"] = m.group(1)
                continue

            m = command_re.search(line)
            if m:
                meta.setdefault("NBASE", m.group(1))
                meta.setdefault("photons/cell/cycle", m.group(2))
                meta.setdefault("steps", m.group(3))
                meta.setdefault("dt", m.group(4))

    if sum_cycles is not None:
        if not cycle_times:
            return None, "no cycle timings found"
        cycle_times.sort(key=lambda x: x[0])
        if sum_cycles == 0:
            selected = cycle_times
            source = f"all {len(selected)} cycles"
        else:
            if len(cycle_times) < sum_cycles:
                return None, (
                    f"only {len(cycle_times)} cycle timings found; "
                    f"need {sum_cycles} for --sum={sum_cycles}"
                )
            selected = cycle_times[-sum_cycles:]
            source = f"last {sum_cycles} cycles"
        return (
            sum(t for _, t in selected),
            meta,
            len(cycle_times),
            source,
        ), None

    if total_time is None:
        return None, "no completed total time"

    return (total_time, meta, len(cycle_times), "total"), None


def find_runs(directory, include_p2p=False, sum_cycles=None):
    pattern = os.path.join(directory, "uniform_WS_*.out")
    files = glob.glob(pattern)

    regular_re = re.compile(r"uniform_WS_(\d+)_n(\d+)\.out$")
    p2p_re = re.compile(r"uniform_WS_P2P_(\d+)_n(\d+)\.out$")

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
        manager = meta.get("manager", "p2p" if is_p2p else "rdma")
        run = Run(
            nprocs=int(m.group(2)),
            job_id=int(m.group(1)),
            filepath=filepath,
            total_time=total_time,
            is_p2p=is_p2p,
            manager=manager,
            steps=parse_int(meta, "steps"),
            dt=parse_float(meta, "dt", parse_float(meta, "DT")),
            nbase=parse_int(meta, "NBASE"),
            nball=parse_int(meta, "NBALL"),
            emitted_per_cycle=parse_int(meta, "emitted/cycle"),
            photons_per_emitter=parse_int(meta, "photons/cell/cycle"),
            radius=parse_float(meta, "radius"),
            target_emitters=parse_int(meta, "target emitters"),
            cycle_count=cycle_count,
            time_source=time_source,
            direction=meta.get("direction", ""),
            last_avg_particle_steps=parse_float(meta, "last avg particle steps"),
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

    if selector == "all":
        return sorted(runs, key=lambda r: (r.nprocs, r.job_id))

    selected = []
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
                nbase=template.nbase,
                nball=template.nball,
                emitted_per_cycle=template.emitted_per_cycle,
                photons_per_emitter=template.photons_per_emitter,
                radius=template.radius,
                target_emitters=template.target_emitters,
                cycle_count=template.cycle_count,
                time_source=template.time_source,
                direction=template.direction,
                last_avg_particle_steps=(
                    statistics.median(r.last_avg_particle_steps for r in group)
                    if selector == "median"
                    else statistics.mean(r.last_avg_particle_steps for r in group)
                ),
                run_date=format_group_dates(group),
            ))
    return selected


def aggregate_for_line(runs, selector):
    if selector == "all":
        return aggregate_runs(runs, "median")
    return aggregate_runs(runs, selector)


def linuniform(start, stop, num=200):
    if num <= 1 or start == stop:
        return [start]
    step = (stop - start) / float(num - 1)
    return [start + i * step for i in range(num)]


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Use --table-only to just parse results.",
              file=sys.stderr)
        sys.exit(1)
    return plt


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


def reference_run(runs, base=None):
    if not runs:
        return None
    if base is not None:
        for run in runs:
            if run.nprocs == base:
                return run
        print(f"Warning: --base={base} not found for "
              f"{'P2P' if runs[0].is_p2p else 'RDMA'}; using minimum nprocs.",
              file=sys.stderr)
    return min(runs, key=lambda r: r.nprocs)


def measurement_label(args):
    if args.sum is None:
        return "Wallclock time (s)"
    if args.sum == 0:
        return "Wallclock time of all available cycles (s)"
    return f"Wallclock time of last {args.sum} cycles (s)"


def output_name(args):
    if args.output:
        return args.output
    if args.efficiency:
        return "uniform_weak_scalability_efficiency.png"
    if args.sum is not None:
        return "uniform_weak_scalability_sum.png"
    if args.optimal:
        return "uniform_weak_scalability_optimal.png"
    return "uniform_weak_scalability.png"


def p2p_acceleration_by_nprocs(p2p_runs):
    return {run.nprocs: run.total_time for run in p2p_runs}


def format_p2p_acceleration(run, p2p_times):
    if not p2p_times:
        return ""
    p2p_time = p2p_times.get(run.nprocs)
    if p2p_time is None or run.total_time <= 0:
        return ""
    return f"{p2p_time / run.total_time:.3g}x"


def print_table(title, runs, tasks_per_node, p2p_times=None):
    if not runs:
        return

    print(title)
    columns = [
        "Processors", "Nodes", "Job", "Run date", "Time(s)", "Measure", "Manager",
        "Cells", "Photons/cell", "Steps", "dt",
        "Emitted/cyc", "AvgSteps(last)",
    ]
    if p2p_times:
        columns.append("P2P/RDMA")
    columns.extend(["Direction", "File"])
    rows = []
    for run in sorted(runs, key=lambda r: (r.nprocs, r.job_id)):
        nodes = run.nprocs / tasks_per_node
        row = [
            str(run.nprocs),
            f"{nodes:.3g}",
            str(run.job_id),
            run.run_date,
            f"{run.total_time:.6g}",
            run.time_source,
            run.manager,
            str(run.nbase) if run.nbase else "",
            str(run.photons_per_emitter) if run.photons_per_emitter else "",
            str(run.steps) if run.steps else "",
            f"{run.dt:.3g}" if run.dt else "",
            str(run.emitted_per_cycle) if run.emitted_per_cycle else "",
            f"{run.last_avg_particle_steps:.6g}" if run.last_avg_particle_steps else "",
        ]
        if p2p_times:
            row.append(format_p2p_acceleration(run, p2p_times))
        row.extend([
            run.direction,
            os.path.basename(run.filepath),
        ])
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


def print_weak_scaling_table(label, runs, ref, tasks_per_node):
    if not runs or ref is None:
        return

    print(f"{label} weak scaling analysis "
          f"(T_ref = {ref.total_time:.6g}s at {ref.nprocs} procs):")
    print(f"  {'Procs':>8}  {'Nodes':>7}  {'Time':>10}  {'Ideal':>10}  "
          f"{'Overhead':>10}  {'Overhead%':>10}  {'Effic%':>7}")
    for run in runs:
        nodes = run.nprocs / tasks_per_node
        ideal = ref.total_time
        overhead = run.total_time - ideal
        pct = overhead / ideal * 100 if ideal > 0 else 0.0
        efficiency = ideal / run.total_time * 100 if run.total_time > 0 else 0.0
        print(f"  {run.nprocs:>8}  {nodes:>7.1f}  {run.total_time:>10.3f}  "
              f"{ideal:>10.3f}  {overhead:>+10.3f}  "
              f"{pct:>+9.1f}%  {efficiency:>6.1f}%")
    print()


def warn_weak_parameter_scaling(label, runs, tasks_per_node):
    nbase_per_node = []
    nball_per_nodes_23 = []
    for run in runs:
        nodes = run.nprocs / tasks_per_node
        if nodes <= 0:
            continue
        if run.nbase:
            nbase_per_node.append(run.nbase / nodes)
        if run.nball:
            nball_per_nodes_23.append(run.nball / (nodes ** (2.0 / 3.0)))

    def summarize(values):
        if not values:
            return None
        return min(values), max(values)

    nbase_range = summarize(nbase_per_node)
    nball_range = summarize(nball_per_nodes_23)
    if nbase_range:
        lo, hi = nbase_range
        if hi > 0 and abs(hi - lo) / hi > 0.01:
            print(f"Warning: {label} NBASE/node is not constant "
                  f"({lo:.3g}..{hi:.3g}).", file=sys.stderr)
        else:
            print(f"{label} weak series: NBASE/node ~= {statistics.mean(nbase_per_node):.0f}")
    if nball_range:
        lo, hi = nball_range
        if hi > 0 and abs(hi - lo) / hi > 0.01:
            print(f"Warning: {label} NBALL/nodes^(2/3) is not constant "
                  f"({lo:.3g}..{hi:.3g}).", file=sys.stderr)
        else:
            print(f"{label} weak series: NBALL/nodes^(2/3) ~= "
                  f"{statistics.mean(nball_per_nodes_23):.0f}")


def warn_mixed_fixed_parameters(label, runs):
    signatures = {
        (run.steps, run.photons_per_emitter, run.radius,
         bool(run.target_emitters), run.direction)
        for run in runs
        if run.steps or run.photons_per_emitter or run.radius or
           run.target_emitters or run.direction
    }
    if len(signatures) > 1:
        formatted = ", ".join(
            f"steps={steps}, photons/cell={photons}, radius={radius:.3g}, "
            f"sampled_source={sampled}, direction={direction or 'unknown'}"
            for steps, photons, radius, sampled, direction in sorted(signatures)
        )
        print(f"Warning: {label} selected runs have mixed fixed parameters: {formatted}",
              file=sys.stderr)


def warn_cross_series_parameters(rdma_runs, p2p_runs):
    if not rdma_runs or not p2p_runs:
        return
    rdma_sig = {(run.steps, run.photons_per_emitter) for run in rdma_runs}
    p2p_sig = {(run.steps, run.photons_per_emitter) for run in p2p_runs}
    if rdma_sig != p2p_sig:
        print("Warning: RDMA and P2P selected curves do not have identical "
              "(steps, photons/cell) metadata.", file=sys.stderr)


def add_nodes_axis(ax, tasks_per_node):
    secax = ax.secondary_xaxis(
        "top",
        functions=(lambda x: x / tasks_per_node,
                   lambda x: x * tasks_per_node),
    )
    secax.set_xlabel("")
    secax.set_xticks([])
    secax.tick_params(axis="x", which="both", top=False, labeltop=False)


def add_ticks(ax, runs, p2p_runs=None):
    ticks = {run.nprocs for run in runs}
    if p2p_runs:
        ticks.update(run.nprocs for run in p2p_runs)
    if ticks:
        ticks = sorted(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(n) for n in ticks], rotation=90)


def plot_time_series(ax, runs, label, marker, color):
    ax.plot([r.nprocs for r in runs], [r.total_time for r in runs],
            marker=marker, linestyle="-", color=color, label=label, markersize=5)


def add_ideal_weak_line(ax, runs, ref, label, color):
    if not runs or ref is None:
        return
    x = linuniform(min(r.nprocs for r in runs), max(r.nprocs for r in runs), 200)
    ax.plot(x, [ref.total_time for _ in x], "--", color=color,
            alpha=IDEAL_LINE_ALPHA,
            label=f"{label} ideal weak scaling (T={ref.total_time:.3g}s)")


def make_efficiency_plot(args, rdma_runs, p2p_runs, rdma_ref, p2p_ref):
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    if rdma_ref is not None:
        rdma_eff = [rdma_ref.total_time / r.total_time * 100 for r in rdma_runs]
        ax.plot([r.nprocs for r in rdma_runs], rdma_eff, "o-",
                color=RDMA_COLOR, label="RDMA efficiency", markersize=5)

    if p2p_runs and p2p_ref is not None:
        p2p_eff = [p2p_ref.total_time / r.total_time * 100 for r in p2p_runs]
        ax.plot([r.nprocs for r in p2p_runs], p2p_eff, "s-",
                color=P2P_COLOR, label="P2P efficiency", markersize=5)

    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8,
               label="Ideal (100%)")
    ax.set_xlabel("Number of processors")
    ax.set_ylabel("Weak scaling efficiency (%)")
    style_axis_text(ax, "Uniform benchmark weak-scaling efficiency")
    ax.grid(True, alpha=0.3)
    style_legend(ax)
    ax.set_xscale("log", base=2)
    add_ticks(ax, rdma_runs, p2p_runs)
    add_nodes_axis(ax, args.tasks_per_node)

    shown = []
    if rdma_ref is not None:
        shown.extend(rdma_ref.total_time / r.total_time * 100 for r in rdma_runs)
    if p2p_runs and p2p_ref is not None:
        shown.extend(p2p_ref.total_time / r.total_time * 100 for r in p2p_runs)
    if shown:
        ymin = args.ymin if args.ymin is not None else min(50.0, min(shown) * 0.2)
        ymax = args.ymax if args.ymax is not None else max(105.0, max(shown) * 1.5)
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = output_name(args)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()


def make_total_time_plot(args, rdma_runs, p2p_runs, rdma_ref, p2p_ref):
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 6))

    suffix = "all cycles" if args.sum == 0 else (
        f"last {args.sum} cycles" if args.sum is not None else "total time"
    )
    plot_time_series(ax, rdma_runs, f"RDMA {suffix}", "o", RDMA_COLOR)
    if p2p_runs:
        plot_time_series(ax, p2p_runs, f"P2P {suffix}", "s", P2P_COLOR)

    if args.optimal:
        add_ideal_weak_line(ax, rdma_runs, rdma_ref, "RDMA", RDMA_COLOR)
        if p2p_runs:
            add_ideal_weak_line(ax, p2p_runs, p2p_ref, "P2P", P2P_COLOR)

    ax.set_xlabel("Number of processors")
    ax.set_ylabel(measurement_label(args))
    style_axis_text(ax, "Uniform benchmark weak scaling")
    ax.grid(True, alpha=0.3)
    ax.grid(True, which="minor", axis="y", alpha=0.15)
    style_legend(ax)
    ax.set_xscale("log", base=2)
    add_ticks(ax, rdma_runs, p2p_runs)
    add_nodes_axis(ax, args.tasks_per_node)

    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="y", which="major", length=7)
    ax.tick_params(axis="y", which="minor", length=4)

    shown_times = [r.total_time for r in rdma_runs]
    shown_times.extend(r.total_time for r in p2p_runs)
    if shown_times:
        ymin = args.ymin if args.ymin is not None else WEAK_TIME_YMIN
        ymax = args.ymax if args.ymax is not None else WEAK_TIME_YMAX
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    out = output_name(args)
    plt.savefig(out, dpi=150)
    print(f"Saved {out}")
    if args.show:
        plt.show()

    if args.optimal:
        print_weak_scaling_table("RDMA", rdma_runs, rdma_ref, args.tasks_per_node)
        if p2p_runs:
            print_weak_scaling_table("P2P", p2p_runs, p2p_ref, args.tasks_per_node)

    if args.optimal and args.plot_diff and rdma_ref is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        rdma_diff = [
            (r.total_time - rdma_ref.total_time) / rdma_ref.total_time * 100
            for r in rdma_runs
        ]
        ax2.plot([r.nprocs for r in rdma_runs], rdma_diff, "o-",
                 label="RDMA", markersize=5, color=RDMA_COLOR)

        if p2p_runs and p2p_ref is not None:
            p2p_diff = [
                (r.total_time - p2p_ref.total_time) / p2p_ref.total_time * 100
                for r in p2p_runs
            ]
            ax2.plot([r.nprocs for r in p2p_runs], p2p_diff, "s-",
                     label="P2P", markersize=5, color=P2P_COLOR)

        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax2.set_xlabel("Number of processors")
        ax2.set_ylabel("Deviation from ideal weak scaling (%)")
        style_axis_text(ax2, "Uniform benchmark deviation from ideal weak scaling")
        ax2.grid(True, alpha=0.3)
        style_legend(ax2)
        ax2.set_xscale("log", base=2)
        add_ticks(ax2, rdma_runs, p2p_runs)
        add_nodes_axis(ax2, args.tasks_per_node)
        plt.tight_layout()
        diff_out = os.path.splitext(out)[0] + "_diff" + os.path.splitext(out)[1]
        plt.savefig(diff_out, dpi=150)
        print(f"Saved {diff_out}")
        if args.show:
            plt.show()


def parse_ignore_set(ignore):
    if not ignore:
        return set()
    return {int(x.strip()) for x in ignore.split(",") if x.strip()}


def main():
    args = parse_args()
    ignore_set = parse_ignore_set(args.ignore)

    rdma_all, p2p_all = find_runs(args.dir, include_p2p=args.p2p,
                                  sum_cycles=args.sum)
    rdma_all = [r for r in rdma_all if r.nprocs not in ignore_set]
    p2p_all = [r for r in p2p_all if r.nprocs not in ignore_set]

    if not rdma_all:
        print("No completed uniform_WS_*.out files found. If jobs are incomplete, "
              "try --sum or --sum=X.", file=sys.stderr)
        sys.exit(1)

    rdma_selected = aggregate_runs(rdma_all, args.select)
    p2p_selected = aggregate_runs(p2p_all, args.select) if args.p2p else []
    rdma_line = aggregate_for_line(rdma_all, args.select)
    p2p_line = aggregate_for_line(p2p_all, args.select) if args.p2p else []

    if args.select == "all":
        p2p_times = p2p_acceleration_by_nprocs(p2p_line) if args.p2p else None
        print_table("RDMA parsed runs:", rdma_selected, args.tasks_per_node, p2p_times)
        if p2p_selected:
            print_table("P2P parsed runs:", p2p_selected, args.tasks_per_node)
        print_table("RDMA median curve:", rdma_line, args.tasks_per_node, p2p_times)
        if p2p_line:
            print_table("P2P median curve:", p2p_line, args.tasks_per_node)
    else:
        p2p_times = p2p_acceleration_by_nprocs(p2p_selected) if args.p2p else None
        print_table(f"RDMA selected runs ({args.select}):", rdma_selected,
                    args.tasks_per_node, p2p_times)
        if p2p_selected:
            print_table(f"P2P selected runs ({args.select}):", p2p_selected,
                        args.tasks_per_node)

    warn_weak_parameter_scaling("RDMA", rdma_line, args.tasks_per_node)
    warn_mixed_fixed_parameters("RDMA", rdma_line)
    if p2p_line:
        warn_weak_parameter_scaling("P2P", p2p_line, args.tasks_per_node)
        warn_mixed_fixed_parameters("P2P", p2p_line)
    warn_cross_series_parameters(rdma_line, p2p_line)

    rdma_ref = reference_run(rdma_line, args.base)
    p2p_ref = reference_run(p2p_line, args.base) if p2p_line else None

    if args.table_only:
        print_weak_scaling_table("RDMA", rdma_line, rdma_ref, args.tasks_per_node)
        if p2p_line:
            print_weak_scaling_table("P2P", p2p_line, p2p_ref,
                                     args.tasks_per_node)
        return

    if args.efficiency:
        print_weak_scaling_table("RDMA", rdma_line, rdma_ref, args.tasks_per_node)
        if p2p_line:
            print_weak_scaling_table("P2P", p2p_line, p2p_ref,
                                     args.tasks_per_node)
        make_efficiency_plot(args, rdma_line, p2p_line, rdma_ref, p2p_ref)
    else:
        make_total_time_plot(args, rdma_line, p2p_line, rdma_ref, p2p_ref)


if __name__ == "__main__":
    main()
