#!/usr/bin/env python3

import argparse
import math
import os
import sys
from typing import Any
from xml.sax.saxutils import escape


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot radius bins and average calls/r^2 from profiling/<rank> files."
    )
    parser.add_argument("--bins", type=int, default=100,
                        help="Number of radial bins.")
    parser.add_argument("--output-prefix", type=str, default="profile",
                        help="Output prefix for the two SVG plots.")
    parser.add_argument("--table-only", action="store_true",
                        help="Only print the binned table; do not write plots.")
    parser.add_argument("profiling_dirs", nargs="*", default=["profiling"],
                        help="Profiling directories to compare. Default: profiling.")
    args = parser.parse_args()

    if args.bins <= 0:
        parser.error("--bins must be positive")

    return args


def profile_files(profile_dir):
    try:
        names = os.listdir(profile_dir)
    except FileNotFoundError:
        print(f"Profile directory not found: {profile_dir}", file=sys.stderr)
        sys.exit(1)

    paths = []
    for name in names:
        path = os.path.join(profile_dir, name)
        if os.path.isfile(path):
            paths.append(path)

    def sort_key(path):
        name = os.path.basename(path)
        return (0, int(name)) if name.isdigit() else (1, name)

    return sorted(paths, key=sort_key)


def read_profile(profile_dir):
    values = []
    skipped = 0
    files = profile_files(profile_dir)

    if not files:
        print(f"No profile files found in {profile_dir}", file=sys.stderr)
        sys.exit(1)

    for path in files:
        with open(path, "r", errors="replace") as profile_file:
            for line in profile_file:
                text = line.strip()
                if not text:
                    continue
                parts = text.replace(",", " ").split()
                if len(parts) < 2:
                    skipped += 1
                    continue
                try:
                    radius = float(parts[0])
                    calls = float(parts[1])
                except ValueError:
                    skipped += 1
                    continue
                if radius < 0.0 or calls < 0.0 or not math.isfinite(radius) or not math.isfinite(calls):
                    skipped += 1
                    continue
                values.append((radius, calls))

    if not values:
        print(f"No valid r,calls rows found in {profile_dir}", file=sys.stderr)
        sys.exit(1)

    return values, skipped, files


def make_bins(values, bin_count, max_radius):
    edges = [0.0 for _ in range(bin_count + 1)]
    counts = [0 for _ in range(bin_count)]
    call_sums = [0.0 for _ in range(bin_count)]

    if max_radius == 0.0:
        counts[0] = len(values)
        call_sums[0] = sum(calls for _, calls in values)
        return edges, counts, call_sums

    width = max_radius / bin_count
    edges = [i * width for i in range(bin_count + 1)]

    for radius, calls in values:
        index = int(radius / width)
        if index == bin_count:
            index = bin_count - 1
        counts[index] += 1
        call_sums[index] += calls

    return edges, counts, call_sums


def bin_centers(edges):
    return [(edges[i] + edges[i + 1]) * 0.5 for i in range(len(edges) - 1)]


def ratio_sums_over_mid_r2(edges, call_sums, counts):
    values = []
    for index, (total, count) in enumerate(zip(call_sums, counts)):
        volume = 4/3 * math.pi * (edges[index + 1]**3 - edges[index]**3)
        if count and volume > 0.0:
            values.append(total / volume)
        else:
            values.append(None)
    return values


def print_table(edges, counts, ratios):
    print("bin\tstart\tend\tradius_count\tsum_calls_over_mid_r2")
    for index, count in enumerate(counts):
        ratio = ratios[index]
        ratio_text = "" if ratio is None else f"{ratio:.17g}"
        print(
            f"{index}\t{edges[index]:.17g}\t{edges[index + 1]:.17g}\t"
            f"{count}\t{ratio_text}"
        )


def nice_range(values):
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return 0.0, 1.0
    low = min(finite)
    high = max(finite)
    if low == high:
        return low, low + 1.0
    return low, high


def write_bar_svg(path, edges, counts, title, y_label):
    width = 1000
    height = 650
    left = 90
    right = 35
    top = 70
    bottom = 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_count = max(max(counts), 1)
    max_radius = edges[-1] if edges else 1.0
    if max_radius == 0.0:
        max_radius = 1.0

    def x_pos(radius):
        return left + plot_width * radius / max_radius

    def y_pos(count):
        return top + plot_height * (1.0 - count / max_count)

    elements = svg_header(width, height, title)
    add_y_grid(elements, left, right, top, plot_height, width, 0.0, float(max_count), y_pos)
    add_x_ticks(elements, left, top, plot_height, max_radius, x_pos)

    for index, count in enumerate(counts):
        bar_x = x_pos(edges[index])
        bar_width = max(1.0, x_pos(edges[index + 1]) - bar_x)
        bar_y = y_pos(count)
        bar_height = top + plot_height - bar_y
        elements.append(f'<rect x="{bar_x:.3f}" y="{bar_y:.3f}" '
                        f'width="{bar_width:.3f}" height="{bar_height:.3f}" '
                        'fill="#4c78a8" stroke="black" stroke-width="0.8"/>')

    add_axes(elements, width, height, left, right, top, plot_height, "r", y_label)
    write_svg(path, elements)


def write_line_svg(path, x_values, y_values, title, y_label):
    points = [
        (x, math.log10(y)) for x, y in zip(x_values, y_values)
        if y is not None and math.isfinite(y) and y > 0.0
    ]
    if not points:
        print(f"No finite values to plot in {path}.", file=sys.stderr)
        return

    width = 1000
    height = 650
    left = 100
    right = 35
    top = 70
    bottom = 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_radius = max(max(x for x, _ in points), 1.0)
    min_y, max_y = nice_range([y for _, y in points])

    def x_pos(radius):
        return left + plot_width * radius / max_radius

    def y_pos(value):
        return top + plot_height * (1.0 - (value - min_y) / (max_y - min_y))

    elements = svg_header(width, height, title)
    add_y_grid(elements, left, right, top, plot_height, width, min_y, max_y, y_pos)
    add_x_ticks(elements, left, top, plot_height, max_radius, x_pos)

    polyline = " ".join(f"{x_pos(x):.3f},{y_pos(y):.3f}" for x, y in points)
    elements.append(f'<polyline points="{polyline}" fill="none" stroke="#4c78a8" stroke-width="2"/>')
    for x, y in points:
        elements.append(f'<circle cx="{x_pos(x):.3f}" cy="{y_pos(y):.3f}" '
                        'r="2.5" fill="#4c78a8"/>')

    add_axes(elements, width, height, left, right, top, plot_height, "r", y_label)
    write_svg(path, elements)


def write_multi_line_svg(path, series, title, y_label, log_y=False):
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
              "#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
    plot_series = []
    all_points = []
    for index, (label, x_values, y_values) in enumerate(series):
        points = []
        for x, y in zip(x_values, y_values):
            if y is None or not math.isfinite(y):
                continue
            if log_y:
                if y <= 0.0:
                    continue
                y = math.log10(y)
            points.append((x, y))
        if points:
            color = colors[index % len(colors)]
            plot_series.append((label, points, color))
            all_points.extend(points)

    if not all_points:
        print(f"No finite values to plot in {path}.", file=sys.stderr)
        return

    width = 1000
    height = 650
    left = 100
    right = 170 if len(plot_series) > 1 else 35
    top = 70
    bottom = 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_radius = max(max(x for x, _ in all_points), 1.0)
    min_y, max_y = nice_range([y for _, y in all_points])

    def x_pos(radius):
        return left + plot_width * radius / max_radius

    def y_pos(value):
        return top + plot_height * (1.0 - (value - min_y) / (max_y - min_y))

    elements = svg_header(width, height, title)
    add_y_grid(elements, left, right, top, plot_height, width, min_y, max_y, y_pos)
    add_x_ticks(elements, left, top, plot_height, max_radius, x_pos)

    for label, points, color in plot_series:
        polyline = " ".join(f"{x_pos(x):.3f},{y_pos(y):.3f}" for x, y in points)
        elements.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in points:
            elements.append(f'<circle cx="{x_pos(x):.3f}" cy="{y_pos(y):.3f}" '
                            f'r="2.5" fill="{color}"/>')

    if len(plot_series) > 1:
        add_legend(elements, width - right + 25, top + 10, plot_series)

    add_axes(elements, width, height, left, right, top, plot_height, "r", y_label)
    write_svg(path, elements)


def scale_second_plot_series(label, ratios):
    if label == "prof_20":
        return [None if value is None else 5.0 * value for value in ratios]
    return ratios


def svg_header(width, height, title):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" '
        f'font-family="sans-serif" font-size="22">{escape(title)}</text>',
    ]


def add_y_grid(elements, left, right, top, plot_height, width, min_y, max_y, y_pos):
    for tick in range(6):
        value = min_y + (max_y - min_y) * tick / 5.0
        y = y_pos(value)
        elements.append(f'<line x1="{left}" y1="{y:.3f}" x2="{width - right}" y2="{y:.3f}" '
                        'stroke="#dddddd" stroke-width="1"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" '
                        f'font-family="sans-serif" font-size="12">{value:.3g}</text>')


def add_x_ticks(elements, left, top, plot_height, max_radius, x_pos):
    for tick in range(6):
        value = max_radius * tick / 5.0
        x = x_pos(value)
        elements.append(f'<line x1="{x:.3f}" y1="{top + plot_height}" '
                        f'x2="{x:.3f}" y2="{top + plot_height + 5}" '
                        'stroke="black" stroke-width="1"/>')
        elements.append(f'<text x="{x:.3f}" y="{top + plot_height + 22}" '
                        f'text-anchor="middle" font-family="sans-serif" '
                        f'font-size="12">{value:.3g}</text>')


def add_axes(elements, width, height, left, right, top, plot_height, x_label, y_label):
    elements.extend([
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" '
        'stroke="black" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" '
        f'y2="{top + plot_height}" stroke="black" stroke-width="1.5"/>',
        f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="15">{escape(x_label)}</text>',
        f'<text x="24" y="{height / 2}" text-anchor="middle" '
        'font-family="sans-serif" font-size="15" '
        'transform="rotate(-90 24 '
        f'{height / 2})">{escape(y_label)}</text>',
        '</svg>',
    ])


def add_legend(elements, x, y, plot_series):
    elements.append(f'<text x="{x}" y="{y}" font-family="sans-serif" '
                    'font-size="13" font-weight="bold">Legend</text>')
    for index, (label, _, color) in enumerate(plot_series):
        item_y = y + 22 + 20 * index
        elements.append(f'<line x1="{x}" y1="{item_y}" x2="{x + 18}" y2="{item_y}" '
                        f'stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{x + 24}" y="{item_y + 4}" font-family="sans-serif" '
                        f'font-size="12">{escape(label)}</text>')


def write_svg(path, elements):
    with open(path, "w") as output_file:
        output_file.write("\n".join(elements))
        output_file.write("\n")
    print(f"Saved {path}")


def main():
    args = parse_args()
    loaded_profiles = []
    max_radius = 0.0
    for profile_dir in args.profiling_dirs:
        values, skipped, files = read_profile(profile_dir)
        max_radius = max(max_radius, max(radius for radius, _ in values))
        loaded_profiles.append((profile_dir, values, skipped, files))

    binned_profiles = []
    for profile_dir, values, skipped, files in loaded_profiles:
        edges, counts, call_sums = make_bins(values, args.bins, max_radius)
        ratios = ratio_sums_over_mid_r2(edges, call_sums, counts)
        label = os.path.basename(os.path.normpath(profile_dir)) or profile_dir
        binned_profiles.append((label, edges, counts, ratios))

        print(f"{profile_dir}: read {len(values)} r,calls rows from {len(files)} files.")
        if skipped:
            print(f"{profile_dir}: skipped {skipped} invalid rows.", file=sys.stderr)
        print_table(edges, counts, ratios)

    if args.table_only:
        return

    centers = bin_centers(binned_profiles[0][1])
    write_multi_line_svg(
        f"{args.output_prefix}_radii.svg",
        [(label, centers, counts) for label, _, counts, _ in binned_profiles],
        "Radius histogram",
        "Number of cells",
    )
    write_multi_line_svg(
        f"{args.output_prefix}_calls_over_r2.svg",
        [
            (label, centers, scale_second_plot_series(label, ratios))
            for label, _, _, ratios in binned_profiles
        ],
        "Sum calls / r^2 by radius",
        "log10(sum calls / r^2)",
        log_y=True,
    )


if __name__ == "__main__":
    main()
