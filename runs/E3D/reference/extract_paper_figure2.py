#!/usr/bin/env python3
"""Digitize Figure 2 of the Brantley--Novellino E3D paper into a reference CSV.

Figure 2 of ``E3D_paper.pdf`` is a vector (matplotlib) figure, so the curves are
stored in the PDF as polylines whose vertices are the plotted data points.  This
script reads those polylines with PyMuPDF, calibrates page coordinates against
the axis tick marks, and writes the long-format CSV consumed by
``analyze_e3d.py``:

    model,time_s,transmission,reflection[,transmission_error,reflection_error]

No values are invented: every number traces back to a vertex in the PDF content
stream.  The only approximation is that transmission (panel a) and reflection
(panel b) are drawn on different time grids, so each curve is resampled onto the
union of its own two grids by linear interpolation in log(time) -- exactly the
interpolation the figure itself performs between vertices.

Usage:

    pip install pymupdf
    python3 extract_paper_figure2.py ../E3D_paper.pdf -o e3d_brantley_reference.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import fitz

# Curve styles as they appear in the PDF content stream.  Colours and stroke
# widths are the only discriminator between the plotted lines; they were read
# back from the legend handles, which sit next to their own legend labels.
CURVE_STYLES = {
    ((0.254902, 0.411765, 0.882353), 1.096): "Atomic Mix",
    ((1.0, 0.0, 0.0), 1.096): "CLS LP",
    ((0.0, 0.501961, 0.0), 1.096): "CLS LRP",
    ((0.501961, 0.0, 0.501961), 0.913): "LRP Ramp+Exp",
    ((0.0, 0.0, 0.0), 0.621): "Benchmark",
    ((0.0, 0.0, 0.0), 0.731): "Benchmark 2sigma",
}

COLOUR_TOLERANCE = 0.01
WIDTH_TOLERANCE = 0.01
# Panel (a) occupies the left half of the page; panel (b) the right half.
PANEL_SPLIT_X = 330.0
# Vertices closer than this in page units are treated as the same point.
MERGE_TOLERANCE = 1.0e-3


class Axis:
    """Maps page coordinates of one panel to data coordinates."""

    def __init__(self, x0, log_t0, x1, log_t1, y0, value0, y1, value1, x_left, x_right):
        self.x0 = x0
        self.log_t0 = log_t0
        self.log_scale = (log_t1 - log_t0) / (x1 - x0)
        self.y0 = y0
        self.value0 = value0
        self.value_scale = (value1 - value0) / (y1 - y0)
        self.x_left = x_left
        self.x_right = x_right

    def log_time(self, x):
        return self.log_t0 + (x - self.x0) * self.log_scale

    def value(self, y):
        return self.value0 + (y - self.y0) * self.value_scale


def _matches(style, colour, width):
    reference_colour, reference_width = style
    if colour is None or width is None:
        return False
    if abs(width - reference_width) > WIDTH_TOLERANCE:
        return False
    return all(abs(a - b) <= COLOUR_TOLERANCE for a, b in zip(colour, reference_colour))


def _classify(colour, width):
    for style, name in CURVE_STYLES.items():
        if _matches(style, colour, width):
            return name
    return None


def _tick_positions(drawings, horizontal, panel_left, panel_right):
    """Return the page coordinates of the major tick marks of one panel.

    Major ticks are the short filled black segments attached to the spines.
    Minor ticks are drawn with a smaller stroke width and are ignored.
    """

    positions = []
    for drawing in drawings:
        items = drawing["items"]
        if len(items) != 1 or items[0][0] != "l" or drawing["fill"] is None:
            continue
        if abs((drawing["width"] or 0.0) - 0.365) > WIDTH_TOLERANCE:
            continue
        start, end = items[0][1], items[0][2]
        if not panel_left <= start.x <= panel_right:
            continue
        length = abs(end.x - start.x) if horizontal else abs(end.y - start.y)
        constant = abs(start.y - end.y) if horizontal else abs(start.x - end.x)
        if constant > 0.01 or not 1.0 < length < 2.0:
            continue
        positions.append(start.y if horizontal else start.x)
    return sorted(set(round(value, 3) for value in positions))


def _spine_box(drawings, panel_left, panel_right):
    """Return (x_left, x_right, y_top, y_bottom) of a panel's axes frame."""

    xs, ys = [], []
    for drawing in drawings:
        items = drawing["items"]
        if len(items) != 1 or items[0][0] != "l" or drawing["fill"] is not None:
            continue
        if abs((drawing["width"] or 0.0) - 0.365) > WIDTH_TOLERANCE:
            continue
        if drawing["color"] != (0.0, 0.0, 0.0):
            continue
        start, end = items[0][1], items[0][2]
        if not panel_left <= start.x <= panel_right:
            continue
        if abs(start.x - end.x) < 0.01 and abs(start.y - end.y) > 50.0:
            xs.append(round(start.x, 3))
        elif abs(start.y - end.y) < 0.01 and abs(start.x - end.x) > 50.0:
            ys.append(round(start.y, 3))
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        raise ValueError("could not locate the axes frame of a panel")
    return min(xs), max(xs), min(ys), max(ys)


def build_axis(drawings, panel_left, panel_right, first_decade, last_time, first_value, value_step):
    """Calibrate one panel from its tick marks.

    ``first_decade`` is log10 of the left-most major x tick, ``last_time`` the
    value of the right-most one, ``first_value`` the bottom-most major y tick and
    ``value_step`` the spacing between successive y ticks.
    """

    x_ticks = _tick_positions(drawings, horizontal=False, panel_left=panel_left, panel_right=panel_right)
    y_ticks = _tick_positions(drawings, horizontal=True, panel_left=panel_left, panel_right=panel_right)
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        raise ValueError("could not locate the major tick marks of a panel")
    x_left, x_right, y_top, y_bottom = _spine_box(drawings, panel_left, panel_right)
    # y ticks are listed top to bottom because PDF y grows downwards.
    bottom_tick = y_ticks[-1]
    top_tick = y_ticks[0]
    top_value = first_value + value_step * (len(y_ticks) - 1)
    return Axis(
        x0=x_ticks[0],
        log_t0=first_decade,
        x1=x_ticks[-1],
        log_t1=math.log10(last_time),
        y0=bottom_tick,
        value0=first_value,
        y1=top_tick,
        value1=top_value,
        x_left=x_left,
        x_right=x_right,
    )


def _polyline(drawing):
    points = []
    for item in drawing["items"]:
        if item[0] != "l":
            raise ValueError("Figure 2 curves are expected to be polylines")
        for point in (item[1], item[2]):
            if points and abs(point.x - points[-1][0]) < MERGE_TOLERANCE and abs(point.y - points[-1][1]) < MERGE_TOLERANCE:
                continue
            points.append((point.x, point.y))
    return points


def _clip_to_axis(points, axis):
    """Clip a polyline to the panel's x range, interpolating the crossings.

    matplotlib clips long paths to the figure bounding box before writing them,
    so vertices outside the axes are drawing artefacts rather than data.  The
    segments themselves are unchanged, so interpolating at the axis limits
    recovers the plotted value there.
    """

    clipped = []
    for index, (x, y) in enumerate(points):
        inside = axis.x_left - 1.0e-6 <= x <= axis.x_right + 1.0e-6
        if inside:
            clipped.append((x, y))
            continue
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(points):
                continue
            nx, ny = points[neighbour]
            if not axis.x_left <= nx <= axis.x_right:
                continue
            boundary = axis.x_left if x < axis.x_left else axis.x_right
            if abs(nx - x) < MERGE_TOLERANCE:
                continue
            fraction = (boundary - x) / (nx - x)
            if 0.0 <= fraction <= 1.0:
                clipped.append((boundary, y + fraction * (ny - y)))
    deduplicated = []
    for x, y in clipped:
        if deduplicated and abs(x - deduplicated[-1][0]) < MERGE_TOLERANCE:
            continue
        deduplicated.append((x, y))
    return deduplicated


def extract_panel(drawings, axis, panel_left, panel_right):
    """Return ``{model: [(log10 time, value), ...]}`` for one panel."""

    curves: dict[str, list[list[tuple[float, float]]]] = {}
    for drawing in drawings:
        if len(drawing["items"]) < 10:
            continue
        name = _classify(drawing["color"], drawing["width"])
        if name is None:
            continue
        points = _polyline(drawing)
        centre = sum(x for x, _ in points) / len(points)
        if not panel_left <= centre <= panel_right:
            continue
        clipped = _clip_to_axis(points, axis)
        if len(clipped) < 2:
            continue
        curves.setdefault(name, []).append([(axis.log_time(x), axis.value(y)) for x, y in clipped])
    return curves


def _interpolate(samples, log_time):
    if log_time <= samples[0][0]:
        return samples[0][1]
    if log_time >= samples[-1][0]:
        return samples[-1][1]
    for index in range(1, len(samples)):
        left_time, left_value = samples[index - 1]
        right_time, right_value = samples[index]
        if log_time <= right_time:
            span = right_time - left_time
            if span <= 0.0:
                return right_value
            fraction = (log_time - left_time) / span
            return left_value + fraction * (right_value - left_value)
    return samples[-1][1]


def _common_grid(transmission, reflection):
    """Union of both panels' time grids restricted to their overlap."""

    low = max(transmission[0][0], reflection[0][0])
    high = min(transmission[-1][0], reflection[-1][0])
    grid = sorted(
        value
        for value in {point[0] for point in transmission} | {point[0] for point in reflection}
        if low - 1.0e-9 <= value <= high + 1.0e-9
    )
    merged = []
    for value in grid:
        if merged and value - merged[-1] < 1.0e-6:
            continue
        merged.append(value)
    return merged


# Relative errors at the final simulation time quoted in Section 3 of the paper.
# They are used to verify the digitization: the axis calibration is only correct
# if the extracted curves reproduce these numbers.
QUOTED_RELATIVE_ERRORS = {
    "Atomic Mix": (-47.0, 12.0),
    "CLS LP": (19.0, -5.0),
    "CLS LRP": (15.0, -3.0),
    "LRP Ramp+Exp": (4.0, -1.0),
}


def validate(final_values):
    """Cross-check the digitized curves against the values quoted in the text.

    Two independent checks are applied.  The relative errors of the approximate
    models at the final time are quoted in Section 3, and transmission plus
    reflection must sum to unity because every photon leaving the slab is tallied
    on one of the two z faces.  The latter is a strong check because the two
    panels are calibrated separately and use different axis scales.
    """

    if "Benchmark" not in final_values:
        return ["Benchmark curve missing; cannot validate"]
    benchmark_transmission, benchmark_reflection = final_values["Benchmark"]
    problems = []
    print("Validation against the values quoted in the paper text:")
    for model, (transmission, reflection) in final_values.items():
        total = transmission + reflection
        if abs(total - 1.0) > 5.0e-3:
            problems.append(f"{model}: transmission + reflection = {total:.5f}, expected 1")
        if model == "Benchmark":
            print(f"  {model:<14} T={transmission:.4f} R={reflection:.4f} T+R={total:.5f}")
            continue
        transmission_error = 100.0 * (transmission - benchmark_transmission) / benchmark_transmission
        reflection_error = 100.0 * (reflection - benchmark_reflection) / benchmark_reflection
        quoted_transmission, quoted_reflection = QUOTED_RELATIVE_ERRORS.get(model, (math.nan, math.nan))
        print(f"  {model:<14} T={transmission:.4f} R={reflection:.4f} T+R={total:.5f} "
              f"dT={transmission_error:+.1f}% (paper {quoted_transmission:+.0f}%) "
              f"dR={reflection_error:+.1f}% (paper {quoted_reflection:+.0f}%)")
        if math.isfinite(quoted_transmission) and abs(transmission_error - quoted_transmission) > 1.0:
            problems.append(f"{model}: transmission error {transmission_error:+.1f}% "
                            f"disagrees with the quoted {quoted_transmission:+.0f}%")
        if math.isfinite(quoted_reflection) and abs(reflection_error - quoted_reflection) > 1.0:
            problems.append(f"{model}: reflection error {reflection_error:+.1f}% "
                            f"disagrees with the quoted {quoted_reflection:+.0f}%")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf", type=Path, nargs="?", default=Path(__file__).resolve().parent.parent / "E3D_paper.pdf")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path(__file__).resolve().parent / "e3d_brantley_reference.csv")
    parser.add_argument("--page", type=int, default=7, help="Zero-based page index holding Figure 2 (default: 7)")
    arguments = parser.parse_args()

    document = fitz.open(arguments.pdf)
    drawings = document[arguments.page].get_drawings()

    transmission_axis = build_axis(drawings, 0.0, PANEL_SPLIT_X,
                                   first_decade=-11.0, last_time=5.0e-9,
                                   first_value=0.0, value_step=0.05)
    reflection_axis = build_axis(drawings, PANEL_SPLIT_X, 1.0e9,
                                 first_decade=-12.0, last_time=5.0e-9,
                                 first_value=0.0, value_step=0.2)

    transmission_curves = extract_panel(drawings, transmission_axis, 0.0, PANEL_SPLIT_X)
    reflection_curves = extract_panel(drawings, reflection_axis, PANEL_SPLIT_X, 1.0e9)

    rows = []
    final_values = {}
    for model in ("Benchmark", "Atomic Mix", "CLS LP", "CLS LRP", "LRP Ramp+Exp"):
        if model not in transmission_curves or model not in reflection_curves:
            print(f"warning: {model} missing from one of the panels", file=sys.stderr)
            continue
        transmission = transmission_curves[model][0]
        reflection = reflection_curves[model][0]
        for log_time in _common_grid(transmission, reflection):
            rows.append(
                [
                    model,
                    f"{10.0 ** log_time:.6e}",
                    f"{_interpolate(transmission, log_time):.6f}",
                    f"{_interpolate(reflection, log_time):.6f}",
                ]
            )
        final_values[model] = (float(rows[-1][2]), float(rows[-1][3]))

    problems = validate(final_values)
    if problems:
        print("Digitization failed its consistency checks:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", newline="") as stream:
        stream.write("# Digitized from Figure 2 of Brantley & Novellino, M&C 2025 (E3D_paper.pdf).\n")
        stream.write("# Produced by extract_paper_figure2.py directly from the PDF vector paths;\n")
        stream.write("# values are the plotted polyline vertices, not hand-read from the image.\n")
        stream.write("# Transmission comes from panel (a) and reflection from panel (b); the two\n")
        stream.write("# panels use different time grids, so each curve is resampled onto the union\n")
        stream.write("# of its grids by linear interpolation in log(time).\n")
        writer = csv.writer(stream)
        writer.writerow(["model", "time_s", "transmission", "reflection"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {arguments.output}")
    for model in sorted({row[0] for row in rows}):
        subset = [row for row in rows if row[0] == model]
        print(f"  {model:<14} points={len(subset):3d} "
              f"t=[{subset[0][1]}, {subset[-1][1]}] "
              f"T_final={subset[-1][2]} R_final={subset[-1][3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
