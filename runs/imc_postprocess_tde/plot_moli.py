#!/usr/bin/env python3
"""Plot a Mollweide projection from an IMC post-processing VTK file.

VERSION: 4.0-consistent-length-units

This script is tailored to the legacy ASCII POLYDATA files written by
RICH's SphericalObserver::writeVTK.  It works with both the multigroup and
Gray/grey outputs because it discovers all POINT_DATA fields dynamically.

Required Python packages:
    numpy, matplotlib, scipy

Examples
--------
List every field in a file::

    python3 plot_imc_mollweide.py observer.vtk --list-fields

Plot the bolometric luminosity::

    python3 plot_imc_mollweide.py observer.vtk \
        --field luminosity --scale log --output luminosity.png

Plot one multigroup luminosity field::

    python3 plot_imc_mollweide.py observer.vtk \
        --field group_12_luminosity --scale log

Plot the gray FLD comparison field and mask zero-valued directions::

    python3 plot_imc_mollweide.py observer_gray.vtk \
        --field fld_surface_luminosity --scale log --mask-le 0

Rescale coordinates and all length-valued plotted fields to solar radii::

    python3 plot_imc_mollweide.py observer.vtk \
        --field photosphere_grey_radius_tau_total --length-unit solar-radius

The default coordinate convention is
    longitude = atan2(y, x), latitude = asin(z / r).
Use --central-longitude, --reverse-longitude, or --rotate to alter the view.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SOLAR_RADIUS_CM = 6.957e10
ASTRONOMICAL_UNIT_CM = 1.495978707e13

# scale_cm is the number of centimeters in one displayed unit.
# ``native`` means no conversion and is useful when the VTK coordinates are
# already nondimensional or their unit is unknown.
LENGTH_UNITS = {
    "native": (1.0, "native VTK length unit", "native VTK length units"),
    "cm": (1.0, "cm", "cm"),
    "km": (1.0e5, "km", "km"),
    "solar-radius": (SOLAR_RADIUS_CM, r"R$_\odot$", "solar radii"),
    "rsun": (SOLAR_RADIUS_CM, r"R$_\odot$", "solar radii"),
    "au": (ASTRONOMICAL_UNIT_CM, "AU", "AU"),
}


@dataclass
class LegacyVtkPointData:
    """Minimal representation of the point data needed for this plot."""

    points: np.ndarray
    fields: Dict[str, np.ndarray]


def _next_nonempty(lines: Sequence[str], index: int) -> Tuple[int, str]:
    while index < len(lines):
        text = lines[index].strip()
        if text:
            return index, text
        index += 1
    raise ValueError("Unexpected end of VTK file")


def _read_numeric_values(
    lines: Sequence[str], start: int, count: int, description: str
) -> Tuple[np.ndarray, int]:
    """Read exactly *count* whitespace-separated numbers from subsequent lines."""

    values: List[float] = []
    index = start
    while index < len(lines) and len(values) < count:
        text = lines[index].strip()
        index += 1
        if not text:
            continue
        tokens = text.split()
        try:
            numbers = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError(
                f"Encountered nonnumeric text while reading {description}: {text!r}"
            ) from exc
        if len(values) + len(numbers) > count:
            raise ValueError(
                f"Too many values on one line while reading {description}; "
                "the legacy VTK layout is not the expected RICH observer format"
            )
        values.extend(numbers)

    if len(values) != count:
        raise ValueError(
            f"Expected {count} values for {description}, found {len(values)}"
        )
    return np.asarray(values, dtype=np.float64), index


def read_legacy_vtk_point_data(path: Path) -> LegacyVtkPointData:
    """Read POINTS and POINT_DATA scalar/vector fields from legacy ASCII VTK."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not an ASCII VTK file. This script expects the legacy "
            "ASCII files produced by imc_postprocess_tde."
        ) from exc

    if not lines or not lines[0].lstrip().startswith("# vtk DataFile"):
        raise ValueError(f"{path} does not look like a legacy VTK file")

    if not any(line.strip().upper() == "ASCII" for line in lines[:10]):
        raise ValueError("Only legacy ASCII VTK files are supported")

    points: np.ndarray | None = None
    point_count: int | None = None
    fields: Dict[str, np.ndarray] = {}

    index = 0
    while index < len(lines):
        text = lines[index].strip()
        upper = text.upper()

        if upper.startswith("POINTS "):
            parts = text.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed POINTS header: {text!r}")
            point_count = int(parts[1])
            raw, index = _read_numeric_values(
                lines, index + 1, 3 * point_count, "VTK points"
            )
            points = raw.reshape(point_count, 3)
            continue

        if upper.startswith("POINT_DATA "):
            parts = text.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed POINT_DATA header: {text!r}")
            data_count = int(parts[1])
            if point_count is None:
                raise ValueError("POINT_DATA appears before POINTS")
            if data_count != point_count:
                raise ValueError(
                    f"POINT_DATA has {data_count} entries, but POINTS has {point_count}"
                )
            index += 1
            break

        index += 1

    if points is None or point_count is None:
        raise ValueError("No POINTS section was found")
    if index >= len(lines):
        raise ValueError("No POINT_DATA section was found")

    # RICH currently writes SCALARS only, but VECTORS support is cheap and useful.
    while index < len(lines):
        line_index, text = _next_nonempty(lines, index)
        upper = text.upper()

        if upper.startswith("SCALARS "):
            parts = text.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed SCALARS header: {text!r}")
            name = parts[1]
            components = int(parts[3]) if len(parts) >= 4 else 1

            lookup_index, lookup = _next_nonempty(lines, line_index + 1)
            if not lookup.upper().startswith("LOOKUP_TABLE"):
                raise ValueError(
                    f"SCALARS {name!r} is not followed by LOOKUP_TABLE"
                )

            raw, index = _read_numeric_values(
                lines,
                lookup_index + 1,
                point_count * components,
                f"scalar field {name!r}",
            )
            array = raw.reshape(point_count, components)
            if components == 1:
                fields[name] = array[:, 0]
            else:
                for component in range(components):
                    fields[f"{name}_{component}"] = array[:, component]
                fields[f"{name}_magnitude"] = np.linalg.norm(array, axis=1)
            continue

        if upper.startswith("VECTORS "):
            parts = text.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed VECTORS header: {text!r}")
            name = parts[1]
            raw, index = _read_numeric_values(
                lines, line_index + 1, point_count * 3, f"vector field {name!r}"
            )
            array = raw.reshape(point_count, 3)
            fields[f"{name}_x"] = array[:, 0]
            fields[f"{name}_y"] = array[:, 1]
            fields[f"{name}_z"] = array[:, 2]
            fields[f"{name}_magnitude"] = np.linalg.norm(array, axis=1)
            continue

        # Ignore connectivity and metadata keywords not relevant to point fields.
        index = line_index + 1

    if not fields:
        raise ValueError("No POINT_DATA scalar or vector fields were found")

    return LegacyVtkPointData(points=points, fields=fields)


def infer_sphere_center(points: np.ndarray) -> Tuple[np.ndarray, float, float, str]:
    """Estimate the observer-sphere center in well-scaled coordinates.

    An algebraic sphere fit is attempted first.  Some observer point sets can
    be reported as rank deficient because of finite output precision or nearly
    symmetric sampling.  For a SphericalObserver map, the centroid is then a
    safe fallback: its directions are distributed over the full sphere, and a
    small center error only causes a correspondingly small angular error.
    """

    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 4:
        raise ValueError("At least four three-dimensional VTK points are required")
    if not np.all(np.isfinite(points)):
        raise ValueError("The VTK point coordinates contain non-finite values")

    origin = np.mean(points, axis=0)
    shifted = points - origin
    scale = float(np.sqrt(np.mean(np.einsum("ij,ij->i", shifted, shifted))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The VTK points have zero or invalid spatial extent")

    normalized = shifted / scale
    matrix = np.column_stack((2.0 * normalized, np.ones(points.shape[0])))
    rhs = np.einsum("ij,ij->i", normalized, normalized)
    solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=1.0e-14)

    method = "algebraic sphere fit"
    center = origin.copy()
    if rank >= 4 and np.all(np.isfinite(solution)):
        center_normalized = solution[:3]
        radius_squared_normalized = solution[3] + np.dot(
            center_normalized, center_normalized
        )
        if np.isfinite(radius_squared_normalized) and radius_squared_normalized > 0.0:
            center = origin + scale * center_normalized
        else:
            method = "centroid fallback"
    else:
        method = "centroid fallback"

    distances = np.linalg.norm(points - center, axis=1)
    radius = float(np.mean(distances))
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("Could not infer a valid observer-sphere radius")
    fractional_scatter = float(np.std(distances) / radius)
    return center, radius, fractional_scatter, method


def rotation_matrix_xyz(angles_degrees: Sequence[float]) -> np.ndarray:
    """Return Rz @ Ry @ Rx for the requested x/y/z rotations."""

    rx, ry, rz = np.deg2rad(np.asarray(angles_degrees, dtype=np.float64))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    matrix_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    matrix_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    matrix_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return matrix_z @ matrix_y @ matrix_x


def wrap_longitude(longitude: np.ndarray | float) -> np.ndarray:
    """Wrap radians to [-pi, pi)."""

    return (np.asarray(longitude) + np.pi) % (2.0 * np.pi) - np.pi


def resolve_field_name(requested: str, fields: Dict[str, np.ndarray]) -> str:
    if requested in fields:
        return requested

    lower_matches = [name for name in fields if name.lower() == requested.lower()]
    if len(lower_matches) == 1:
        return lower_matches[0]

    substring_matches = [name for name in fields if requested.lower() in name.lower()]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if len(substring_matches) > 1:
        joined = "\n  ".join(sorted(substring_matches))
        raise ValueError(
            f"Field selector {requested!r} is ambiguous. Matching fields:\n  {joined}"
        )

    available = "\n  ".join(sorted(fields))
    raise ValueError(
        f"Field {requested!r} was not found. Available fields:\n  {available}"
    )


def interpolate_on_sphere(
    sample_directions: np.ndarray,
    sample_values: np.ndarray,
    query_directions: np.ndarray,
    method: str,
    neighbors: int,
    power: float,
) -> np.ndarray:
    """Interpolate using 3-D chord distance, which is seam- and pole-safe."""

    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for spherical interpolation. Install it with "
            "'python3 -m pip install scipy'."
        ) from exc

    tree = cKDTree(sample_directions)
    if method == "nearest":
        _, indices = tree.query(query_directions, k=1)
        return sample_values[indices]

    k = max(1, min(int(neighbors), sample_directions.shape[0]))
    try:
        distances, indices = tree.query(query_directions, k=k, workers=-1)
    except TypeError:  # Compatibility with older SciPy versions.
        distances, indices = tree.query(query_directions, k=k)

    if k == 1:
        return sample_values[indices]

    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices)
    result = np.empty(query_directions.shape[0], dtype=np.float64)

    exact = distances[:, 0] <= 32.0 * np.finfo(np.float64).eps
    result[exact] = sample_values[indices[exact, 0]]

    nonexact = ~exact
    weights = 1.0 / np.maximum(distances[nonexact], 1.0e-15) ** power
    neighbor_values = sample_values[indices[nonexact]]
    result[nonexact] = np.sum(weights * neighbor_values, axis=1) / np.sum(
        weights, axis=1
    )
    return result


def field_statistics(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "no finite values"
    return (
        f"finite={finite.size}/{values.size}, "
        f"min={np.min(finite):.8e}, max={np.max(finite):.8e}, "
        f"mean={np.mean(finite):.8e}"
    )


def list_fields(dataset: LegacyVtkPointData) -> None:
    width = max(len(name) for name in dataset.fields)
    for name in sorted(dataset.fields):
        print(f"{name:<{width}}  {field_statistics(dataset.fields[name])}")


def infer_field_dimension(field_name: str) -> str:
    """Infer whether a VTK scalar is a length.

    RICH's observer VTK files use explicit names for all current length-valued
    outputs, notably ``photosphere_*_radius_*``.  The broader distance/length
    checks make the behavior useful for future observer fields as well.
    """

    name = field_name.lower()
    length_tokens = ("radius", "distance", "_dist", "length", "height")
    return "length" if any(token in name for token in length_tokens) else "none"


def infer_field_unit_label(field_name: str, length_unit_label: str) -> str | None:
    """Return a human-readable physical unit for known observer fields."""

    name = field_name.lower()
    if infer_field_dimension(name) == "length":
        return length_unit_label

    # These diagnostics are dimensionless even when the base field name also
    # contains words such as luminosity or flux.
    dimensionless_tokens = (
        "count", "valid", "relerr", "neff", "snr", "polarization_degree",
        "mismatch", "weight",
    )
    if any(token in name for token in dimensionless_tokens):
        return None
    if "solid_angle" in name:
        return "sr"
    if "polarization_angle" in name:
        return "rad"
    if name.startswith("log10_") or "_log10_" in name:
        if "flux" in name:
            return r"log$_{10}$(erg s$^{-1}$ cm$^{-2}$)"
        if "luminosity" in name:
            return r"log$_{10}$(erg s$^{-1}$)"
        return "dex"
    if "flux" in name:
        # Flux fields are written in CGS and are not changed by --length-unit.
        return r"erg s$^{-1}$ cm$^{-2}$"
    if "luminosity" in name:
        return r"erg s$^{-1}$"
    if name == "energy" or name.endswith("_energy") or "packet_energy" in name:
        return "erg"
    return None


def label_with_unit(label: str, unit: str | None) -> str:
    if not unit:
        return label
    # Respect a custom label that already appears to contain a unit.
    if "[" in label and "]" in label:
        return label
    return f"{label} [{unit}]"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot any POINT_DATA field from a RICH imc_postprocess_tde VTK "
            "observer file on a Mollweide projection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("vtk_file", type=Path, help="Legacy ASCII observer VTK file")
    parser.add_argument("--field", help="Field name to plot; use --list-fields to inspect")
    parser.add_argument(
        "--list-fields", action="store_true", help="List fields and exit unless --field is also given"
    )
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Open an interactive plot window")

    parser.add_argument(
        "--scale", choices=("linear", "log", "symlog"), default="linear",
        help="Color normalization",
    )
    parser.add_argument("--vmin", type=float, help="Lower color limit")
    parser.add_argument("--vmax", type=float, help="Upper color limit")
    parser.add_argument(
        "--clip-percentiles", nargs=2, type=float, metavar=("LOW", "HIGH"),
        help="Set missing vmin/vmax from these percentiles of the interpolated map",
    )
    parser.add_argument(
        "--linthresh", type=float,
        help="Linear half-width for symlog; inferred from the data when omitted",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument("--label", help="Colorbar label; defaults to the field name")
    parser.add_argument("--title", help="Plot title")

    parser.add_argument(
        "--method", choices=("idw", "nearest"), default="idw",
        help="Spherical interpolation method",
    )
    parser.add_argument("--neighbors", type=int, default=8, help="IDW neighbor count")
    parser.add_argument("--power", type=float, default=2.0, help="IDW distance exponent")
    parser.add_argument(
        "--resolution", nargs=2, type=int, default=(720, 360),
        metavar=("NLON", "NLAT"), help="Longitude/latitude grid size",
    )

    parser.add_argument(
        "--center", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help=(
            "Observer-sphere center in the original VTK units; otherwise it is "
            "inferred from the points"
        ),
    )
    parser.add_argument(
        "--length-unit",
        choices=tuple(LENGTH_UNITS),
        default="native",
        help=(
            "Displayed length unit. Coordinates are converted before sphere fitting, "
            "and length-valued fields are converted to the same unit. The VTK "
            "coordinates/length fields are assumed to be in cm for km, "
            "solar-radius/rsun, and au"
        ),
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        help=(
            "Custom number of original VTK length units per rescaled unit; "
            "coordinates are divided by this value. Overrides --length-unit"
        ),
    )
    parser.add_argument(
        "--field-dimension",
        choices=("auto", "length", "none"),
        default="auto",
        help=(
            "Whether the selected field is length-valued. In auto mode, fields "
            "whose names contain radius/distance/length are converted to "
            "--length-unit. Use length or none to override detection"
        ),
    )
    parser.add_argument(
        "--rotate", nargs=3, type=float, default=(0.0, 0.0, 0.0),
        metavar=("RX", "RY", "RZ"),
        help="View rotation in degrees, applied about x then y then z",
    )
    parser.add_argument(
        "--central-longitude", type=float, default=0.0,
        help="Longitude at the center of the map, in degrees",
    )
    parser.add_argument(
        "--reverse-longitude", action="store_true",
        help="Make increasing longitude run from right to left",
    )
    parser.add_argument(
        "--show-points", action="store_true",
        help="Overlay the original observer sample locations",
    )

    parser.add_argument(
        "--mask-le", type=float, metavar="VALUE",
        help="Discard samples whose field value is <= VALUE before interpolation",
    )
    parser.add_argument(
        "--mask-ge", type=float, metavar="VALUE",
        help="Discard samples whose field value is >= VALUE before interpolation",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Saved image resolution")
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=(12.0, 6.8),
        metavar=("WIDTH", "HEIGHT"), help="Figure size in inches",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.vtk_file.is_file():
        parser.error(f"VTK file does not exist: {args.vtk_file}")
    if args.neighbors < 1:
        parser.error("--neighbors must be at least 1")
    if args.power <= 0.0:
        parser.error("--power must be positive")
    if args.resolution[0] < 8 or args.resolution[1] < 4:
        parser.error("--resolution is too small")
    if args.clip_percentiles is not None:
        low, high = args.clip_percentiles
        if not (0.0 <= low < high <= 100.0):
            parser.error("--clip-percentiles must satisfy 0 <= LOW < HIGH <= 100")
    if args.length_scale is not None and (
        not np.isfinite(args.length_scale) or args.length_scale <= 0.0
    ):
        parser.error("--length-scale must be a positive finite number")

    try:
        dataset = read_legacy_vtk_point_data(args.vtk_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.list_fields:
        list_fields(dataset)
        if args.field is None:
            return 0
    if args.field is None:
        parser.error("--field is required unless only --list-fields is requested")

    try:
        field_name = resolve_field_name(args.field, dataset.fields)
    except ValueError as exc:
        parser.error(str(exc))

    raw_values = np.asarray(dataset.fields[field_name], dtype=np.float64)

    if args.length_scale is not None:
        coordinate_divisor = float(args.length_scale)
        length_unit_label = "custom length unit"
        length_unit_words = (
            f"custom units (1 unit = {coordinate_divisor:.12e} original VTK units)"
        )
    else:
        coordinate_divisor, length_unit_label, length_unit_words = LENGTH_UNITS[
            args.length_unit
        ]
        coordinate_divisor = float(coordinate_divisor)

    points = np.asarray(dataset.points, dtype=np.float64) / coordinate_divisor

    detected_dimension = infer_field_dimension(field_name)
    field_dimension = (
        detected_dimension if args.field_dimension == "auto" else args.field_dimension
    )
    values = raw_values.copy()
    if field_dimension == "length":
        values /= coordinate_divisor
    field_unit = infer_field_unit_label(field_name, length_unit_label)
    if field_dimension == "length":
        field_unit = length_unit_label

    if coordinate_divisor != 1.0:
        print(
            f"Rescaled VTK coordinates by 1/{coordinate_divisor:.12e}; "
            f"displayed length unit: {length_unit_words}"
        )
    if field_dimension == "length":
        print(
            f"Converted field {field_name!r} by 1/{coordinate_divisor:.12e}; "
            f"displayed field unit: {length_unit_words}"
        )

    if args.center is None:
        try:
            center, radius, radial_scatter, center_method = infer_sphere_center(points)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            "Inferred sphere: center=("
            + ", ".join(f"{component:.12e}" for component in center)
            + f"), radius={radius:.12e}, fractional radial scatter={radial_scatter:.3e}, "
            + f"method={center_method}, unit={length_unit_words}"
        )
        if center_method == "centroid fallback":
            print(
                "Warning: algebraic sphere fitting was rank deficient; using the "
                "observer-point centroid as the center.",
                file=sys.stderr,
            )
        if radial_scatter > 1.0e-4:
            print(
                "Warning: points have appreciable radial scatter; consider supplying --center.",
                file=sys.stderr,
            )
    else:
        # --center remains in original VTK units for backward compatibility.
        center = np.asarray(args.center, dtype=np.float64) / coordinate_divisor
        supplied_distances = np.linalg.norm(points - center, axis=1)
        radius = float(np.mean(supplied_distances))
        radial_scatter = float(np.std(supplied_distances) / radius)
        center_method = "user supplied"

    relative = points - center
    radii = np.linalg.norm(relative, axis=1)
    valid_geometry = np.isfinite(radii) & (radii > 0.0)
    directions = np.empty_like(relative)
    directions[valid_geometry] = relative[valid_geometry] / radii[valid_geometry, None]

    rotation = rotation_matrix_xyz(args.rotate)
    directions = directions @ rotation.T

    valid = valid_geometry & np.isfinite(values)
    if args.mask_le is not None:
        valid &= values > args.mask_le
    if args.mask_ge is not None:
        valid &= values < args.mask_ge
    if args.scale == "log":
        valid &= values > 0.0

    if not np.any(valid):
        parser.error("No samples remain after applying geometry, finite-value, and mask filters")

    sample_directions = directions[valid]
    sample_values = values[valid]

    nlon, nlat = args.resolution
    plot_longitude = np.linspace(-np.pi, np.pi, nlon)
    latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, nlat)
    plot_lon_grid, lat_grid = np.meshgrid(plot_longitude, latitude)

    central_longitude = math.radians(args.central_longitude)
    longitude_sign = -1.0 if args.reverse_longitude else 1.0
    source_longitude = wrap_longitude(
        central_longitude + longitude_sign * plot_lon_grid
    )
    cos_latitude = np.cos(lat_grid)
    query_directions = np.column_stack(
        (
            (cos_latitude * np.cos(source_longitude)).ravel(),
            (cos_latitude * np.sin(source_longitude)).ravel(),
            np.sin(lat_grid).ravel(),
        )
    )

    try:
        interpolated = interpolate_on_sphere(
            sample_directions,
            sample_values,
            query_directions,
            args.method,
            args.neighbors,
            args.power,
        ).reshape(nlat, nlon)
    except RuntimeError as exc:
        parser.error(str(exc))

    finite_map = interpolated[np.isfinite(interpolated)]
    if args.scale == "log":
        finite_for_limits = finite_map[finite_map > 0.0]
    else:
        finite_for_limits = finite_map
    if finite_for_limits.size == 0:
        parser.error("The interpolated map has no values valid for the selected scale")

    vmin = args.vmin
    vmax = args.vmax
    if args.clip_percentiles is not None:
        percentile_low, percentile_high = np.percentile(
            finite_for_limits, args.clip_percentiles
        )
        if vmin is None:
            vmin = float(percentile_low)
        if vmax is None:
            vmax = float(percentile_high)
    if vmin is None:
        vmin = float(np.min(finite_for_limits))
    if vmax is None:
        vmax = float(np.max(finite_for_limits))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        parser.error(f"Invalid color limits: vmin={vmin}, vmax={vmax}")
    if args.scale == "log" and vmin <= 0.0:
        parser.error("Log scaling requires vmin > 0")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm, Normalize, SymLogNorm
    except ImportError as exc:
        parser.error(
            "matplotlib is required. Install it with "
            "'python3 -m pip install matplotlib'."
        )

    if args.scale == "linear":
        norm = Normalize(vmin=vmin, vmax=vmax)
    elif args.scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        interpolated = np.ma.masked_less_equal(interpolated, 0.0)
    else:
        max_abs = max(abs(vmin), abs(vmax))
        linthresh = args.linthresh
        if linthresh is None:
            nonzero = np.abs(finite_for_limits[finite_for_limits != 0.0])
            linthresh = (
                float(np.percentile(nonzero, 5.0))
                if nonzero.size
                else max(1.0e-12, 1.0e-3 * max_abs)
            )
        if linthresh <= 0.0:
            parser.error("--linthresh must be positive")
        norm = SymLogNorm(
            linthresh=linthresh, linscale=1.0, vmin=vmin, vmax=vmax, base=10
        )

    figure = plt.figure(figsize=tuple(args.figsize), constrained_layout=True)
    axis = figure.add_subplot(111, projection="mollweide")
    mesh = axis.pcolormesh(
        plot_lon_grid,
        lat_grid,
        interpolated,
        shading="auto",
        cmap=args.cmap,
        norm=norm,
        rasterized=True,
    )
    axis.grid(True, alpha=0.35)

    if args.show_points:
        sample_lon = np.arctan2(sample_directions[:, 1], sample_directions[:, 0])
        sample_lat = np.arcsin(np.clip(sample_directions[:, 2], -1.0, 1.0))
        sample_plot_lon = longitude_sign * wrap_longitude(
            sample_lon - central_longitude
        )
        axis.scatter(
            sample_plot_lon,
            sample_lat,
            s=3.0,
            c="black",
            alpha=0.35,
            linewidths=0.0,
        )

    colorbar = figure.colorbar(mesh, ax=axis, orientation="horizontal", pad=0.08)
    base_colorbar_label = args.label if args.label else field_name
    colorbar.set_label(label_with_unit(base_colorbar_label, field_unit))

    title = args.title
    if title is None:
        title = f"{field_name} — {args.vtk_file.name}"
    center_text = ", ".join(f"{component:.4g}" for component in center)
    geometry_caption = (
        f"observer center = ({center_text}) {length_unit_label}; "
        f"radius = {radius:.6g} {length_unit_label}"
    )
    axis.set_title(f"{title}\n{geometry_caption}", pad=18.0)

    output = args.output
    if output is None:
        safe_field = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in field_name
        )
        output = args.vtk_file.with_name(
            f"{args.vtk_file.stem}_{safe_field}_mollweide.png"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {output}")
    print(f"Plotted field: {label_with_unit(field_name, field_unit)} ({field_statistics(values)})")
    print(f"Used {sample_values.size}/{values.size} observer samples")

    if args.show:
        plt.show()
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())