#!/usr/bin/env python3
"""Overlay centered-interface IMC and DDMC profiles with Densmore Fig. 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KEV_K = 1.602176634e-9 / 1.380649e-16


def load_profile(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile not found: {path}")
    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    if raw.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    return raw[:, 0], raw[:, 1] / KEV_K


def load_reference(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Reference data not found: {path}")
    raw = np.loadtxt(path, delimiter=",", comments="#")
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    if raw.shape[1] < 2:
        raise ValueError(f"Expected two columns in {path}")
    return raw[:, 0], raw[:, 1]


def save_overlay(
    output_stem: Path,
    x_mc: np.ndarray,
    t_mc: np.ndarray,
    x_ddmc: np.ndarray,
    t_ddmc: np.ndarray,
    x_ref: np.ndarray,
    t_ref: np.ndarray,
    *,
    zoom: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        x_ref,
        t_ref,
        linewidth=2.0,
        label="Densmore 2012 Fig. 4 reference",
    )
    ax.plot(
        x_mc,
        t_mc,
        linestyle="-",
        marker="o",
        markersize=2.5,
        markerfacecolor="none",
        linewidth=1.0,
        label="RICH IMC, equal interface cells",
    )
    ax.plot(
        x_ddmc,
        t_ddmc,
        linestyle="-",
        marker="s",
        markersize=2.5,
        markerfacecolor="none",
        linewidth=1.0,
        label="RICH DDMC, equal interface cells",
    )
    ax.axvline(2.0, linestyle="--", linewidth=1.0, label="Opacity interface")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Material temperature [keV]")
    if zoom:
        ax.set_title("Densmore 2012 centered interface: IMC vs DDMC (interface zoom)")
        ax.set_xlim(1.85, 2.45)
        ax.set_ylim(0.0, 0.85)
    else:
        ax.set_title("Densmore 2012 centered interface: IMC vs DDMC")
        ax.set_xlim(0.0, 3.0)
        ax.set_ylim(0.0, 1.0)

    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    suffix = "_interface_zoom" if zoom else ""
    png = output_stem.with_name(output_stem.name + suffix).with_suffix(".png")
    pdf = output_stem.with_name(output_stem.name + suffix).with_suffix(".pdf")
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", required=True, type=Path, help="Centered IMC profile")
    parser.add_argument("--ddmc", required=True, type=Path, help="Centered DDMC profile")
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Digitized Densmore Fig. 4 CSV: x [cm], T [keV]",
    )
    parser.add_argument(
        "--output-stem",
        required=True,
        type=Path,
        help="Output path without extension",
    )
    args = parser.parse_args()

    x_mc, t_mc = load_profile(args.mc)
    x_ddmc, t_ddmc = load_profile(args.ddmc)
    x_ref, t_ref = load_reference(args.reference)

    save_overlay(
        args.output_stem,
        x_mc,
        t_mc,
        x_ddmc,
        t_ddmc,
        x_ref,
        t_ref,
        zoom=False,
    )
    save_overlay(
        args.output_stem,
        x_mc,
        t_mc,
        x_ddmc,
        t_ddmc,
        x_ref,
        t_ref,
        zoom=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
