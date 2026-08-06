#!/usr/bin/env python3
"""Plot gray scattering opacity divided by MG Planck-mean scattering opacity."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


KELVIN_PER_EV = 11604.5
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = (SCRIPT_DIR / "../..").resolve()


def read_vector(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=float)
    return np.atleast_1d(values)


def bilinear(x_grid: np.ndarray, y_grid: np.ndarray, values: np.ndarray, x: float, y: float) -> float:
    ix = int(np.searchsorted(x_grid, x, side="right") - 1)
    iy = int(np.searchsorted(y_grid, y, side="right") - 1)
    ix = max(0, min(ix, len(x_grid) - 2))
    iy = max(0, min(iy, len(y_grid) - 2))

    x0, x1 = x_grid[ix], x_grid[ix + 1]
    y0, y1 = y_grid[iy], y_grid[iy + 1]
    tx = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)

    z00 = values[ix, iy]
    z10 = values[ix + 1, iy]
    z01 = values[ix, iy + 1]
    z11 = values[ix + 1, iy + 1]
    return float((1.0 - tx) * (1.0 - ty) * z00 + tx * (1.0 - ty) * z10 + (1.0 - tx) * ty * z01 + tx * ty * z11)


def load_gray_scattering(grey_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    log_t = read_vector(grey_dir / "T.txt")
    log_rho = read_vector(grey_dir / "rho.txt")
    # STA gray tables store log(opacity); C++ Interpolate2DTable exponentiates
    # the interpolated value before STAgreyOpacity applies the density scale.
    table = read_vector(grey_dir / "scatter.txt").reshape((len(log_t), len(log_rho)))
    return log_t, log_rho, table


def gray_scattering(log_t_grid: np.ndarray, log_rho_grid: np.ndarray, table: np.ndarray, temp_k: float, rho: float) -> float:
    log_t = float(np.log(temp_k))
    log_rho = float(np.log(rho))
    density_scale = 1.0
    if log_rho < log_rho_grid[0]:
        density_scale = rho / np.exp(log_rho_grid[0])
        log_rho = float(log_rho_grid[0])
    elif log_rho > log_rho_grid[-1]:
        density_scale = rho / np.exp(log_rho_grid[-1])
        log_rho = float(log_rho_grid[-1])
    log_t = float(np.clip(log_t, log_t_grid[0], log_t_grid[-1]))
    return float(np.exp(bilinear(log_t_grid, log_rho_grid, table, log_t, log_rho)) * density_scale)


def load_mg_scattering(mg_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temp_ev_grid = read_vector(mg_dir / "T.txt")
    log_t_grid = np.log(temp_ev_grid * KELVIN_PER_EV)
    rho_grid = read_vector(mg_dir / "rho.txt")
    log_rho_grid = np.log(rho_grid)
    edges_ev = read_vector(mg_dir / "frequency_edges.txt")

    group_tables: list[np.ndarray] = []
    for group in range(len(edges_ev) - 1):
        raw = read_vector(mg_dir / f"sigma_scattering_planck_{group + 1}.txt")
        # The C++ STAMGopacityMC table is indexed as [rho][T] and stores
        # log(sigma_scattering_planck) + log(rho).
        table = raw.reshape((len(log_rho_grid), len(log_t_grid)))
        group_tables.append(np.log(np.maximum(table, np.finfo(float).tiny)) + log_rho_grid[:, None])

    return log_t_grid, log_rho_grid, edges_ev, np.asarray(group_tables)


def mg_group_scattering(
    log_t_grid: np.ndarray,
    log_rho_grid: np.ndarray,
    group_tables: np.ndarray,
    group: int,
    temp_k: float,
    rho: float,
) -> float:
    log_t = float(np.clip(np.log(temp_k), log_t_grid[0], log_t_grid[-1]))
    log_rho = float(np.log(rho))
    density_scale = 1.0
    if log_rho < log_rho_grid[0]:
        density_scale = rho / np.exp(log_rho_grid[0])
        log_rho = float(log_rho_grid[0])
    elif log_rho > log_rho_grid[-1]:
        density_scale = rho / np.exp(log_rho_grid[-1])
        log_rho = float(log_rho_grid[-1])
    return float(np.exp(bilinear(log_rho_grid, log_t_grid, group_tables[group], log_rho, log_t)) * density_scale)


def planck_integrand(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    small = x < 1.0e-6
    mid = (x >= 1.0e-6) & (x < 700.0)
    out[small] = x[small] ** 2
    out[mid] = x[mid] ** 3 / np.expm1(x[mid])
    return out


def planck_group_weights(edges_ev: np.ndarray, temp_k: float, quadrature_order: int = 64) -> np.ndarray:
    temp_ev = temp_k / KELVIN_PER_EV
    x_edges = edges_ev / temp_ev
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    result = np.zeros(len(edges_ev) - 1)
    for group, (lo, hi) in enumerate(zip(x_edges[:-1], x_edges[1:])):
        center = 0.5 * (lo + hi)
        half_width = 0.5 * (hi - lo)
        if not np.isfinite(center) or not np.isfinite(half_width) or half_width <= 0.0:
            continue
        x = center + half_width * nodes
        result[group] = half_width * float(np.sum(weights * planck_integrand(x)))
    return result


def mg_planck_mean_scattering(
    log_t_grid: np.ndarray,
    log_rho_grid: np.ndarray,
    edges_ev: np.ndarray,
    group_tables: np.ndarray,
    temp_k: float,
    rho: float,
) -> float:
    weights = planck_group_weights(edges_ev, temp_k)
    total_weight = float(np.sum(weights))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return np.nan

    weighted_opacity = 0.0
    for group, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        weighted_opacity += weight * mg_group_scattering(log_t_grid, log_rho_grid, group_tables, group, temp_k, rho)
    return weighted_opacity / total_weight


def make_ratio_grid(grey_dir: Path, mg_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grey_log_t, grey_log_rho, grey_table = load_gray_scattering(grey_dir)
    mg_log_t, mg_log_rho, mg_edges_ev, mg_group_tables = load_mg_scattering(mg_dir)

    temp_grid = np.logspace(np.log10(7000.0), np.log10(1.0e9), 32)
    rho_grid = np.logspace(-17.0, -8.0, 32)
    ratio = np.full((len(temp_grid), len(rho_grid)), np.nan)

    for i, temp_k in enumerate(temp_grid):
        for j, rho in enumerate(rho_grid):
            grey = gray_scattering(grey_log_t, grey_log_rho, grey_table, temp_k, rho)
            mg = mg_planck_mean_scattering(mg_log_t, mg_log_rho, mg_edges_ev, mg_group_tables, temp_k, rho)
            if mg > 0.0 and np.isfinite(grey) and np.isfinite(mg):
                ratio[i, j] = grey / mg

    return temp_grid, rho_grid, ratio


def plot_ratio(temp_grid: np.ndarray, rho_grid: np.ndarray, ratio: np.ndarray, output: Path) -> None:
    finite = ratio[np.isfinite(ratio) & (ratio > 0.0)]
    if finite.size == 0:
        raise RuntimeError("No finite positive ratios were computed")

    vmin = max(np.percentile(finite, 2.0), np.finfo(float).tiny)
    vmax = max(np.percentile(finite, 98.0), vmin * 1.001)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    mesh = ax.pcolormesh(rho_grid, temp_grid, ratio, shading="auto", norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\rho$ [g cm$^{-3}$]")
    ax.set_ylabel(r"$T$ [K]")
    ax.set_title("Gray scattering opacity / MG Planck-mean scattering opacity")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"$\sigma_{s,\mathrm{gray}} / \langle\sigma_{s,\mathrm{MG}}\rangle_\mathrm{Planck}$")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grey-dir", type=Path, default=REPO_ROOT / "data/STA")
    parser.add_argument("--mg-dir", type=Path, default=REPO_ROOT / "data/STA/MG")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "output/figures/gray_vs_mg_planck_scattering_ratio.png",
    )
    args = parser.parse_args()

    temp_grid, rho_grid, ratio = make_ratio_grid(args.grey_dir, args.mg_dir)
    plot_ratio(temp_grid, rho_grid, ratio, args.output)

    finite = ratio[np.isfinite(ratio)]
    print(f"wrote {args.output}")
    print(f"ratio min/median/max = {finite.min():.6e} / {np.median(finite):.6e} / {finite.max():.6e}")


if __name__ == "__main__":
    main()
