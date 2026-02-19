#!/usr/bin/env python3
"""
Plot regression test results against analytical/reference solutions.

Inspects the latest regression_results/<timestamp> directory to determine
which tests were run, then reads profile data from the case directories
and generates comparison plots.

Usage:
    python3 regression_tests/plot_results.py [--results-dir DIR] [--output-dir DIR]
"""

import argparse
import os
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np

# Defer matplotlib import so --help works without a display
_plt = None


def _get_plt():
    global _plt
    if _plt is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _plt = plt
    return _plt


def _save_fig(fig, out_dir: Path, name: str, dpi: int = 150) -> None:
    """Save a figure as both PNG and PDF."""
    fig.savefig(str(out_dir / f"{name}.png"), dpi=dpi)
    fig.savefig(str(out_dir / f"{name}.pdf"))


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_latest_results(root: Path) -> Path | None:
    results_dir = root / "regression_results"
    if not results_dir.is_dir():
        return None
    timestamps = sorted(
        [d for d in results_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )
    return timestamps[-1] if timestamps else None


def tests_in_results(results_path: Path) -> set[str]:
    if results_path is None:
        return set()
    return {d.name for d in results_path.iterdir() if d.is_dir()}


# --------------------------------------------------------------------------- #
# Individual test plotters
# --------------------------------------------------------------------------- #


def plot_sod(root: Path, out_dir: Path) -> bool:
    """Sod 1D: density and pressure vs x compared to exact Riemann solution."""
    profile = root / "regression_tests" / "cases" / "sod_1d" / "sod_profile.txt"
    if not profile.exists():
        print(f"  [sod_1d] profile not found: {profile}")
        return False

    enrs_path = root / "analytic" / "enrs.py"
    enrs = SourceFileLoader("enrs", str(enrs_path)).load_module()

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x = raw[:, 0]
    density = raw[:, 1]
    pressure = raw[:, 2]

    left = enrs.Primitive(1.0, 1.0, 0.0)
    right = enrs.Primitive(0.125, 0.1, 0.0)
    rp = enrs.RiemannProfile(left, right, 1.4)
    t = 0.2
    offset = 0.5

    x_fine = np.linspace(float(x.min()), float(x.max()), 1000)
    density_exact = np.array([rp.CalcPrim((xi - offset) / t).Density for xi in x_fine])
    pressure_exact = np.array([rp.CalcPrim((xi - offset) / t).Pressure for xi in x_fine])

    plt = _get_plt()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(x, density, "k.", markersize=2, label="Numeric")
    ax1.plot(x_fine, density_exact, "r-", linewidth=1.5, label="Exact")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Density")
    ax1.set_title("Sod -- Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, pressure, "k.", markersize=2, label="Numeric")
    ax2.plot(x_fine, pressure_exact, "r-", linewidth=1.5, label="Exact")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Pressure")
    ax2.set_title("Sod -- Pressure")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, "sod_1d")
    plt.close(fig)
    print(f"  [sod_1d] saved sod_1d.png/pdf")
    return True


def plot_sedov(root: Path, out_dir: Path) -> bool:
    """Sedov 3D: density vs r compared to Sedov-Taylor ODE solution."""
    profile = root / "regression_tests" / "cases" / "sedov_3d_mpi" / "sedov_profile.txt"
    if not profile.exists():
        print(f"  [sedov_3d_mpi] profile not found: {profile}")
        return False

    sedov_path = root / "analytic" / "sedov_taylor.py"
    sedov = SourceFileLoader("sedov_taylor", str(sedov_path)).load_module()

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    r = raw[:, 0]
    density = raw[:, 1]
    pressure = raw[:, 2]
    velocity = raw[:, 3]

    gamma = 5.0 / 3.0
    w = 0.0
    n = 3

    # Find shock front at max pressure
    idx_shock = int(np.argmax(pressure))
    shock_front = {
        "radius": float(r[idx_shock]),
        "density": float(density[idx_shock]),
        "pressure": float(pressure[idx_shock]),
        "velocity": float(velocity[idx_shock]),
    }

    # Upstream state from far field
    far_mask = r > 0.8 * float(np.max(r))
    if not np.any(far_mask):
        far_mask = r > np.median(r)
    upstream = {
        "density": float(np.median(density[far_mask])),
        "pressure": float(np.median(pressure[far_mask])),
        "velocity": float(np.median(velocity[far_mask])),
    }

    # Build Sedov-Taylor profiles via ODE integration tables
    nip = 3000
    ssv = np.linspace(1e-6 + 1.0 / gamma, 2.0 / (gamma + 1.0), num=nip)
    shock_radius = shock_front["radius"]

    # Use theoretical strong-shock density at the front:
    # rho_shock = (gamma+1)/(gamma-1) * rho_upstream
    compression = (gamma + 1.0) / (gamma - 1.0)
    shock_density_theory = upstream["density"] * compression

    radius_table = np.array([shock_radius * sedov.vtoz(v, w, gamma, n) for v in ssv])
    density_table = np.array([
        shock_density_theory * sedov.vtod(v, w, gamma, n) / compression
        for v in ssv
    ])

    r_fine = np.linspace(0, float(r.max()), 500)
    density_analytic = np.array([
        float(np.interp(ri, radius_table, density_table)) if ri <= shock_radius
        else upstream["density"]
        for ri in r_fine
    ])

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, density, "k.", markersize=2, label="Numeric (binned)")
    ax.plot(r_fine, density_analytic, "r-", linewidth=1.5, label="Sedov-Taylor ODE")
    ax.set_xlabel("r")
    ax.set_ylabel("Density")
    ax.set_title("Sedov 3D -- Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "sedov_3d_mpi")
    plt.close(fig)
    print(f"  [sedov_3d_mpi] saved sedov_3d_mpi.png/pdf")
    return True


def plot_lane(root: Path, out_dir: Path) -> bool:
    """Lane-Emden: density vs r compared to initial (analytic) Lane-Emden profile."""
    profile = root / "regression_tests" / "cases" / "lane_self_gravity" / "lane_profile.txt"
    if not profile.exists():
        print(f"  [lane_self_gravity] profile not found: {profile}")
        return False

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    r = raw[:, 0]
    density = raw[:, 1]
    density_analytic = raw[:, 2]

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(r, density, "k.", markersize=3, label="Numeric (binned)")
    ax.plot(r, density_analytic, "r-", linewidth=1.5, label="Initial (Lane-Emden)")
    ax.set_xlabel("r [cm]")
    ax.set_ylabel("Density [g/cm$^3$]")
    ax.set_title("Lane-Emden Self-Gravity -- Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "lane_self_gravity")
    plt.close(fig)
    print(f"  [lane_self_gravity] saved lane_self_gravity.png/pdf")
    return True


def plot_till(root: Path, out_dir: Path) -> bool:
    """Till Compton: Tgas and Trad vs time, with IN-FBC reference from McGraw et al."""
    case_dir = root / "regression_tests" / "cases" / "till_compton"
    time_file = case_dir / "time.txt"
    tgas_file = case_dir / "Tgas.txt"
    trad_file = case_dir / "Trad.txt"
    for f in (time_file, tgas_file, trad_file):
        if not f.exists():
            print(f"  [till_compton] file not found: {f}")
            return False

    time = np.loadtxt(str(time_file))
    tgas = np.loadtxt(str(tgas_file))
    trad = np.loadtxt(str(trad_file))

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, tgas, "r-", linewidth=1.5, label="$T_{\\mathrm{gas}}$ (RICH)")
    ax.plot(time, trad, "b-", linewidth=1.5, label="$T_{\\mathrm{rad}}$ (RICH)")

    ref_file = case_dir / "data" / "in_fbc_reference.txt"
    if ref_file.exists():
        ref = np.loadtxt(str(ref_file))
        ax.plot(ref[:, 0], ref[:, 1], "k^", markersize=5,
                label="$T_{\\mathrm{gas}}$ (IN-FBC)")
        ax.plot(ref[:, 0], ref[:, 2], "ks", markersize=4,
                label="$T_{\\mathrm{rad}}$ (IN-FBC)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 3e-9)
    ax.set_title("Till Compton -- Gas & Radiation Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "till_compton")
    plt.close(fig)
    print(f"  [till_compton] saved till_compton.png/pdf")
    return True


def _load_nlte_solver(root: Path):
    """Import the NLTE radiative shock solver."""
    solver_dir = root / "analysis_files" / "radiative_shock"
    if not solver_dir.is_dir():
        raise FileNotFoundError(f"Analytical solver directory not found: {solver_dir}")
    if str(solver_dir) not in sys.path:
        sys.path.insert(0, str(solver_dir))
    from nlte_radiative_shock import NLTERadiativeShock  # noqa: E402
    return NLTERadiativeShock


def _plot_mach2(root: Path, out_dir: Path, case_name: str, label: str) -> bool:
    """Mach2 radiative shock: density, Tgas, Trad vs x with NLTE analytical."""
    profile = root / "regression_tests" / "cases" / case_name / "mach2_profile.txt"
    if not profile.exists():
        print(f"  [{case_name}] profile not found: {profile}")
        return False

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_num = raw[:, 0]
    rho_num = raw[:, 1]
    T_num = raw[:, 2]
    Trad_num = raw[:, 3] if raw.shape[1] > 3 else None

    NLTERadiativeShock = _load_nlte_solver(root)

    gamma = 5.0 / 3.0
    k_boltz = 1.380649e-16
    mu = 1.67e-24
    cv = k_boltz / (mu * (gamma - 1.0))
    rho_left = 5.45887e-13
    v_left = 2.3547e5
    T_left = 100.0
    sigma_ross = 0.848902
    sigma_abs = 3.93e-5
    cs_left = np.sqrt(gamma * (gamma - 1.0) * cv * T_left)
    M0 = v_left / cs_left
    sim_time = 0.01

    solver = NLTERadiativeShock(
        M0=M0,
        gamma=gamma,
        sigma_ross=lambda T, rho: sigma_ross,
        sigma_abs=lambda T, rho: sigma_abs,
        cv=cv,
        rho_left=rho_left,
        v_left=v_left,
        T_left=T_left,
        eps_nlte_solver=1e-4,
    )
    solution = solver.solve_profiles(time=sim_time, x=x_num)
    rho_ana = solution["density"]
    T_ana = solution["temperature"]
    Trad_ana = solution["radiation_temperature"]

    plt = _get_plt()

    ncols = 3 if Trad_num is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    ax = axes[0]
    ax.plot(x_num, rho_num, "k.", markersize=1, label="Numeric")
    ax.plot(x_num, rho_ana, "r-", linewidth=1.5, label="NLTE Analytical")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Density [g/cm$^3$]")
    ax.set_title(f"{label} -- Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(x_num, T_num, "k.", markersize=1, label="Numeric $T_{\\mathrm{gas}}$")
    ax.plot(x_num, T_ana, "r-", linewidth=1.5, label="NLTE Analytical")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("$T_{\\mathrm{gas}}$ [K]")
    ax.set_title(f"{label} -- Gas Temperature")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if Trad_num is not None:
        ax = axes[2]
        ax.plot(x_num, Trad_num, "k.", markersize=1, label="Numeric $T_{\\mathrm{rad}}$")
        ax.plot(x_num, Trad_ana, "b-", linewidth=1.5, label="NLTE Analytical")
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("$T_{\\mathrm{rad}}$ [K]")
        ax.set_title(f"{label} -- Radiation Temperature")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, case_name)
    plt.close(fig)
    print(f"  [{case_name}] saved {case_name}.png/pdf")
    return True


def plot_mach2_diffusion(root: Path, out_dir: Path) -> bool:
    return _plot_mach2(root, out_dir, "mach2_diffusion", "Mach2 Gray Diffusion")


def _plot_mach2_spectrum(root: Path, out_dir: Path) -> bool:
    """Mach2 multigroup: radiation spectrum of hottest cell vs Planck at Tgas."""
    spectrum_file = root / "regression_tests" / "cases" / "mach2_spectrum.txt"
    if not spectrum_file.exists():
        print(f"  [mach2_multigroup] spectrum file not found: {spectrum_file}")
        return False

    # Parse the spectrum file
    tgas = None
    density = None
    num_groups = None
    group_data = []
    with open(str(spectrum_file)) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Tgas "):
                tgas = float(line.split()[1])
            elif line.startswith("density "):
                density = float(line.split()[1])
            elif line.startswith("num_groups "):
                num_groups = int(line.split()[1])
            else:
                parts = line.split()
                if len(parts) == 3:
                    group_data.append((float(parts[0]), float(parts[1]), float(parts[2])))

    if tgas is None or density is None or len(group_data) == 0:
        print(f"  [mach2_multigroup] could not parse spectrum file")
        return False

    E_low = np.array([g[0] for g in group_data])
    E_high = np.array([g[1] for g in group_data])
    Eg_numeric = np.array([g[2] for g in group_data])  # erg/cm^3 per group
    E_center = 0.5 * (E_low + E_high)
    dE = E_high - E_low

    # Compute Planck energy density per group at Tgas
    k_B = 1.380649e-16  # erg/K
    a_rad = 7.565732690980505e-15  # erg cm^-3 K^-4

    def planck_clark_taylor(x):
        x2 = x * x
        x3 = x2 * x
        return x3 * (1.0/3.0 + x * (-1.0/8.0 + x * (1.0/60.0 + x2 * (-1.0/5040.0 +
               x2 * (1.0/272160.0 + x2 * (-1.0/13305600.0 + x2 / 622702080.0))))))

    def planck_clark_series(x, N=5):
        x2 = x * x
        x3 = x2 * x
        s = 0.0
        for n in range(1, N + 1):
            inv_n = 1.0 / n
            s += inv_n * (x3 + inv_n * (3.0 * x2 + 6.0 * inv_n * (x + inv_n))) * np.exp(-x * n)
        return -s

    def planck_integral_fn(a, b):
        coeff = 15.0 / (np.pi ** 4)
        x_clark = 2.0
        if a > x_clark:
            return coeff * (planck_clark_series(b) - planck_clark_series(a))
        if b < x_clark:
            return coeff * (planck_clark_taylor(b) - planck_clark_taylor(a))
        return 1.0 + coeff * (planck_clark_series(b) - planck_clark_taylor(a))

    kT = k_B * tgas
    Eg_planck = np.array([
        a_rad * tgas**4 * planck_integral_fn(el / kT, eh / kT)
        for el, eh in zip(E_low, E_high)
    ])

    # Plot as energy density per unit energy: Eg/dE
    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(E_center / k_B, Eg_numeric / dE, where="mid", color="k", linewidth=1.5,
            label="Numeric spectrum")
    ax.step(E_center / k_B, Eg_planck / dE, where="mid", color="r", linewidth=1.5,
            linestyle="--", label=f"Planck at $T_{{\\mathrm{{gas}}}}$ = {tgas:.1f} K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e3, top=2e8)
    ax.set_xlabel("$E / k_B$ [K]")
    ax.set_ylabel("Energy density per unit energy [erg cm$^{-3}$ erg$^{-1}$]")
    ax.set_title("Mach2 Multigroup -- Spectrum at Hottest Cell")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    _save_fig(fig, out_dir, "mach2_multigroup_spectrum")
    plt.close(fig)
    print(f"  [mach2_multigroup] saved mach2_multigroup_spectrum.png/pdf")
    return True


def plot_mach2_multigroup(root: Path, out_dir: Path) -> bool:
    ok1 = _plot_mach2(root, out_dir, "mach2_multigroup", "Mach2 Multigroup")
    ok2 = _plot_mach2_spectrum(root, out_dir)
    return ok1 or ok2


# --------------------------------------------------------------------------- #
# Marshak wave plotters (Problems 1-4)
# --------------------------------------------------------------------------- #


def _plot_marshak_wave(root: Path, out_dir: Path, prob_num: int) -> bool:
    """Plot Marshak wave profile: RICH Tgas, Trad vs analytical."""
    case_dir = root / "regression_tests" / "cases" / f"marshak_wave_{prob_num}"
    profile = case_dir / "marshak_profile.txt"
    if not profile.exists():
        print(f"  [marshak_wave_{prob_num}] profile not found: {profile}")
        return False

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x_sim = raw[:, 0]
    tgas_sim = raw[:, 1]
    trad_sim = raw[:, 2]

    keV_K = 1.602176634e-9 / 1.380649e-16

    # Compute analytical solution using the checker module
    checker_path = root / "regression_tests" / "lib" / "check_marshak_wave.py"
    checker = SourceFileLoader("check_marshak_wave", str(checker_path)).load_module()
    try:
        x_ref, tgas_ref, trad_ref = checker.solve_marshak(prob_num)
    except Exception as exc:
        print(f"  [marshak_wave_{prob_num}] analytical solver failed: {exc}")
        x_ref, tgas_ref, trad_ref = None, None, None

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_sim, tgas_sim / keV_K, "bs", markersize=3, fillstyle="none", linewidth=0, label="$T_{\\mathrm{gas}}$ RICH")
    ax.plot(x_sim, trad_sim / keV_K, "rs", markersize=3, fillstyle="none", linewidth=0, label="$T_{\\mathrm{rad}}$ RICH")
    if x_ref is not None:
        ax.plot(x_ref, tgas_ref / keV_K, "b--", linewidth=1.5, label="$T_{\\mathrm{gas}}$ Analytic")
        ax.plot(x_ref, trad_ref / keV_K, "r--", linewidth=1.5, label="$T_{\\mathrm{rad}}$ Analytic")
    x_max_map = {1: 0.2, 2: 0.2, 3: 1.0, 4: 1.0}
    ax.set_xlim(0, x_max_map[prob_num])
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("T [keV]")
    ax.set_title(f"Marshak Wave Problem {prob_num}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, f"marshak_wave_{prob_num}")
    plt.close(fig)
    print(f"  [marshak_wave_{prob_num}] saved marshak_wave_{prob_num}.png/pdf")
    return True


def plot_marshak_wave_1(root: Path, out_dir: Path) -> bool:
    return _plot_marshak_wave(root, out_dir, 1)

def plot_marshak_wave_2(root: Path, out_dir: Path) -> bool:
    return _plot_marshak_wave(root, out_dir, 2)

def plot_marshak_wave_3(root: Path, out_dir: Path) -> bool:
    return _plot_marshak_wave(root, out_dir, 3)

def plot_marshak_wave_4(root: Path, out_dir: Path) -> bool:
    return _plot_marshak_wave(root, out_dir, 4)


# --------------------------------------------------------------------------- #
# Gresho vortex plotters
# --------------------------------------------------------------------------- #


def _azimuthal_velocity_analytic(r):
    r = np.asarray(r, dtype=float)
    vtheta = np.zeros_like(r)
    mask1 = r < 0.2
    mask2 = (r >= 0.2) & (r <= 0.4)
    vtheta[mask1] = 5.0 * r[mask1]
    vtheta[mask2] = 2.0 - 5.0 * r[mask2]
    return vtheta


def _plot_gresho(root: Path, out_dir: Path, test_id: str, label: str) -> bool:
    case_dir = root / "regression_tests" / "cases" / test_id
    profile = case_dir / "gresho_profile.txt"
    if not profile.exists():
        print(f"  [{test_id}] profile not found: {profile}")
        return False

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    x = raw[:, 0]
    y = raw[:, 1]
    vol = raw[:, 2]
    pressure = raw[:, 3]
    vx = raw[:, 4]
    vy = raw[:, 5]

    r = np.sqrt(x**2 + y**2)
    safe_r = np.where(r > 1e-10, r, 1e-10)
    vtheta = (-vx * y + vy * x) / safe_r

    plt = _get_plt()
    from matplotlib.tri import Triangulation

    # Figure 1: Pressure in xy plane
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    tri = Triangulation(x, y)
    tc1 = ax1.tripcolor(tri, pressure, shading="flat", cmap="viridis")
    fig1.colorbar(tc1, ax=ax1, label="Pressure")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title(f"Gresho {label} -- Pressure")
    ax1.set_aspect("equal")
    fig1.tight_layout()
    _save_fig(fig1, out_dir, f"{test_id}_pressure")
    plt.close(fig1)

    # Figure 2: Azimuthal velocity in xy plane
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    tc2 = ax2.tripcolor(tri, vtheta, shading="flat", cmap="RdBu_r")
    fig2.colorbar(tc2, ax=ax2, label="$v_\\theta$")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title(f"Gresho {label} -- Azimuthal Velocity")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    _save_fig(fig2, out_dir, f"{test_id}_vtheta_2d")
    plt.close(fig2)

    # Figure 3: Azimuthal velocity vs r (volume-averaged, binned)
    nbins = 30
    r_edges = np.linspace(0, 0.5, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    vtheta_binned = np.zeros(nbins)
    for i in range(nbins):
        mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
        if np.any(mask):
            vtheta_binned[i] = np.sum(vtheta[mask] * vol[mask]) / np.sum(vol[mask])

    vtheta_analytic = _azimuthal_velocity_analytic(r_centers)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(r_centers, vtheta_binned, "ko-", markersize=4, label=f"RICH t=5 ({label})")
    ax3.plot(r_centers, vtheta_analytic, "r-", linewidth=2, label="Initial condition")
    ax3.set_xlabel("r")
    ax3.set_ylabel("$v_\\theta$")
    ax3.set_title(f"Gresho {label} -- Azimuthal Velocity Profile")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    _save_fig(fig3, out_dir, f"{test_id}_vtheta_r")
    plt.close(fig3)

    print(f"  [{test_id}] saved {test_id}_pressure/vtheta_2d/vtheta_r .png/.pdf")
    return True


def plot_gresho_euler(root: Path, out_dir: Path) -> bool:
    return _plot_gresho(root, out_dir, "gresho_euler", "Euler")

def plot_gresho_lagrangian(root: Path, out_dir: Path) -> bool:
    return _plot_gresho(root, out_dir, "gresho_lagrangian", "Lagrangian")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

ALL_PLOTTERS = {
    "sod_1d": plot_sod,
    "sedov_3d_mpi": plot_sedov,
    "lane_self_gravity": plot_lane,
    "till_compton": plot_till,
    "mach2_diffusion": plot_mach2_diffusion,
    "mach2_multigroup": plot_mach2_multigroup,
    "marshak_wave_1": plot_marshak_wave_1,
    "marshak_wave_2": plot_marshak_wave_2,
    "marshak_wave_3": plot_marshak_wave_3,
    "marshak_wave_4": plot_marshak_wave_4,
    "gresho_euler": plot_gresho_euler,
    "gresho_lagrangian": plot_gresho_lagrangian,
}


def main():
    parser = argparse.ArgumentParser(description="Plot regression test results vs analytical solutions.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Path to a specific regression_results/<timestamp> directory. "
             "If omitted, the latest timestamp is used.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots in (default: regression_tests/plots/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot all tests regardless of what is in regression_results.",
    )
    args = parser.parse_args()

    root = repo_root()

    # Determine which tests to plot
    if args.results_dir:
        results_path = Path(args.results_dir)
    else:
        results_path = find_latest_results(root)

    if args.all:
        tests_to_plot = set(ALL_PLOTTERS.keys())
    elif results_path and results_path.is_dir():
        tests_to_plot = tests_in_results(results_path)
        print(f"Using results from: {results_path}")
        if not tests_to_plot:
            print("No test subdirectories found; falling back to plotting all available data.")
            tests_to_plot = set(ALL_PLOTTERS.keys())
    else:
        print("No regression_results directory found; plotting all tests with available data.")
        tests_to_plot = set(ALL_PLOTTERS.keys())

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = root / "regression_tests" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving plots to: {out_dir}\n")

    generated = []
    skipped = []
    for test_id, plotter in ALL_PLOTTERS.items():
        if test_id not in tests_to_plot:
            continue
        try:
            ok = plotter(root, out_dir)
            if ok:
                generated.append(test_id)
            else:
                skipped.append(test_id)
        except Exception as exc:
            print(f"  [{test_id}] ERROR: {exc}")
            skipped.append(test_id)

    print(f"\nSummary: {len(generated)} plots generated, {len(skipped)} skipped.")
    if generated:
        print(f"  Generated: {', '.join(generated)}")
    if skipped:
        print(f"  Skipped:   {', '.join(skipped)}")

    return 0 if generated else 1


if __name__ == "__main__":
    sys.exit(main())
