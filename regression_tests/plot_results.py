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
    """Sedov 3D: density, pressure, and velocity vs r compared to Sedov-Taylor ODE."""
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
    sim_time = 0.0075

    idx_shock = int(np.argmax(pressure))
    shock_radius = float(r[idx_shock])

    far_mask = r > 0.8 * float(np.max(r))
    if not np.any(far_mask):
        far_mask = r > np.median(r)
    rho_0 = float(np.median(density[far_mask]))
    p_upstream = float(np.median(pressure[far_mask]))
    v_upstream = float(np.median(velocity[far_mask]))

    v_s = (2.0 / 5.0) * shock_radius / sim_time

    nip = 3000
    ssv = np.linspace(1e-6 + 1.0 / gamma, 2.0 / (gamma + 1.0), num=nip)

    radius_table = np.array([shock_radius * sedov.vtoz(v, w, gamma, n) for v in ssv])
    density_table = np.array([
        rho_0 * sedov.vtod(v, w, gamma, n) for v in ssv
    ])
    pressure_table = np.array([
        rho_0 * v_s ** 2 * sedov.vtop(v, w, gamma, n) for v in ssv
    ])
    velocity_table = np.array([
        v_s * sedov.vtoz(v, w, gamma, n) * v for v in ssv
    ])

    r_fine = np.linspace(0, float(r.max()), 500)

    def _interp_field(table, upstream_val):
        return np.array([
            float(np.interp(ri, radius_table, table)) if ri <= shock_radius
            else upstream_val
            for ri in r_fine
        ])

    density_analytic = _interp_field(density_table, rho_0)
    pressure_analytic = _interp_field(pressure_table, p_upstream)
    velocity_analytic = _interp_field(velocity_table, v_upstream)

    plt = _get_plt()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    ax1.plot(r, density, "k.", markersize=2, label="Numeric (binned)")
    ax1.plot(r_fine, density_analytic, "r-", linewidth=1.5, label="Sedov-Taylor ODE")
    ax1.set_xlabel("r")
    ax1.set_ylabel("Density")
    ax1.set_title("Sedov 3D -- Density")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(r, pressure, "k.", markersize=2, label="Numeric (binned)")
    ax2.plot(r_fine, pressure_analytic, "r-", linewidth=1.5, label="Sedov-Taylor ODE")
    ax2.set_xlabel("r")
    ax2.set_ylabel("Pressure")
    ax2.set_title("Sedov 3D -- Pressure")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.plot(r, velocity, "k.", markersize=2, label="Numeric (binned)")
    ax3.plot(r_fine, velocity_analytic, "r-", linewidth=1.5, label="Sedov-Taylor ODE")
    ax3.set_xlabel("r")
    ax3.set_ylabel("Radial Velocity")
    ax3.set_title("Sedov 3D -- Velocity")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, out_dir, "sedov_3d_mpi")
    plt.close(fig)
    print(f"  [sedov_3d_mpi] saved sedov_3d_mpi.png/pdf")
    return True


def plot_lane(root: Path, out_dir: Path) -> bool:
    """Lane-Emden: density vs r compared to initial (analytic) Lane-Emden profile."""
    quad_profile = root / "regression_tests" / "cases" / "lane_self_gravity" / "lane_profile.txt"
    fmm_profile = root / "regression_tests" / "cases" / "lane_self_gravity_fmm" / "lane_profile.txt"

    if not quad_profile.exists() and not fmm_profile.exists():
        print(f"  [lane_self_gravity] no lane profile found under {root / 'regression_tests' / 'cases'}")
        return False

    def _load_profile(path: Path):
        raw = np.loadtxt(str(path))
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        return raw[:, 0], raw[:, 1], raw[:, 2]

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    plotted = False
    if quad_profile.exists():
        r, density, density_analytic = _load_profile(quad_profile)
        ax.plot(r, density, "k.", markersize=3, label="Quadrupole tree (binned)")
        plotted = True
    else:
        density_analytic = None

    if fmm_profile.exists():
        r_fmm, density_fmm, density_analytic_fmm = _load_profile(fmm_profile)
        ax.plot(r_fmm, density_fmm, "b.", markersize=3, label="FMM P=3, θ=0.9 (binned)")
        if density_analytic is None:
            density_analytic = density_analytic_fmm
            r = r_fmm
        plotted = True

    if density_analytic is not None:
        analytic_r = r if quad_profile.exists() else r_fmm
        ax.plot(analytic_r, density_analytic, "r-", linewidth=1.5, label="Initial (Lane-Emden)")

    if not plotted:
        return False

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


def plot_till_mc(root: Path, out_dir: Path) -> bool:
    """Till Compton MC: Tgas and Trad vs time (IMC transport)."""
    case_dir = root / "regression_tests" / "cases" / "till_compton_mc"
    time_file = case_dir / "time.txt"
    tgas_file = case_dir / "Tgas.txt"
    trad_file = case_dir / "Trad.txt"
    for f in (time_file, tgas_file, trad_file):
        if not f.exists():
            print(f"  [till_compton_mc] file not found: {f}")
            return False

    time = np.loadtxt(str(time_file))
    tgas = np.loadtxt(str(tgas_file))
    trad = np.loadtxt(str(trad_file))

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, tgas, "r-", linewidth=1.5, label="$T_{\\mathrm{gas}}$ (RICH IMC)")
    ax.plot(time, trad, "b-", linewidth=1.5, label="$T_{\\mathrm{rad}}$ (RICH IMC)")

    ref_file = root / "regression_tests" / "cases" / "till_compton" / "data" / "in_fbc_reference.txt"
    if ref_file.exists():
        ref = np.loadtxt(str(ref_file))
        ax.plot(ref[:, 0], ref[:, 1], "k^", markersize=5,
                label="$T_{\\mathrm{gas}}$ (IN-FBC)")
        ax.plot(ref[:, 0], ref[:, 2], "ks", markersize=4,
                label="$T_{\\mathrm{rad}}$ (IN-FBC)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 3e-8)
    ax.set_title("Till Compton MC -- Gas & Radiation Temperature (IMC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "till_compton_mc")
    plt.close(fig)
    print(f"  [till_compton_mc] saved till_compton_mc.png/pdf")
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
    spectrum_file = root / "regression_tests" / "cases" / "mach2_multigroup" / "mach2_spectrum.txt"
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
# Densmore 2012 MC plotter
# --------------------------------------------------------------------------- #


def plot_desmore2012_mc(root: Path, out_dir: Path) -> bool:
    """Densmore 2012 heterogeneous MC: MPI (no RW), serial (RW), DDMC vs reference."""
    cases = root / "regression_tests" / "cases"
    profile_mpi = cases / "desmore2012_mc" / "desmore2012_mc_profile.txt"
    profile_serial = cases / "desmore2012_mc_serial" / "desmore2012_mc_serial_profile.txt"
    profile_ddmc = cases / "desmore2012_mc_ddmc" / "desmore2012_mc_ddmc_profile.txt"
    ref_file = cases / "desmore2012_mc" / "data" / "densmore2012_fig4_mc.csv"

    if not profile_mpi.exists() and not profile_serial.exists() and not profile_ddmc.exists():
        print(f"  [desmore2012_mc] no profile found for any variant")
        return False

    keV_K = 1.602176634e-9 / 1.380649e-16

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))

    if ref_file.exists():
        ref = np.loadtxt(str(ref_file), delimiter=",", comments="#")
        ax.plot(ref[:, 0], ref[:, 1], "b-", linewidth=1.5,
                label="Densmore 2012 Fig.\u20094 (MC)")

    if profile_mpi.exists():
        raw = np.loadtxt(str(profile_mpi))
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        ax.plot(raw[:, 0], raw[:, 1] / keV_K, "ko", markersize=3,
                markerfacecolor="none", label="RICH MC (MPI, no RW)")

    if profile_serial.exists():
        raw = np.loadtxt(str(profile_serial))
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        ax.plot(raw[:, 0], raw[:, 1] / keV_K, "rx", markersize=3,
                label="RICH MC (serial, RW)")

    if profile_ddmc.exists():
        raw = np.loadtxt(str(profile_ddmc))
        if raw.ndim == 1:
            raw = np.expand_dims(raw, axis=0)
        ax.plot(raw[:, 0], raw[:, 1] / keV_K, "gs", markersize=3,
                markerfacecolor="none", label="RICH MC (MPI, DDMC)")

    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Material Temperature [keV]")
    ax.set_title("Densmore 2012 Heterogeneous Step-Opacity -- MC IMC")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "desmore2012_mc")
    plt.close(fig)
    print(f"  [desmore2012_mc] saved desmore2012_mc.png/pdf")
    return True


# --------------------------------------------------------------------------- #
# Yee isentropic vortex plotters
# --------------------------------------------------------------------------- #


def _yee_density_analytic(x, y):
    gamma = 1.4
    beta = 5.0
    r2 = x**2 + y**2
    T = 1.0 - (gamma - 1.0) * beta**2 / (8.0 * gamma * np.pi**2) * np.exp(1.0 - r2)
    return T ** (1.0 / (gamma - 1.0))


def _parse_density_l1(check_log: Path) -> float | None:
    """Extract DENSITY_L1 value from a vortex_check.stdout.log file."""
    if not check_log.exists():
        return None
    with open(str(check_log)) as f:
        for line in f:
            if line.startswith("DENSITY_L1="):
                try:
                    return float(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
    return None


def plot_yee_isentropic_vortex(root: Path, out_dir: Path) -> bool:
    """Yee isentropic vortex: density/pressure 2D, density vs r, L1 convergence."""
    cases = root / "regression_tests" / "cases"
    profile_64 = cases / "yee_vortex_64" / "vortex_profile.txt"
    profile_128 = cases / "yee_vortex_128" / "vortex_profile.txt"

    if not profile_128.exists():
        print(f"  [yee_vortex] 128x128 profile not found: {profile_128}")
        return False

    plt = _get_plt()
    from matplotlib.tri import Triangulation

    any_ok = False

    # Load 128 data for 2D plots
    raw128 = np.loadtxt(str(profile_128))
    if raw128.ndim == 1:
        raw128 = np.expand_dims(raw128, axis=0)
    x128 = raw128[:, 0]
    y128 = raw128[:, 1]
    vol128 = raw128[:, 2]
    rho128 = raw128[:, 3]
    p128 = raw128[:, 4]

    tri128 = Triangulation(x128, y128)

    # Figure 1: Density in xy-plane (128x128)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    tc1 = ax1.tripcolor(tri128, rho128, shading="flat", cmap="viridis")
    fig1.colorbar(tc1, ax=ax1, label="Density")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Yee Vortex 128$\\times$128 -- Density")
    ax1.set_aspect("equal")
    fig1.tight_layout()
    _save_fig(fig1, out_dir, "yee_vortex_density_2d")
    plt.close(fig1)
    any_ok = True

    # Figure 2: Pressure in xy-plane (128x128)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    tc2 = ax2.tripcolor(tri128, p128, shading="flat", cmap="viridis")
    fig2.colorbar(tc2, ax=ax2, label="Pressure")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("Yee Vortex 128$\\times$128 -- Pressure")
    ax2.set_aspect("equal")
    fig2.tight_layout()
    _save_fig(fig2, out_dir, "yee_vortex_pressure_2d")
    plt.close(fig2)

    # Figure 3: Density vs r (radially binned, both resolutions)
    r128 = np.sqrt(x128**2 + y128**2)
    rho_exact_128 = _yee_density_analytic(x128, y128)

    nbins = 40
    r_max = 5.0
    r_edges = np.linspace(0, r_max, nbins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    def _radial_bin(r, rho, vol):
        binned = np.zeros(nbins)
        for i in range(nbins):
            mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
            if np.any(mask):
                binned[i] = np.sum(rho[mask] * vol[mask]) / np.sum(vol[mask])
        return binned

    rho_binned_128 = _radial_bin(r128, rho128, vol128)
    rho_exact_binned = _radial_bin(r128, rho_exact_128, vol128)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(r_centers, rho_exact_binned, "r-", linewidth=2, label="Analytical IC")

    if profile_64.exists():
        raw64 = np.loadtxt(str(profile_64))
        if raw64.ndim == 1:
            raw64 = np.expand_dims(raw64, axis=0)
        x64 = raw64[:, 0]
        y64 = raw64[:, 1]
        vol64 = raw64[:, 2]
        rho64 = raw64[:, 3]
        r64 = np.sqrt(x64**2 + y64**2)
        rho_binned_64 = _radial_bin(r64, rho64, vol64)
        ax3.plot(r_centers, rho_binned_64, "bs-", markersize=3,
                 linewidth=1, label="RICH 64$\\times$64")

    ax3.plot(r_centers, rho_binned_128, "ko-", markersize=3,
             linewidth=1, label="RICH 128$\\times$128")
    ax3.set_xlabel("r")
    ax3.set_ylabel("Density")
    ax3.set_title("Yee Isentropic Vortex -- Density Profile")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    _save_fig(fig3, out_dir, "yee_vortex_density_r")
    plt.close(fig3)

    # Figure 4: L1 convergence log-log plot
    l1_64 = _parse_density_l1(cases / "yee_vortex_64" / "vortex_check.stdout.log")
    l1_128 = _parse_density_l1(cases / "yee_vortex_128" / "vortex_check.stdout.log")

    if l1_64 is not None and l1_128 is not None and l1_64 > 0 and l1_128 > 0:
        Ns = np.array([64, 128])
        L1s = np.array([l1_64, l1_128])

        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.loglog(Ns, L1s, "ko-", markersize=8, linewidth=2, label="RICH")

        N_ref = np.array([48, 192])
        L1_ref = L1s[0] * (Ns[0] / N_ref) ** 2
        ax4.loglog(N_ref, L1_ref, "r--", linewidth=1.5, label="2nd order")

        ax4.set_xlabel("N (cells per side)")
        ax4.set_ylabel("$L_1$ density error")
        ax4.set_title("Yee Isentropic Vortex -- Convergence")
        ax4.legend()
        ax4.grid(True, alpha=0.3, which="both")
        fig4.tight_layout()
        _save_fig(fig4, out_dir, "yee_vortex_convergence")
        plt.close(fig4)
        print(f"  [yee_vortex] saved convergence plot (L1_64={l1_64:.3e}, L1_128={l1_128:.3e})")
    else:
        print(f"  [yee_vortex] skipped convergence plot (L1 values not available)")

    print(f"  [yee_vortex] saved yee_vortex_density_2d/pressure_2d/density_r .png/.pdf")
    return any_ok


# --------------------------------------------------------------------------- #
# Rayleigh-Taylor plotter
# --------------------------------------------------------------------------- #


def plot_rayleigh_taylor(root: Path, out_dir: Path) -> bool:
    """Rayleigh-Taylor: Ek_z(t) with fitted growth rate, and density slice."""
    case_dir = root / "regression_tests" / "cases" / "rayleigh_taylor_mpi"
    ek_file = case_dir / "rt_kinetic_energy.txt"
    slice_file = case_dir / "rt_density_slice.txt"

    if not ek_file.exists():
        print(f"  [rayleigh_taylor_mpi] kinetic energy file not found: {ek_file}")
        return False

    checker_path = root / "regression_tests" / "lib" / "check_rayleigh_taylor.py"
    checker = SourceFileLoader("check_rayleigh_taylor", str(checker_path)).load_module()

    raw = np.loadtxt(str(ek_file))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    time = raw[:, 0]
    ekz = raw[:, 1]

    positive = ekz > 0
    time = time[positive]
    ekz = ekz[positive]

    sigma_analytical = checker.analytical_growth_rate()
    sigma_fit, log_C, mask = checker.fit_growth_rate(time, ekz)

    checker.make_plots(time, ekz, sigma_fit, log_C, mask,
                       sigma_analytical,
                       str(slice_file) if slice_file.exists() else None,
                       str(out_dir))

    print(f"  [rayleigh_taylor_mpi] saved rayleigh_taylor_mpi_ekz/slice .png/.pdf")
    return True


def plot_eulerian_diffusion_freefree_1d(root: Path, out_dir: Path) -> bool:
    """1D free-free diffusion test: plot gas temperature versus x."""
    profile = root / "regression_tests" / "cases" / "eulerian_diffusion_freefree_1d" / "temperature_profile.txt"
    if not profile.exists():
        print(f"  [eulerian_diffusion_freefree_1d] profile not found: {profile}")
        return False

    raw = np.loadtxt(str(profile))
    if raw.ndim == 1:
        raw = np.expand_dims(raw, axis=0)
    if raw.shape[1] < 4:
        print("  [eulerian_diffusion_freefree_1d] expected columns: x density Tgas Trad")
        return False

    x = raw[:, 0]
    tgas = raw[:, 2]
    trad = raw[:, 3]
    tgas_kev = tgas * 8.617333262145e-8
    trad_kev = trad * 8.617333262145e-8
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(tgas_kev)) or not np.all(np.isfinite(trad_kev)):
        print("  [eulerian_diffusion_freefree_1d] non-finite x/Tgas/Trad values")
        return False

    order = np.argsort(x)
    x = x[order]
    tgas_kev = tgas_kev[order]
    trad_kev = trad_kev[order]

    if np.any(tgas_kev <= 0) or np.any(trad_kev <= 0):
        print("  [eulerian_diffusion_freefree_1d] Tgas/Trad must be positive for log plots")
        return False

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, tgas_kev, color="tab:red", linewidth=1.2)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Gas temperature [keV]")
    ax.set_title("1D Eulerian diffusion (free-free): Tgas vs x")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "eulerian_diffusion_freefree_1d")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, trad_kev, color="tab:blue", linewidth=1.2)
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("Radiation temperature [keV]")
    ax.set_title("1D Eulerian diffusion (free-free): Trad vs x")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, "eulerian_diffusion_freefree_1d_trad")
    plt.close(fig)
    print("  [eulerian_diffusion_freefree_1d] saved eulerian_diffusion_freefree_1d(.png/.pdf) and _trad(.png/.pdf)")
    return True


# --------------------------------------------------------------------------- #
# Spherical collapse -- xy-plane scatter of density and internal energy
# --------------------------------------------------------------------------- #


def plot_spherical_collapse(root: Path, out_dir: Path) -> bool:
    """xy-plane interpolated plots of density and internal energy from collapse_xy_slice.txt."""
    from scipy.interpolate import griddata

    case_dir = root / "regression_tests" / "cases" / "spherical_collapse"
    slice_file = case_dir / "collapse_xy_slice.txt"
    if not slice_file.exists():
        print(f"  [spherical_collapse] slice file not found: {slice_file}")
        return False

    raw = np.loadtxt(str(slice_file))
    if raw.ndim != 2 or raw.shape[1] < 4:
        print("  [spherical_collapse] expected columns: x y density internal_energy")
        return False

    x, y, rho, ie = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    if len(x) == 0:
        print("  [spherical_collapse] empty slice file")
        return False

    box = 1.1
    mask = (np.abs(x) <= box) & (np.abs(y) <= box)
    x, y, rho, ie = x[mask], y[mask], rho[mask], ie[mask]

    ngrid = 512
    xi = np.linspace(-box, box, ngrid)
    yi = np.linspace(-box, box, ngrid)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    rho_grid = griddata((x, y), rho, (xi_grid, yi_grid), method="linear")
    ie_grid = griddata((x, y), ie, (xi_grid, yi_grid), method="linear")

    plt = _get_plt()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(rho_grid, extent=[-box, box, -box, box], origin="lower", cmap="inferno")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$\rho$", fontsize=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title("Density — $z \\approx 0$ slice (final)", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, out_dir, "collapse_xy_density")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(ie_grid, extent=[-box, box, -box, box], origin="lower", cmap="magma")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$e_{\mathrm{int}}$", fontsize=12)
    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("y", fontsize=11)
    ax.set_title("Internal energy — $z \\approx 0$ slice (final)", fontsize=12)
    ax.set_aspect("equal")
    fig.tight_layout()
    _save_fig(fig, out_dir, "collapse_xy_internal_energy")
    plt.close(fig)

    print("  [spherical_collapse] saved collapse_xy_density and collapse_xy_internal_energy (.png/.pdf)")
    return True


def _plot_till_variant(root: Path, out_dir: Path, case_name: str, label: str) -> bool:
    """Generic Till Compton plot for variant cases (small-dt etc)."""
    case_dir = root / "regression_tests" / "cases" / case_name
    time_file = case_dir / "time.txt"
    tgas_file = case_dir / "Tgas.txt"
    trad_file = case_dir / "Trad.txt"
    for f in (time_file, tgas_file, trad_file):
        if not f.exists():
            print(f"  [{case_name}] file not found: {f}")
            return False

    time = np.loadtxt(str(time_file))
    tgas = np.loadtxt(str(tgas_file))
    trad = np.loadtxt(str(trad_file))

    plt = _get_plt()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time, tgas, "r-", linewidth=1.5, label="$T_{\\mathrm{gas}}$")
    ax.plot(time, trad, "b-", linewidth=1.5, label="$T_{\\mathrm{rad}}$")

    ref_file = root / "regression_tests" / "cases" / "till_compton" / "data" / "in_fbc_reference.txt"
    if ref_file.exists():
        ref = np.loadtxt(str(ref_file))
        ax.plot(ref[:, 0], ref[:, 1], "k^", markersize=5, label="$T_{\\mathrm{gas}}$ (IN-FBC)")
        ax.plot(ref[:, 0], ref[:, 2], "ks", markersize=4, label="$T_{\\mathrm{rad}}$ (IN-FBC)")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Temperature [K]")
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 3e-8)
    ax.set_title(f"Till Compton -- {label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, out_dir, case_name)
    plt.close(fig)
    print(f"  [{case_name}] saved {case_name}.png/pdf")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

ALL_PLOTTERS = {
    "sod_1d": plot_sod,
    "sedov_3d_mpi": plot_sedov,
    "lane_self_gravity": plot_lane,
    "lane_self_gravity_fmm": plot_lane,
    "till_compton": plot_till,
    "till_compton_mc": plot_till_mc,
    "mach2_diffusion": plot_mach2_diffusion,
    "mach2_multigroup": plot_mach2_multigroup,
    "marshak_wave_1": plot_marshak_wave_1,
    "marshak_wave_2": plot_marshak_wave_2,
    "marshak_wave_3": plot_marshak_wave_3,
    "marshak_wave_4": plot_marshak_wave_4,
    "gresho_euler": plot_gresho_euler,
    "gresho_lagrangian": plot_gresho_lagrangian,
    "desmore2012_mc": plot_desmore2012_mc,
    "desmore2012_mc_serial": plot_desmore2012_mc,
    "desmore2012_mc_ddmc": plot_desmore2012_mc,
    "yee_vortex_64": plot_yee_isentropic_vortex,
    "yee_vortex_128": plot_yee_isentropic_vortex,
    "rayleigh_taylor_mpi": plot_rayleigh_taylor,
    "eulerian_diffusion_freefree_1d": plot_eulerian_diffusion_freefree_1d,
    "spherical_collapse": plot_spherical_collapse,
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
    parser.add_argument(
        "--test",
        action="append",
        choices=sorted(ALL_PLOTTERS.keys()),
        help="Plot only this test id. May be passed more than once.",
    )
    args = parser.parse_args()

    root = repo_root()

    # Determine which tests to plot
    if args.results_dir:
        results_path = Path(args.results_dir)
    else:
        results_path = find_latest_results(root)

    if args.test:
        tests_to_plot = set(args.test)
    elif args.all:
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
