#!/usr/bin/env python3
"""
Plot the Mach 45 radiative shock in the style of Figure 9(b) from
Steinberg & Heizler (2021), arXiv:2108.13453, Section 5.2
(original problem: Lowrie & Edwards 2008).

Four panels:
  (a) Material temperature  T_gas (keV)
  (b) Radiation temperature T_rad (keV)
  (c) Density rho (g/cc)
  (d) Velocity v_x (cm/s)

All plotted vs x (cm), zoomed on the shock region.

Usage:
    python plot_mach45.py                                  # auto-find shared profiles
    python plot_mach45.py <file1.txt> [file2.txt ...]      # explicit files
    python plot_mach45.py --dir /path/to/run               # scan directory
    python plot_mach45.py --wide                            # full domain view
    python plot_mach45.py --match <file.txt>                 # shift analytic to best-fit first file
"""

import sys
import os
import re
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Physical constants (CGS) ---
CLIGHT = 2.99792458e10
ARAD = 4.0 * 5.670374419e-5 / CLIGHT
KEV_K = 1.160451812e7

# --- Mach 45 parameters (arXiv:2108.13453, Section 5.2) ---
GAMMA = 5.0 / 3.0
CV = 1.45e15 / KEV_K                # erg/(g K), ~1.25e8

T_UP_KEV = 0.1
T_DN_KEV = 8.36
T_UP_K = T_UP_KEV * KEV_K
T_DN_K = T_DN_KEV * KEV_K

RHO_UP = 1.0
RHO_DN = 6.43
V_SHOCK = 5.71e8                    # cm/s
V_UP = V_SHOCK                      # upstream in shock frame (rightward)
V_DN = V_SHOCK - 4.82e8             # downstream in shock frame (rightward)

SHOCK_X0 = 2300.0                   # initial shock position
SHARED_MACH45_ROOT = Path("/data/shared/maorm/MC_results/Mach45")


def default_profile_directory():
    """Choose the default shared Mach45 output directory."""
    output_dir = os.environ.get("RICH_OUTPUT_DIR")
    if output_dir:
        return output_dir

    if SHARED_MACH45_ROOT.is_dir():
        dated_dirs = sorted(
            path
            for path in SHARED_MACH45_ROOT.iterdir()
            if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:_\d{2}-\d{2}-\d{2})?", path.name)
        )
        if dated_dirs:
            return str(dated_dirs[-1])

    # Keep local-directory behavior as a fallback on machines without the share.
    return "."


def load_profile(filepath):
    """Load a Mach45 profile txt file."""
    t_us, Np, cycle = None, None, None
    with open(filepath) as f:
        has_data = False
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                m = re.search(r"t_us=([\d.e+-]+)", stripped)
                if m:
                    t_us = float(m.group(1))
                m = re.search(r"cycle=(\d+)", stripped)
                if m:
                    cycle = int(m.group(1))
                m = re.search(r"Np=(\d+)", stripped)
                if m:
                    Np = int(m.group(1))
            else:
                has_data = True
                break

    if not has_data:
        raise ValueError(
            f"Profile file is empty or contains no data rows: {filepath}"
        )

    try:
        data = np.loadtxt(filepath, delimiter=",", comments="#")
    except ValueError as exc:
        raise ValueError(f"Could not parse profile file {filepath}: {exc}") from exc

    # np.loadtxt returns a 1-D array for a one-row profile.
    data = np.atleast_2d(data)
    if data.shape[1] < 5:
        raise ValueError(
            f"Profile file {filepath} has {data.shape[1]} columns; expected at least 5"
        )

    return dict(
        x=data[:, 0],
        rho=data[:, 1],
        T_gas=data[:, 2],
        T_rad=data[:, 3],
        vx=data[:, 4],
        t_us=t_us,
        cycle=cycle,
        Np=Np,
        path=filepath,
    )


def find_shock_position(prof):
    """Locate shock front as the steepest density gradient,
    excluding boundary cells."""
    rho = prof["rho"]
    x = prof["x"]
    margin = max(10, len(rho) // 50)
    drho = np.abs(np.diff(rho))
    drho[:margin] = 0
    drho[-margin:] = 0
    idx = np.argmax(drho)
    return 0.5 * (x[idx] + x[idx + 1])


def find_profile_files(directory="."):
    """Find all mach45 profile .txt files."""
    patterns = [
        os.path.join(directory, "mach45_mc_?????.txt"),
        os.path.join(directory, "*_?????.txt"),
    ]
    files = set()
    for pat in patterns:
        files.update(glob.glob(pat))
    for suffix in ("_init.txt", "_final.txt"):
        files.update(glob.glob(os.path.join(directory, f"*{suffix}")))
    return sorted(files)


def load_profiles(files, skip_invalid=False):
    """Load profiles, optionally skipping incomplete output files."""
    profiles = []
    for filepath in files:
        try:
            profiles.append(load_profile(filepath))
        except (OSError, ValueError) as exc:
            if not skip_invalid:
                raise
            print(f"Skipping {filepath}: {exc}", file=sys.stderr)
    return profiles


# =========================================================================
#  Semi-analytic Lowrie & Edwards (2008) NLTE solution for Mach 45
# =========================================================================

def compute_mach45_analytic(x_plot, x_shock_plot):
    """
    Compute the Lowrie & Edwards (2008) NLTE (2-temperature) semi-analytic
    diffusion solution for the Mach 45 radiative shock
    (arXiv:2108.13453 Section 5.2), shifted so the shock front aligns
    with *x_shock_plot*.

    Returns dict with 'T_gas', 'T_rad' (keV), 'rho' (g/cc), 'vx' (cm/s).
    """
    import scipy.optimize
    import scipy.integrate

    gamma = GAMMA
    M0 = V_SHOCK / np.sqrt(gamma * (gamma - 1.0) * CV * T_UP_K)

    T_left_K = T_UP_K
    sie_left = CV * T_left_K
    cs_left = np.sqrt(gamma * (gamma - 1.0) * sie_left)
    P0 = ARAD * T_left_K ** 4 / (RHO_UP * cs_left ** 2)
    Cp = 1.0 / (gamma - 1.0)
    Km = 3.0 * (gamma * M0 ** 2 + 1.0) + gamma * P0

    # --- dimensionless opacity helpers ---
    def _sigma_a_dim(T_K, rho_dim):
        return 0.0142 * rho_dim ** 2 * (T_K / KEV_K) ** (-3.5)

    def _sigma_s_dim(rho_dim):
        return 0.4006 * rho_dim

    def kappa_nd(T, rho):
        """Diffusion coefficient (uses total opacity sigma_a + sigma_s)."""
        rho_d = rho * RHO_UP
        T_K = T * T_left_K
        sig = _sigma_a_dim(T_K, rho_d) + _sigma_s_dim(rho_d)
        return CLIGHT / (3.0 * sig * cs_left)

    def sigma_nd(T, rho):
        """Absorption rate (Planck opacity only, for energy exchange)."""
        rho_d = rho * RHO_UP
        T_K = T * T_left_K
        sig_a = _sigma_a_dim(T_K, rho_d)
        return sig_a * CLIGHT / cs_left

    # --- jump conditions (LE08 Eqs. 12-13) ---
    def _f1(T):
        return (3.0 * (gamma + 1.0) * (T - 1.0)
                - P0 * gamma * (gamma - 1.0) * (7.0 + T ** 4))

    def _f2(T):
        return (12.0 * (gamma - 1.0) ** 2 * T
                * (3.0 + gamma * P0 * (1.0 + 7.0 * T ** 4)))

    def rho_of_T(T):
        return (_f1(T) + np.sqrt(np.maximum(_f1(T) ** 2 + _f2(T), 0.0))) / (
            6.0 * (gamma - 1.0) * T)

    def _jump_residual(T):
        rho = rho_of_T(T)
        return (3.0 * rho * (rho * T - 1.0) +
                gamma * P0 * rho * (T ** 4 - 1.0) -
                3.0 * gamma * (rho - 1.0) * M0 ** 2)

    # Robust root-finding for downstream temperature
    T1 = scipy.optimize.brentq(_jump_residual, 1.01, M0 ** 2,
                                xtol=1e-12, maxiter=200)
    rho1 = rho_of_T(T1)
    v1 = M0 / rho1

    # --- NLTE helper functions ---
    def theta_fTM(T, rho):
        tmp = (Km - 3.0 * gamma * M0 ** 2 / rho - 3.0 * T * rho) / (
            gamma * P0)
        return np.sqrt(np.sqrt(max(tmp, 0.0)))

    def dthetadx_f(v, rho, T, theta):
        num = v * (6.0 * Cp * rho * (T - 1.0)
                   + 3.0 * rho * (v ** 2 - M0 ** 2)
                   + 8.0 * P0 * (theta ** 4 - rho))
        den = 24.0 * P0 * kappa_nd(T, rho) * theta ** 3
        if abs(den) < 1e-300:
            return 0.0
        return num / den

    # --- LE08 Eqs. 37-38: ODE system dx/dM, dT/dM ---
    def ode_func(y, M):
        T = y[1]
        if T <= 0:
            return np.array([0.0, 0.0])
        v = M * np.sqrt(T)
        rho = M0 / v
        theta = theta_fTM(T, rho)
        g = dthetadx_f(v, rho, T, theta)
        r = 3.0 * rho * sigma_nd(T, rho) * (theta ** 4 - T ** 4)
        ZD = (4.0 * M0 * theta ** 3 * g
              + (gamma - 1.0) / (gamma + 1.0)
              * (gamma * M ** 2 + 1.0) * r)
        ZN = 4.0 * M0 * theta ** 3 * g + (gamma * M ** 2 - 1.0) * r
        if abs(ZD) < 1e-300:
            return np.array([0.0, 0.0])
        dxdM = (-6.0 * M0 * rho * T /
                ((gamma + 1.0) * P0 * M) * ((M ** 2 - 1.0) / ZD))
        dTdM = -2.0 * (gamma - 1.0) / (gamma + 1.0) * T * ZN / (M * ZD)
        return np.array([dxdM, dTdM])

    # --- Branch integration (LE08 Sec. 5) ---
    NPTS = 2048

    def solve_branch(rho0, T0, M_start, eps_val, epsasp_val, root_sign):
        theta0 = T0
        b = Km - gamma * P0 * theta0 ** 4
        d = np.sqrt(max(b ** 2 - 36.0 * gamma * M0 ** 2 * T0, 0.0))
        root = root_sign

        drhodT = -1.0 / T0 * (rho0 + root * 3.0 * gamma * M0 ** 2 / max(d, 1e-30))
        drhodth = (-2.0 / 3.0 * P0 * gamma * theta0 ** 3 / T0
                   * (1.0 + root * (Km - gamma * P0 * theta0 ** 4)
                      / max(d, 1e-30)))
        k0 = kappa_nd(T0, rho0)
        c1 = M0 / (24.0 * P0 * k0 * rho0 ** 2 * theta0 ** 3) if abs(k0) > 1e-30 else 0
        c2 = P0 / (3.0 * Cp * M0 * (M_start ** 2 - 1.0)) if abs(M_start ** 2 - 1.0) > 1e-30 else 0

        dGdT = c1 * (6.0 * Cp * rho0 * (2.0 * drhodT * (T0 - 1.0) + rho0)
                      - 6.0 * M0 ** 2 * rho0 * drhodT
                      + 8.0 * P0 * drhodT * (theta0 ** 4 - 2.0 * rho0))
        dGdth = c1 * (12.0 * Cp * drhodth * rho0 * (T0 - 1.0)
                       - 6.0 * M0 ** 2 * rho0 * drhodth
                       + 8.0 * P0 * (drhodth * (theta0 ** 4 - 2.0 * rho0)
                                     + 4.0 * rho0 * theta0 ** 3))

        sig0 = sigma_nd(T0, rho0)
        dFdT = c2 * (4.0 * M_start * np.sqrt(T0) * theta0 ** 3 * dGdT
                      - 12.0 * sig0
                      * (gamma * M_start ** 2 - 1.0) * T0 ** 3)
        dFdth = c2 * (4.0 * M_start * np.sqrt(T0) * theta0 ** 3 * dGdth
                       + 12.0 * sig0
                       * (gamma * M_start ** 2 - 1.0) * theta0 ** 3)

        disc = (dFdT - dGdth) ** 2 + 4.0 * dGdT * dFdth
        root2 = root
        sq = np.sqrt(max(disc, 0.0))
        denom_slope = 2.0 * dGdT if abs(dGdT) > 1e-30 else 1e-30
        dTdtheta = (dFdT - dGdth - root2 * sq) / denom_slope
        if (drhodT * dTdtheta + drhodth) <= 0:
            root2 = -root2
            dTdtheta = (dFdT - dGdth - root2 * sq) / denom_slope

        theps = theta0 + eps_val
        Teps = T0 + eps_val * dTdtheta
        if Teps <= 0:
            Teps = T0 * (1.0 + 1e-8 * np.sign(eps_val))
        beps = Km - gamma * P0 * theps ** 4
        dsc2 = max(beps ** 2 - 36.0 * gamma * M0 ** 2 * Teps, 0.0)
        rhoeps = (beps + root * np.sqrt(dsc2)) / (6.0 * Teps)
        if rhoeps <= 0:
            rhoeps = rho0
        Meps = M0 / (rhoeps * np.sqrt(Teps))

        g_init = dthetadx_f(M0 / rhoeps, rhoeps, Teps, theps)
        x0 = -eps_val / g_init if abs(g_init) > 1e-30 else 0.0

        target = 1.0 + epsasp_val
        if (Meps > 1.0 and target < Meps) or (Meps < 1.0 and target > Meps):
            Minteg = np.geomspace(Meps, target, NPTS)
        else:
            Minteg = np.linspace(Meps, target, NPTS)

        res = scipy.integrate.odeint(ode_func, np.array([0.0, Teps]), Minteg,
                                     full_output=False, mxstep=20000)

        xa = np.concatenate([[x0], res[:, 0]])
        Ma = np.concatenate([[M_start], Minteg])
        Ta = np.concatenate([[T0], res[:, 1]])
        rhoa = M0 / (Ma * np.sqrt(np.maximum(Ta, 1e-30)))
        tha = np.array([theta_fTM(Ti, ri) for Ti, ri in zip(Ta, rhoa)])
        return xa, Ta, tha, rhoa, Ma

    # --- solve both branches ---
    x_pre, T_pre, th_pre, rho_pre, M_pre = solve_branch(
        1.0, 1.0, M0, 1e-5, 1e-5, -1)
    x_rel, T_rel, th_rel, rho_rel, M_rel = solve_branch(
        rho1, T1, v1 / np.sqrt(T1), -1e-5, -1e-5, 1)

    # --- connect via embedded hydrodynamic shock ---
    p_pre = rho_pre * T_pre / gamma
    p_rel = rho_rel * T_rel / gamma
    rho_hugoniot = (rho_pre * ((gamma + 1.0) * p_rel +
                               (gamma - 1.0) * p_pre) /
                              ((gamma + 1.0) * p_pre +
                               (gamma - 1.0) * p_rel))
    RHO_REL, RHO_HUG = np.meshgrid(rho_rel, rho_hugoniot)
    TH_REL, TH_PRE = np.meshgrid(th_rel, th_pre)
    cost_grid = (np.abs(RHO_REL - RHO_HUG) / (np.abs(RHO_REL) + 1e-30) +
                 np.abs(TH_REL - TH_PRE) / (np.abs(TH_REL) + 1e-30))
    best_ip, best_ir = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)

    best_ip = min(int(best_ip) + 1, len(x_pre) - 2)
    best_ir = min(int(best_ir) + 1, len(x_rel) - 2)
    best_ip = max(best_ip, 2)
    best_ir = max(best_ir, 2)

    dxpre = -x_pre[best_ip]
    dxrel = -x_rel[best_ir]
    x_full = np.concatenate([(x_pre + dxpre)[:best_ip],
                             ((x_rel + dxrel)[:best_ir])[::-1]])
    T_full = np.concatenate([T_pre[:best_ip], T_rel[:best_ir][::-1]])
    th_full = np.concatenate([th_pre[:best_ip], th_rel[:best_ir][::-1]])
    rho_full = np.concatenate([rho_pre[:best_ip], rho_rel[:best_ir][::-1]])

    order = np.argsort(x_full)
    x_full = x_full[order]
    T_full = T_full[order]
    th_full = th_full[order]
    rho_full = rho_full[order]

    # --- convert to dimensional and interpolate ---
    x_dim = x_full + x_shock_plot
    T_mat_keV = T_full * T_UP_KEV
    T_rad_keV = th_full * T_UP_KEV
    rho_dim = rho_full * RHO_UP
    v_dim = M0 * cs_left / rho_full  # shock frame velocity

    x_arr = np.asarray(x_plot)
    return dict(
        T_gas=np.interp(x_arr, x_dim, T_mat_keV,
                        left=T_UP_KEV, right=T_DN_KEV),
        T_rad=np.interp(x_arr, x_dim, T_rad_keV,
                        left=T_UP_KEV, right=T_DN_KEV),
        rho=np.interp(x_arr, x_dim, rho_dim,
                      left=RHO_UP, right=RHO_DN),
        vx=np.interp(x_arr, x_dim, v_dim,
                     left=M0 * cs_left, right=M0 * cs_left / rho1),
    )


def find_best_shift(profile):
    """Find the shift K that minimizes the L2 density residual between
    analytic(x - K) and the first data profile.

    Returns K (cm) such that analytic evaluated at x_shock = SHOCK_X0 + K
    best matches the data.
    """
    import scipy.optimize

    x_data = profile["x"]
    rho_data = profile["rho"]

    x_wide = np.linspace(x_data[0] - 300, x_data[-1] + 300, 10000)
    analytic = compute_mach45_analytic(x_wide, SHOCK_X0)
    rho_analytic = analytic["rho"]

    def cost(K):
        rho_shifted = np.interp(x_data, x_wide + K, rho_analytic,
                                left=RHO_UP, right=RHO_DN)
        return np.sum((rho_shifted - rho_data) ** 2)

    result = scipy.optimize.minimize_scalar(cost, bounds=(-300, 300),
                                            method='bounded')
    return result.x


# =========================================================================
#  Plotting
# =========================================================================

def plot_mach45(profiles, outfile="mach45_figure9.png", wide=False,
                match_shift=None):
    """
    Plot T_gas, T_rad, density, velocity vs x.
    Style follows Figure 9(b) of arXiv:2108.13453.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_Tg = axes[0, 0]
    ax_Tr = axes[0, 1]
    ax_rho = axes[1, 0]
    ax_vx = axes[1, 1]

    cmap = plt.cm.viridis
    n = len(profiles)

    x_shock = SHOCK_X0
    if match_shift is not None:
        x_shock = SHOCK_X0 + match_shift

    for i, prof in enumerate(profiles):
        color = cmap(0.15 + 0.7 * i / max(n - 1, 1)) if n > 1 else "C0"
        parts = []
        if prof.get("cycle") is not None:
            parts.append(f"cycle {prof['cycle']}")
        if prof["t_us"] is not None:
            parts.append(f"t={prof['t_us']:.3f} μs")
        t_str = ", ".join(parts) if parts else Path(prof["path"]).stem
        lw = 1.6 if n <= 5 else 1.0

        x = prof["x"]
        ax_Tg.plot(x, prof["T_gas"], color=color, lw=lw, label=t_str)
        ax_Tr.plot(x, prof["T_rad"], color=color, lw=lw, label=t_str)
        ax_rho.plot(x, prof["rho"], color=color, lw=lw, label=t_str)
        ax_vx.plot(x, prof["vx"] / 1e8, color=color, lw=lw, label=t_str)

    # Reference lines for upstream/downstream
    ref_kw = dict(color="gray", ls="--", lw=0.7, alpha=0.5)
    for ax in (ax_Tg, ax_Tr):
        ax.axhline(T_UP_KEV, **ref_kw)
        ax.axhline(T_DN_KEV, **ref_kw)
    ax_rho.axhline(RHO_UP, **ref_kw)
    ax_rho.axhline(RHO_DN, **ref_kw)
    ax_vx.axhline(V_UP / 1e8, **ref_kw)
    ax_vx.axhline(V_DN / 1e8, **ref_kw)

    if wide:
        xlim = (1950.0, 2450.0)
    else:
        margin = 120
        xlim = (x_shock - margin, x_shock + 30)

    # --- Analytical solution (use same x_shock for consistent positioning) ---
    try:
        x_fine = np.linspace(xlim[0] - 20, xlim[1] + 20, 4000)
        analytic = compute_mach45_analytic(x_fine, x_shock)
        akw = dict(ls="-", lw=2, zorder=0)
        ax_Tg.plot(x_fine, analytic["T_gas"], color="magenta", **akw,
                   label="Analytic")
        ax_Tr.plot(x_fine, analytic["T_rad"], color="red", ls="--", lw=2,
                   label=r"Analytic $T_r$", zorder=0)
        ax_rho.plot(x_fine, analytic["rho"], color="magenta", **akw,
                    label="Analytic")
        ax_vx.plot(x_fine, np.array(analytic["vx"]) / 1e8,
                   color="magenta", **akw, label="Analytic")
    except Exception as e:
        print(f"Warning: could not compute analytical solution: {e}")
        import traceback
        traceback.print_exc()

    ax_Tg.set(ylabel=r"$T_{\mathrm{mat}}$ (keV)",
              title="(a) Material temperature")
    ax_Tr.set(ylabel=r"$T_{\mathrm{rad}}$ (keV)",
              title="(b) Radiation temperature")
    ax_rho.set(ylabel=r"$\rho$ (g/cc)", title="(c) Density")
    ax_vx.set(ylabel=r"$v_x$ ($10^8$ cm/s)", title="(d) Velocity")

    for ax in axes.flat:
        ax.set_xlabel("x (cm)")
        ax.set_xlim(xlim)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2)

    time_str = ""
    if profiles and profiles[-1]["t_us"] is not None:
        time_str = f",  $t = {profiles[-1]['t_us']:.4f}$ μs"

    shift_str = ""
    if match_shift is not None:
        shift_str = f",  analytic shift $K = {match_shift:+.2f}$ cm"

    fig.suptitle(
        r"Mach 45 Radiative Shock — cf. Steinberg & Heizler (2021) Fig. 9(b)"
        "\n"
        r"$\sigma_a = 0.0142\,\rho^2\,(T/\mathrm{keV})^{-3.5}$,  "
        r"$\sigma_s = 0.4006\,\rho$,  "
        r"$\gamma = 5/3$,  "
        r"$C_v = 1.45 \times 10^{15}$ erg/(g keV)"
        + time_str + shift_str,
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.show()


def export_analytic_profile(outfile, Np=4000, xmin=1950.0, xmax=2450.0,
                            shock_x=SHOCK_X0):
    """Export the analytic profile to a .dat file that test.cpp can read.

    Columns: x(cm), rho(g/cc), T_gas(K), Erad(erg/cc), v_x(cm/s)
    """
    x_cells = np.linspace(xmin, xmax, Np, endpoint=False) + 0.5 * (xmax - xmin) / Np
    analytic = compute_mach45_analytic(x_cells, shock_x)

    T_gas_K = np.array(analytic["T_gas"]) * KEV_K
    T_rad_K = np.array(analytic["T_rad"]) * KEV_K
    rho = np.array(analytic["rho"])
    vx = np.array(analytic["vx"])
    Erad = ARAD * T_rad_K**4

    with open(outfile, "w") as f:
        f.write(f"# Mach45 analytic profile (Lowrie & Edwards 2008)\n")
        f.write(f"# Np={Np}  xmin={xmin}  xmax={xmax}  shock_x={shock_x}\n")
        f.write(f"# x(cm), rho(g/cc), T_gas(K), Erad(erg/cc), v_x(cm/s)\n")
        for j in range(len(x_cells)):
            f.write(f"{x_cells[j]}, {rho[j]}, {T_gas_K[j]}, {Erad[j]}, {vx[j]}\n")
    print(f"Exported analytic profile to {outfile} ({len(x_cells)} points)")


def main():
    directory = default_profile_directory()
    explicit_files = []
    wide = False
    export_dat = None
    do_match = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dir" and i + 1 < len(args):
            directory = args[i + 1]
            i += 2
        elif args[i] == "--wide":
            wide = True
            i += 1
        elif args[i] == "--match":
            do_match = True
            i += 1
        elif args[i] == "--export-profile":
            export_dat = args[i + 1] if i + 1 < len(args) else "mach45_analytic.dat"
            i += 2 if i + 1 < len(args) else i + 1
        else:
            explicit_files.append(args[i])
            i += 1

    if export_dat:
        export_analytic_profile(export_dat)
        return

    if explicit_files:
        try:
            profiles = load_profiles(explicit_files)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"Error loading profile: {exc}") from exc
    else:
        files = find_profile_files(directory)
        if not files:
            print(f"No profile files found in {directory}")
            print("Usage: python plot_mach45.py [file1.txt ...] "
                  "[--dir path] [--wide] [--export-profile file.dat]")
            sys.exit(1)

        print(f"Scanning profile directory: {directory}")
        profiles = load_profiles(files, skip_invalid=True)
        if not profiles:
            print(f"No readable profile files found in {directory}", file=sys.stderr)
            sys.exit(1)
        print(f"Found {len(profiles)} profile file(s):")
        for p in profiles:
            if p["t_us"] is not None:
                t_str = f"t = {p['t_us']:.4f} μs"
            else:
                t_str = "t = ?"
            print(f"  {Path(p['path']).name:35s}  {t_str}")

    match_shift = None
    if do_match:
        match_shift = find_best_shift(profiles[0])
        print(f"Best-fit analytic shift: K = {match_shift:+.4f} cm")

    plot_mach45(profiles, wide=wide, match_shift=match_shift)


if __name__ == "__main__":
    main()
