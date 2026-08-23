#!/usr/bin/env python3
"""
Plot the Mach 2 radiative shock in the style of Figure 9(a) from
Steinberg & Heizler (2021), arXiv:2108.13453, Section 5.1
(original problem: Lowrie & Edwards 2008).

Four panels:
  (a) Material temperature  T_gas (keV)
  (b) Radiation temperature T_rad (keV)
  (c) Density rho (g/cc)
  (d) Velocity v_x (cm/s)

All plotted vs x (cm), zoomed on the shock region.

Usage:
    python plot_mach2.py                                  # auto-find profiles
    python plot_mach2.py <file1.txt> [file2.txt ...]      # explicit files
    python plot_mach2.py --dir /path/to/run               # scan directory
    python plot_mach2.py --wide                            # full domain view
"""

import sys
import os
import re
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

T_UP = 0.122   # keV
T_DN = 0.253   # keV
RHO_UP = 1.0   # g/cc
RHO_DN = 2.29  # g/cc
V_UP = 3.4616e7       # cm/s, stationary shock frame
V_DN = 1.5116e7       # cm/s, stationary shock frame
PAPER_SHOCK = -0.1730126582  # x-coordinate of the digitized Fig. 9(a) jump
DEFAULT_PAPER_DATA = Path(__file__).with_name("paper_fig9a.csv")


def load_profile(filepath):
    """Load a Mach2 profile txt file."""
    t_ns, Np = None, None
    with open(filepath) as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(r"t_ns=([\d.e+-]+)", line)
                if m:
                    t_ns = float(m.group(1))
                m = re.search(r"Np=(\d+)", line)
                if m:
                    Np = int(m.group(1))
            else:
                break
    data = np.loadtxt(filepath, delimiter=",", comments="#")
    return dict(
        x=data[:, 0],
        rho=data[:, 1],
        T_gas=data[:, 2],
        T_rad=data[:, 3],
        vx=data[:, 4],
        t_ns=t_ns,
        Np=Np,
        path=filepath,
    )


def load_paper_data(filepath=DEFAULT_PAPER_DATA):
    """Load the digitized magenta analytic curves from Fig. 9(a)."""
    data = np.loadtxt(filepath, delimiter=",", comments="#")
    return dict(
        x=data[:, 0],
        T_gas=data[:, 1],
        T_rad=data[:, 2],
        rho=data[:, 3],
        vx=data[:, 4],
        path=str(filepath),
    )


def find_shock_position(prof):
    """Locate the physical shock using the density midpoint.

    A late-time radiative shock is broad, and small boundary waves can have a
    steeper individual cell-to-cell gradient than the shock itself.
    """
    rho = prof["rho"]
    x = prof["x"]
    margin = max(10, len(rho) // 50)
    valid = np.arange(margin, len(rho) - margin)
    target = 0.5 * (RHO_UP + RHO_DN)
    idx = valid[np.argmin(np.abs(rho[valid] - target))]
    return x[idx]


def find_profile_files(directory="."):
    """Find the current run's initial/final profiles and numbered dumps."""
    patterns = [os.path.join(directory, "mach2_storm_?????.txt")]
    files = set()
    for pat in patterns:
        files.update(glob.glob(pat))

    for suffix in ("_init.txt", "_final.txt"):
        files.update(glob.glob(os.path.join(directory, f"*{suffix}")))

    return sorted(files)


def compute_mach2_analytic(x_plot, x_shock_plot):
    """
    Compute the Lowrie & Edwards (2008) NLTE (2-temperature) semi-analytic
    diffusion solution for the Mach 2 radiative shock
    (arXiv:2108.13453 Section 5.1), shifted so the shock front aligns
    with *x_shock_plot*.

    Based on:
      Lowrie & Edwards 2008, Shock Waves 18:129-143 (LE08)
    ODE integration via scipy.integrate.odeint (standalone functions to
    avoid stack-overflow in the Noebauer class hierarchy).

    Returns dict with 'T_gas' (keV), 'T_rad' (keV), 'rho' (g/cc),
    and lab-frame 'vx' (cm/s). The returned analytic coordinate is in
    the paper convention: upstream on the left, downstream on the right.
    """
    import scipy.optimize
    import scipy.integrate

    CLIGHT = 2.99792458e10
    ARAD = 4.0 * 5.670374419e-5 / CLIGHT
    KEV_K = 1.160451812e7

    gamma = 5.0 / 3.0
    cv = 1.91e8
    M0 = 2.0
    T_left_K = T_UP * KEV_K

    sie_left = cv * T_left_K
    cs_left = np.sqrt(gamma * (gamma - 1.0) * sie_left)
    P0 = ARAD * T_left_K ** 4 / (RHO_UP * cs_left ** 2)
    Cp = 1.0 / (gamma - 1.0)
    Km = 3.0 * (gamma * M0 ** 2 + 1.0) + gamma * P0

    # --- dimensionless opacity functions ---
    def kappa_nd(T, rho):
        sig = 0.362 * rho * RHO_UP * ((T * T_left_K) / KEV_K) ** (-3.5)
        return CLIGHT / (3.0 * sig * cs_left)

    def sigma_nd(T, rho):
        sig = 0.362 * rho * RHO_UP * ((T * T_left_K) / KEV_K) ** (-3.5)
        return sig * CLIGHT / cs_left

    # --- jump conditions (LR07 Eqs. 12-13) ---
    def _f1(T):
        return (3.0 * (gamma + 1.0) * (T - 1.0)
                - P0 * gamma * (gamma - 1.0) * (7.0 + T ** 4))

    def _f2(T):
        return (12.0 * (gamma - 1.0) ** 2 * T
                * (3.0 + gamma * P0 * (1.0 + 7.0 * T ** 4)))

    def rho_of_T(T):
        return (_f1(T) + np.sqrt(_f1(T) ** 2 + _f2(T))) / (
            6.0 * (gamma - 1.0) * T)

    T1 = scipy.optimize.newton(
        lambda T: (lambda rho: 3.0 * rho * (rho * T - 1.0) +
                   gamma * P0 * rho * (T ** 4 - 1.0) -
                   3.0 * gamma * (rho - 1.0) * M0 ** 2)(rho_of_T(T)),
        M0 ** 2, tol=1e-12)
    rho1 = rho_of_T(T1)
    v1 = M0 / rho1

    # --- NLTE helper functions (LE08) ---
    def theta_fTM(T, rho):
        tmp = (Km - 3.0 * gamma * M0 ** 2 / rho - 3.0 * T * rho) / (
            gamma * P0)
        return np.sqrt(np.sqrt(max(tmp, 0.0)))

    def dthetadx_f(v, rho, T, theta):
        return (v * (6.0 * Cp * rho * (T - 1.0)
                     + 3.0 * rho * (v ** 2 - M0 ** 2)
                     + 8.0 * P0 * (theta ** 4 - rho))
                / (24.0 * P0 * kappa_nd(T, rho) * theta ** 3))

    # --- LE08 Eqs. 37-38: ODE system dx/dM, dT/dM ---
    def ode_func(y, M):
        T = y[1]
        v = M * np.sqrt(T)
        rho = M0 / v
        theta = theta_fTM(T, rho)
        g = dthetadx_f(v, rho, T, theta)
        r = 3.0 * rho * sigma_nd(T, rho) * (theta ** 4 - T ** 4)
        ZD = (4.0 * M0 * theta ** 3 * g
              + (gamma - 1.0) / (gamma + 1.0) * (gamma * M ** 2 + 1.0) * r)
        ZN = 4.0 * M0 * theta ** 3 * g + (gamma * M ** 2 - 1.0) * r
        dxdM = -6.0 * M0 * rho * T / (
            (gamma + 1.0) * P0 * M) * ((M ** 2 - 1.0) / ZD)
        dTdM = -2.0 * (gamma - 1.0) / (gamma + 1.0) * T * ZN / (M * ZD)
        return np.array([dxdM, dTdM])

    # --- LE08 Sec. 5: epsilon-state & branch integration ---
    def solve_branch(rho0, T0, M_start, eps_val, epsasp_val, root_sign):
        theta0 = T0
        b = Km - gamma * P0 * theta0 ** 4
        d = np.sqrt(b ** 2 - 36.0 * gamma * M0 ** 2 * T0)
        root = root_sign
        v0 = M_start * np.sqrt(T0)

        drhodT = -1.0 / T0 * (rho0 + root * 3.0 * gamma * M0 ** 2 / d)
        drhodth = (-2.0 / 3.0 * P0 * gamma * theta0 ** 3 / T0
                   * (1.0 + root * (Km - gamma * P0 * theta0 ** 4) / d))
        c1 = M0 / (24.0 * P0 * kappa_nd(T0, rho0) * rho0 ** 2 * theta0 ** 3)
        c2 = P0 / (3.0 * Cp * M0 * (M_start ** 2 - 1.0))

        dGdT = c1 * (6.0 * Cp * rho0 * (2.0 * drhodT * (T0 - 1.0) + rho0)
                      - 6.0 * M0 ** 2 * rho0 * drhodT
                      + 8.0 * P0 * drhodT * (theta0 ** 4 - 2.0 * rho0))
        dGdth = c1 * (12.0 * Cp * drhodth * rho0 * (T0 - 1.0)
                       - 6.0 * M0 ** 2 * rho0 * drhodth
                       + 8.0 * P0 * (drhodth * (theta0 ** 4 - 2.0 * rho0)
                                     + 4.0 * rho0 * theta0 ** 3))
        dFdT = c2 * (4.0 * v0 * theta0 ** 3 * dGdT
                      - 12.0 * sigma_nd(T0, rho0)
                        * (gamma * M_start ** 2 - 1.0) * T0 ** 3)
        dFdth = c2 * (4.0 * v0 * theta0 ** 3 * dGdth
                       + 12.0 * sigma_nd(T0, rho0)
                         * (gamma * M_start ** 2 - 1.0) * theta0 ** 3)

        disc = (dFdT - dGdth) ** 2 + 4.0 * dGdT * dFdth
        root2 = root
        dTdtheta = (dFdT - dGdth - root2 * np.sqrt(max(disc, 0.0))) / (
            2.0 * dGdT)
        if (drhodT * dTdtheta + drhodth) <= 0:
            root2 = -root2
            dTdtheta = (dFdT - dGdth - root2 * np.sqrt(max(disc, 0.0))) / (
                2.0 * dGdT)

        theps = theta0 + eps_val
        Teps = T0 + eps_val * dTdtheta
        beps = Km - gamma * P0 * theps ** 4
        rhoeps = (beps + root * np.sqrt(
            max(beps ** 2 - 36.0 * gamma * M0 ** 2 * Teps, 0.0))) / (
            6.0 * Teps)
        Meps = M0 / (rhoeps * np.sqrt(Teps))
        x0 = -eps_val / dthetadx_f(M0 / rhoeps, rhoeps, Teps, theps)

        Minteg = np.geomspace(Meps, 1.0 + epsasp_val, 1024)
        res = scipy.integrate.odeint(ode_func, np.array([0.0, Teps]), Minteg)

        xa = np.concatenate([[x0], res[:, 0]])
        Ma = np.concatenate([[M_start], Minteg])
        Ta = np.concatenate([[T0], res[:, 1]])
        rhoa = M0 / (Ma * np.sqrt(Ta))
        tha = np.array([theta_fTM(Ti, ri) for Ti, ri in zip(Ta, rhoa)])
        return xa, Ta, tha, rhoa, Ma

    # --- solve both branches ---
    x_pre, T_pre, th_pre, rho_pre, M_pre = solve_branch(
        1.0, 1.0, M0, 1e-5, 1e-5, -1)
    x_rel, T_rel, th_rel, rho_rel, M_rel = solve_branch(
        rho1, T1, v1 / np.sqrt(T1), -1e-5, -1e-5, 1)

    # --- connect domains via embedded hydrodynamic shock (LE08 Sec. 5) ---
    # Match precursor and relaxation branches where:
    #   (1) theta (radiation temperature) is continuous
    #   (2) density satisfies the pressure-based Hugoniot jump condition
    # Follows the Noebauer reference implementation exactly.
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

    best_ip = int(best_ip)
    best_ir = int(best_ir)
    best_ir += 1
    best_ip += 1

    dxpre = -x_pre[best_ip]
    dxrel = -x_rel[best_ir]
    x_full = np.concatenate([(x_pre + dxpre)[:best_ip],
                             ((x_rel + dxrel)[:best_ir])[::-1]])
    T_full = np.concatenate([T_pre[:best_ip], T_rel[:best_ir][::-1]])
    th_full = np.concatenate([th_pre[:best_ip], th_rel[:best_ir][::-1]])
    rho_full = np.concatenate([rho_pre[:best_ip], rho_rel[:best_ir][::-1]])

    order = np.argsort(x_full)
    x_full, T_full = x_full[order], T_full[order]
    th_full, rho_full = th_full[order], rho_full[order]

    # --- convert to dimensional and interpolate ---
    # The matching placed the shock at x=0; shift to align with simulation.
    x_dim = x_full + x_shock_plot
    T_mat_keV = T_full * T_UP
    T_rad_keV = th_full * T_UP
    rho_dim = rho_full * RHO_UP

    # The benchmark is plotted in the stationary shock frame, as in Fig. 9(a).
    v_shock = M0 * cs_left / rho_full
    vx_dim = np.interp(np.asarray(x_plot), x_dim, v_shock,
                       left=V_UP, right=V_DN)

    x_arr = np.asarray(x_plot)
    return dict(
        T_gas=np.interp(x_arr, x_dim, T_mat_keV, left=T_UP, right=T_DN),
        T_rad=np.interp(x_arr, x_dim, T_rad_keV, left=T_UP, right=T_DN),
        rho=np.interp(x_arr, x_dim, rho_dim, left=RHO_UP, right=RHO_DN),
        vx=vx_dim,
    )


def plot_mach2(profiles, outfile="mach2_figure9.png", wide=False,
               paper_path=DEFAULT_PAPER_DATA):
    """
    Plot T_gas, T_rad, density, velocity vs x.
    Style follows Figure 9(a) of arXiv:2108.13453.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax_Tg = axes[0, 0]
    ax_Tr = axes[0, 1]
    ax_rho = axes[1, 0]
    ax_vx = axes[1, 1]

    cmap = plt.cm.viridis
    n = len(profiles)

    simulation_shock = find_shock_position(profiles[-1]) if profiles else 0.0
    x_shock = PAPER_SHOCK if profiles else 0.0

    for i, prof in enumerate(profiles):
        color = cmap(0.15 + 0.7 * i / max(n - 1, 1)) if n > 1 else "C0"
        if prof["t_ns"] is not None:
            t_str = f"t = {prof['t_ns']:.2f} ns"
        else:
            t_str = Path(prof["path"]).stem
        lw = 1.6 if n <= 5 else 1.0

        x = prof["x"] + (PAPER_SHOCK - simulation_shock)

        ax_Tg.plot(x, prof["T_gas"], color=color, lw=lw, label=t_str)
        ax_Tr.plot(x, prof["T_rad"], color=color, lw=lw, label=t_str)
        ax_rho.plot(x, prof["rho"], color=color, lw=lw, label=t_str)
        ax_vx.plot(x, prof["vx"] / 1e7, color=color, lw=lw, label=t_str)

    # The paper's panel uses the opposite x orientation from the simulation:
    # upstream is on the left and downstream on the right. Reflect the
    # digitized reference around the measured simulation shock before plotting.
    try:
        paper = load_paper_data(paper_path)
        paper_x = paper["x"]
        paper_kw = dict(color="black", ls=":", lw=1.4, alpha=0.9,
                        label="Paper Fig. 9(a), digitized")
        ax_Tg.plot(paper_x, paper["T_gas"], **paper_kw)
        ax_Tr.plot(paper_x, paper["T_rad"], **paper_kw)
        ax_rho.plot(paper_x, paper["rho"], **paper_kw)
        ax_vx.plot(paper_x, -paper["vx"] / 1e7, **paper_kw)
    except Exception as e:
        print(f"Warning: could not load paper reference: {e}")

    ref_kw = dict(color="gray", ls="--", lw=0.7, alpha=0.5)
    for ax in (ax_Tg, ax_Tr):
        ax.axhline(T_UP, **ref_kw)
        ax.axhline(T_DN, **ref_kw)
    ax_rho.axhline(RHO_UP, **ref_kw)
    ax_rho.axhline(RHO_DN, **ref_kw)
    ax_vx.axhline(V_UP / 1e7, **ref_kw)
    ax_vx.axhline(V_DN / 1e7, **ref_kw)

    if wide:
        xlim = (-0.22, 0.72)
    else:
        margin = 0.12
        xlim = (x_shock - margin, x_shock + margin)

    # --- Analytical solution on all panels ---
    shock_pos = x_shock
    try:
        x_fine = np.linspace(xlim[0] - 0.05, xlim[1] + 0.05, 2000)
        analytic = compute_mach2_analytic(x_fine, PAPER_SHOCK)
        akw = dict(ls='-', lw=2, zorder=0)
        ax_Tg.plot(x_fine, analytic['T_gas'], color='magenta', **akw,
                   label=r'Analytic $T_m$')
        ax_Tg.plot(x_fine, analytic['T_rad'], color='red', ls='--', lw=2,
                   label=r'Analytic $T_r$', zorder=0)
        ax_Tr.plot(x_fine, analytic['T_rad'], color='red', ls='--', lw=2,
                   label=r'Analytic $T_r$', zorder=0)
        ax_rho.plot(x_fine, analytic['rho'], color='magenta', **akw,
                    label='Analytic')
        ax_vx.plot(x_fine, np.array(analytic['vx']) / 1e7, color='magenta', **akw,
                   label='Analytic')
    except Exception as e:
        print(f"Warning: could not compute analytical solution: {e}")

    ax_Tg.set(ylabel=r"$T_{\mathrm{mat}}$ (keV)", title="(a) Material temperature")
    ax_Tr.set(ylabel=r"$T_{\mathrm{rad}}$ (keV)", title="(b) Radiation temperature")
    ax_rho.set(ylabel=r"$\rho$ (g/cc)", title="(c) Density")
    ax_vx.set(ylabel=r"$-v_x$ ($10^7$ cm/s)", title="(d) Velocity")

    for ax in axes.flat:
        ax.set_xlabel("x (cm)")
        ax.set_xlim(xlim)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(labelsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Mach 2 Radiative Shock — cf. Steinberg & Heizler (2021) Fig. 9(a)\n"
        r"$\sigma_a = 0.362\,\rho\,(T/\mathrm{keV})^{-3.5}$,  "
        r"$\gamma = 5/3$,  "
        r"$C_v = 1.91 \times 10^8$ erg/(g K)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved {outfile}")
    plt.close(fig)


def main():
    directory = "."
    explicit_files = []
    wide = False
    outfile = "mach2_figure9.png"
    paper_path = DEFAULT_PAPER_DATA

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--dir" and i + 1 < len(args):
            directory = args[i + 1]
            i += 2
        elif args[i] == "--wide":
            wide = True
            i += 1
        elif args[i] == "--output" and i + 1 < len(args):
            outfile = args[i + 1]
            i += 2
        elif args[i] == "--paper" and i + 1 < len(args):
            paper_path = args[i + 1]
            i += 2
        else:
            explicit_files.append(args[i])
            i += 1

    if explicit_files:
        profiles = [load_profile(f) for f in explicit_files]
    else:
        files = find_profile_files(directory)
        if not files:
            print(f"No profile files found in {directory}")
            print("Usage: python plot_mach2.py [file1.txt ...] [--dir path] [--wide] [--output file]")
            sys.exit(1)

        profiles = [load_profile(f) for f in files]
        print(f"Found {len(profiles)} profile file(s):")
        for p in profiles:
            t_str = f"t = {p['t_ns']:.3f} ns" if p["t_ns"] is not None else "t = ?"
            print(f"  {Path(p['path']).name:30s}  {t_str}")

    plot_mach2(profiles, outfile=outfile, wide=wide, paper_path=paper_path)


if __name__ == "__main__":
    main()
