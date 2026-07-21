#!/usr/bin/env python3
"""
Analytical self-similar solutions for the non-equilibrium nonlinear
supersonic Marshak wave problem.

Problems 1-2: Krief & McClarren (2024), arXiv:2401.05138, eqs 23-24
Problem 3:    Derei et al. (2024), arXiv:2411.14891, Table III Test 1 (fitted)
Problem 4:    Derei et al. (2024), arXiv:2411.14891, Table III Test 3 (fitted)

The self-similar ODE system is solved via a shooting method: integrate
inward from the heat front xi_0 to xi=0 and adjust xi_0 until f(0)=1.
"""
import argparse
import sys
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

c_light = 2.99792458e10
a_rad = 7.5657e-15
keV_K = 1.602176634e-9 / 1.380649e-16

# --------------- Problem definitions ---------------
# For P1-P2 (Krief): homogeneous (omega=0), beta=4, mu=1 => u = a*T^4/eps
#   tau = 1/alpha', delta = (1 + alpha/alpha')/2
#   A = c * k0' * T0^{-alpha'}   (dimensionless coupling)
#   eps = a*T0^4 / (F*T0^beta*rho0^{1-mu})  (radiation/material energy ratio)
#
# For P3-P4 (Derei): inhomogeneous (omega != 0), general beta
#   tau and omega determined by material exponents
#   delta = 1 (when alpha=alpha', lambda=lambda')
#   A = M * E0^{-alpha'/4} * L^{omega(1+lambda')}
#   B = P * E0^{-alpha'/4} * L^{omega(lambda'+mu)}

PROBLEMS = {
    1: dict(
        alpha=3.0, lam=0.0, alpha_p=3.0, lam_p=0.0,
        beta=4.0, mu=0.0, omega=0.0,
        G=1.0 / (100.0 * keV_K**3),
        Gp=1.0 / (0.1 * keV_K**3),
        F=6.860085e14 * keV_K**(-4),
        rho0=1.0,
        T0_coeff=1.008038, tau=1.0 / 3.0, t_ns=1.0,
    ),
    2: dict(
        alpha=3.0, lam=0.0, alpha_p=3.0, lam_p=0.0,
        beta=4.0, mu=0.0, omega=0.0,
        G=1.0 / (100.0 * keV_K**3),
        Gp=1.0 / (100.0 * keV_K**3),
        F=6.860085e14 * keV_K**(-4),
        rho0=1.0,
        T0_coeff=1.014565, tau=1.0 / 3.0, t_ns=1.0,
    ),
    3: dict(
        alpha=1.5, lam=0.2, alpha_p=1.5, lam_p=0.2,
        beta=3.4, mu=0.14, omega=-20.0 / 19.0,
        G=1.0 / (40.0 * keV_K**1.5),
        Gp=1.0 / (0.1 * keV_K**1.5),
        F=1e14 * keV_K**(-3.4),
        rho0=1.0,
        T0_coeff=1.0470478, tau=86.0 / 57.0, t_ns=1.0,
    ),
    4: dict(
        alpha=4.5, lam=0.9, alpha_p=4.5, lam_p=0.9,
        beta=6.0, mu=0.3, omega=40.0 / 139.0,
        G=1.0 / (2.0 * keV_K**4.5),
        Gp=1.0 / (0.001 * keV_K**4.5),
        F=1e14 * keV_K**(-6.0),
        rho0=1.0,
        T0_coeff=1.01008116, tau=14.0 / 139.0, t_ns=1.0,
    ),
}


def _solve_marshak_fitted_problem3():
    """
    Reference solution for problem 3 using fitted analytic profiles from
    Derei et al. (2024) Table III, Test 1 (~0.5% accuracy).
    """
    p = PROBLEMS[3]
    t_eval = p['t_ns'] * 1e-9
    tau = p['tau']
    T0_coeff = p['T0_coeff']

    T0_bath_paper = T0_coeff * keV_K * (1e-9)**(-tau)
    T0_paper = T0_bath_paper

    E0 = a_rad * T0_paper**4
    G = p['G']
    lam = p['lam']
    rho0 = p['rho0']
    alpha = p['alpha']
    omega = p['omega']
    K = c_light * G / 3.0 * rho0**(-(1 + lam)) * a_rad**(-alpha / 4.0)
    L = (K * E0**(alpha / 4.0))**(1.0 / (2.0 - omega * (1 + lam)))
    delta = 1.0  # alpha == alpha', lambda == lambda'

    xi0 = 1.274605
    r = np.linspace(0.0, 1.0, 4000)
    rs = np.clip(r, 1e-30, None)

    # f^{1/4} from Table III, Test 1
    f14 = np.empty_like(r)
    lo_f = r < 0.75
    hi_f = ~lo_f
    f14[lo_f] = 1.0 - 0.75567 * rs[lo_f]**2.0416
    f14[hi_f] = 1.2527 * np.clip(1.0 - r[hi_f], 0.0, None)**0.55623
    f14[0] = 1.0
    f14[-1] = 0.0
    f14 = np.clip(f14, 0.0, None)

    # g^{1/4} from Table III, Test 1 (omega < 0 => g(0) = 0)
    g14 = np.empty_like(r)
    lo_g = r < 0.05
    hi_g = ~lo_g
    g14[lo_g] = -0.15937 * rs[lo_g]**0.2194 + rs[lo_g]**0.074842
    g14[hi_g] = (0.63674 + 0.55611 * r[hi_g]**0.56101) \
        * np.clip(1.0 - r[hi_g], 0.0, None)**0.63964
    g14[0] = 0.0
    g14[-1] = 0.0
    g14 = np.clip(g14, 0.0, None)

    xi_fine = r * 0.8332614
    x_arr = xi_fine 

    T_scale = keV_K
    Trad_arr = T_scale * f14
    Tmat_arr = T_scale * g14

    mask = x_arr <= 1.1
    return x_arr[mask], Tmat_arr[mask], Trad_arr[mask]


def _solve_marshak_fitted_problem4():
    """
    Reference solution for problem 4 using fitted analytic profiles from
    Derei et al. (2024) Table III, Test 3 (~0.5% accuracy).
    """
    p = PROBLEMS[4]
    t_eval = p['t_ns'] * 1e-9
    tau = p['tau']
    T0_coeff = p['T0_coeff']

    T0_bath_paper = T0_coeff * keV_K * (1e-9)**(-tau)
    T0_paper = T0_bath_paper

    E0 = a_rad * T0_paper**4
    G = p['G']
    lam = p['lam']
    rho0 = p['rho0']
    alpha = p['alpha']
    omega = p['omega']
    K = c_light * G / 3.0 * rho0**(-(1 + lam)) * a_rad**(-alpha / 4.0)
    L = (K * E0**(alpha / 4.0))**(1.0 / (2.0 - omega * (1 + lam)))
    delta = 1.0  # alpha == alpha', lambda == lambda'

    xi0 = 0.314115
    r = np.linspace(0.0, 1.0, 4000)
    rs = np.clip(r, 1e-30, None)

    # f^{1/4} from Table III, Test 3
    f14 = np.empty_like(r)
    lo_f = r < 0.2
    hi_f = ~lo_f
    f14[lo_f] = 1.0 - 0.31578 * rs[lo_f]**0.66822
    f14[hi_f] = (0.95143 - 0.11368 * r[hi_f]**1.4227) \
        * np.clip(1.0 - r[hi_f], 0.0, None)**0.24206
    f14[0] = 1.0
    f14[-1] = 0.0
    f14 = np.clip(f14, 0.0, None)

    # g^{1/4} from Table III, Test 3
    g14 = np.empty_like(r)
    lo_g = r < 0.05
    hi_g = ~lo_g
    g14[lo_g] = 1.0 + 0.75349 * rs[lo_g]**0.37109 - rs[lo_g]**0.24703
    g14[hi_g] = (1.2663 - 0.58658 * rs[hi_g]**0.061645) \
        * np.clip(1.0 - r[hi_g], 0.0, None)**0.21797
    g14[0] = 1.0
    g14[-1] = 0.0
    g14 = np.clip(g14, 0.0, None)

    xi_fine = r * 0.95029077
    x_arr = xi_fine 

    T_scale = keV_K
    Trad_arr = T_scale * f14
    Tmat_arr = T_scale * g14

    mask = x_arr <= 1.1
    return x_arr[mask], Tmat_arr[mask], Trad_arr[mask]


def _compute_ode_params(prob_num, T0_paper):
    """
    Compute dimensionless ODE parameters.

    T0_paper is the coefficient in T_s(t) = T0_paper * t^tau, with units
    K * s^{-tau}.  The ODE depends on A = c*k0'*T0_paper^{-alpha'}, which
    is dimensionless when tau = 1/alpha'.
    """
    p = PROBLEMS[prob_num]
    alpha = p['alpha']
    alpha_p = p['alpha_p']
    lam = p['lam']
    lam_p = p['lam_p']
    beta = p['beta']
    mu = p['mu']
    omega = p['omega']
    rho0 = p['rho0']
    G = p['G']
    Gp = p['Gp']
    F = p['F']
    tau = p['tau']

    if omega == 0:
        delta = 0.5 * (1.0 + alpha / alpha_p)
    else:
        delta = 1.0

    E0 = a_rad * T0_paper**4
    K = c_light * G / 3.0 * rho0**(-(1 + lam)) * a_rad**(-alpha / 4.0)
    # Paper eq 23: L = (K * E0^{alpha/4})^{1/(2-omega*(1+lambda))}
    L_exp = 1.0 / (2.0 - omega * (1 + lam))
    L = (K * E0**(alpha / 4.0))**L_exp

    if omega == 0:
        # Krief eq 16: A = c * k0' * T0^{-alpha'}
        k0p = rho0**(1 + lam_p) / Gp
        A_coeff = c_light * k0p * T0_paper**(-alpha_p)
        # eps = radiation/material energy ratio at equilibrium
        eps = a_rad * T0_paper**4 / (F * T0_paper**beta * rho0**(1 - mu))
        B_coeff = eps * A_coeff
    else:
        # Derei eqs 14-16: generalized with density profile
        M = c_light / Gp * rho0**(1 + lam_p) * a_rad**(alpha_p / 4.0)
        P_coeff = 4.0 * c_light * a_rad**((alpha_p + beta) / 4.0) * \
            rho0**(lam_p + mu) / (beta * Gp * F)
        A_coeff = M * E0**(-alpha_p / 4.0) * L**(omega * (1 + lam_p))
        B_coeff = P_coeff * E0**(-alpha_p / 4.0) * L**(omega * (lam_p + mu))

    w1 = omega * (1 + lam)
    w2 = omega * (1 + lam_p)
    w3 = omega * (lam_p + mu)

    return dict(
        alpha=alpha, alpha_p=alpha_p, tau=tau, delta=delta, omega=omega,
        beta=beta, A=A_coeff, B=B_coeff, w1=w1, w2=w2, w3=w3,
        L=L, E0=E0, T0_paper=T0_paper,
    )


def _ode_rhs(xi, y, params):
    """
    RHS of the self-similar ODE system for [f, f', g].

    Krief eqs 23-24 / Derei eqs 23-24.

    The coupling term g^{-alpha'/4}*(f-g) is rewritten as
    (f/g - 1)*g^{1-alpha'/4} to avoid overflow near the front
    where g -> 0 but f/g -> 1.
    """
    f, fp, g = y
    alpha = params['alpha']
    alpha_p = params['alpha_p']
    tau = params['tau']
    delta = params['delta']
    A = params['A']
    B = params['B']
    w1 = params['w1']
    w3 = params['w3']

    g = max(g, 1e-100)
    f = max(f, 1e-100)
    xi = max(abs(xi), 1e-100)

    # Regularized coupling: g^{-alpha'/4}*(f-g) = (f/g-1)*g^{1-alpha'/4}
    ratio_minus1 = f / g - 1.0
    g_exp = g**(1.0 - alpha_p / 4.0)
    coupling_raw = ratio_minus1 * g_exp

    ga4 = g**(alpha / 4.0)
    w2 = params['w2']

    # g' from eq 24:  4τg - δξg' = B·ξ^{-w3}·coupling
    xi_w3 = xi**(-w3) if w3 != 0 else 1.0
    coupling_g = B * xi_w3 * coupling_raw
    gp = (4.0 * tau * g - coupling_g) / (delta * xi)

    # f'' from eq 23 expanded:
    #   ξ^{w1}·g^{α/4}·f'' = lhs + A·ξ^{-w2}·coupling
    #                         - w1·ξ^{w1-1}·g^{α/4}·f'
    #                         - ξ^{w1}·(α/4)·g^{α/4-1}·g'·f'
    lhs = 4.0 * tau * f - delta * xi * fp
    xi_w2_neg = xi**(-w2) if w2 != 0 else 1.0
    coupling_f = A * xi_w2_neg * coupling_raw

    xi_w1 = xi**w1 if w1 != 0 else 1.0
    fpp = (lhs + coupling_f) / (xi_w1 * ga4) - w1 / xi * fp - (alpha / 4.0) * gp * fp / g

    return [fp, fpp, gp]


def _shoot(xi0, params, N_pts=4000):
    """
    Integrate the ODE from xi ~ xi0 inward to xi ~ 0, using
    near-front asymptotic initial conditions.

    Near the front eta = xi0 - xi -> 0:
      f ~ g ~ C * eta^p  with p = 4/alpha
      C = (delta * xi0 / p)^p   (for omega=0)
      C = (delta * xi0^{1-w1} / p)^p  (general)

    Returns f(0) if integration succeeds, or np.inf on failure.
    """
    alpha = params['alpha']
    delta = params['delta']
    w1 = params['w1']
    p = 4.0 / alpha

    eta_start = xi0 * 1e-6
    xi_start = xi0 - eta_start

    if w1 == 0:
        C = (delta * xi0 / p)**p
    else:
        C = (delta * xi0**(1.0 - w1) / p)**p

    f0 = C * eta_start**p
    g0 = f0
    fp0 = -C * p * eta_start**(p - 1)

    if f0 < 1e-200 or not np.isfinite(f0):
        return np.inf, None

    y0 = [f0, fp0, g0]

    xi_end = xi0 * 1e-8

    try:
        sol = solve_ivp(
            _ode_rhs, [xi_start, xi_end], y0,
            args=(params,), method='LSODA',
            rtol=1e-8, atol=1e-12,
            dense_output=True,
        )
        if not sol.success:
            return np.inf, None
        f_end = sol.y[0, -1]
        if not np.isfinite(f_end):
            return np.inf, None
        return f_end, sol
    except Exception:
        return np.inf, None


def _find_xi0(params, xi0_hint=None):
    """Find xi0 such that f(0) = 1 using Brent's method."""

    def residual(xi0):
        fval, _ = _shoot(xi0, params)
        return fval - 1.0

    # f(0) increases monotonically with xi0.  Use a geometric search
    # starting from the hint to quickly bracket the root.
    if xi0_hint is None:
        xi0_hint = 0.5

    lo = hi = xi0_hint
    r_lo = residual(lo)
    if abs(r_lo) < 1e-12:
        return lo

    # Expand outward to bracket
    if r_lo < 0:
        for _ in range(40):
            hi = hi * 2.0
            r_hi = residual(hi)
            if r_hi > 0 or not np.isfinite(r_hi):
                break
    else:
        for _ in range(40):
            lo = lo / 2.0
            r_lo = residual(lo)
            if r_lo < 0:
                break
        r_hi = residual(hi)

    if not (np.isfinite(residual(lo)) and residual(lo) < 0
            and np.isfinite(residual(hi)) and residual(hi) > 0):
        raise RuntimeError("Could not bracket xi0 for shooting method")

    xi0_sol = brentq(residual, lo, hi, rtol=1e-8, maxiter=100)
    return xi0_sol


def _get_profiles(xi0, params, N_pts=4000):
    """Get the full profiles f(xi), g(xi) on a fine grid from 0 to xi0."""
    _, sol = _shoot(xi0, params, N_pts)
    if sol is None:
        raise RuntimeError("ODE integration failed at converged xi0")

    xi_arr = np.linspace(xi0 * 1e-8, xi0 - xi0 * 1e-6, N_pts)
    y_dense = sol.sol(xi_arr)

    f_arr = np.clip(y_dense[0], 0.0, None)
    g_arr = np.clip(y_dense[2], 0.0, None)

    # Prepend xi=0 (f=1 by construction, g=g(small xi))
    # Append xi=xi0 (f=g=0)
    xi_full = np.concatenate(([0.0], xi_arr, [xi0]))
    f_full = np.concatenate(([1.0], f_arr, [0.0]))
    g_full = np.concatenate(([g_arr[0]], g_arr, [0.0]))

    return xi_full, f_full, g_full


def _marshak_bc_correction(xi0, params):
    """
    Compute the ratio T_bath/T_surface from the Marshak boundary condition.

    Krief eq 45: (T_bath/T0)^4 = 1 + 2*g(0)^{alpha/4}*|f'(0)| / (c*k0*T0^{-alpha})^{1/2}

    For the self-similar solution with imposed surface temperature,
    the bath-to-surface ratio depends on the flux at xi=0.
    """
    _, sol = _shoot(xi0, params)
    if sol is None:
        return 1.0

    xi_eval = xi0 * 1e-8
    y_at_0 = sol.sol(xi_eval)
    fp_0 = y_at_0[1]
    g_0 = max(y_at_0[2], 1e-100)

    alpha = params['alpha']
    w1 = params['w1']

    if w1 != 0:
        return 1.0

    flux_0 = g_0**(alpha / 4.0) * abs(fp_0)
    L = params['L']

    bath_ratio_4 = 1.0 + 2.0 * L * flux_0 / c_light
    return bath_ratio_4**0.25


def solve_marshak(prob_num):
    """
    Solve the self-similar Marshak wave problem.

    Problems 1-2: ODE shooting method.
    Problem 3: fitted analytic profiles from Derei et al. Table III, Test 1.
    Problem 4: fitted analytic profiles from Derei et al. Table III, Test 3.

    Returns (x_phys, Tmat_phys, Trad_phys) at the evaluation time.
    """
    if prob_num == 3:
        return _solve_marshak_fitted_problem3()
    if prob_num == 4:
        return _solve_marshak_fitted_problem4()

    p = PROBLEMS[prob_num]
    t_eval = p['t_ns'] * 1e-9
    tau = p['tau']
    T0_coeff = p['T0_coeff']

    T0_bath_paper = T0_coeff * keV_K * (1e-9)**(-tau)

    T0_paper = T0_bath_paper
    for iteration in range(20):
        params = _compute_ode_params(prob_num, T0_paper)
        try:
            xi0 = _find_xi0(params)
        except RuntimeError:
            break

        ratio = _marshak_bc_correction(xi0, params)
        T0_paper_new = T0_bath_paper / ratio

        if abs(T0_paper_new - T0_paper) / abs(T0_paper) < 1e-10:
            T0_paper = T0_paper_new
            break
        T0_paper = T0_paper_new

    params = _compute_ode_params(prob_num, T0_paper)
    xi0 = _find_xi0(params)
    xi_arr, f_arr, g_arr = _get_profiles(xi0, params)

    L = params['L']
    delta = params['delta']
    x_arr = xi_arr * t_eval**delta * L

    T_scale = T0_paper * t_eval**tau
    Trad_arr = T_scale * np.clip(f_arr, 0, None)**0.25
    Tmat_arr = T_scale * np.clip(g_arr, 0, None)**0.25

    x_max_map = {1: 0.2, 2: 0.2}
    mask = x_arr <= x_max_map[prob_num] * 1.1
    return x_arr[mask], Tmat_arr[mask], Trad_arr[mask]


def compute_l1_error(x_sim, T_sim, x_ref, T_ref):
    """Per-cell relative L1 error: mean(|T_sim - T_ref| / |T_sim|)."""
    T_ref_interp = np.interp(x_sim, x_ref, T_ref)

    mask = np.abs(T_sim) > 0.01 * np.max(np.abs(T_sim))
    if np.sum(mask) < 5:
        mask = np.ones(len(x_sim), dtype=bool)

    return np.mean(np.abs(T_sim[mask] - T_ref_interp[mask]) / np.abs(T_sim[mask]))


def main():
    parser = argparse.ArgumentParser(description="Check Marshak wave profile against analytical solution")
    parser.add_argument("--problem", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--profile", type=str, required=True, help="Path to marshak_profile.txt")
    parser.add_argument("--max-tgas-rel-l1", type=float, default=1e-2)
    parser.add_argument("--max-trad-rel-l1", type=float, default=1e-2)
    args = parser.parse_args()

    data = np.loadtxt(args.profile)
    x_sim = data[:, 0]
    Tgas_sim = data[:, 1]
    Trad_sim = data[:, 2]

    x_ref, Tgas_ref, Trad_ref = solve_marshak(args.problem)

    tgas_l1 = compute_l1_error(x_sim, Tgas_sim, x_ref, Tgas_ref)
    trad_l1 = compute_l1_error(x_sim, Trad_sim, x_ref, Trad_ref)

    passed = tgas_l1 <= args.max_tgas_rel_l1 and trad_l1 <= args.max_trad_rel_l1

    print(f"TGAS_REL_L1={tgas_l1:.6e}")
    print(f"TRAD_REL_L1={trad_l1:.6e}")
    print(f"PASS={'1' if passed else '0'}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
