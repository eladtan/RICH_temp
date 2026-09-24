#!/usr/bin/env python3
"""
KRTI-S semi-analytic reference solver (CGS canonical specification).

This script solves the proposed pure-scattering incompressible KRTI-S benchmark
by reducing the x,t dependence analytically to exp(s t + i k x), solving the
remaining radiation transport with step characteristics + deterministic angular
quadrature, coupling the resulting radiation force to a 1-D incompressible
hydrodynamic boundary-value problem, and finding the unstable root D(s)=0.

The solver also exports background fields and eigenfunctions suitable for
initializing a numerical calculation.

Important scope:
  * gray coherent isotropic scattering only
  * no absorption/emission
  * no O(v/c) transport terms
  * incompressible two-fluid reference model
  * sharp interface at z=0

The RICH low-Mach ideal-gas realization described in the companion PDF is an
approximation to this incompressible reference, not mathematically identical.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.linalg import solve_banded

C_LIGHT = 2.99792458e10  # cm/s
PI = math.pi


@dataclass(frozen=True)
class Case:
    name: str
    rho_minus: float = 1.0       # g/cm^3
    rho_plus: float = 3.0        # g/cm^3
    g: float = 1.0e8             # cm/s^2
    H: float = 1.0               # cm
    k: float = 4.0               # cm^-1; kH=4
    alpha: float = 0.5            # radiation support fraction
    theta: float = 1.0            # kappa*rho_ref/k
    c: float = C_LIGHT            # cm/s
    p_interface: float = 1.0e12   # dyn/cm^2; arbitrary offset for incompressible reference
    k_eta0: float = 1.0e-3        # initial dimensionless interface amplitude
    gamma_proxy: float = 5.0 / 3.0  # only for low-Mach compressible realization notes
    Ib_fixed: float | None = None   # erg/(cm^2 s sr); if set, this is the frozen source BC

    @property
    def rho_ref(self) -> float:
        return 0.5 * (self.rho_minus + self.rho_plus)

    @property
    def atwood(self) -> float:
        return (self.rho_plus - self.rho_minus) / (self.rho_plus + self.rho_minus)

    @property
    def Lx(self) -> float:
        return 2.0 * PI / self.k

    @property
    def kappa(self) -> float:
        return self.theta * self.k / self.rho_ref  # cm^2/g

    @property
    def chi_minus(self) -> float:
        return self.kappa * self.rho_minus  # cm^-1

    @property
    def chi_plus(self) -> float:
        return self.kappa * self.rho_plus

    @property
    def g_eff(self) -> float:
        return self.g * (1.0 - self.alpha)

    @property
    def F0_target(self) -> float:
        return self.alpha * self.c * self.g / self.kappa  # erg/(cm^2 s)

    @property
    def eta0(self) -> float:
        return self.k_eta0 / self.k  # cm

    @property
    def s_classical_finite(self) -> float:
        return math.sqrt(self.atwood * self.g_eff * self.k * math.tanh(self.k * self.H))

    @property
    def t_efold_classical(self) -> float:
        return 1.0 / self.s_classical_finite

    def p0(self, z: np.ndarray | float) -> np.ndarray | float:
        zarr = np.asarray(z)
        rho = np.where(zarr < 0.0, self.rho_minus, self.rho_plus)
        out = self.p_interface - rho * self.g_eff * zarr
        return float(out) if np.ndim(z) == 0 else out


CASES: Dict[str, Case] = {
    "T": Case(name="KRTI-S-T", theta=0.1),
    "X": Case(name="KRTI-S-X", theta=1.0, Ib_fixed=1.6856350e18),
    "D": Case(name="KRTI-S-D", theta=30.0),
}


@dataclass
class Numerics:
    nz_per_layer: int = 64
    nmu: int = 16
    nphi: int = 16
    gmres_rtol: float = 2.0e-9
    gmres_atol: float = 1.0e-12
    gmres_restart: int = 80
    gmres_maxiter: int = 400


class SlabGrid:
    def __init__(self, case: Case, nz_per_layer: int):
        self.case = case
        self.n = int(nz_per_layer)
        self.dz = case.H / self.n
        self.N = 2 * self.n
        # Cell centers; interface z=0 is exactly a cell boundary.
        self.z = np.concatenate([
            -case.H + (np.arange(self.n) + 0.5) * self.dz,
            0.0 + (np.arange(self.n) + 0.5) * self.dz,
        ])
        self.chi = np.concatenate([
            np.full(self.n, case.chi_minus),
            np.full(self.n, case.chi_plus),
        ])
        self.rho = np.concatenate([
            np.full(self.n, case.rho_minus),
            np.full(self.n, case.rho_plus),
        ])


@dataclass
class BackgroundSolution:
    z: np.ndarray
    J: np.ndarray
    Fz: np.ndarray
    I_mu_z: np.ndarray
    mu: np.ndarray
    wmu: np.ndarray
    I_interface_mu: np.ndarray
    J_interface: float
    Ib: float
    F0: float
    F_unit: float
    gmres_info: int


@dataclass
class EigenSolution:
    s: float
    z: np.ndarray
    Jhat: np.ndarray
    Fxhat: np.ndarray
    Fzhat: np.ndarray
    f_x: np.ndarray
    f_z: np.ndarray
    w: np.ndarray
    u: np.ndarray
    phat: np.ndarray
    I_hat: np.ndarray
    mu: np.ndarray
    phi: np.ndarray
    xi: np.ndarray
    ang_wavg: np.ndarray
    D: complex


def _char_step(Iin: complex, J: complex, chi: float, a: complex, ds: float, abs_mu: float) -> Tuple[complex, complex]:
    """Return cell-AVERAGED intensity and outgoing boundary intensity.

    J is taken piecewise constant in the cell. Using the analytic cell average, rather
    than the midpoint intensity, makes conservative pure scattering conserve the
    discrete-ordinates flux cell by cell.
    """
    tau = a * ds / abs_mu
    ef = np.exp(-tau)
    # h = average of exp(-a*s/mu) across the cell. expm1 is stable for small tau.
    if abs(tau) < 1.0e-8:
        h = 1.0 - 0.5*tau + tau*tau/6.0
    else:
        h = -np.expm1(-tau) / tau
    source_factor = chi / a
    Iavg = Iin * h + source_factor * J * (1.0 - h)
    Iout = Iin * ef + source_factor * J * (1.0 - ef)
    return Iavg, Iout


def _sweep_1d_axisymmetric(grid: SlabGrid, J: np.ndarray, mu: float,
                           bottom_in: complex = 0.0, top_in: complex = 0.0,
                           a_extra: complex = 0.0,
                           interface_jump: complex = 0.0) -> Tuple[np.ndarray, complex]:
    """Step-characteristic sweep for a single mu; returns cell-center I and interface I.

    interface_jump is defined as I_plus - I_minus at z=0 for this direction.
    """
    N, n, dz = grid.N, grid.n, grid.dz
    Icen = np.zeros(N, dtype=complex)
    if mu > 0:
        Ibd = complex(bottom_in)
        Iint = None
        for i in range(N):
            chi = grid.chi[i]
            a = chi + a_extra
            Icen[i], Iout = _char_step(Ibd, J[i], chi, a, dz, mu)
            Ibd = Iout
            if i == n - 1:
                Iint = Ibd
                Ibd = Ibd + interface_jump
        assert Iint is not None
        return Icen, complex(Iint)
    else:
        amu = -mu
        Ibd = complex(top_in)
        Iint = None
        for i in range(N - 1, -1, -1):
            chi = grid.chi[i]
            a = chi + a_extra
            Icen[i], Iout = _char_step(Ibd, J[i], chi, a, dz, amu)
            Ibd = Iout
            if i == n:
                Iint = Ibd
                # I_plus - I_minus = jump => I_minus = I_plus - jump
                Ibd = Ibd - interface_jump
        assert Iint is not None
        return Icen, complex(Iint)


def _direction_response_matrix(grid: SlabGrid, mu: float, a_extra: complex,
                               bottom_in: complex, top_in: complex,
                               interface_jump: complex):
    """Return R,b where I_center = R @ J + b for one discrete direction."""
    N,n,dz=grid.N,grid.n,grid.dz
    R=np.zeros((N,N),dtype=complex)
    b=np.zeros(N,dtype=complex)
    resp=np.zeros(N,dtype=complex)
    scalar=complex(bottom_in if mu>0 else top_in)
    if mu>0:
        amu=mu
        it=range(N)
        iface_index=n-1
        sign_jump=+1.0
    else:
        amu=-mu
        it=range(N-1,-1,-1)
        iface_index=n
        sign_jump=-1.0
    for i in it:
        chi=grid.chi[i]
        a=chi+a_extra
        tau=a*dz/amu
        ef=np.exp(-tau)
        if abs(tau)<1.0e-8:
            havg=1.0-0.5*tau+tau*tau/6.0
        else:
            havg=-np.expm1(-tau)/tau
        qavg=(chi/a)*(1.0-havg)
        qf=(chi/a)*(1.0-ef)
        R[i,:]=resp*havg
        R[i,i]+=qavg
        b[i]=scalar*havg
        resp=resp*ef
        resp[i]+=qf
        scalar*=ef
        if i==iface_index:
            scalar += sign_jump*interface_jump
    return R,b


def solve_background(case: Case, num: Numerics) -> BackgroundSolution:
    grid = SlabGrid(case, num.nz_per_layer)
    mu, wmu = leggauss(num.nmu)
    if np.any(np.isclose(mu, 0.0)):
        raise ValueError("Use an even nmu so no ordinate has mu=0")
    K=np.zeros((grid.N,grid.N),dtype=float)
    b=np.zeros(grid.N,dtype=float)
    for mm,ww in zip(mu,wmu):
        R,bd=_direction_response_matrix(grid,float(mm),0.0,
                                       1.0 if mm>0 else 0.0,0.0,0.0)
        K += 0.5*ww*np.real(R)
        b += 0.5*ww*np.real(bd)
    Junit=np.linalg.solve(np.eye(grid.N)-K,b)

    # reconstruct moments and interface intensity
    Jnew=np.zeros(grid.N); Fz_unit=np.zeros(grid.N)
    Iall_unit=np.zeros((num.nmu,grid.N)); Iint_unit=np.zeros(num.nmu)
    for m,(mm,ww) in enumerate(zip(mu,wmu)):
        Icen,Iiface=_sweep_1d_axisymmetric(grid,Junit,float(mm),
                                           1.0 if mm>0 else 0.0,0.0)
        Ireal=np.real(Icen); Iall_unit[m]=Ireal; Iint_unit[m]=np.real(Iiface)
        Jnew += 0.5*ww*Ireal
        Fz_unit += 2.0*PI*ww*mm*Ireal
    F_unit=float(np.mean(Fz_unit))
    if F_unit<=0: raise RuntimeError(f"Computed nonpositive unit net flux: {F_unit}")
    scale = case.Ib_fixed if case.Ib_fixed is not None else case.F0_target/F_unit
    J=Junit*scale; Fz=Fz_unit*scale; Iall=Iall_unit*scale; Iint=Iint_unit*scale
    Ib=scale; Jint=float(0.5*np.sum(wmu*Iint))
    return BackgroundSolution(grid.z.copy(),J,Fz,Iall,mu,wmu,Iint,Jint,Ib,float(np.mean(Fz)),F_unit,0)

def angular_quadrature(nmu: int, nphi: int):
    mu, wmu = leggauss(nmu)
    if np.any(np.isclose(mu, 0.0)):
        raise ValueError("Use even nmu")
    phi = 2.0 * PI * (np.arange(nphi) + 0.5) / nphi
    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    XI = np.sqrt(np.maximum(0.0, 1.0 - MU * MU)) * np.cos(PHI)
    # normalized angular-average weights: (1/4pi) dOmega
    WAVG = np.repeat((wmu / (2.0 * nphi))[:, None], nphi, axis=1)
    return MU.ravel(), PHI.ravel(), XI.ravel(), WAVG.ravel(), mu, wmu


def _interp_background_iface(background: BackgroundSolution, mu_dir: float) -> complex:
    # Background is azimuth-independent. Interpolate in mu through the discrete ordinates.
    # Linear interpolation is sufficient because only the interface jump source needs this value;
    # convergence is checked with nmu refinement.
    return complex(np.interp(mu_dir, background.mu, background.I_interface_mu))


def solve_radiation_perturbation(case: Case, num: Numerics, background: BackgroundSolution,
                                 s: complex, eta_hat: complex = 1.0):
    grid = SlabGrid(case, num.nz_per_layer)
    MU, PHI, XI, WAVG, _, _ = angular_quadrature(num.nmu, num.nphi)
    nang = MU.size
    delta_chi = case.chi_plus - case.chi_minus
    jumps=np.empty(nang,dtype=complex)
    for a in range(nang):
        mm=MU[a]; Iint0=_interp_background_iface(background,mm)
        jumps[a]=-eta_hat*delta_chi*(background.J_interface-Iint0)/mm

    K=np.zeros((grid.N,grid.N),dtype=complex)
    b=np.zeros(grid.N,dtype=complex)
    for a in range(nang):
        extra=s/case.c+1j*case.k*XI[a]
        R,bd=_direction_response_matrix(grid,float(MU[a]),extra,0.0,0.0,jumps[a])
        K += WAVG[a]*R
        b += WAVG[a]*bd
    Jhat=np.linalg.solve(np.eye(grid.N,dtype=complex)-K,b)

    Jnew=np.zeros(grid.N,dtype=complex); Fx=np.zeros(grid.N,dtype=complex); Fz=np.zeros(grid.N,dtype=complex)
    Ihat=np.zeros((nang,grid.N),dtype=complex)
    for a in range(nang):
        extra=s/case.c+1j*case.k*XI[a]
        Icen,_=_sweep_1d_axisymmetric(grid,Jhat,float(MU[a]),0.0,0.0,
                                      a_extra=extra,interface_jump=jumps[a])
        Ihat[a]=Icen; wa=WAVG[a]
        Jnew += wa*Icen
        Fx += 4.0*PI*wa*XI[a]*Icen
        Fz += 4.0*PI*wa*MU[a]*Icen
    return grid,Jhat,Fx,Fz,Ihat,MU,PHI,XI,WAVG

def _solve_layer_bvp(zrad: np.ndarray, fxrad: np.ndarray, fzrad: np.ndarray,
                     a: float, b: float, rho: float, s: complex, k: float,
                     w_a: complex, w_b: complex, n_nodes: int):
    """Solve w''-k^2 w=q on [a,b] with Dirichlet data using a 1-D FD BVP."""
    z = np.linspace(a, b, n_nodes)
    # Cubic complex interpolation by separate real/imag parts.
    cs_fx_r = CubicSpline(zrad, np.real(fxrad), extrapolate=True)
    cs_fx_i = CubicSpline(zrad, np.imag(fxrad), extrapolate=True)
    cs_fz_r = CubicSpline(zrad, np.real(fzrad), extrapolate=True)
    cs_fz_i = CubicSpline(zrad, np.imag(fzrad), extrapolate=True)
    fx = cs_fx_r(z) + 1j * cs_fx_i(z)
    fz = cs_fz_r(z) + 1j * cs_fz_i(z)
    dfx = cs_fx_r(z, 1) + 1j * cs_fx_i(z, 1)
    q = -(k * k * fz + 1j * k * dfx) / (rho * s)

    h = z[1] - z[0]
    nint = n_nodes - 2
    # banded matrix for second derivative - k^2.
    ab = np.zeros((3, nint), dtype=complex)
    ab[0, 1:] = 1.0 / h**2
    ab[1, :] = -2.0 / h**2 - k**2
    ab[2, :-1] = 1.0 / h**2
    rhs = q[1:-1].copy()
    rhs[0] -= w_a / h**2
    rhs[-1] -= w_b / h**2
    wint = solve_banded((1, 1), ab, rhs)
    w = np.empty(n_nodes, dtype=complex)
    w[0], w[-1] = w_a, w_b
    w[1:-1] = wint

    # Fourth-order one-sided derivative when possible; otherwise second order.
    if n_nodes >= 5:
        dw_left = (-25*w[0] + 48*w[1] - 36*w[2] + 16*w[3] - 3*w[4]) / (12*h)
        dw_right = (25*w[-1] - 48*w[-2] + 36*w[-3] - 16*w[-4] + 3*w[-5]) / (12*h)
    else:
        dw_left = (-3*w[0] + 4*w[1] - w[2]) / (2*h)
        dw_right = (3*w[-1] - 4*w[-2] + w[-3]) / (2*h)

    # Gradient at every node for u and pressure.
    dw = np.gradient(w, h, edge_order=2)
    u = 1j * dw / k
    p = -rho * s * dw / (k*k) - 1j * fx / k
    return z, w, u, p, fx, fz, dw_left, dw_right


def solve_hydro_and_dispersion(case: Case, num: Numerics, background: BackgroundSolution,
                               s: complex, eta_hat: complex = 1.0, export_intensity: bool = True) -> EigenSolution:
    grid, Jhat, Fx, Fz, Ihat, MU, PHI, XI, WAVG = solve_radiation_perturbation(
        case, num, background, s, eta_hat=eta_hat)
    f_x = grid.chi * Fx / case.c
    f_z = grid.chi * Fz / case.c

    n = grid.n
    # Lower and upper radiation centers.
    zl, zu = grid.z[:n], grid.z[n:]
    fxl, fxu = f_x[:n], f_x[n:]
    fzl, fzu = f_z[:n], f_z[n:]
    n_h = max(65, num.nz_per_layer + 1)
    zL, wL, uL, pL, fxL, fzL, dwL_a, dwL_b = _solve_layer_bvp(
        zl, fxl, fzl, -case.H, 0.0, case.rho_minus, s, case.k, 0.0, s*eta_hat, n_h)
    zU, wU, uU, pU, fxU, fzU, dwU_a, dwU_b = _solve_layer_bvp(
        zu, fxu, fzu, 0.0, case.H, case.rho_plus, s, case.k, s*eta_hat, 0.0, n_h)

    # Re-evaluate interface pressure with high-order one-sided w' and interpolated fx at z=0.
    cs_fxl_r = CubicSpline(zl, np.real(fxl), extrapolate=True)
    cs_fxl_i = CubicSpline(zl, np.imag(fxl), extrapolate=True)
    cs_fxu_r = CubicSpline(zu, np.real(fxu), extrapolate=True)
    cs_fxu_i = CubicSpline(zu, np.imag(fxu), extrapolate=True)
    fx0_minus = cs_fxl_r(0.0) + 1j*cs_fxl_i(0.0)
    fx0_plus = cs_fxu_r(0.0) + 1j*cs_fxu_i(0.0)
    p0_minus = -case.rho_minus * s * dwL_b / (case.k**2) - 1j * fx0_minus / case.k
    p0_plus = -case.rho_plus * s * dwU_a / (case.k**2) - 1j * fx0_plus / case.k
    D = p0_plus - p0_minus - (case.rho_plus - case.rho_minus) * case.g_eff * eta_hat

    # Merge hydro arrays without duplicating interface; pressure at interface retained from each side is
    # physically matched only at the root. For export, average the two interface values at the root.
    zhyd = np.concatenate([zL[:-1], zU])
    w = np.concatenate([wL[:-1], wU])
    u = np.concatenate([uL[:-1], uU])
    p = np.concatenate([pL[:-1], pU])
    if abs(D) < 1e-5 * max(1.0, abs((case.rho_plus-case.rho_minus)*case.g_eff*eta_hat)):
        # Replace interface p by average for cleaner initialization.
        p[-len(zU)] = 0.5 * (p0_minus + p0_plus)

    return EigenSolution(float(np.real(s)), grid.z.copy(), Jhat, Fx, Fz, f_x, f_z,
                         w, u, p, Ihat if export_intensity else np.empty((0,0), complex),
                         MU, PHI, XI, WAVG, D)


def dispersion_real(case: Case, num: Numerics, background: BackgroundSolution, s: float, verbose=False) -> float:
    eig = solve_hydro_and_dispersion(case, num, background, complex(s, 0.0), export_intensity=False)
    if verbose:
        print(f"  s={s:.8e}  D={eig.D.real:.8e} + {eig.D.imag:.3e}i")
    return float(np.real(eig.D))


def find_unstable_root(case: Case, num: Numerics, background: BackgroundSolution, verbose=False) -> float:
    s0 = case.s_classical_finite
    lo, hi = 0.5*s0, 2.0*s0
    flo = dispersion_real(case,num,background,lo,verbose=verbose)
    fhi = dispersion_real(case,num,background,hi,verbose=verbose)
    # Safeguarded expansion; normally the canonical branch is already bracketed.
    for _ in range(6):
        if flo*fhi < 0.0:
            break
        if abs(flo) < abs(fhi):
            hi *= 1.75; fhi=dispersion_real(case,num,background,hi,verbose=verbose)
        else:
            lo *= 0.5; flo=dispersion_real(case,num,background,lo,verbose=verbose)
    else:
        raise RuntimeError(f"Could not bracket unstable root: D({lo})={flo}, D({hi})={fhi}")
    return float(brentq(lambda x: dispersion_real(case,num,background,x),lo,hi,
                        rtol=2e-8,xtol=1e-7,maxiter=40))

def export_case_parameters(case: Case, background: BackgroundSolution, root: float, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    d = {
        "benchmark": case.name,
        "units": "CGS",
        "rho_minus_g_cm3": case.rho_minus,
        "rho_plus_g_cm3": case.rho_plus,
        "g_cm_s2": case.g,
        "c_cm_s": case.c,
        "H_cm": case.H,
        "Lx_cm": case.Lx,
        "k_cm_inv": case.k,
        "Atwood": case.atwood,
        "alpha": case.alpha,
        "theta": case.theta,
        "kappa_s_cm2_g": case.kappa,
        "chi_minus_cm_inv": case.chi_minus,
        "chi_plus_cm_inv": case.chi_plus,
        "F0_target_erg_cm2_s": case.F0_target,
        "F0_discrete_erg_cm2_s": background.F0,
        "Ib_erg_cm2_s_sr": background.Ib,
        "Ib_is_frozen_boundary_input": case.Ib_fixed is not None,
        "g_eff_cm_s2": case.g_eff,
        "p_interface_dyn_cm2": case.p_interface,
        "p_bottom_dyn_cm2": case.p0(-case.H),
        "p_top_dyn_cm2": case.p0(case.H),
        "eta0_cm": case.eta0,
        "k_eta0": case.k_eta0,
        "root_s_inv_s": root,
        "efold_time_s": 1.0/root,
        "classical_finite_s_inv_s": case.s_classical_finite,
        "radiation_bottom_BC": "I(z=-H,Omega,t)=Ib for Omega_z>0; no imposed outgoing intensity",
        "radiation_top_BC": "I(z=+H,Omega,t)=0 for Omega_z<0; no imposed outgoing intensity",
        "radiation_x_BC": "periodic with period Lx",
        "hydro_bottom_BC": "impermeable free-slip wall: v_z=0; tangential velocity free/slip",
        "hydro_top_BC": "impermeable free-slip wall: v_z=0; tangential velocity free/slip",
        "hydro_x_BC": "periodic with period Lx",
    }
    (outdir / f"{case.name}_parameters.json").write_text(json.dumps(d, indent=2) + "\n")


def export_background(case: Case, background: BackgroundSolution, outdir: Path):
    with (outdir / f"{case.name}_background.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["z_cm", "rho_g_cm3", "p0_dyn_cm2", "J0_erg_cm2_s_sr", "F0z_erg_cm2_s"])
        for z, J, F in zip(background.z, background.J, background.Fz):
            rho = case.rho_minus if z < 0 else case.rho_plus
            wr.writerow([f"{z:.17e}", f"{rho:.17e}", f"{case.p0(z):.17e}",
                         f"{J:.17e}", f"{F:.17e}"])


    with (outdir / f"{case.name}_background_intensity.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["mu_Omega_z", "z_cm", "I0_erg_cm2_s_sr"])
        for m,mu in enumerate(background.mu):
            for i,z in enumerate(background.z):
                wr.writerow([f"{mu:.17e}", f"{z:.17e}", f"{background.I_mu_z[m,i]:.17e}"])


def export_eigenfunctions(case: Case, background: BackgroundSolution, eig: EigenSolution, eta_amp: float, outdir: Path):
    # Radiation eigenfunctions live at radiation cell centers; normalized to eta_hat=1 cm.
    with (outdir / f"{case.name}_radiation_eigenfunction.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["z_cm", "Jhat_re_per_cm", "Jhat_im_per_cm",
                     "Fxhat_re_per_cm", "Fxhat_im_per_cm", "Fzhat_re_per_cm", "Fzhat_im_per_cm",
                     "fxhat_re_dyn_cm3_per_cm", "fxhat_im_dyn_cm3_per_cm",
                     "fzhat_re_dyn_cm3_per_cm", "fzhat_im_dyn_cm3_per_cm"])
        for i,z in enumerate(eig.z):
            wr.writerow([f"{z:.17e}", f"{eig.Jhat[i].real:.17e}", f"{eig.Jhat[i].imag:.17e}",
                         f"{eig.Fxhat[i].real:.17e}", f"{eig.Fxhat[i].imag:.17e}",
                         f"{eig.Fzhat[i].real:.17e}", f"{eig.Fzhat[i].imag:.17e}",
                         f"{eig.f_x[i].real:.17e}", f"{eig.f_x[i].imag:.17e}",
                         f"{eig.f_z[i].real:.17e}", f"{eig.f_z[i].imag:.17e}"])

    # Hydro arrays were built on a nodal grid spanning both layers.
    n_h = (len(eig.w) + 1)//2
    zL = np.linspace(-case.H, 0.0, n_h)
    zU = np.linspace(0.0, case.H, n_h)
    zh = np.concatenate([zL[:-1], zU])
    with (outdir / f"{case.name}_hydro_eigenfunction.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["z_cm", "what_re_cm_s_per_cm", "what_im_cm_s_per_cm",
                     "uhat_re_cm_s_per_cm", "uhat_im_cm_s_per_cm",
                     "phat_re_dyn_cm2_per_cm", "phat_im_dyn_cm2_per_cm"])
        for i,z in enumerate(zh):
            wr.writerow([f"{z:.17e}", f"{eig.w[i].real:.17e}", f"{eig.w[i].imag:.17e}",
                         f"{eig.u[i].real:.17e}", f"{eig.u[i].imag:.17e}",
                         f"{eig.phat[i].real:.17e}", f"{eig.phat[i].imag:.17e}"])

    # Direction-resolved intensity eigenfunction can be large but is needed for exact MC initialization.
    with (outdir / f"{case.name}_intensity_eigenfunction.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["angle_index", "mu_Omega_z", "phi_rad", "xi_Omega_x", "z_cm",
                     "Ihat_re_per_cm", "Ihat_im_per_cm"])
        for a in range(eig.I_hat.shape[0]):
            for i,z in enumerate(eig.z):
                val = eig.I_hat[a,i]
                wr.writerow([a, f"{eig.mu[a]:.17e}", f"{eig.phi[a]:.17e}", f"{eig.xi[a]:.17e}",
                             f"{z:.17e}", f"{val.real:.17e}", f"{val.imag:.17e}"])

    # A compact initialization recipe at t=0 for real fields:
    init = {
        "eta0_cm": eta_amp,
        "phase_convention": "physical perturbation = Re[ eta0 * qhat(z,Omega) * exp(i*k*x) ]",
        "interface": "z_interface(x,0)=eta0*cos(k*x)",
        "velocity": "v'(x,z,0)=Re[eta0*(uhat(z),what(z))*exp(i*k*x)]",
        "pressure": "p'(x,z,0)=Re[eta0*phat(z)*exp(i*k*x)]",
        "radiation": "I'(x,z,Omega,0)=Re[eta0*Ihat(z,Omega)*exp(i*k*x)]",
        "time_evolution_linear": "multiply every perturbation by exp(s*t)",
    }
    (outdir / f"{case.name}_initialization_recipe.json").write_text(json.dumps(init, indent=2)+"\n")


    # Compact machine-readable package for exact eigenmode initialization.
    np.savez_compressed(outdir / f"{case.name}_eigenfunctions.npz",
        s_inv_s=eig.s, k_cm_inv=case.k, eta0_cm=eta_amp,
        z_rad_cm=eig.z, J0=background.J, F0z=background.Fz,
        mu_background=background.mu, I0_mu_z=background.I_mu_z,
        Jhat_per_cm=eig.Jhat, Fxhat_per_cm=eig.Fxhat, Fzhat_per_cm=eig.Fzhat,
        Ihat_per_cm=eig.I_hat, mu=eig.mu, phi_rad=eig.phi, xi=eig.xi, angular_average_weight=eig.ang_wavg,
        z_hydro_cm=zh, what_cm_s_per_cm=eig.w, uhat_cm_s_per_cm=eig.u, phat_dyn_cm2_per_cm=eig.phat)


def write_space_time_slice(case: Case, eig: EigenSolution, eta0: float, outdir: Path,
                           nx: int = 129, times=None):
    if times is None:
        times = [0.0, 0.5/eig.s, 1.0/eig.s, 2.0/eig.s]
    n_h = (len(eig.w) + 1)//2
    zL = np.linspace(-case.H, 0.0, n_h)
    zU = np.linspace(0.0, case.H, n_h)
    zh = np.concatenate([zL[:-1], zU])
    x = np.linspace(0.0, case.Lx, nx, endpoint=False)
    # Keep file compact: material interface, and hydro fields on x-z grid.
    with (outdir / f"{case.name}_space_time_hydro.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["time_s","x_cm","z_cm","rho_background_g_cm3","p0_dyn_cm2",
                     "u_cm_s","w_cm_s","p_dyn_cm2"])
        for t in times:
            growth = eta0 * math.exp(eig.s*t)
            phase_x = np.exp(1j*case.k*x)
            for ix,xx in enumerate(x):
                phase = growth * phase_x[ix]
                for iz,zz in enumerate(zh):
                    rho = case.rho_minus if zz < 0 else case.rho_plus
                    u = (phase*eig.u[iz]).real
                    w = (phase*eig.w[iz]).real
                    p = case.p0(zz) + (phase*eig.phat[iz]).real
                    wr.writerow([f"{t:.17e}",f"{xx:.17e}",f"{zz:.17e}",f"{rho:.17e}",
                                 f"{case.p0(zz):.17e}",f"{u:.17e}",f"{w:.17e}",f"{p:.17e}"])
    with (outdir / f"{case.name}_interface_vs_time.csv").open("w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["time_s","x_cm","eta_cm"])
        for t in times:
            amp = eta0 * math.exp(eig.s*t)
            for xx in x:
                wr.writerow([f"{t:.17e}",f"{xx:.17e}",f"{amp*math.cos(case.k*xx):.17e}"])


def run_case(case_key: str, num: Numerics, outdir: Path, verbose=False):
    case = CASES[case_key]
    print(f"Solving {case.name}: theta={case.theta}, kappa={case.kappa:.8g} cm^2/g")
    bg = solve_background(case, num)
    flux_rel_spread = np.std(bg.Fz) / abs(np.mean(bg.Fz))
    print(f"  Background: Ib={bg.Ib:.9e}, F0={bg.F0:.9e}, target={case.F0_target:.9e}, rel flux spread={flux_rel_spread:.3e}")
    root = find_unstable_root(case, num, bg, verbose=verbose)
    print(f"  Root s={root:.9e} 1/s; classical-effective={case.s_classical_finite:.9e}; ratio={root/case.s_classical_finite:.9f}")
    eig = solve_hydro_and_dispersion(case, num, bg, complex(root,0), eta_hat=1.0, export_intensity=True)
    print(f"  D(root)={eig.D.real:.4e}+{eig.D.imag:.4e}i")
    cdir = outdir / case.name
    cdir.mkdir(parents=True, exist_ok=True)
    export_case_parameters(case, bg, root, cdir)
    export_background(case, bg, cdir)
    export_eigenfunctions(case, bg, eig, case.eta0, cdir)
    write_space_time_slice(case, eig, case.eta0, cdir)
    return case, bg, eig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=["T","X","D","all"], default="X")
    ap.add_argument("--outdir", default="krti_s_reference_output")
    ap.add_argument("--nz", type=int, default=64, help="cells per fluid layer")
    ap.add_argument("--nmu", type=int, default=16, help="Gauss-Legendre mu ordinates (even)")
    ap.add_argument("--nphi", type=int, default=16, help="azimuth ordinates")
    ap.add_argument("--verbose-root", action="store_true")
    args = ap.parse_args()
    num = Numerics(nz_per_layer=args.nz, nmu=args.nmu, nphi=args.nphi)
    out = Path(args.outdir)
    keys = ["T","X","D"] if args.case == "all" else [args.case]
    summary=[]
    for key in keys:
        case,bg,eig=run_case(key,num,out,args.verbose_root)
        summary.append({"case":case.name,"theta":case.theta,"kappa":case.kappa,
                        "Ib":bg.Ib,"F0":bg.F0,"s":eig.s,
                        "s_over_classical_effective":eig.s/case.s_classical_finite,
                        "D_re":eig.D.real,"D_im":eig.D.imag})
    out.mkdir(parents=True, exist_ok=True)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__":
    main()
