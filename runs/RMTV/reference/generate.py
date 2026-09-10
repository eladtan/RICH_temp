#!/usr/bin/env python3
"""Generate dimensionless RMTV shape tables from pinned LANL ExactPack.

Capture the two upstream ODE integrations with dense output, rather than
reintegrating for every radius. The unmodified upstream file is in vendor/.
"""
from pathlib import Path
import json
import argparse
import sys
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0, str(Path(__file__).resolve().parent / 'vendor'))
import timmes

ROOT = Path(__file__).resolve().parent
ALPHA = 9 / 13
KAPPA = -19 / 9
BETA = 7.197534e7
ZETA = ((0.5 * BETA)**(1 / 12) / ALPHA)**ALPHA


def generate(core=0.02):
    solutions = []
    def capture(*args, **kwargs):
        sol = solve_ivp(*args, **kwargs, dense_output=True)
        if not sol.success:
            raise RuntimeError(sol.message)
        solutions.append(sol)
        return sol
    original = timmes.solve_ivp
    timmes.solve_ivp = capture
    try:
        timmes.rmtv_1d(core, -2, 6.5, 1, 1.25, 1, 2, 2, 1, BETA, 1)
    finally:
        timmes.solve_ivp = original
    outer, inner = solutions
    xi_outer_end = np.exp(outer.t[0])
    xi_inner = np.geomspace(core, 1, 2401)
    xi_outer = 2 - np.geomspace(2-xi_outer_end, 1, 2401)[::-1]
    def shape(sol, xi):
        u,h,w,theta = sol.sol(np.log(xi))
        return np.column_stack((xi, xi**(KAPPA+timmes.sigma)*h,
                                xi*u, xi*xi*theta))
    inside = shape(inner, xi_inner)
    outside = shape(outer, xi_outer)
    # Regular central continuation; do not evaluate upstream at r=0.
    center = np.array([[0., inside[0,1], 0., inside[0,3]]])
    table = np.vstack((center, inside, outside))
    np.savetxt(ROOT / 'shape.dat', table, fmt='%.17e',
               header='xi rho_shape velocity_shape temperature_shape; duplicate xi=1 preserves shock')
    metadata = dict(upstream='lanl/ExactPack', revision='9bacc477eb791e959c9621e771fc7a725db755fa',
                    alpha=ALPHA, kappa=KAPPA, beta0=BETA, zeta=ZETA,
                    core_xi=core, inner_rows=len(inside)+1, outer_rows=len(outside),
                    upstream_rtol=4e-10, upstream_atol=4e-10)
    (ROOT / 'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # Isothermal shock: mass and momentum fluxes continuous in shock frame.
    l,r=inside[-1],outside[0]
    mass=np.array([l[1]*(l[2]-1),r[1]*(r[2]-1)])
    momentum=np.array([l[1]*((l[2]-1)**2+l[3]),r[1]*((r[2]-1)**2+r[3])])
    assert np.allclose(mass[0],mass[1],rtol=1e-10)
    assert np.allclose(momentum[0],momentum[1],rtol=1e-10)
    assert np.isclose(l[3],r[3],rtol=1e-12)
    assert np.all(np.isfinite(table)) and np.all(table[:,1]>0) and np.all(table[:,3]>0)
    print(json.dumps(metadata,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--core-xi",type=float,default=0.02)
    args=parser.parse_args()
    if not 0<args.core_xi<1:parser.error("core-xi must lie in (0,1)")
    generate(args.core_xi)
