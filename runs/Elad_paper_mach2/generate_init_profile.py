#!/usr/bin/env python3
"""
Generate the Lowrie & Edwards (2008) NLTE analytical steady-state profile
for the Mach 2 radiative shock, in the format expected by test.cpp.

Output: mach2_analytic_ic.dat
Columns: x(cm)  rho(g/cc)  T_mat(K)  T_rad(K)  vx(cm/s)

The shock is placed at x = 0 (matching the simulation convention in test.cpp).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_mach2 import compute_mach2_analytic, T_UP, T_DN, V_DN, RHO_UP, RHO_DN

KEV_K = 1.160451812e7

XMIN, XMAX = -0.21, 0.25
MARGIN = 0.1
NPTS = 4000

outfile = sys.argv[1] if len(sys.argv) > 1 else "mach2_analytic_ic.dat"

# The analytical solver and simulation use the same stationary shock-frame
# convention: upstream is at small x and downstream at large x.
x_sim = np.linspace(XMIN - MARGIN, XMAX + MARGIN, NPTS)
prof = compute_mach2_analytic(x_sim, x_shock_plot=0.0)

T_mat_K = prof["T_gas"] * KEV_K
T_rad_K = prof["T_rad"] * KEV_K

with open(outfile, "w") as f:
    f.write(f"# Lowrie & Edwards (2008) NLTE analytic IC for Mach 2 radiative shock\n")
    f.write(f"# Columns: x(cm)  rho(g/cc)  T_mat(K)  T_rad(K)  vx(cm/s)\n")
    f.write(f"# Shock at x = 0; stationary shock-frame coordinates (upstream left, downstream right)\n")
    f.write(f"# {NPTS} points\n")
    for i in range(NPTS):
        f.write(f"{x_sim[i]:.10e}  {prof['rho'][i]:.10e}  {T_mat_K[i]:.10e}  "
                f"{T_rad_K[i]:.10e}  {prof['vx'][i]:.10e}\n")

print(f"Wrote {outfile}  ({NPTS} points, x=[{x_sim[0]:.4f}, {x_sim[-1]:.4f}] cm)")

# Sanity check: left boundary should be upstream, right boundary downstream
print(f"  Left  (x={x_sim[0]:.3f}, upstream):    rho={prof['rho'][0]:.4f}  "
      f"T_mat={T_mat_K[0]:.1f} K  vx={prof['vx'][0]:.2e}")
print(f"  Right (x={x_sim[-1]:.3f}, downstream): rho={prof['rho'][-1]:.4f}  "
      f"T_mat={T_mat_K[-1]:.1f} K  vx={prof['vx'][-1]:.2e}")
