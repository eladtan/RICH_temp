#!/usr/bin/env python3
"""
Generate a standalone LaTeX report documenting the RICH regression test suite.

The script:
  1. Optionally runs plot_results.py --all to produce comparison plots.
  2. Writes a .tex file with one section per test (description, initial/boundary
     conditions, mesh movement, execution mode, pass criteria, plots).
  3. Optionally compiles the .tex to PDF via pdflatex.

Usage:
    python3 regression_tests/generate_test_report.py
    python3 regression_tests/generate_test_report.py --output-dir /tmp/report
    python3 regression_tests/generate_test_report.py --no-compile
    python3 regression_tests/generate_test_report.py --no-plots
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Test categories (ordered)
# --------------------------------------------------------------------------- #

TEST_GROUPS: dict[str, tuple[str, list[str]]] = {
    "gresho": ("Gresho Vortex", ["gresho_euler", "gresho_lagrangian"]),
    "yee_vortex": ("Yee Isentropic Vortex", ["yee_vortex_64", "yee_vortex_128"]),
    "marshak_waves": ("Marshak Wave Problems", [
        "marshak_wave_1_diffusion", "marshak_wave_2_diffusion", "marshak_wave_3_diffusion", "marshak_wave_4_diffusion",
    ]),
    "mach2": ("Mach~2 Radiative Shock", ["mach2_diffusion", "mach2_multigroup"]),
    "eulerian_diffusion": ("1D Eulerian Diffusion Suite", [
        "eulerian_diffusion_freefree_suite",
        "eulerian_diffusion_freefree_multigroup_suite",
    ]),
    "till_compton_group": ("Till Compton Equilibration", ["till_compton"]),
    "densmore": ("Densmore 2012 Heterogeneous Step-Opacity (Monte Carlo)", [
        "desmore2012_mc", "desmore2012_mc_serial",
    ]),
    "lsq_gradient": ("LSQ Gradient Verification", [
        "spherical_gauss_linear", "cartesian_gauss_linear",
    ]),
}

TEST_CATEGORIES: list[tuple[str, list[str]]] = [
    ("Pure Hydrodynamics", [
        "sod_1d",
        "sedov_3d_mpi",
        "gresho",
        "yee_vortex",
        "spherical_collapse",
        "rayleigh_taylor_mpi",
    ]),
    ("Radiation Transport -- Diffusion", [
        "marshak_waves",
        "mach2",
        "eulerian_diffusion",
    ]),
    ("Radiation Transport -- Monte Carlo", [
        "till_compton_group",
        "densmore",
        "moving_slab",
    ]),
    ("Self-Gravity", [
        "lane_self_gravity",
        "lane_self_gravity_fmm",
    ]),
    ("Mesh and Infrastructure", [
        "amr_random",
        "voronoi_volume",
        "lsq_gradient",
    ]),
]

# --------------------------------------------------------------------------- #
# Test metadata
# --------------------------------------------------------------------------- #

TESTS = [
    {
        "id": "sod_1d",
        "title": "Sod Shock Tube (1D)",
        "description": (
            "The Sod shock tube \\cite{sod1978} is a classical one-dimensional "
            "Riemann problem that produces a left-moving rarefaction wave, a "
            "contact discontinuity, and a right-moving shock. It is one of the "
            "most widely used verification benchmarks for compressible "
            "hydrodynamics codes \\cite{toro2009}.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Riemann solver (HLLC):} The test directly "
            "exercises the approximate Riemann solver that computes inter-cell "
            "fluxes. Correct shock speed, contact velocity, and rarefaction "
            "fan structure all depend on the solver functioning properly.\n"
            "  \\item \\textbf{Spatial reconstruction (PLM):} The piecewise-linear "
            "reconstruction with slope limiting determines the order of accuracy "
            "and controls spurious oscillations near discontinuities. A faulty "
            "limiter would produce either excessive diffusion or non-physical "
            "overshoots at the shock and contact.\n"
            "  \\item \\textbf{Equation of state:} The ideal-gas EOS "
            "($\\gamma$-law) is tested through the pressure--density--energy "
            "coupling across all three wave families.\n"
            "  \\item \\textbf{Time integration and CFL condition:} The Courant "
            "time-step limiter and the conservative update must correctly advance "
            "the solution without introducing instability.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} The Sod problem is the most fundamental "
            "hydrodynamics test. Because it has an exact analytical solution, it "
            "provides an unambiguous correctness check. A code that fails this "
            "test cannot be trusted for any compressible-flow simulation. Its "
            "one-dimensional, Eulerian nature also makes it a fast, inexpensive "
            "smoke test that can be run on every commit."
        ),
        "initial_conditions": (
            r"The domain is $x \in [0,\,1]$ with 400 uniformly spaced cells. "
            r"The initial discontinuity is at $x = 0.5$:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item \textbf{Left state} ($x < 0.5$): $\rho = 1$, $P = 1$, $u = 0$." "\n"
            r"  \item \textbf{Right state} ($x \ge 0.5$): $\rho = 0.125$, $P = 0.1$, $u = 0$." "\n"
            r"\end{itemize}" "\n"
            r"The adiabatic index is $\gamma = 1.4$. "
            r"The simulation runs to $t = 0.2$."
        ),
        "boundary_conditions": "Rigid (reflective) walls at both ends of the domain.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"The numerical density and pressure profiles are compared to the exact "
            r"Riemann solution (computed via the \texttt{enrs.py} solver). "
            r"The goodness-of-fit (GOF) metric must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Density GOF $\le 0.02$." "\n"
            r"  \item Pressure GOF $\le 0.02$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["sod_1d.png"],
        "plot_caption": (
            "Sod shock tube at $t = 0.2$: numerical results (black dots) "
            "vs.\\ exact Riemann solution (red line) for density (left) and "
            "pressure (right)."
        ),
    },
    {
        "id": "sedov_3d_mpi",
        "title": "Sedov--Taylor Blast Wave (3D, MPI)",
        "description": (
            "The Sedov--Taylor blast wave \\cite{sedov1959,taylor1950} is a "
            "point-explosion problem with a self-similar analytical solution "
            "obtained by integrating an ODE \\cite{kamm2007}. It is a standard "
            "verification problem for multi-dimensional Lagrangian and ALE "
            "hydrodynamics.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{3D Voronoi tessellation:} The unstructured Voronoi "
            "mesh is generated from $\\sim\\!5\\times10^6$ random seed points and "
            "must correctly compute cell volumes, face areas, and geometric "
            "connectivity in three dimensions.\n"
            "  \\item \\textbf{Lagrangian mesh motion:} Cell generators move with "
            "the local fluid velocity, with periodic cell-rounding corrections to "
            "maintain mesh quality. This tests the full mesh-motion pipeline "
            "including velocity interpolation, generator advection, and "
            "incremental re-tessellation.\n"
            "  \\item \\textbf{Spherical symmetry on an unstructured mesh:} A "
            "strong spherical shock must propagate symmetrically despite the "
            "inherently irregular Voronoi geometry. This stresses the isotropy "
            "of the numerical scheme.\n"
            "  \\item \\textbf{MPI domain decomposition:} The problem is "
            "distributed across 128 MPI ranks. Correct ghost-cell exchange, "
            "parallel tessellation, and flux communication are all exercised.\n"
            "  \\item \\textbf{Strong-shock hydrodynamics:} The Sedov blast "
            "wave involves an infinite-strength shock with a compression ratio "
            "of $(\\gamma+1)/(\\gamma-1) = 4$ (for $\\gamma = 5/3$). Correctly "
            "capturing the peak density and the post-shock profile is demanding.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} This is the primary multi-dimensional, "
            "Lagrangian, parallel verification test. It simultaneously checks "
            "the geometric machinery (Voronoi), the physics (strong-shock "
            "hydrodynamics), and the parallel infrastructure (MPI). Failures "
            "here can indicate bugs ranging from tessellation errors to "
            "conservation violations across processor boundaries."
        ),
        "initial_conditions": (
            r"The domain is a cube $[-1,\,1]^3$ filled with a Voronoi tessellation "
            r"from $\sim\!5\times10^6$ random points. "
            r"Cells with centroid $r < 0.2$ receive high energy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item \textbf{Inner region} ($r < 0.1$): $\rho = 1$, $e_{\mathrm{int}} = 8 \times 10^5$." "\n"
            r"  \item \textbf{Outer region} ($r \ge 0.1$): $\rho = 1$, $e_{\mathrm{int}} = 0.1$." "\n"
            r"\end{itemize}" "\n"
            r"The adiabatic index is $\gamma = 5/3$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all six faces of the cube.",
        "mesh_movement": "Lagrangian (cells move with the fluid velocity), with cell rounding.",
        "execution": "MPI, 128~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Radially binned profiles of density, pressure, and velocity are "
            r"compared to the Sedov--Taylor ODE solution (scaled from theory "
            r"using $v_s = \tfrac{2}{5}\,R_s/t$, not normalized to numerical peaks). "
            r"The volume-weighted relative $L_1$ error (weights $\propto r^2$) must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Density: relative $L_1 \le 0.50$." "\n"
            r"  \item Pressure: relative $L_1 \le 0.30$." "\n"
            r"  \item Velocity: relative $L_1 \le 0.60$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["sedov_3d_mpi.png"],
        "plot_caption": (
            "Sedov--Taylor blast wave: radially binned numerical profiles (black dots) "
            "vs.\\ the ODE self-similar solution (red line) for density (left), "
            "pressure (center), and radial velocity (right)."
        ),
    },
    {
        "id": "till_compton",
        "title": "Till Compton Equilibration",
        "description": (
            "This test verifies the radiation--matter energy exchange via Compton "
            "scattering and absorption/emission in a uniform, static medium, "
            "following the problem setup of Till \\cite{till2021}. "
            "The numerical results are compared against the IN-FBC (Inline "
            "Full-Boltzmann Compton) reference data from McGraw et al.\\ "
            "\\cite{mcgraw2023}. "
            "Starting from mismatched gas and radiation temperatures, the system "
            "should relax to thermal equilibrium.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Compton scattering:} The dominant energy-exchange "
            "mechanism at keV temperatures. The test verifies that the Compton "
            "heating operator correctly transfers energy between the radiation "
            "field and the electrons, including the induced-scattering nonlinearity "
            "inherent to Bose--Einstein statistics.\n"
            "  \\item \\textbf{Absorption and emission:} The Kramers free--free "
            "(bremsstrahlung) opacity couples each energy group to the material "
            "temperature. Correct group-by-group absorption/emission rates are "
            "essential for the system to relax to the right equilibrium.\n"
            "  \\item \\textbf{Multigroup radiation transport:} The 32 "
            "logarithmically spaced energy groups must each evolve consistently, "
            "and the total radiation energy density must be conserved to within "
            "the implicit solver tolerance.\n"
            "  \\item \\textbf{Implicit time integration:} The radiation--matter "
            "coupling is stiff (the Compton and absorption time scales are much "
            "shorter than the equilibration time). The implicit solver must "
            "remain stable and converge even for large time steps.\n"
            "  \\item \\textbf{Energy conservation:} Total energy (radiation + "
            "material) must be conserved. Any leak indicates a bug in the "
            "coupling operator or the implicit solve.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Compton scattering is the primary radiation--matter "
            "coupling mechanism in high-energy-density physics (HEDP) and "
            "astrophysical applications. This test is the only one in the suite "
            "that exercises the Compton-scattering module in isolation (without "
            "hydrodynamic motion), making it essential for verifying the "
            "radiation physics independently of the hydro solver."
        ),
        "initial_conditions": (
            r"A single Voronoi cell in a unit domain with density "
            r"$\rho = 1\;\mathrm{g\,cm^{-3}}$ of hydrogen ($Z=1$). "
            r"The initial gas temperature is $T_{\mathrm{gas}} = 1\;\mathrm{keV}$ "
            r"and the initial radiation temperature is $T_{\mathrm{rad}} = 10\;\mathrm{keV}$. "
            r"Multigroup radiation transport is used with 32 logarithmically spaced "
            r"energy groups from $10^{-1}\;\mathrm{eV}$ to $10^{3}\;\mathrm{keV}$. "
            r"Hydrodynamics is disabled." "\n\n"
            r"The absorption opacity is the Kramers free--free (bremsstrahlung) formula:" "\n"
            r"\[" "\n"
            r"  \sigma_a(\nu,\,T) = 3.7\times10^{8}\;"
            r"Z^{3}\,n_e\,n_i\;T^{-1/2}\,"
            r"\bigl(1 - e^{-h\nu/k_BT}\bigr)\,\nu^{-3}"
            r"\quad[\mathrm{cm^{-1}}]," "\n"
            r"\]" "\n"
            r"where $n_e = n_i = \rho\,N_A \approx 6.022\times10^{23}\;\mathrm{cm^{-3}}$ "
            r"for fully ionized hydrogen at $\rho = 1\;\mathrm{g\,cm^{-3}}$. "
            r"With $Z=1$ the leading constant evaluates to "
            r"$3.7\times10^{8}\times(6.022\times10^{23})^{2} "
            r"\approx 1.342\times10^{56}\;\mathrm{cm^{-1}\,K^{1/2}\,Hz^{3}}$. "
            r"Scattering opacity is zero."
        ),
        "boundary_conditions": "Rigid walls (irrelevant since hydrodynamics is off).",
        "mesh_movement": "Lagrangian, but the cell is effectively static (single cell, no flow).",
        "execution": "Serial, 1~CPU, direct execution. Built with \\texttt{--energy\\_groups\\_num=32}.",
        "pass_criteria": (
            r"The simulation produces time histories of $T_{\mathrm{gas}}(t)$, "
            r"$T_{\mathrm{rad}}(t)$, and the total specific energy "
            r"$E_{\mathrm{tot}}(t) = e_{\mathrm{int}} + E_{\mathrm{rad}}/\rho$. "
            r"The test passes when:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item No NaN or Inf values appear in the output." "\n"
            r"  \item The final gas and radiation temperatures agree within 1\%:" "\n"
            r"        $|T_{\mathrm{gas}} - T_{\mathrm{rad}}| / "
            r"\max(T_{\mathrm{gas}},\,T_{\mathrm{rad}}) < 10^{-2}$." "\n"
            r"  \item Total energy (thermal + radiation) is conserved:" "\n"
            r"        $|E_{\mathrm{tot}}^{\mathrm{final}} - E_{\mathrm{tot}}^{\mathrm{initial}}|"
            r" / E_{\mathrm{tot}}^{\mathrm{initial}} < 10^{-8}$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["till_compton.png"],
        "plot_caption": (
            "Till Compton equilibration: gas temperature (red) and radiation "
            "temperature (blue) vs.\\ time, converging to thermal equilibrium. "
            "Black markers show the IN-FBC reference data digitized from "
            "McGraw, Till \\& Warsa, JCP 478 (2023) 111980, Figure~2(a)."
        ),
    },
    {
        "id": "amr_random",
        "title": "AMR Random Refine/Coarsen Consistency",
        "description": (
            "This test checks that adaptive mesh refinement (AMR) operations "
            "(random cell splitting and merging) preserve the hydrodynamic state "
            "of a uniform gas at rest. Any drift from the baseline indicates a "
            "conservation or interpolation bug in the AMR machinery.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Cell refinement (splitting):} When a Voronoi cell "
            "is refined, the parent cell's conserved quantities (mass, momentum, "
            "energy) must be exactly partitioned among the children. The test "
            "verifies that the volume-weighted interpolation and the geometric "
            "insertion of new seed points are correct.\n"
            "  \\item \\textbf{Cell coarsening (merging):} When cells are merged, "
            "the reverse operation must exactly recover the combined conserved "
            "quantities. Errors here would cause mass or energy to be created or "
            "destroyed.\n"
            "  \\item \\textbf{Voronoi re-tessellation:} After adding or removing "
            "seed points, the tessellation must be rebuilt consistently. "
            "Cell volumes, face areas, and neighbour connectivity must all be "
            "updated without introducing geometric artefacts.\n"
            "  \\item \\textbf{MPI consistency:} In the parallel variant, cells "
            "near processor boundaries may be refined or coarsened while their "
            "ghost copies reside on a different rank. The test checks that "
            "cross-rank AMR operations maintain consistency.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} AMR is a critical capability for production "
            "simulations that need to resolve local features (shocks, contact "
            "discontinuities, collapse regions) without paying the cost of a "
            "globally fine mesh. Because AMR touches the deepest layers of the "
            "data structures---cell connectivity, conserved-variable storage, "
            "MPI communication patterns---even small bugs can cause silent "
            "conservation errors that accumulate over thousands of time steps. "
            "A uniform-gas test is the most stringent check: the answer must be "
            "exactly unchanged to machine precision."
        ),
        "initial_conditions": (
            r"The domain is a cube $[-1,\,1]^3$ with a Voronoi tessellation from "
            r"random points. The gas is uniform and at rest: $\rho = 1$, "
            r"$e_{\mathrm{int}} = 2.5$, $\mathbf{v} = 0$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces.",
        "mesh_movement": (
            "Lagrangian with cell rounding. AMR randomly refines and coarsens cells "
            "during the simulation."
        ),
        "execution": (
            "Both serial and MPI. Serial: 1~CPU, direct execution. "
            "MPI: 64~CPUs via SLURM (partition \\texttt{bigrun}, exclusive)."
        ),
        "pass_criteria": (
            r"After AMR operations, the maximum relative drift in density, "
            r"internal energy, pressure, and velocity must stay small:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Serial: $\texttt{max\_drift} \le 10^{-8}$." "\n"
            r"  \item MPI: $\texttt{max\_drift} \le 10^{-6}$." "\n"
            r"\end{itemize}"
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "voronoi_volume",
        "title": "Voronoi Volume Conservation",
        "description": (
            "A pure-geometry test that verifies the Voronoi tessellation correctly "
            "partitions the domain. The sum of all cell volumes must match the "
            "known analytical box volume to machine precision.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Voronoi geometry kernel:} The convex-hull and "
            "face-clipping algorithms that compute each cell's vertices, faces, "
            "and volume are tested end-to-end. A bug in vertex computation or "
            "face orientation would cause the total volume to deviate from the "
            "box volume.\n"
            "  \\item \\textbf{Boundary handling:} Cells adjacent to the domain "
            "boundary are clipped against the bounding box. Correct clipping is "
            "essential for volume conservation and for imposing boundary "
            "conditions in production simulations.\n"
            "  \\item \\textbf{Parallel tessellation:} In the MPI variant, each "
            "rank builds a local Voronoi diagram with ghost layers from "
            "neighbouring ranks. The sum of all local volumes must still equal "
            "the global box volume, verifying that no gaps or overlaps exist at "
            "processor boundaries.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} The Voronoi tessellation is the geometric "
            "foundation of the entire code. Every flux calculation, gradient "
            "reconstruction, and cell-centred update depends on correct cell "
            "volumes and face areas. This test catches low-level geometric bugs "
            "(floating-point degeneracies, clipping errors, parallel "
            "inconsistencies) that would otherwise manifest as mysterious "
            "conservation violations in physics simulations. It is also extremely "
            "cheap to run, making it ideal as a first-line diagnostic."
        ),
        "initial_conditions": (
            r"The domain is a unit cube $[0,\,1]^3$ filled with a Voronoi "
            r"tessellation from random points. No hydrodynamics or time evolution "
            r"is performed."
        ),
        "boundary_conditions": "Rigid walls (domain boundary).",
        "mesh_movement": "Static (no time evolution).",
        "execution": (
            "Both serial and MPI. Serial: 1~CPU, direct execution. "
            "MPI: 64~CPUs via SLURM (partition \\texttt{bigrun}, exclusive)."
        ),
        "pass_criteria": (
            r"The relative volume error must be negligible:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item $\texttt{rel\_error} = |V_{\mathrm{total}} - V_{\mathrm{box}}|"
            r" / V_{\mathrm{box}} < 10^{-10}$." "\n"
            r"\end{itemize}"
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "lane_self_gravity",
        "title": "Lane--Emden Self-Gravity Equilibrium",
        "description": (
            "This test initializes a polytropic star in hydrostatic equilibrium "
            "according to the Lane--Emden equation \\cite{chandrasekhar1939,lane1870} "
            "(polytropic index $n = 1.5$) "
            "and evolves it with self-gravity. The density profile should remain "
            "close to the analytical solution. This is a standard benchmark for "
            "codes coupling hydrodynamics and Newtonian gravity.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Self-gravity solver:} The Newtonian gravitational "
            "potential and its gradient must be computed accurately enough that "
            "the gravitational acceleration balances the pressure gradient, "
            "maintaining hydrostatic equilibrium. Errors in the gravity solver "
            "would cause the star to spuriously expand or collapse.\n"
            "  \\item \\textbf{Hydrostatic balance on an unstructured mesh:} "
            "Maintaining a static equilibrium with steep density gradients on a "
            "Voronoi mesh is challenging. The pressure-gradient reconstruction "
            "and the gravity-source coupling must be well-balanced to avoid "
            "generating spurious velocities.\n"
            "  \\item \\textbf{Spherical geometry on a Cartesian domain:} The "
            "spherical star is embedded in a Cartesian bounding box. The test "
            "checks that the boundary between the star's interior and the "
            "low-density exterior is handled cleanly.\n"
            "  \\item \\textbf{Conservation under long-time evolution:} The "
            "simulation runs for multiple dynamical times. Slow drifts in total "
            "energy or mass, or gradual distortion of the density profile, "
            "would indicate conservation or well-balancing defects.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Self-gravity is essential for astrophysical "
            "applications (star formation, stellar structure, compact-object "
            "mergers). The Lane--Emden equilibrium is one of the few "
            "configurations with an analytical density profile, making it "
            "invaluable for verifying the gravity module. It is also a "
            "sensitive test of the code's ability to maintain a static "
            "equilibrium---a prerequisite for reliable long-duration "
            "simulations of gravitationally bound systems."
        ),
        "initial_conditions": (
            r"A sphere of mass $M = 2\times10^{33}\;\mathrm{g}$ and radius "
            r"$R = 7\times10^{10}\;\mathrm{cm}$ is set up with the Lane--Emden "
            r"density profile (read from tabulated data files "
            r"\texttt{data/xsi32.txt} and \texttt{data/theta32.txt}). "
            r"The gravitational constant is $G = 6.674\times10^{-8}\;\mathrm{cgs}$. "
            r"The Voronoi mesh uses stratified random sampling inside the sphere."
        ),
        "boundary_conditions": "Rigid walls at the outer boundary.",
        "mesh_movement": "Lagrangian with cell rounding. Gravity via a conservative force module.",
        "execution": "MPI, 512~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"The volume-weighted average density error vs.\ the analytical "
            r"Lane--Emden profile must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item $|\texttt{final\_metric}| = "
            r"\frac{\sum |\rho_{\mathrm{num}} - \rho_{\mathrm{analytic}}|\,V}"
            r"{\sum V} < 4\times10^{-2}$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["lane_self_gravity.png"],
        "plot_caption": (
            "Lane--Emden self-gravity: numerical density profile (black dots) "
            "vs.\\ the analytical Lane--Emden solution (red line) as a function "
            "of radius."
        ),
    },
    {
        "id": "lane_self_gravity_fmm",
        "title": "Lane--Emden Self-Gravity Equilibrium (FMM)",
        "description": (
            "This test is identical to the Lane--Emden self-gravity benchmark, "
            "but uses the distributed fast multipole method (FMM) gravity solver "
            "instead of the quadrupole Barnes--Hut tree. The expansion order is "
            "$P = 3$ and the multipole acceptance criterion is "
            "$\\theta = 0.9$.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{FMM gravity in hydro:} The "
            "\\texttt{FastMultipoleAcceleration3D} adapter must compute "
            "consistent gravitational accelerations on the Voronoi mesh and "
            "couple correctly to the conservative force module.\n"
            "  \\item \\textbf{Hydrostatic balance with approximate gravity:} "
            "The star should remain near the analytical Lane--Emden profile "
            "despite using an approximate FMM solver rather than the direct "
            "quadrupole tree.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Same as \\texttt{lane\\_self\\_gravity}: a polytropic star with "
            r"$M = 2\times10^{33}\;\mathrm{g}$, $R = 7\times10^{10}\;\mathrm{cm}$, "
            r"and Lane--Emden tables \\texttt{data/xsi32.txt} and "
            r"\\texttt{data/theta32.txt}."
        ),
        "boundary_conditions": "Rigid walls at the outer boundary.",
        "mesh_movement": "Lagrangian with cell rounding. Gravity via FMM conservative force.",
        "execution": "MPI, 512~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"The volume-weighted average density error vs.\ the analytical "
            r"Lane--Emden profile must satisfy "
            r"$|\texttt{final\_metric}| < 4\times10^{-2}$."
        ),
        "plots": ["lane_self_gravity.png"],
        "plot_caption": (
            "Lane--Emden self-gravity: FMM density profile (blue dots) overlaid "
            "on the quadrupole-tree result (black dots, if available) vs.\\ the "
            "analytical Lane--Emden solution (red line)."
        ),
    },
    {
        "id": "mach2_diffusion",
        "title": "Mach~2 Radiative Shock -- Gray Diffusion",
        "description": (
            "A Mach~2 radiative shock with gray (single-group) flux-limited "
            "diffusion \\cite{lowrie2008,lowrie1999}. Radiative shocks exhibit a "
            "rich structure including a precursor region where radiation preheats "
            "the upstream material and a relaxation zone behind the shock "
            "\\cite{mihalas1984,zel2002}. "
            "The numerical profiles are compared to the NLTE "
            "analytical solution for density, gas temperature, and radiation "
            "temperature.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Radiation--hydrodynamics coupling:} This is the "
            "first test in the suite that couples radiation transport and "
            "hydrodynamics in a non-trivial flow. The radiation precursor heats "
            "the upstream gas before the shock arrives, modifying the shock "
            "structure. Getting both the precursor length and the post-shock "
            "relaxation zone correct requires accurate coupling between the "
            "radiation and hydro solvers.\n"
            "  \\item \\textbf{Flux-limited diffusion:} The gray radiation "
            "diffusion equation with a flux limiter must transition smoothly "
            "between the diffusion regime (optically thick) and the "
            "free-streaming regime (optically thin). The flux limiter is "
            "critical for preventing superluminal radiation transport.\n"
            "  \\item \\textbf{Implicit radiation solve:} The radiation "
            "diffusion equation is solved implicitly. The linear solver, "
            "matrix assembly, and convergence criteria are all exercised.\n"
            "  \\item \\textbf{Non-equilibrium radiation:} The gas and radiation "
            "temperatures differ throughout the precursor and relaxation regions "
            "($T_{\\mathrm{gas}} \\neq T_{\\mathrm{rad}}$). The test verifies "
            "that the code correctly handles non-equilibrium conditions and "
            "evolves toward the correct downstream equilibrium.\n"
            "  \\item \\textbf{Steady-state shock structure:} The simulation "
            "must reach a steady state in the shock frame, requiring correct "
            "inflow boundary conditions and long enough run time.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Radiative shocks are ubiquitous in HEDP "
            "experiments and astrophysical phenomena (accretion shocks, "
            "supernova remnants, stellar atmospheres). The Mach~2 Lowrie "
            "benchmark is one of the few radiation-hydrodynamics problems "
            "with a semi-analytical solution, making it a cornerstone "
            "verification test. Passing this test gives confidence that the "
            "code can simulate radiation-dominated flows where the radiation "
            "field significantly modifies the hydrodynamic structure."
        ),
        "initial_conditions": (
            r"A 1D Cartesian mesh with 1024 cells spanning "
            r"$x \in [-10^3,\;2\times10^3]\;\mathrm{cm}$. "
            r"The left and right states are:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item \textbf{Left (upstream):} $\rho = 5.459\times10^{-13}\;\mathrm{g/cm^3}$, "
            r"$v = 2.355\times10^{5}\;\mathrm{cm/s}$, $T = 100\;\mathrm{K}$." "\n"
            r"  \item \textbf{Right (downstream):} $\rho = 1.248\times10^{-12}\;\mathrm{g/cm^3}$, "
            r"$v = 1.03\times10^{5}\;\mathrm{cm/s}$, $T = 207.8\;\mathrm{K}$." "\n"
            r"\end{itemize}" "\n"
            r"Adiabatic index $\gamma = 5/3$; Rosseland opacity "
            r"$\sigma_{\mathrm{Ross}} = 0.849\;\mathrm{cm^{-1}}$; "
            r"absorption opacity $\sigma_{\mathrm{abs}} = 3.93\times10^{-5}\;\mathrm{cm^{-1}}$. "
            r"The simulation time is $t = 0.01\;\mathrm{s}$."
        ),
        "boundary_conditions": "Inflow boundaries at left and right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "MPI, 8~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Profiles are compared to the NLTE analytical solution. "
            r"The relative $L_1$ error must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Density: relative $L_1 \le 0.025$." "\n"
            r"  \item Temperature: relative $L_1 \le 0.025$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["mach2_diffusion.png"],
        "plot_caption": (
            "Mach~2 gray radiative shock: numerical profiles (black dots) vs.\\ "
            "NLTE analytical solution (colored lines) for density, gas temperature, "
            "and radiation temperature."
        ),
    },
    {
        "id": "mach2_multigroup",
        "title": "Mach~2 Radiative Shock -- Multigroup",
        "description": (
            "The same Mach~2 radiative shock as the gray diffusion test "
            "\\cite{lowrie2008,lowrie1999}, but using "
            "multigroup radiation transport with 32 logarithmically spaced energy "
            "groups. This tests the frequency-dependent radiation solver and "
            "verifies that the multigroup solution reproduces the correct spatial "
            "profiles. In addition, the radiation spectrum of the "
            "hottest cell is compared to a Planck distribution at the local gas "
            "temperature to check spectral equilibrium.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Multigroup radiation transport:} Unlike the gray "
            "test, the radiation field is resolved into 32 frequency groups. "
            "Each group has its own diffusion equation with frequency-dependent "
            "opacities. The test verifies that the group-by-group solver, the "
            "opacity evaluation, and the energy-group coupling are all correct.\n"
            "  \\item \\textbf{Spectral equilibrium:} In the post-shock region "
            "where radiation and matter are in thermal equilibrium, the radiation "
            "spectrum should follow a Planck distribution at the local gas "
            "temperature. Comparing the computed spectrum to the Planck function "
            "checks that the multigroup solver correctly thermalises the radiation "
            "field.\n"
            "  \\item \\textbf{Consistency with gray solution:} The multigroup "
            "spatial profiles (density, $T_{\\mathrm{gas}}$, $T_{\\mathrm{rad}}$) "
            "should closely match the gray solution. Significant deviations would "
            "indicate a bug in the frequency-dependent opacity averaging or "
            "group-boundary treatment.\n"
            "  \\item \\textbf{Planck and Rosseland mean opacities:} The "
            "effective gray opacity emerges from averaging the frequency-dependent "
            "values. The test implicitly checks that the Planck and Rosseland "
            "means are computed consistently.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Production radiation-hydrodynamics simulations "
            "almost always use multigroup (or full-spectrum) transport. The gray "
            "approximation is a useful simplification for verification, but "
            "real applications require frequency dependence to capture "
            "line-driven effects, spectral hardening, and non-Planckian radiation "
            "fields. This test bridges the gap between the simple gray benchmark "
            "and production-level multigroup capability, ensuring that the "
            "frequency-dependent machinery does not introduce errors beyond those "
            "already present in the gray solution."
        ),
        "initial_conditions": (
            r"Identical to the gray diffusion case (1024-cell Cartesian mesh, same "
            r"left/right states and opacities). The energy groups span "
            r"$E_{\min} = 10^{-3}\,k_B\!\cdot\!200\;\mathrm{K}$ to "
            r"$E_{\max} = 10^{3}\,k_B\!\cdot\!200\;\mathrm{K}$, logarithmically spaced."
        ),
        "boundary_conditions": "Inflow boundaries at left and right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": (
            "MPI, 8~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive). "
            "Built with \\texttt{--energy\\_groups\\_num=32}."
        ),
        "pass_criteria": (
            r"Same spatial profile thresholds as the gray case:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Density: relative $L_1 \le 0.025$." "\n"
            r"  \item Temperature: relative $L_1 \le 0.025$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["mach2_multigroup.png", "mach2_multigroup_spectrum.png"],
        "plot_caption": (
            "Mach~2 multigroup radiative shock. Top: spatial profiles of density, "
            "gas temperature, and radiation temperature (numerical vs.\\ NLTE analytical). "
            "Bottom: radiation energy spectrum at the hottest cell (numerical vs.\\ "
            "Planck distribution at $T_{\\mathrm{gas}}$)."
        ),
    },
    {
        "id": "marshak_wave_1_diffusion",
        "title": "Marshak Wave Problem~1 (Non-Equilibrium, Uniform Density)",
        "description": (
            "Problem~1 from Giron et~al.\\ (2026, arXiv:2601.05120), originally "
            "defined as Test~2 in Krief \\& McClarren (2024, arXiv:2401.05138). "
            "A supersonic non-equilibrium Marshak wave propagates into a cold, "
            "uniform-density medium. The opacity is strongly temperature-dependent: "
            "$\\kappa_R = 100\\,(T/\\mathrm{keV})^{-3}$ and the Planck opacity is "
            "$\\kappa_P = 0.001\\,\\kappa_R$ (far from equilibrium). "
            "The material energy is $u = 6.86\\times10^{14}\\,(T/\\mathrm{keV})^4$.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Grey flux-limited diffusion (FLD):} Exercises the "
            "implicit diffusion solver with power-law opacities and no flux limiter.\n"
            "  \\item \\textbf{Non-equilibrium radiation--matter coupling:} With "
            "$\\kappa_P \\ll \\kappa_R$, the radiation and gas temperatures decouple "
            "significantly, testing the absorption/emission coupling.\n"
            "  \\item \\textbf{Time-dependent Marshak boundary condition.}\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Domain $x \in [0,\,0.2]$ with 512 uniform cells. "
            r"$\rho = 1$ (uniform), initially cold ($T \approx 0$). "
            r"The bath temperature at $x=0$ is "
            r"$T_{\mathrm{bath}}(t) = 1.008038\,(t/\mathrm{ns})^{1/3}\;\mathrm{keV}$."
        ),
        "boundary_conditions": "Marshak (radiation inflow) at left, zero-flux at right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"Relative $L_1$ error for both $T_{\mathrm{gas}}$ and $T_{\mathrm{rad}}$ "
            r"must be $\le 10^{-2}$, compared to the self-similar analytical solution."
        ),
        "plots": ["marshak_wave_1_diffusion.png"],
        "plot_caption": (
            "Marshak wave Problem~1 at $t=1$~ns: gas and radiation temperature "
            "profiles (RICH vs.\\ analytical self-similar solution)."
        ),
    },
    {
        "id": "marshak_wave_2_diffusion",
        "title": "Marshak Wave Problem~2 (Equilibrium Limit, Uniform Density)",
        "description": (
            "Problem~2 from Giron et~al.\\ (2026), originally Test~3 in "
            "Krief \\& McClarren (2024). Same as Problem~1 except the Planck "
            "opacity equals the Rosseland opacity ($\\kappa_P = \\kappa_R$), "
            "placing the system in the equilibrium (LTE) limit where "
            "$T_{\\mathrm{rad}} \\approx T_{\\mathrm{gas}}$.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Equilibrium radiation diffusion:} Tests that the "
            "code correctly recovers the LTE limit where matter and radiation "
            "temperatures converge.\n"
            "  \\item \\textbf{Marshak boundary with time-dependent driving temperature.}\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Domain $x \in [0,\,0.2]$ with 512 uniform cells, $\rho=1$, initially cold. "
            r"$T_{\mathrm{bath}}(t) = 1.014565\,(t/\mathrm{ns})^{1/3}\;\mathrm{keV}$."
        ),
        "boundary_conditions": "Marshak (radiation inflow) at left, zero-flux at right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"Relative $L_1$ error for both $T_{\mathrm{gas}}$ and $T_{\mathrm{rad}}$ "
            r"must be $\le 10^{-2}$."
        ),
        "plots": ["marshak_wave_2_diffusion.png"],
        "plot_caption": (
            "Marshak wave Problem~2 (equilibrium) at $t=1$~ns: gas and radiation "
            "temperature profiles (RICH vs.\\ analytical)."
        ),
    },
    {
        "id": "marshak_wave_3_diffusion",
        "title": "Marshak Wave Problem~3 (Non-Uniform Density, Power-Law Profile)",
        "description": (
            "Problem~3 from Giron et~al.\\ (2026), originally Test~1 in "
            "Derei et~al.\\ (2024, arXiv:2411.14891). The material density "
            "varies as $\\rho(x) = x^{20/19}$, and both the opacity and the "
            "energy depend on density: "
            "$\\kappa_R = 40\\,(T/\\mathrm{keV})^{-1.5}\\,\\rho^{1.2}$, "
            "$\\kappa_P = 0.0025\\,\\kappa_R$, "
            "$u = 10^{14}\\,(T/\\mathrm{keV})^{3.4}\\,\\rho^{0.86}$.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Density-dependent opacities and EOS:} Tests the "
            "code's ability to handle coupled density--temperature dependence "
            "in the diffusion coefficients.\n"
            "  \\item \\textbf{Non-uniform initial density profile.}\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Domain $x \in [0,\,1]$, 512 uniform cells. "
            r"$\rho(x) = x^{20/19}$, initially cold. "
            r"$T_{\mathrm{bath}}(t) = 1.0470478\,(t/\mathrm{ns})^{86/57}\;\mathrm{keV}$."
        ),
        "boundary_conditions": "Marshak at left, zero-flux at right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"Relative $L_1$ error for both $T_{\mathrm{gas}}$ and $T_{\mathrm{rad}}$ "
            r"must be $\le 10^{-2}$."
        ),
        "plots": ["marshak_wave_3_diffusion.png"],
        "plot_caption": (
            "Marshak wave Problem~3 (non-uniform density) at $t=1$~ns."
        ),
    },
    {
        "id": "marshak_wave_4_diffusion",
        "title": "Marshak Wave Problem~4 (Divergent Density, Stretched Grid)",
        "description": (
            "Problem~4 from Giron et~al.\\ (2026), originally Test~3 in "
            "Derei et~al.\\ (2024). The density diverges toward the origin as "
            "$\\rho(x) = x^{-40/139}$. A stretched grid (geometric progression) "
            "is used to resolve the steep gradients near $x=0$. "
            "$\\kappa_R = 2\\,(T/\\mathrm{keV})^{-4.5}\\,\\rho^{1.9}$, "
            "$\\kappa_P = 5{\\times}10^{-4}\\,\\kappa_R$, "
            "$u = 10^{14}\\,(T/\\mathrm{keV})^{6}\\,\\rho^{0.7}$.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Stretched (non-uniform) grid:} The geometric "
            "progression mesh tests the diffusion solver on highly non-uniform "
            "cell sizes.\n"
            "  \\item \\textbf{Divergent density at origin:} Stresses the "
            "opacity and energy evaluations near singular density profiles.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Domain with 512 stretched cells (geometric progression). "
            r"$\rho(x) = x^{-40/139}$, initially cold. "
            r"$T_{\mathrm{bath}}(t) = 1.01008116\,(t/\mathrm{ns})^{14/139}\;\mathrm{keV}$."
        ),
        "boundary_conditions": "Marshak at left, zero-flux at right.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"Relative $L_1$ error for both $T_{\mathrm{gas}}$ and $T_{\mathrm{rad}}$ "
            r"must be $\le 10^{-2}$."
        ),
        "plots": ["marshak_wave_4_diffusion.png"],
        "plot_caption": (
            "Marshak wave Problem~4 (divergent density, stretched grid) at $t=1$~ns."
        ),
    },
    {
        "id": "gresho_euler",
        "title": "Gresho Vortex (Eulerian)",
        "description": (
            "The Gresho vortex \\cite{liska2003} is a stationary azimuthal flow "
            "in centrifugal equilibrium: the pressure gradient exactly balances "
            "the centripetal acceleration, so the exact solution at any time is "
            "the initial condition. Numerical dissipation and dispersion cause "
            "the vortex to decay over time, making this an excellent test of a "
            "code's ability to preserve rotating structures.\n\n"
            "This test uses an \\textbf{Eulerian (fixed) mesh}, which subjects "
            "the solution to advection errors as the vortex sweeps across the "
            "grid. The 3D setup uses a single cell in $z$ to approximate a 2D "
            "problem.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Euler fluxes on a fixed mesh:} The vortex tests "
            "whether the Riemann solver and spatial reconstruction preserve a "
            "smooth, subsonic rotating flow.\n"
            "  \\item \\textbf{Numerical dissipation:} The rate of vortex decay "
            "is a direct measure of the scheme's numerical viscosity.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Domain $[-0.5,\,0.5]^2 \times [0,\,\Delta z]$, Cartesian $50\times50\times1$. "
            r"$\gamma = 5/3$, $\rho = 1$. "
            r"Azimuthal velocity: $v_\theta = 5r$ for $r<0.2$, "
            r"$v_\theta = 2-5r$ for $0.2 \le r \le 0.4$, $v_\theta = 0$ for $r>0.4$. "
            r"Pressure set for centrifugal equilibrium. "
            r"Simulation runs to $t = 5$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"Relative $L_1$ error for the azimuthal velocity profile "
            r"(volume-averaged, radially binned) at $t=5$ compared to the initial "
            r"condition must be $\le 0.1$."
        ),
        "plots": [
            "gresho_euler_pressure.png",
            "gresho_euler_vtheta_2d.png",
            "gresho_euler_vtheta_r.png",
        ],
        "plot_caption": (
            "Gresho vortex (Eulerian) at $t=5$: pressure field (top left), "
            "azimuthal velocity field (top right), and radially binned azimuthal "
            "velocity profile vs.\\ initial condition (bottom)."
        ),
    },
    {
        "id": "gresho_lagrangian",
        "title": "Gresho Vortex (Lagrangian + RoundCells)",
        "description": (
            "Same Gresho vortex problem as the Eulerian case, but run with a "
            "\\textbf{Lagrangian mesh} augmented by the Arepo-style cell-rounding "
            "correction (\\texttt{RoundCells3D}). Mesh point motion is restricted "
            "to the $xy$-plane.\n\n"
            "Because the mesh co-rotates with the fluid, advection errors are "
            "eliminated. The primary error source becomes the mesh deformation "
            "and the cell-rounding correction. The Lagrangian scheme should "
            "preserve the vortex significantly better than the Eulerian scheme.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Lagrangian mesh motion:} Verifies that the moving "
            "mesh correctly tracks the rotating flow.\n"
            "  \\item \\textbf{Cell rounding:} The RoundCells correction must "
            "maintain mesh quality without introducing excessive dissipation.\n"
            "  \\item \\textbf{Custom point-motion wrapper:} The $z$-velocity "
            "zeroing wrapper is tested for correctness.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Identical to the Eulerian case: $50\times50\times1$ Cartesian mesh, "
            r"$\gamma = 5/3$, same velocity and pressure profiles, $t_{\mathrm{end}} = 5$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces.",
        "mesh_movement": "Lagrangian + RoundCells (restricted to $xy$-plane).",
        "execution": "MPI, 8~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Relative $L_1$ error for the azimuthal velocity profile must be $\le 0.05$."
        ),
        "plots": [
            "gresho_lagrangian_pressure.png",
            "gresho_lagrangian_vtheta_2d.png",
            "gresho_lagrangian_vtheta_r.png",
        ],
        "plot_caption": (
            "Gresho vortex (Lagrangian + RoundCells) at $t=5$: pressure, azimuthal "
            "velocity field, and radially binned azimuthal velocity vs.\\ initial condition."
        ),
    },
    {
        "id": "desmore2012_mc",
        "title": "Densmore 2012 Heterogeneous Step-Opacity (Monte Carlo IMC)",
        "description": (
            "The first heterogeneous test problem from Densmore et al.\\ "
            "\\cite{densmore2012} exercises the Monte Carlo Implicit Monte Carlo "
            "(IMC) radiation transport solver with frequency-dependent opacities "
            "in a two-material slab geometry. The domain contains an optically "
            "thin region ($x < 2$~cm, $\\sigma_0 = 10$~keV$^{7/2}$/cm) and an "
            "optically thick region ($x \\ge 2$~cm, $\\sigma_0 = 1000$~keV$^{7/2}$/cm), "
            "with a Planck source at 1~keV driving radiation from the left boundary.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Monte Carlo IMC transport:} Particle creation, "
            "transport, absorption, and re-emission in a frequency-dependent setting.\n"
            "  \\item \\textbf{Multigroup opacities:} Correct frequency-group "
            "averaging of the power-law opacity $\\sigma = \\sigma_0 / (\\sqrt{kT}\\,E^3)$.\n"
            "  \\item \\textbf{MPI parallelism:} The problem is distributed across "
            "32 MPI ranks with domain decomposition and particle communication.\n"
            "  \\item \\textbf{Heterogeneous material interface:} The sharp opacity "
            "jump at $x = 2$~cm produces a characteristic temperature spike that "
            "tests the interface treatment.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} This is the primary verification test for the "
            "Monte Carlo IMC radiation solver with frequency-dependent opacities. "
            "The reference solution from the Milagro IMC code \\cite{densmore2012} "
            "provides an independent, published benchmark."
        ),
        "initial_conditions": (
            r"The domain is $x \in [0,\,3]$~cm with 256 uniformly spaced cells. "
            r"The opacity follows $\sigma(\nu) = \sigma_0 / (\sqrt{kT}\,E^3)$ with:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item $x < 2$~cm: $\sigma_0 = 10$~keV$^{7/2}$/cm (optically thin)." "\n"
            r"  \item $x \ge 2$~cm: $\sigma_0 = 1000$~keV$^{7/2}$/cm (optically thick)." "\n"
            r"\end{itemize}" "\n"
            r"Initial temperature $T = 1$~eV, density $\rho = 1$~g/cm$^3$, "
            r"ideal gas EOS with $\gamma = 1.4$ and $C_v = 10^{15}/T_{\mathrm{keV}}$. "
            r"The simulation runs to $t = 1$~ns with $\Delta t = 5 \times 10^{-12}$~s."
        ),
        "boundary_conditions": (
            r"Left boundary: isotropic Planck source at $T = 1$~keV. "
            r"Right boundary: reflective."
        ),
        "mesh_movement": "Eulerian (fixed mesh), no hydrodynamics.",
        "execution": "MPI, 32 ranks, SLURM (bigrun partition, exclusive).",
        "pass_criteria": (
            r"The gas temperature profile $T_{\mathrm{gas}}(x)$ is compared to "
            r"the Monte Carlo curve digitized from Figure~4 of Densmore et al.\ "
            r"\cite{densmore2012}. The $L_1$ norm of the absolute error in keV "
            r"must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item $L_1 = \mathrm{mean}(|T_{\mathrm{sim}} - T_{\mathrm{ref}}|) \le 0.05$~keV." "\n"
            r"\end{itemize}"
        ),
        "plots": ["desmore2012_mc.png"],
        "plot_caption": (
            "Densmore 2012 heterogeneous step-opacity at $t = 1$~ns: RICH MPI MC "
            "without random walk (black circles), RICH serial MC with random walk "
            "(red crosses), and digitized reference from Figure~4 of "
            "Densmore et al.\\ (2012) (blue line)."
        ),
    },
    {
        "id": "desmore2012_mc_serial",
        "title": "Densmore 2012 Heterogeneous Step-Opacity (Serial MC, Random Walk)",
        "description": (
            "The same heterogeneous problem as \\texttt{desmore2012\\_mc} but run "
            "serially with the random walk acceleration enabled. This exercises the "
            "serial (non-MPI) execution path and RW acceleration of the Monte Carlo IMC solver.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Serial execution path:} Verifies that the MC solver "
            "produces correct results without MPI domain decomposition.\n"
            "  \\item \\textbf{Cross-validation:} Together with the MPI (no-RW) variant, "
            "this test confirms that both random walk acceleration and MPI parallelism "
            "produce accurate results independently.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Identical to \texttt{desmore2012\_mc}: 256 cells, $x \in [0,\,3]$~cm, "
            r"$\sigma_0 = 10 / 1000$~keV$^{7/2}$/cm, $T_0 = 1$~eV, "
            r"$t_{\mathrm{end}} = 1$~ns, $\Delta t = 5 \times 10^{-12}$~s."
        ),
        "boundary_conditions": (
            r"Left boundary: isotropic Planck source at $T = 1$~keV. "
            r"Right boundary: reflective."
        ),
        "mesh_movement": "Eulerian (fixed mesh), no hydrodynamics.",
        "execution": "Serial (single core).",
        "pass_criteria": (
            r"Same as \texttt{desmore2012\_mc}: "
            r"$L_1 = \mathrm{mean}(|T_{\mathrm{sim}} - T_{\mathrm{ref}}|) \le 0.05$~keV."
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "yee_vortex_64",
        "title": "Yee Isentropic Vortex (64$\\times$64, Lagrangian)",
        "description": (
            "Stationary isentropic vortex (Yee et al.\\ 1999) on a Lagrangian mesh "
            "with RoundCells correction. The vortex is an exact steady-state solution "
            "of the compressible Euler equations: a smooth velocity field with matching "
            "isentropic density and pressure perturbations.\n\n"
            "The analytical solution at any time $t$ equals the initial condition, so "
            "the $L_1$ density error at $t=10$ measures numerical dissipation and mesh "
            "deformation artifacts. This is the lower-resolution run of the convergence "
            "pair (see also \\texttt{yee\\_vortex\\_128}).\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Lagrangian mesh motion:} Verifies that the moving "
            "mesh tracks a smooth rotational flow over many revolutions.\n"
            "  \\item \\textbf{Isentropic flow preservation:} Tests the scheme's "
            "ability to maintain entropy in a smooth, vortex-dominated flow.\n"
            "  \\item \\textbf{Density convergence:} Paired with the 128$\\times$128 "
            "run to verify second-order spatial convergence.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"$64\times64\times1$ Cartesian mesh on $[-5,\,5]^2$, $\gamma = 1.4$. "
            r"Vortex strength $\beta = 5$, centred at the origin. "
            r"$T = 1 - \frac{(\gamma-1)\beta^2}{8\gamma\pi^2} e^{1-r^2}$, "
            r"$\rho = T^{1/(\gamma-1)}$, $p = T^{\gamma/(\gamma-1)}$, "
            r"$v_x = -\frac{\beta}{2\pi} y\, e^{(1-r^2)/2}$, "
            r"$v_y = \frac{\beta}{2\pi} x\, e^{(1-r^2)/2}$. "
            r"$t_{\mathrm{end}} = 10$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces.",
        "mesh_movement": "Lagrangian + RoundCells (restricted to $xy$-plane).",
        "execution": "MPI, 8~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Volume-weighted $L_1$ density error at $t=10$ compared to the "
            r"analytical initial condition must be $\le 0.05$."
        ),
        "plots": [
            "yee_vortex_density_2d.png",
            "yee_vortex_pressure_2d.png",
            "yee_vortex_density_r.png",
            "yee_vortex_convergence.png",
        ],
        "plot_caption": (
            "Yee isentropic vortex at $t=10$: density field (top left), pressure "
            "field (top right), radially binned density profile vs.\\ analytical "
            "IC for both resolutions (bottom left), and $L_1$ density error "
            "convergence with second-order reference (bottom right)."
        ),
    },
    {
        "id": "yee_vortex_128",
        "title": "Yee Isentropic Vortex (128$\\times$128, Lagrangian)",
        "description": (
            "Higher-resolution companion to \\texttt{yee\\_vortex\\_64}. "
            "Same isentropic vortex problem on a $128\\times128\\times1$ mesh. "
            "Together with the 64$\\times$64 run, this establishes the spatial "
            "convergence rate of the Lagrangian scheme for smooth flows."
        ),
        "initial_conditions": (
            r"$128\times128\times1$ Cartesian mesh on $[-5,\,5]^2$, same "
            r"$\gamma$, $\beta$, velocity and thermodynamic profiles as the "
            r"64$\times$64 case. $t_{\mathrm{end}} = 10$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces.",
        "mesh_movement": "Lagrangian + RoundCells (restricted to $xy$-plane).",
        "execution": "MPI, 16~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Volume-weighted $L_1$ density error at $t=10$ must be $\le 0.05$."
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "spherical_gauss_linear",
        "title": "Spherical LSQ Gradient -- Linear Field Verification",
        "description": (
            "A pure-reconstruction test that verifies the weighted least-squares "
            "(LSQ) gradient in spherical coordinate space implemented in "
            "\\texttt{SphericalLinearGauss3D}. Fields that are exactly linear in "
            "$(r,\\,\\theta)$ are prescribed at every cell centre and then "
            "reconstructed to face centroids. Because the LSQ gradient can "
            "represent linear functions exactly, the interpolated values must "
            "agree with the exact values to machine precision.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{LSQ gradient accuracy:} The 3\\texttimes3 normal "
            "system and its analytic inverse must yield the exact coordinate "
            "derivatives $\\partial f/\\partial r$, $\\partial f/\\partial\\theta$, "
            "$\\partial f/\\partial\\phi$.\n"
            "  \\item \\textbf{Velocity frame conversion:} Each cell's velocity "
            "is projected onto its own local spherical basis; the interpolated "
            "velocity at the face centroid is converted back to Cartesian using "
            "the basis at that point.\n"
            "  \\item \\textbf{Coordinate-space interpolation:} The "
            "displacement $(\\Delta r,\\,\\Delta\\theta,\\,\\Delta\\phi)$ between "
            "cell centre and face centroid must be computed correctly.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"A Voronoi tessellation from $\sim$4\,000 random points in a "
            r"spherical shell $1 < r < 2$ embedded in a $[-4,\,4]^3$ box. "
            r"Scalar fields (density, internal energy) are linear in "
            r"$(r,\,\theta)$; velocity components $(v_r,\,v_\theta,\,v_\phi)$ "
            r"are also linear in $(r,\,\theta)$ (no $\phi$ dependence to avoid "
            r"angle-wrapping artefacts). The slope limiter is disabled so that "
            r"the reconstruction is purely linear."
        ),
        "boundary_conditions": "Rigid walls at box faces.",
        "mesh_movement": "Static (no time evolution).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"For interior faces (both cell CMs in $1.1 < r < 1.9$):"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Scalar max relative error $< 10^{-8}$." "\n"
            r"  \item Velocity max relative error $< 0.1$." "\n"
            r"\end{itemize}"
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "cartesian_gauss_linear",
        "title": "Cartesian LSQ Gradient -- Linear Field Verification",
        "description": (
            "Complementary to \\texttt{spherical\\_gauss\\_linear}: this test "
            "verifies the standard Cartesian weighted least-squares gradient "
            "in \\texttt{LinearGauss3D}. Fields that are exactly linear in "
            "$(x,\\,y,\\,z)$ are prescribed at every cell centre and "
            "reconstructed to face centroids. Because the LSQ gradient can "
            "represent linear functions exactly, the Cartesian interpolated "
            "values must agree with the exact values to machine precision.\n\n"
            "The test also evaluates the \\emph{spherical} reconstruction "
            "(\\texttt{SphericalLinearGauss3D}) on the same Cartesian-linear "
            "fields and checks that the Cartesian mode achieves a strictly "
            "lower error, confirming that each reconstruction is optimal for "
            "its own coordinate family.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Cartesian LSQ gradient accuracy:} The normal "
            "system must yield the exact derivatives $\\partial f/\\partial x$, "
            "$\\partial f/\\partial y$, $\\partial f/\\partial z$ to roundoff.\n"
            "  \\item \\textbf{Reconstruction mode comparison:} Cartesian "
            "reconstruction must outperform spherical for Cartesian-linear "
            "fields, validating the coordinate-selection logic.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"A Voronoi tessellation from $\sim$4\,000 random points in a "
            r"spherical shell $1 < r < 2$ embedded in a $[-4,\,4]^3$ box. "
            r"Scalar fields (density, pressure, internal energy) are linear in "
            r"$(x,\,y,\,z)$; velocity components $(v_x,\,v_y,\,v_z)$ are "
            r"also linear in $(x,\,y,\,z)$. The slope limiter is disabled "
            r"so that the reconstruction is purely linear."
        ),
        "boundary_conditions": "Rigid walls at box faces.",
        "mesh_movement": "Static (no time evolution).",
        "execution": "Serial, 1~CPU, direct execution.",
        "pass_criteria": (
            r"For interior faces (both cell CMs in $1.1 < r < 1.9$):"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Cartesian scalar max relative error $< 10^{-6}$." "\n"
            r"  \item Cartesian velocity max relative error $< 0.1$." "\n"
            r"  \item Cartesian scalar error strictly less than spherical scalar error." "\n"
            r"\end{itemize}"
        ),
        "plots": [],
        "plot_caption": "",
    },
    {
        "id": "spherical_collapse",
        "title": "Spherical Shell Collapse (Symmetry Test)",
        "description": (
            "A dense spherical shell ($0.9 < r < 1.0$, $\\rho=10$, $P=0.1$) is "
            "given an inward radial velocity $v_r = -1$ and embedded in a "
            "low-density ambient medium ($\\rho=10^{-3}$, "
            "$P=10^{-5}$).  The shell collapses inward on a fixed Eulerian "
            "mesh. The mesh is constructed by generating a well-rounded "
            "template sphere (\\texttt{RandSphereSurfaceRounded}, $\\sim$10\\,000 "
            "points) and replicating it at logarithmically spaced radii from "
            "$R=1.1$ down to $R=0.05$ with constant $\\Delta R/R$, ensuring "
            "near-unity aspect ratio. The inner core ($r<0.05$) and the "
            "region between the outer sphere and the box walls are filled with "
            "random points.\n\n"
            "The simulation runs until the volume-averaged inward velocity at "
            "$r=0.05$ reaches unity. Snapshots are written each time the "
            "shock front advances inward by $0.1$ in radius.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Spherical symmetry preservation:} The primary "
            "metric is the angular scatter (standard deviation / mean) of "
            "density and radial velocity in each radial bin. A well-behaved "
            "code should maintain low scatter throughout the collapse.\n"
            "  \\item \\textbf{Eulerian advection on unstructured mesh:} The "
            "fixed mesh forces all dynamics to be captured by inter-cell "
            "fluxes, stressing the Riemann solver and reconstruction.\n"
            "  \\item \\textbf{Strong shock convergence:} The collapsing shell "
            "drives a converging shock that amplifies any asymmetry.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"Dense shell: $\rho = 10$, $P = 0.1$, $v_r = -1$ for $0.9 < r < 1.0$; "
            r"ambient: $\rho = 10^{-3}$, $P = 10^{-5}$, zero velocity elsewhere. "
            r"$\gamma = 5/3$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces of the $[-1.55, 1.55]^3$ box.",
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": "MPI, 64~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Maximum angular density scatter $< 0.1$ and maximum angular "
            r"velocity scatter $< 0.1$ across all radial bins with "
            r"significant density ($\rho > 0.01$) or velocity ($|v_r| > 0.01$)."
        ),
        "plots": ["collapse_xy_density.png", "collapse_xy_internal_energy.png"],
        "plot_caption": (
            "Spherical collapse at termination ($z \\approx 0$ slice): "
            "density and internal energy scatter in the $xy$-plane."
        ),
    },
    {
        "id": "rayleigh_taylor_mpi",
        "title": "Rayleigh--Taylor Instability (3D, MPI)",
        "description": (
            "The Rayleigh--Taylor instability \\cite{rayleigh1883,taylor1950rt} "
            "develops when a heavy fluid is supported against gravity above a "
            "lighter fluid. A small sinusoidal perturbation at the interface "
            "grows exponentially in the linear regime with a growth rate "
            "determined by the Atwood number, gravitational acceleration, and "
            "perturbation wavenumber \\cite{chandrasekhar1961}.\n\n"
            "The initial condition places $\\rho_{\\mathrm{heavy}}=2$ above "
            "$\\rho_{\\mathrm{light}}=1$ in a $[0,1]^2 \\times [0,2]$ box with "
            "a flat interface at $z=1$ in exact hydrostatic equilibrium. "
            "A two-mode cosine velocity perturbation in $v_z$ (amplitude "
            "$0.03$, Gaussian-localised near the interface) seeds the "
            "instability. Constant downward gravity $g=0.5$ is applied.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Source term coupling:} The conservative gravity "
            "source term (\\texttt{ConservativeForce3D}) must correctly inject "
            "momentum and energy to maintain hydrostatic equilibrium and drive "
            "the instability.\n"
            "  \\item \\textbf{Lagrangian mesh motion with RoundCells:} The "
            "moving mesh must track the developing interface while keeping "
            "cells well-shaped.\n"
            "  \\item \\textbf{MPI scalability:} The test runs on 128~MPI tasks "
            "with $\\sim$10$^6$ cells, exercising the parallel domain "
            "decomposition and communication.\n"
            "\\end{itemize}"
        ),
        "initial_conditions": (
            r"$\rho_{\mathrm{light}}=1$ for $z < 1$, "
            r"$\rho_{\mathrm{heavy}}=2$ for $z \geq 1$ (flat interface). "
            r"Hydrostatic pressure with $P_0=10$, $g=0.5$. $\gamma=5/3$. "
            r"Velocity perturbation: "
            r"$v_z = 0.03\,[\cos(2\pi x) + \cos(2\pi y)]\,"
            r"\exp[-(z-1)^2/0.04]$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all faces of the $[0,1]^2 \\times [0,2]$ box.",
        "mesh_movement": "Lagrangian + RoundCells (\\texttt{Lagrangian3D} wrapped in \\texttt{RoundCells3D}).",
        "execution": "MPI, 128~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Fitted linear growth rate $\sigma_{\mathrm{fit}}$ (from $t=2$ "
            r"to $t=3$) within 25\% of analytical "
            r"$\sigma = \sqrt{A\,g\,k}$ where $A=1/3$, $g=0.5$, $k=2\pi$."
        ),
        "plots": [
            "rayleigh_taylor_mpi_ekz.png",
            "rayleigh_taylor_mpi_slice.png",
        ],
        "plot_caption": (
            "Rayleigh--Taylor instability: vertical kinetic energy $E_{k,z}(t)$ "
            "with best-fit exponential growth rate (left), and density slice in "
            "the $xz$-plane at $y=0.5$ at $t=3$ (right)."
        ),
    },
    {
        "id": "eulerian_diffusion_freefree_suite",
        "title": "1D Eulerian Diffusion Suite (Gray Free--Free, Cooling Limiter)",
        "description": (
            "A four-case resolution and cooling-limiter study for 1D Eulerian "
            "radiation-hydrodynamics with gray flux-limited diffusion, free--free "
            "(bremsstrahlung) absorption opacity, and Thomson scattering. "
            "Two counter-propagating streams ($v_x = \\pm 10^8\\,\\mathrm{cm\\,s^{-1}}$) "
            "collide at the domain midpoint, driving a radiative shock. "
            "The suite runs the same physics at two resolutions "
            "(512 and 32~cells) and with the cooling-time opacity limiter toggled "
            "on and off, producing a four-way comparison.\n\n"
            "\\textbf{The cooling-time opacity limiter.}\\enspace "
            "When the post-shock cooling length is under-resolved, the "
            "finite-volume scheme cannot capture the thin radiative relaxation "
            "layer behind the shock. Because energy is radiated away over fewer "
            "cells than the physics requires, the post-shock temperature in an "
            "under-resolved run is artificially elevated: the gas cools too "
            "fast relative to the compression time, violating the "
            "resolution-independent structure of the shock.\n\n"
            "The limiter addresses this by comparing two time scales in every "
            "cell that is undergoing compression:\n"
            "\\begin{enumerate}\n"
            "  \\item \\textbf{Hydrodynamic heating time:} "
            "$t_{\\mathrm{hydro}} = 1/\\max(-\\nabla\\!\\cdot\\!\\mathbf{v},\\,\\epsilon)$, "
            "measuring how fast compression is depositing energy.\n"
            "  \\item \\textbf{Radiative cooling time:} "
            "$t_{\\mathrm{cool}} = \\rho\\,e / (P_{\\mathrm{Planck}} + P_{\\mathrm{Compton}})$, "
            "where $P_{\\mathrm{Planck}} = c\\,\\sigma_P\\,(aT^4 - E_r)$ and "
            "$P_{\\mathrm{Compton}}$ is the Compton exchange rate (if Compton "
            "scattering is enabled).\n"
            "\\end{enumerate}\n"
            "The limiter activates only in cells where the compression speed "
            "exceeds a fraction of the bulk velocity "
            "($(-\\nabla\\!\\cdot\\!\\mathbf{v})\\,\\Delta x > 0.25\\,|\\mathbf{v}|$). "
            "If $t_{\\mathrm{cool}} < 2\\,t_{\\mathrm{hydro}}$, both the Planck "
            "absorption and scattering opacities are scaled down by a common "
            "factor so that the effective cooling time equals "
            "$2\\,t_{\\mathrm{hydro}}$. This preserves the relative weighting "
            "of absorption and scattering while preventing the cooling rate "
            "from exceeding what the grid can resolve.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Eulerian radiation-hydrodynamics coupling:} "
            "Tests the full hydro + gray diffusion pipeline in a "
            "shock-forming setup with counter-propagating inflows.\n"
            "  \\item \\textbf{Free--free opacity + Thomson scattering:} "
            "Validates the gray Kramers free--free absorption and constant "
            "Thomson scattering opacity paths.\n"
            "  \\item \\textbf{Resolution convergence:} Comparing 512 vs.\\ "
            "32~cells reveals the sensitivity of the post-shock structure to "
            "resolution. The high-resolution run serves as a reference.\n"
            "  \\item \\textbf{Cooling-limiter effectiveness:} With the limiter "
            "enabled, the 32-cell $T_{\\mathrm{gas}}$ peak should be closer to "
            "the 512-cell result, demonstrating that the limiter successfully "
            "prevents resolution-dependent over-cooling.\n"
            "  \\item \\textbf{Mixed radiation boundary conditions:} The "
            "radiation boundary is open on the $x$-faces and closed on $y/z$, "
            "testing a directional boundary implementation.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Under-resolved cooling layers are one of the "
            "most common failure modes in production radiation-hydrodynamics "
            "simulations. This suite directly tests the optional limiter that "
            "mitigates this problem and quantifies its effect across an "
            "order-of-magnitude resolution difference."
        ),
        "initial_conditions": (
            r"Domain $x \in [0,\,2\times10^{12}]\,\mathrm{cm}$ with uniform cells. "
            r"The gas is initialised with $\rho = 2\times10^{-13}\,\mathrm{g\,cm^{-3}}$, "
            r"$T = 2\times10^{5}\,\mathrm{K}$. "
            r"The left half has $v_x = +10^8\,\mathrm{cm\,s^{-1}}$ and the right half "
            r"$v_x = -10^8\,\mathrm{cm\,s^{-1}}$, forming a symmetric collision. "
            r"The adiabatic index is $\gamma = 5/3$." "\n\n"
            r"The four runs are:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item 512 cells, limiter \textbf{off} (reference)." "\n"
            r"  \item 512 cells, limiter \textbf{on}." "\n"
            r"  \item 32 cells, limiter \textbf{off}." "\n"
            r"  \item 32 cells, limiter \textbf{on}." "\n"
            r"\end{itemize}" "\n"
            r"The run terminates when the shock front (maximum $x$ with "
            r"$\rho > 2\rho_0$) reaches $0.75\,L$, or when $t = 9\times10^4\,\mathrm{s}$."
        ),
        "boundary_conditions": (
            "Hydrodynamics: left/right inflow ghost states matching the "
            "respective initial half-domain states; rigid walls on $y/z$ faces. "
            "Radiation: open (streaming) on the two $x$-boundary faces, closed "
            "on $y/z$."
        ),
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": (
            "MPI, 16~CPUs for 512-cell runs, 4~CPUs for 32-cell runs, "
            "submitted via SLURM (partition \\texttt{bigrun}, exclusive)."
        ),
        "pass_criteria": (
            r"Each of the four sub-runs must complete without fatal errors and "
            r"produce a valid \texttt{temperature\_profile.txt} with at least two "
            r"data points. The suite then generates four-way comparison plots "
            r"(gas temperature, radiation temperature, density, and velocity "
            r"vs.\ $x$) which are inspected visually. Quantitative acceptance:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item All profile files contain finite, positive temperatures." "\n"
            r"  \item The 32-cell limited peak $T_{\mathrm{gas}}$ is closer to "
            r"the 512-cell reference than the 32-cell unlimited peak." "\n"
            r"\end{itemize}"
        ),
        "plot_dir": "eulerian_diffusion_freefree_compare",
        "plots": [
            "temperature_vs_x_compare_512_512_limited_32_32_limited.png",
            "trad_vs_x_compare_512_512_limited_32_32_limited.png",
            "density_vs_x_compare_512_512_limited_32_32_limited.png",
            "velocity_vs_x_compare_512_512_limited_32_32_limited.png",
        ],
        "plot_caption": (
            "Gray free--free diffusion suite: four-way comparison of 512-cell "
            "(solid), 512-limited (dash-dot), 32-cell (dashed), and 32-limited "
            "(dotted) runs. Top row: gas temperature (left) and radiation "
            "temperature (right). Bottom row: density (left) and $x$-velocity "
            "(right). The cooling limiter brings the low-resolution peak "
            "$T_{\\mathrm{gas}}$ closer to the high-resolution reference."
        ),
    },
    {
        "id": "eulerian_diffusion_freefree_multigroup_suite",
        "title": "1D Eulerian Diffusion Suite (Multigroup Free--Free, Cooling Limiter)",
        "description": (
            "The multigroup counterpart of the gray free--free suite. The same "
            "symmetric collision problem is run with 32 logarithmically spaced "
            "energy groups, frequency-dependent free--free absorption, and "
            "Compton scattering. The same four-variant pattern is used "
            "(512/32~cells $\\times$ limiter on/off) and a four-way comparison "
            "of profiles is produced.\n\n"
            "\\textbf{The multigroup cooling limiter.}\\enspace "
            "In multigroup mode the cooling-time estimate sums over all energy "
            "groups: the Planck exchange is accumulated group-by-group as "
            "$c\\,\\sigma_{P,g}\\,(B_g(T) - E_g)$ and the Compton exchange is "
            "computed from the full Compton transfer matrix $S_{g,g'}$ rather "
            "than a gray approximation. When the net cooling time is shorter "
            "than $2\\,t_{\\mathrm{hydro}}$, all group-level absorption "
            "opacities are scaled by a common factor and the Compton coupling "
            "matrix entries ($S$ and $\\mathrm{d}S/\\mathrm{d}U_m$) are "
            "multiplied by a stored per-cell scale factor, ensuring that the "
            "full multigroup exchange is consistently limited.\n\n"
            "\\textbf{Code and physics aspects verified:}\n"
            "\\begin{itemize}\n"
            "  \\item \\textbf{Multigroup diffusion with frequency-dependent "
            "opacities:} Each of the 32 energy groups evolves with its own "
            "absorption coefficient, testing the group-by-group solver.\n"
            "  \\item \\textbf{Compton scattering matrix:} The full "
            "Kompaneets-based Compton transfer matrix is exercised in a "
            "dynamic (non-static) setting with strong temperature gradients.\n"
            "  \\item \\textbf{Multigroup cooling limiter:} Verifies that "
            "the per-cell Compton-limiter scale factor is correctly stored, "
            "applied to the $S$ and $\\mathrm{d}S/\\mathrm{d}U_m$ matrices, "
            "and produces the same qualitative resolution-convergence "
            "improvement seen in the gray case.\n"
            "  \\item \\textbf{Consistency with gray results:} The spatial "
            "profiles should be qualitatively similar to the gray suite, "
            "with frequency-dependent effects introducing only modest "
            "quantitative differences.\n"
            "\\end{itemize}\n\n"
            "\\textbf{Importance:} Production simulations typically use "
            "multigroup transport and Compton scattering. This suite ensures "
            "that the cooling limiter---originally developed for the gray "
            "solver---extends correctly to the frequency-dependent case "
            "including the Compton coupling terms."
        ),
        "initial_conditions": (
            r"Identical domain and gas state to the gray suite "
            r"($L = 2\times10^{12}\,\mathrm{cm}$, "
            r"$\rho = 2\times10^{-13}\,\mathrm{g\,cm^{-3}}$, "
            r"$T = 2\times10^{5}\,\mathrm{K}$, symmetric $\pm 10^8\,\mathrm{cm\,s^{-1}}$). "
            r"Radiation transport uses 32 logarithmically spaced energy groups "
            r"with free--free opacity (frequency-dependent Kramers formula) "
            r"and Compton scattering." "\n\n"
            r"The four runs are:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item 512 cells, limiter \textbf{off} (reference)." "\n"
            r"  \item 512 cells, limiter \textbf{on}." "\n"
            r"  \item 32 cells, limiter \textbf{off}." "\n"
            r"  \item 32 cells, limiter \textbf{on}." "\n"
            r"\end{itemize}"
        ),
        "boundary_conditions": (
            "Same as the gray suite: inflow on $x$, rigid on $y/z$; "
            "radiation open on $x$, closed on $y/z$."
        ),
        "mesh_movement": "Eulerian (fixed mesh).",
        "execution": (
            "MPI, 16~CPUs for 512-cell runs, 4~CPUs for 32-cell runs, "
            "submitted via SLURM (partition \\texttt{bigrun}, exclusive). "
            "Built with \\texttt{--energy\\_groups\\_num=32}."
        ),
        "pass_criteria": (
            r"Same structural criteria as the gray suite: all four sub-runs "
            r"must complete and produce valid profiles. Additionally:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item All profile files contain finite, positive temperatures." "\n"
            r"  \item The 32-cell limited peak $T_{\mathrm{gas}}$ is closer to "
            r"the 512-cell reference than the 32-cell unlimited peak." "\n"
            r"\end{itemize}"
        ),
        "plot_dir": "eulerian_diffusion_freefree_multigroup_compare",
        "plots": [
            "temperature_vs_x_compare_mg32_512_512_limited_32_32_limited.png",
            "trad_vs_x_compare_mg32_512_512_limited_32_32_limited.png",
            "density_vs_x_compare_mg32_512_512_limited_32_32_limited.png",
            "velocity_vs_x_compare_mg32_512_512_limited_32_32_limited.png",
        ],
        "plot_caption": (
            "Multigroup free--free diffusion suite (32~energy groups): "
            "four-way comparison analogous to the gray suite. "
            "Top row: gas temperature (left) and radiation temperature (right). "
            "Bottom row: density (left) and $x$-velocity (right). "
            "The cooling limiter reduces the low-resolution temperature excess "
            "as in the gray case, while the multigroup Compton coupling "
            "introduces modest quantitative differences."
        ),
    },
]


# --------------------------------------------------------------------------- #
# Metric parsers -- read achieved results from test output files
# --------------------------------------------------------------------------- #


def _parse_kv_equals(filepath: Path) -> dict[str, str]:
    """Parse a file with KEY=value lines (checker stdout logs)."""
    result = {}
    if not filepath.is_file():
        return result
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if "=" in line:
            key, _, val = line.partition("=")
            result[key.strip()] = val.strip()
    return result


def _parse_kv_space(filepath: Path) -> dict[str, str]:
    """Parse a file with 'key value' lines (metrics files)."""
    result = {}
    if not filepath.is_file():
        return result
    for line in filepath.read_text().splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    return result


def _last_number(filepath: Path) -> Optional[float]:
    """Return the last numeric value in a file."""
    if not filepath.is_file():
        return None
    text = filepath.read_text()
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1])
    except ValueError:
        return None


# Each parser returns a list of (label, achieved, threshold, passed_bool) tuples,
# or an empty list if the output files are not found.

MetricRow = tuple[str, str, str, bool]


def _read_sod_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / "sod_1d" / "sod_check.stdout.log")
    if not kv:
        return []
    rows = []
    for field, key, max_key in [
        ("Density GOF", "SOD_DENSITY_GOF", "SOD_MAX_DENSITY_GOF"),
        ("Pressure GOF", "SOD_PRESSURE_GOF", "SOD_MAX_PRESSURE_GOF"),
    ]:
        val = kv.get(key)
        thr = kv.get(max_key)
        if val is not None and thr is not None:
            passed = float(val) <= float(thr)
            rows.append((field, val, thr, passed))
    return rows


def _read_sedov_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / "sedov_3d_mpi" / "sedov_check.stdout.log")
    if not kv:
        return []
    rows = []
    for field, key, max_key in [
        ("Density rel. $L_1$", "SEDOV_DENSITY_REL_L1", "SEDOV_MAX_DENSITY_REL_L1"),
        ("Pressure rel. $L_1$", "SEDOV_PRESSURE_REL_L1", "SEDOV_MAX_PRESSURE_REL_L1"),
        ("Velocity rel. $L_1$", "SEDOV_VELOCITY_REL_L1", "SEDOV_MAX_VELOCITY_REL_L1"),
    ]:
        val = kv.get(key)
        thr = kv.get(max_key)
        if val is not None and thr is not None:
            passed = float(val) <= float(thr)
            rows.append((field, val, thr, passed))
    return rows


def _read_till_metrics(cases_dir: Path) -> list[MetricRow]:
    case = cases_dir / "till_compton"
    tgas = _last_number(case / "Tgas.txt")
    trad = _last_number(case / "Trad.txt")
    if tgas is None or trad is None:
        return []
    denom = max(tgas, trad)
    if denom <= 0:
        denom = 1e-99
    rel_diff = abs(tgas - trad) / denom
    passed = rel_diff < 1e-2
    rows: list[MetricRow] = [
        ("Final $T_{\\mathrm{gas}}$", f"{tgas:.6e}", "---", True),
        ("Final $T_{\\mathrm{rad}}$", f"{trad:.6e}", "---", True),
        ("Relative difference", f"{rel_diff:.6e}", "$ < 10^{-2}$", passed),
    ]
    # Energy conservation
    etotal_file = case / "Etotal.txt"
    if etotal_file.is_file():
        try:
            lines = [l.strip() for l in etotal_file.read_text().splitlines() if l.strip()]
            e_init = float(lines[0])
            e_final = float(lines[-1])
            if e_init > 0:
                e_rel = abs(e_final - e_init) / e_init
                e_passed = e_rel < 1e-8
                rows.append(("Energy conservation (rel.)", f"{e_rel:.6e}",
                             "$ < 10^{-8}$", e_passed))
        except (ValueError, IndexError):
            pass
    return rows


def _read_amr_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "amr_random" / "amr_random_metrics.txt")
    if not kv:
        return []
    mode = kv.get("mode", "?")
    drift = kv.get("max_drift")
    threshold = kv.get("threshold")
    pass_flag = kv.get("pass")
    if drift is None or threshold is None:
        return []
    passed = pass_flag == "1" and float(drift) <= float(threshold)
    return [
        ("Mode", mode, "---", True),
        ("Max drift", drift, threshold, passed),
    ]


def _read_voronoi_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "voronoi_volume" / "voronoi_volume_metrics.txt")
    if not kv:
        return []
    rel_error = kv.get("rel_error")
    pass_flag = kv.get("pass")
    if rel_error is None:
        return []
    thr = "$ < 10^{-10}$"
    passed = pass_flag == "1" and float(rel_error) < 1e-10
    return [("Relative volume error", rel_error, thr, passed)]


def _read_lane_metrics(cases_dir: Path, case_name: str = "lane_self_gravity") -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / case_name / "lane_gravity_metrics.txt")
    if not kv:
        return []
    metric = kv.get("final_metric")
    pass_flag = kv.get("pass")
    if metric is None:
        return []
    thr_val = 4e-2
    passed = pass_flag == "1" and abs(float(metric)) < thr_val
    return [
        ("|final\\_metric|", f"{abs(float(metric)):.6e}", f"{thr_val:.2e}", passed),
    ]


def _read_mach2_metrics(cases_dir: Path, case_name: str) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / case_name / "mach2_check.stdout.log")
    if not kv:
        return []
    rows = []
    for field, key, max_key in [
        ("Density rel. $L_1$", "MACH2_DENSITY_REL_L1", "MACH2_MAX_DENSITY_REL_L1"),
        ("Temperature rel. $L_1$", "MACH2_TEMPERATURE_REL_L1", "MACH2_MAX_TEMPERATURE_REL_L1"),
    ]:
        val = kv.get(key)
        thr = kv.get(max_key)
        if val is not None and thr is not None:
            passed = float(val) <= float(thr)
            rows.append((field, val, thr, passed))
    trad_val = kv.get("MACH2_TRAD_REL_L1")
    trad_thr = kv.get("MACH2_MAX_TRAD_REL_L1")
    if trad_val is not None and trad_thr is not None:
        passed = float(trad_val) <= float(trad_thr)
        rows.append(("$T_{\\mathrm{rad}}$ rel. $L_1$", trad_val, trad_thr, passed))
    return rows


def _read_marshak_wave_metrics(cases_dir: Path, prob_num: int) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / f"marshak_wave_{prob_num}" / "marshak_check.stdout.log")
    if not kv:
        return []
    rows = []
    for field, key in [
        ("$T_{\\mathrm{gas}}$ rel. $L_1$", "TGAS_REL_L1"),
        ("$T_{\\mathrm{rad}}$ rel. $L_1$", "TRAD_REL_L1"),
    ]:
        val = kv.get(key)
        if val is not None:
            passed = float(val) <= 1e-2
            rows.append((field, val, "1e-2", passed))
    return rows


def _read_gresho_metrics(cases_dir: Path, test_id: str) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / test_id / "gresho_check.stdout.log")
    if not kv:
        return []
    rows = []
    val = kv.get("VTHETA_REL_L1")
    if val is not None:
        thr = "0.05" if "lagrangian" in test_id else "0.1"
        passed = float(val) <= float(thr)
        rows.append(("$v_\\theta$ rel. $L_1$", val, thr, passed))
    return rows


def _read_desmore2012_mc_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / "desmore2012_mc" / "desmore2012_mc_check.stdout.log")
    if not kv:
        return []
    rows = []
    val = kv.get("DESMORE2012_MC_TGAS_L1")
    if val is not None:
        passed = float(val) <= 0.05
        rows.append(("$T_{\\mathrm{gas}}$ $L_1$ [keV]", val, "0.05", passed))
    return rows


def _read_desmore2012_mc_serial_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / "desmore2012_mc_serial" / "desmore2012_mc_serial_check.stdout.log")
    if not kv:
        return []
    rows = []
    val = kv.get("DESMORE2012_MC_TGAS_L1")
    if val is not None:
        passed = float(val) <= 0.05
        rows.append(("$T_{\\mathrm{gas}}$ $L_1$ [keV]", val, "0.05", passed))
    return rows



def _read_yee_vortex_metrics(cases_dir: Path, test_id: str) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / test_id / "vortex_check.stdout.log")
    if not kv:
        return []
    rows = []
    val = kv.get("DENSITY_L1")
    if val is not None:
        thr = "0.05"
        passed = float(val) <= float(thr)
        rows.append(("Density $L_1$", val, thr, passed))
    return rows


def _read_spherical_gauss_linear_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "spherical_gauss_linear" / "gauss_linear_metrics.txt")
    if not kv:
        return []
    rows = []
    for field, key, thr_val in [
        ("Scalar max rel. error", "scalar_max_rel_error", 1e-8),
        ("Velocity max rel. error", "velocity_max_rel_error", 0.1),
    ]:
        val = kv.get(key)
        if val is not None:
            passed = float(val) < thr_val
            rows.append((field, val, f"{thr_val:.0e}", passed))
    faces = kv.get("faces_checked")
    if faces is not None:
        rows.append(("Faces checked", faces, "> 0", int(faces) > 0))
    return rows


def _read_collapse_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "spherical_collapse" / "collapse_metrics.txt")
    if not kv:
        return []
    rows = []
    for field, key, thr_val in [
        ("Max density scatter", "max_density_scatter", 0.1),
        ("Max velocity scatter", "max_velocity_scatter", 0.1),
    ]:
        val = kv.get(key)
        if val is not None:
            passed = float(val) < thr_val
            rows.append((field, val, f"{thr_val}", passed))
    return rows


def _read_rt_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_equals(cases_dir / "rayleigh_taylor_mpi" / "rt_check.stdout.log")
    if not kv:
        return []
    rows = []
    sigma_fit = kv.get("RT_SIGMA_FIT")
    sigma_ana = kv.get("RT_SIGMA_ANALYTICAL")
    rel_err = kv.get("RT_GROWTH_RATE_REL_ERROR")
    thr = kv.get("RT_MAX_GROWTH_RATE_REL_ERROR", "0.25")
    if rel_err is not None:
        passed = float(rel_err) <= float(thr)
        rows.append(("Growth rate rel. error", rel_err, thr, passed))
    if sigma_fit is not None and sigma_ana is not None:
        rows.append(("Fitted $\\sigma$", sigma_fit, f"(analytical: {sigma_ana})", True))
    return rows


def _read_freefree_suite_metrics(cases_dir: Path,
                                  case_prefix: str) -> list[MetricRow]:
    """Read metrics for a 4-run free-free suite (gray or multigroup)."""
    rows: list[MetricRow] = []
    variants = [
        ("512", f"{case_prefix}_1d"),
        ("512 limited", f"{case_prefix}_1d_512_limited"),
        ("32", f"{case_prefix}_1d_32"),
        ("32 limited", f"{case_prefix}_1d_32_limited"),
    ]
    for label, case_name in variants:
        profile = cases_dir / case_name / "temperature_profile.txt"
        if profile.is_file():
            try:
                import numpy as np
                raw = np.loadtxt(str(profile))
                if raw.ndim == 1:
                    raw = np.expand_dims(raw, axis=0)
                n_points = raw.shape[0]
                tgas_col = raw[:, 2] if raw.shape[1] >= 3 else np.array([])
                tgas_max = float(np.max(tgas_col)) if len(tgas_col) > 0 else 0.0
                kev = 8.617333262145e-8
                rows.append((f"{label}: points", str(n_points), "$\\ge 2$",
                             n_points >= 2))
                rows.append((f"{label}: peak $T_{{\\mathrm{{gas}}}}$",
                             f"{tgas_max * kev:.4f} keV", "finite",
                             np.isfinite(tgas_max) and tgas_max > 0))
            except Exception:
                rows.append((f"{label}", "error reading profile", "---", False))
        else:
            rows.append((f"{label}", "profile not found", "---", False))
    return rows


def _read_freefree_gray_suite_metrics(cases_dir: Path) -> list[MetricRow]:
    return _read_freefree_suite_metrics(
        cases_dir, "eulerian_diffusion_freefree")


def _read_freefree_mg_suite_metrics(cases_dir: Path) -> list[MetricRow]:
    return _read_freefree_suite_metrics(
        cases_dir, "eulerian_diffusion_freefree_multigroup")


def _read_cartesian_gauss_linear_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "cartesian_gauss_linear" / "cart_gauss_linear_metrics.txt")
    if not kv:
        return []
    rows = []
    for field, key, thr_val in [
        ("Cart. scalar max rel. error", "cart_scalar_max_rel_error", 1e-6),
        ("Cart. velocity max rel. error", "cart_velocity_max_rel_error", 0.1),
    ]:
        val = kv.get(key)
        if val is not None:
            passed = float(val) < thr_val
            rows.append((field, val, f"{thr_val:.0e}", passed))
    faces = kv.get("faces_checked")
    if faces is not None:
        rows.append(("Faces checked", faces, "> 0", int(faces) > 0))
    sph_s = kv.get("sph_scalar_max_rel_error")
    cart_s = kv.get("cart_scalar_max_rel_error")
    if sph_s is not None and cart_s is not None:
        passed = float(cart_s) < float(sph_s)
        rows.append(("Cart < Sph scalar error", cart_s, sph_s, passed))
    return rows


METRIC_READERS: dict[str, object] = {
    "sod_1d": lambda cd: _read_sod_metrics(cd),
    "sedov_3d_mpi": lambda cd: _read_sedov_metrics(cd),
    "till_compton": lambda cd: _read_till_metrics(cd),
    "amr_random": lambda cd: _read_amr_metrics(cd),
    "voronoi_volume": lambda cd: _read_voronoi_metrics(cd),
    "lane_self_gravity": lambda cd: _read_lane_metrics(cd),
    "lane_self_gravity_fmm": lambda cd: _read_lane_metrics(cd, "lane_self_gravity_fmm"),
    "mach2_diffusion": lambda cd: _read_mach2_metrics(cd, "mach2_diffusion"),
    "mach2_multigroup": lambda cd: _read_mach2_metrics(cd, "mach2_multigroup"),
    "marshak_wave_1_diffusion": lambda cd: _read_marshak_wave_metrics(cd, 1),
    "marshak_wave_2_diffusion": lambda cd: _read_marshak_wave_metrics(cd, 2),
    "marshak_wave_3_diffusion": lambda cd: _read_marshak_wave_metrics(cd, 3),
    "marshak_wave_4_diffusion": lambda cd: _read_marshak_wave_metrics(cd, 4),
    "gresho_euler": lambda cd: _read_gresho_metrics(cd, "gresho_euler"),
    "gresho_lagrangian": lambda cd: _read_gresho_metrics(cd, "gresho_lagrangian"),
    "desmore2012_mc": lambda cd: _read_desmore2012_mc_metrics(cd),
    "desmore2012_mc_serial": lambda cd: _read_desmore2012_mc_serial_metrics(cd),
    "yee_vortex_64": lambda cd: _read_yee_vortex_metrics(cd, "yee_vortex_64"),
    "yee_vortex_128": lambda cd: _read_yee_vortex_metrics(cd, "yee_vortex_128"),
    "cartesian_gauss_linear": lambda cd: _read_cartesian_gauss_linear_metrics(cd),
    "spherical_collapse": lambda cd: _read_collapse_metrics(cd),
    "spherical_gauss_linear": lambda cd: _read_spherical_gauss_linear_metrics(cd),
    "rayleigh_taylor_mpi": lambda cd: _read_rt_metrics(cd),
    "eulerian_diffusion_freefree_suite": lambda cd: _read_freefree_gray_suite_metrics(cd),
    "eulerian_diffusion_freefree_multigroup_suite": lambda cd: _read_freefree_mg_suite_metrics(cd),
}


def _metrics_table_tex(rows: list[MetricRow]) -> str:
    """Return a LaTeX table showing achieved metrics vs thresholds."""
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l r r c}")
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Metric} & \\textbf{Achieved} & \\textbf{Threshold} "
        "& \\textbf{Status} \\\\"
    )
    lines.append("\\midrule")
    for label, achieved, threshold, passed in rows:
        status = (
            "{\\color{green!60!black}PASS}" if passed
            else "{\\color{red}FAIL}"
        )
        lines.append(f"{label} & {achieved} & {threshold} & {status} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# LaTeX generation
# --------------------------------------------------------------------------- #


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}

% Prevent graphicx from loading epstopdf-base (often missing in minimal
% TeX Live installs).  Declaring a fake version satisfies the dependency.
\makeatletter
\@namedef{ver@epstopdf-base.sty}{2024/01/01 stub}
\makeatother

\usepackage{graphicx}
\usepackage{geometry}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{xcolor}

\geometry{margin=2.5cm}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt plus 2pt minus 1pt}

\hypersetup{
    colorlinks=true,
    linkcolor=blue!60!black,
    urlcolor=blue!60!black,
    citecolor=green!50!black,
}

\pagestyle{headings}

\title{\textbf{RICH Regression Test Suite} \\ \large Documentation and Results}
\author{Auto-generated by \texttt{generate\_test\_report.py}}
\date{\today}

\begin{document}

\maketitle
\tableofcontents
\newpage
"""

BIBLIOGRAPHY = r"""
\begin{thebibliography}{99}

% --- V&V methodology ---

\bibitem{roache1998}
P.~J. Roache,
\textit{Verification and Validation in Computational Science and Engineering},
Hermosa Publishers, Albuquerque, NM, 1998.

\bibitem{oberkampf2010}
W.~L. Oberkampf and C.~J. Roy,
\textit{Verification and Validation in Scientific Computing},
Cambridge University Press, 2010.

\bibitem{aiaa1998}
AIAA,
``Guide for the Verification and Validation of Computational Fluid Dynamics
Simulations,'' AIAA-G-077-1998, American Institute of Aeronautics and
Astronautics, 1998.

\bibitem{asme2006}
ASME,
``Guide for Verification and Validation in Computational Solid Mechanics,''
ASME V\&V~10-2006, American Society of Mechanical Engineers, 2006.

\bibitem{salari2000}
K.~Salari and P.~Knupp,
``Code Verification by the Method of Manufactured Solutions,''
Sandia Report SAND2000-1444, Sandia National Laboratories, 2000.

\bibitem{roache2002}
P.~J. Roache,
``Code Verification by the Method of Manufactured Solutions,''
\textit{J.~Fluids Eng.}, 124(1):4--10, 2002.

\bibitem{richardson1911}
L.~F. Richardson,
``The Approximate Arithmetical Solution by Finite Differences of Physical
Problems Involving Differential Equations, with an Application to the Stresses
in a Masonry Dam,''
\textit{Philos. Trans. R. Soc. London, Ser.~A}, 210:307--357, 1911.

% --- Sod shock tube ---

\bibitem{sod1978}
G.~A. Sod,
``A Survey of Several Finite Difference Methods for Systems of Nonlinear
Hyperbolic Conservation Laws,''
\textit{J.~Comput. Phys.}, 27(1):1--31, 1978.

\bibitem{toro2009}
E.~F. Toro,
\textit{Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical
Introduction}, 3rd~ed., Springer, 2009.

% --- Sedov--Taylor blast wave ---

\bibitem{sedov1959}
L.~I. Sedov,
\textit{Similarity and Dimensional Methods in Mechanics},
Academic Press, New York, 1959.

\bibitem{taylor1950}
G.~I. Taylor,
``The Formation of a Blast Wave by a Very Intense Explosion. I.~Theoretical
Discussion,''
\textit{Proc. R. Soc. London, Ser.~A}, 201(1065):159--174, 1950.

\bibitem{kamm2007}
J.~R. Kamm and F.~X. Timmes,
``On Efficient Generation of Numerically Robust Sedov Solutions,''
Los Alamos Report LA-UR-07-2849, 2007.

% --- Till Compton equilibration ---

\bibitem{till2021}
A.~Till,
``Exploring Compton Scattering Options for Thermal Radiative Transfer,''
in \textit{Proc. ANS M\&C 2021}, LA-UR-20-29430, 2021.

\bibitem{mcgraw2023}
C.~McGraw, A.~Till, and J.~Warsa,
``A New Operator-Split Compton Scattering Method,''
\textit{J.~Comput. Phys.}, 478:111980, 2023.

% --- Lane--Emden self-gravity ---

\bibitem{chandrasekhar1939}
S.~Chandrasekhar,
\textit{An Introduction to the Study of Stellar Structure},
University of Chicago Press, 1939.

\bibitem{lane1870}
J.~Homer Lane,
``On the Theoretical Temperature of the Sun, under the Hypothesis of a Gaseous
Mass Maintaining Its Volume by Its Internal Heat, and Depending on the Laws of
Gases as Known to Terrestrial Experiment,''
\textit{Am. J. Sci.}, 2nd Series, 50(148):57--74, 1870.

% --- Mach 2 radiative shock ---

\bibitem{lowrie2008}
R.~B. Lowrie and J.~D. Edwards,
``Radiative Shock Solutions with Grey Nonequilibrium Diffusion,''
\textit{Shock Waves}, 18(2):129--143, 2008.

\bibitem{lowrie1999}
R.~B. Lowrie, J.~E. Morel, and J.~A. Hittinger,
``The Coupling of Radiation and Hydrodynamics,''
\textit{Astrophys. J.}, 521(1):432--450, 1999.

\bibitem{mihalas1984}
D.~Mihalas and B.~W. Mihalas,
\textit{Foundations of Radiation Hydrodynamics},
Oxford University Press, 1984.

\bibitem{zel2002}
Ya.~B. Zel'dovich and Yu.~P. Raizer,
\textit{Physics of Shock Waves and High-Temperature Hydrodynamic Phenomena},
Dover Publications, 2002 (reprint).

% --- Densmore 2012 (MC IMC) ---

\bibitem{densmore2012}
J.~D. Densmore, K.~G. Thompson, and T.~J. Urbatsch,
``A hybrid transport-diffusion Monte Carlo method for frequency-dependent
radiative-transfer simulations,''
\textit{J.~Comput. Phys.}, 231(20):6924--6934, 2012.

% --- Gresho vortex ---

\bibitem{liska2003}
R.~Liska and B.~Wendroff,
``Comparison of Several Difference Schemes on 1D and 2D Test Problems for the
Euler Equations,''
\textit{SIAM J.~Sci.~Comput.}, 25(3):995--1017, 2003.

% --- Rayleigh--Taylor instability ---

\bibitem{rayleigh1883}
Lord Rayleigh,
``Investigation of the Character of the Equilibrium of an Incompressible Heavy
Fluid of Variable Density,''
\textit{Proc.~London Math.~Soc.}, s1-14(1):170--177, 1883.

\bibitem{taylor1950rt}
G.~I.~Taylor,
``The Instability of Liquid Surfaces when Accelerated in a Direction
Perpendicular to their Planes.~I,''
\textit{Proc.~R.~Soc.~Lond.~A}, 201(1065):192--196, 1950.

\bibitem{chandrasekhar1961}
S.~Chandrasekhar,
\textit{Hydrodynamic and Hydromagnetic Stability},
Oxford University Press, 1961.

\end{thebibliography}
"""

POSTAMBLE = r"""
\end{document}
"""


def _section_for_test(test: dict, plots_dir: Path, cases_dir: Path,
                      out_dir: Optional[Path] = None, *,
                      level: str = "section") -> str:
    """Return the LaTeX source for one test section.

    *level* controls heading depth: "section" (default) or "subsection".
    """
    lines = []
    tid = test["id"]
    title = test["title"]

    if level == "section":
        sec, subsec = "section", "subsection*"
    else:
        sec, subsec = "subsection", "subsubsection*"

    lines.append(f"\\{sec}{{{title}}}")
    lines.append(f"\\label{{sec:{tid}}}")
    lines.append("")

    lines.append(f"\\{subsec}{{Description}}")
    lines.append(test["description"])
    lines.append("")

    lines.append(f"\\{subsec}{{Initial Conditions}}")
    lines.append(test["initial_conditions"])
    lines.append("")

    lines.append(f"\\{subsec}{{Boundary Conditions}}")
    lines.append(test["boundary_conditions"])
    lines.append("")

    lines.append(f"\\{subsec}{{Mesh Movement}}")
    lines.append(test["mesh_movement"])
    lines.append("")

    lines.append(f"\\{subsec}{{Execution}}")
    lines.append(test["execution"])
    lines.append("")

    lines.append(f"\\{subsec}{{Pass Criteria}}")
    lines.append(test["pass_criteria"])
    lines.append("")

    # Achieved results (metrics table)
    reader = METRIC_READERS.get(tid)
    if reader is not None:
        metric_rows = reader(cases_dir)
        if metric_rows:
            lines.append(f"\\{subsec}{{Achieved Results}}")
            lines.append(_metrics_table_tex(metric_rows))
            lines.append("")
        else:
            lines.append(f"\\{subsec}{{Achieved Results}}")
            lines.append(
                "\\textit{No metric output files found --- "
                "run the test suite first.}"
            )
            lines.append("")

    # Plots -- prefer PDF (vector), fall back to PNG (raster)
    # Search in plots_dir first, then in an optional per-test plot_dir under cases_dir.
    plot_files = test.get("plots", [])
    caption = test.get("plot_caption", "")
    extra_plot_dir = test.get("plot_dir")
    if plot_files:
        available = []
        search_dirs = [plots_dir]
        if extra_plot_dir:
            search_dirs.append(cases_dir / extra_plot_dir)
        for pf in plot_files:
            stem = Path(pf).stem
            found = False
            for sd in search_dirs:
                pdf_path = sd / f"{stem}.pdf"
                png_path = sd / pf
                if pdf_path.is_file():
                    if out_dir is not None:
                        available.append(Path(os.path.relpath(pdf_path.resolve(), out_dir.resolve())))
                    else:
                        available.append(pdf_path.resolve())
                    found = True
                    break
                elif png_path.is_file():
                    if out_dir is not None:
                        available.append(Path(os.path.relpath(png_path.resolve(), out_dir.resolve())))
                    else:
                        available.append(png_path.resolve())
                    found = True
                    break
        if available:
            lines.append(f"\\{subsec}{{Plots}}")
            lines.append("\\begin{figure}[htbp]")
            lines.append("  \\centering")
            n_plots = len(available)
            if n_plots == 1:
                lines.append(
                    f"  \\includegraphics[width=0.95\\textwidth]{{{available[0]}}}"
                )
            elif n_plots == 2:
                for abs_path in available:
                    lines.append(
                        f"  \\includegraphics[width=0.48\\textwidth]{{{abs_path}}}"
                    )
                    lines.append("  \\hfill")
            else:
                for idx, abs_path in enumerate(available):
                    lines.append(
                        f"  \\includegraphics[width=0.48\\textwidth]{{{abs_path}}}"
                    )
                    if idx % 2 == 0 and idx + 1 < n_plots:
                        lines.append("  \\hfill")
                    elif idx + 1 < n_plots:
                        lines.append("  \\\\[6pt]")
            if caption:
                lines.append(f"  \\caption{{{caption}}}")
            lines.append(f"  \\label{{fig:{tid}}}")
            lines.append("\\end{figure}")
            lines.append("")
        else:
            lines.append(
                f"\\{subsec}{{Plots}}"
            )
            lines.append(
                "\\textit{Plots not available --- "
                "run the test suite and the plot generator first.}"
            )
            lines.append("")
    else:
        lines.append(f"\\{subsec}{{Plots}}")
        lines.append("\\textit{This test does not produce comparison plots.}")
        lines.append("")

    lines.append("\\newpage")
    lines.append("")
    return "\n".join(lines)


_SUMMARY_TABLE_ROWS: dict[str, tuple[str, str, str, str, str]] = {
    "sod_1d": ("Sod 1D", "Serial", "1", "Eulerian", "Yes"),
    "sedov_3d_mpi": ("Sedov 3D", "MPI", "128", "Lagrangian", "Yes"),
    "gresho_euler": ("Gresho Vortex (Euler)", "Serial", "1", "Eulerian", "Yes"),
    "gresho_lagrangian": ("Gresho Vortex (Lagrangian)", "MPI", "8", "Lagrangian", "Yes"),
    "yee_vortex_64": ("Yee Vortex ($64^2$)", "MPI", "8", "Lagrangian", "Yes"),
    "yee_vortex_128": ("Yee Vortex ($128^2$)", "MPI", "16", "Lagrangian", "No"),
    "spherical_collapse": ("Spherical Collapse", "MPI", "64", "Eulerian", "Yes"),
    "rayleigh_taylor_mpi": ("Rayleigh--Taylor", "MPI", "128", "Lagrangian", "Yes"),
    "marshak_wave_1_diffusion": ("Marshak Wave 1", "Serial", "1", "Eulerian", "Yes"),
    "marshak_wave_2_diffusion": ("Marshak Wave 2", "Serial", "1", "Eulerian", "Yes"),
    "marshak_wave_3_diffusion": ("Marshak Wave 3", "Serial", "1", "Eulerian", "Yes"),
    "marshak_wave_4_diffusion": ("Marshak Wave 4", "Serial", "1", "Eulerian", "Yes"),
    "mach2_diffusion": ("Mach 2 Gray", "MPI", "8", "Eulerian", "Yes"),
    "mach2_multigroup": ("Mach 2 Multigroup", "MPI", "8", "Eulerian", "Yes"),
    "eulerian_diffusion_freefree_suite": ("Gray Free--Free Suite", "MPI", "4--16", "Eulerian", "Yes"),
    "eulerian_diffusion_freefree_multigroup_suite": ("Multigroup Free--Free Suite", "MPI", "4--16", "Eulerian", "Yes"),
    "till_compton": ("Till Compton", "Serial", "1", "Lagrangian", "Yes"),
    "desmore2012_mc": ("Densmore 2012 MC (MPI, no RW)", "MPI", "32", "Eulerian", "Yes"),
    "desmore2012_mc_serial": ("Densmore 2012 MC (serial+RW)", "Serial", "1", "Eulerian", "No"),
    "lane_self_gravity": ("Lane--Emden", "MPI", "512", "Lagrangian", "Yes"),
    "lane_self_gravity_fmm": ("Lane--Emden FMM", "MPI", "512", "Lagrangian", "Yes"),
    "amr_random": ("AMR Random", "Serial + MPI", "1 / 64", "Lagrangian + AMR", "No"),
    "voronoi_volume": ("Voronoi Volume", "Serial + MPI", "1 / 64", "Static", "No"),
    "spherical_gauss_linear": ("Spherical LSQ Gradient", "Serial", "1", "Static", "No"),
    "cartesian_gauss_linear": ("Cartesian LSQ Gradient", "Serial", "1", "Static", "No"),
}


def _summary_table() -> str:
    """Return a LaTeX summary table of all tests, grouped by category."""
    lines = []
    lines.append("\\section{Summary}")
    lines.append("\\label{sec:summary}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Overview of all regression tests.}")
    lines.append("\\label{tab:summary}")
    lines.append(
        "\\begin{tabular}{l l l l l}"
    )
    lines.append("\\toprule")
    lines.append(
        "\\textbf{Test} & \\textbf{Mode} & \\textbf{CPUs} "
        "& \\textbf{Mesh} & \\textbf{Plots} \\\\"
    )
    lines.append("\\midrule")

    for cat_idx, (cat_name, cat_ids) in enumerate(TEST_CATEGORIES):
        lines.append("\\multicolumn{5}{l}{\\textit{" + cat_name + "}} \\\\")
        for entry in cat_ids:
            if entry in TEST_GROUPS:
                _, sub_ids = TEST_GROUPS[entry]
                for tid in sub_ids:
                    row = _SUMMARY_TABLE_ROWS[tid]
                    lines.append("\\quad " + " & ".join(row) + " \\\\")
            else:
                row = _SUMMARY_TABLE_ROWS[entry]
                lines.append("\\quad " + " & ".join(row) + " \\\\")
        if cat_idx < len(TEST_CATEGORIES) - 1:
            lines.append("\\addlinespace")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\newpage")
    lines.append("")
    return "\n".join(lines)


VV_INTRODUCTION = r"""
\section{Verification and Validation}

\subsection{Background and Definitions}

Verification and Validation (V\&V) is the primary means by which confidence in
computational simulations is established \cite{roache1998,oberkampf2010}.
Although the two terms are frequently paired, they address fundamentally
distinct questions:

\begin{itemize}
  \item \textbf{Verification} asks: \emph{``Are we solving the equations
    correctly?''}  It is a \emph{mathematics} activity that checks whether the
    numerical implementation faithfully reproduces the intended mathematical
    model.  Verification is typically decomposed into \emph{code verification}
    (detecting programming errors by comparing against known analytical or
    manufactured solutions) and \emph{solution verification} (estimating
    numerical discretisation errors through convergence studies)
    \cite{roache1998,salari2000}.
  \item \textbf{Validation} asks: \emph{``Are we solving the correct
    equations?''}  It is a \emph{physics} activity that compares simulation
    predictions with experimental or observational data to assess whether the
    mathematical model adequately represents the physical phenomena of interest
    \cite{oberkampf2010}.
\end{itemize}

The distinction was formalised by the American Institute of Aeronautics and
Astronautics (AIAA) \cite{aiaa1998} and later expanded into comprehensive
guidelines by the American Society of Mechanical Engineers (ASME)
\cite{asme2006}.  Together, V\&V provides a systematic framework for
quantifying and reducing uncertainty in simulation results, an essential step
before codes are used in predictive settings such as high-energy-density physics
(HEDP), inertial confinement fusion (ICF), astrophysics, and engineering
design.

\subsection{The V\&V Process}

A robust V\&V programme typically proceeds through several stages
\cite{oberkampf2010}:

\begin{enumerate}
  \item \textbf{Code verification with exact solutions.}
    Problems that possess known closed-form or semi-analytical solutions---such
    as the Sod shock tube \cite{sod1978}, the Sedov--Taylor blast wave
    \cite{sedov1959,taylor1950}, or Lane--Emden polytropic equilibria
    \cite{chandrasekhar1939}---are used to confirm that the code reproduces the
    correct answer.  Error norms are monitored and, where feasible, the order of
    convergence is verified against the theoretical rate of the discretisation.

  \item \textbf{Code verification with manufactured solutions.}
    When analytical solutions are unavailable, the Method of Manufactured
    Solutions (MMS) \cite{roache2002,salari2000} introduces a prescribed
    solution and derives the source terms required to sustain it.  The code is
    then run with those source terms, and the discretisation error is checked
    for the expected convergence rate.

  \item \textbf{Solution verification.}
    Even after code correctness is established, discretisation and iteration
    errors remain.  Grid-convergence studies (e.g.\ Richardson extrapolation
    \cite{richardson1911,roache1998}) and time-step refinement are used to
    estimate and bound these errors in production calculations.

  \item \textbf{Validation against experimental or observational data.}
    Finally, simulations are compared with real-world measurements.  Agreement
    gives confidence; discrepancies drive model improvement.  Validation is
    inherently problem-specific and requires careful attention to experimental
    uncertainties \cite{oberkampf2010}.
\end{enumerate}

In practice, \emph{regression testing}---re-running a fixed set of verification
problems after every code change---is the mechanism that ensures
previously passing tests continue to pass.  Regression testing does not replace
formal convergence studies or validation campaigns, but it provides a
continuous, automated safety net that catches inadvertent errors early in the
development cycle.

\subsection{Role of the RICH Regression Test Suite}

The test suite documented in this report serves the \emph{code verification}
and \emph{regression testing} aspects of V\&V for the RICH code.  Each
regression test exercises a specific subset of the code's physics
capabilities---compressible hydrodynamics, flux-limited radiation diffusion,
multigroup radiation transport, Compton scattering, Newtonian self-gravity, and
adaptive mesh refinement (AMR)---and compares the results against known
analytical solutions, published reference data, or strict conservation
invariants.

The suite is designed with the following principles:

\begin{enumerate}
  \item \textbf{Correctness.}  Detect coding errors, regressions, or
    platform-dependent numerical issues as early as possible by comparing
    simulation output to known solutions.

  \item \textbf{Reproducibility.}  Every test is fully automated, deterministic,
    and produces machine-readable pass/fail signals (exit codes) so that the
    suite can be run routinely in continuous-integration or pre-merge workflows.

  \item \textbf{Coverage.}  The suite spans serial and MPI execution, Eulerian
    and Lagrangian mesh motion, single- and multi-group radiation, and multiple
    spatial dimensions, providing broad coverage of the code's feature space.

  \item \textbf{Documentation.}  For each test, this report records the problem
    definition (initial and boundary conditions, mesh movement, material
    properties), the quantitative pass criteria with explicit thresholds, and
    the achieved numerical results, forming a living document that evolves with
    the code.
\end{enumerate}

The individual tests are summarised in the table below and described in detail
in the sections that follow.  For each test we provide references to the
underlying analytical solutions or prior work on which the problem setup is
based.

\newpage
"""


def generate_tex(plots_dir: Path, cases_dir: Path,
                  out_dir: Optional[Path] = None) -> str:
    """Return the full LaTeX document as a string."""
    test_by_id = {t["id"]: t for t in TESTS}
    parts = [PREAMBLE, VV_INTRODUCTION, _summary_table()]
    for cat_name, cat_ids in TEST_CATEGORIES:
        parts.append(f"\n\\part{{{cat_name}}}\n\\setcounter{{section}}{{0}}\n\\newpage\n")
        for entry in cat_ids:
            if entry in TEST_GROUPS:
                group_title, sub_ids = TEST_GROUPS[entry]
                parts.append(f"\\section{{{group_title}}}")
                parts.append(f"\\label{{sec:{entry}}}")
                parts.append("")
                for sub_tid in sub_ids:
                    test = test_by_id[sub_tid]
                    parts.append(_section_for_test(
                        test, plots_dir, cases_dir, out_dir, level="subsection"))
            else:
                test = test_by_id[entry]
                parts.append(_section_for_test(test, plots_dir, cases_dir, out_dir))
    parts.append(BIBLIOGRAPHY)
    parts.append(POSTAMBLE)
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX report for the RICH regression test suite."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write .tex and .pdf into (default: regression_tests/).",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Only generate the .tex file; do not run pdflatex.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip running plot_results.py (assume plots already exist).",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Directory containing plot PNGs (default: regression_tests/plots/).",
    )
    args = parser.parse_args()

    root = repo_root()
    regression_dir = root / "regression_tests"

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = regression_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Plots directory
    if args.plots_dir:
        plots_dir = Path(args.plots_dir)
    else:
        plots_dir = regression_dir / "plots"

    # Step 1: generate plots
    if not args.no_plots:
        print("Generating plots via plot_results.py ...")
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_script = regression_dir / "plot_results.py"
        if plot_script.is_file():
            result = subprocess.run(
                [
                    sys.executable,
                    str(plot_script),
                    "--all",
                    "--output-dir",
                    str(plots_dir),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            print(result.stdout)
            if result.returncode != 0:
                print(
                    f"Warning: plot_results.py exited with code {result.returncode}",
                    file=sys.stderr,
                )
                if result.stderr:
                    print(result.stderr, file=sys.stderr)
        else:
            print(f"Warning: plot script not found at {plot_script}", file=sys.stderr)
    else:
        print("Skipping plot generation (--no-plots).")

    # Step 2: generate .tex
    cases_dir = regression_dir / "cases"
    tex_path = out_dir / "test_report.tex"
    tex_content = generate_tex(plots_dir, cases_dir, out_dir)
    tex_path.write_text(tex_content, encoding="utf-8")
    print(f"Wrote {tex_path}")

    # Step 3: compile to PDF
    if not args.no_compile:
        pdflatex = shutil.which("pdflatex")
        if pdflatex is None:
            print(
                "pdflatex not found; skipping PDF compilation. "
                "Install a TeX distribution to compile the .tex file.",
                file=sys.stderr,
            )
            return 0

        print("Compiling LaTeX to PDF ...")
        for pass_num in (1, 2):
            result = subprocess.run(
                [
                    pdflatex,
                    "-interaction=nonstopmode",
                    "-output-directory",
                    str(out_dir),
                    str(tex_path),
                ],
                cwd=str(out_dir),
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(
                    f"pdflatex pass {pass_num} exited with code {result.returncode}",
                    file=sys.stderr,
                )
                log_file = out_dir / "test_report.log"
                if log_file.is_file():
                    print(f"See {log_file} for details.", file=sys.stderr)
                if pass_num == 1:
                    return 1

        pdf_path = out_dir / "test_report.pdf"
        if pdf_path.is_file():
            print(f"PDF generated: {pdf_path}")
        else:
            print("Warning: PDF file not found after compilation.", file=sys.stderr)
    else:
        print("Skipping PDF compilation (--no-compile).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
