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
import math
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
# Test metadata
# --------------------------------------------------------------------------- #

TESTS = [
    {
        "id": "sod_1d",
        "title": "Sod Shock Tube (1D)",
        "description": (
            "The Sod shock tube is a classical one-dimensional Riemann problem "
            "that produces a left-moving rarefaction wave, a contact discontinuity, "
            "and a right-moving shock. It exercises the hydrodynamic Riemann solver "
            "and the PLM reconstruction scheme in a simple geometry."
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
            "The Sedov--Taylor blast wave is a point-explosion problem with a "
            "self-similar analytical solution obtained by integrating an ODE. "
            "This test exercises the 3D Voronoi mesh, Lagrangian hydrodynamics, "
            "and MPI domain decomposition with 5 million cells."
        ),
        "initial_conditions": (
            r"The domain is a cube $[-1,\,1]^3$ filled with a Voronoi tessellation "
            r"from $\sim\!5\times10^6$ random points. "
            r"Cells with centroid $r < 0.2$ receive high energy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item \textbf{Inner region} ($r < 0.2$): $\rho = 1$, $e_{\mathrm{int}} = 10^5$." "\n"
            r"  \item \textbf{Outer region} ($r \ge 0.2$): $\rho = 1$, $e_{\mathrm{int}} = 0.1$." "\n"
            r"\end{itemize}" "\n"
            r"The adiabatic index is $\gamma = 5/3$."
        ),
        "boundary_conditions": "Rigid (reflective) walls on all six faces of the cube.",
        "mesh_movement": "Lagrangian (cells move with the fluid velocity), with cell rounding.",
        "execution": "MPI, 128~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
        "pass_criteria": (
            r"Radially binned profiles of density, pressure, and velocity are "
            r"compared to the Sedov--Taylor ODE solution. "
            r"The relative $L_1$ error must satisfy:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item Density: relative $L_1 \le 0.30$." "\n"
            r"  \item Pressure: relative $L_1 \le 0.30$." "\n"
            r"  \item Velocity: relative $L_1 \le 0.30$." "\n"
            r"\end{itemize}"
        ),
        "plots": ["sedov_3d_mpi.png"],
        "plot_caption": (
            "Sedov--Taylor blast wave: radially binned numerical density (black dots) "
            "vs.\\ the ODE self-similar solution (red line)."
        ),
    },
    {
        "id": "till_compton",
        "title": "Till Compton Equilibration",
        "description": (
            "This test verifies the radiation--matter energy exchange via Compton "
            "scattering and absorption/emission in a uniform, static medium. "
            "Starting from mismatched gas and radiation temperatures, the system "
            "should relax to thermal equilibrium."
        ),
        "initial_conditions": (
            r"A single Voronoi cell in a unit domain. "
            r"The initial gas temperature is $T_{\mathrm{gas}} = 1\;\mathrm{keV}$ "
            r"and the initial radiation temperature is $T_{\mathrm{rad}} = 10\;\mathrm{keV}$. "
            r"Multigroup radiation transport is used with 32 energy groups. "
            r"Hydrodynamics is disabled."
        ),
        "boundary_conditions": "Rigid walls (irrelevant since hydrodynamics is off).",
        "mesh_movement": "Lagrangian, but the cell is effectively static (single cell, no flow).",
        "execution": "Serial, 1~CPU, direct execution. Built with \\texttt{--energy\\_groups\\_num=32}.",
        "pass_criteria": (
            r"The simulation produces time histories of $T_{\mathrm{gas}}(t)$ and "
            r"$T_{\mathrm{rad}}(t)$. The test passes when:"
            "\n"
            r"\begin{itemize}" "\n"
            r"  \item No NaN or Inf values appear in the output." "\n"
            r"  \item The final gas and radiation temperatures agree within 1\%:" "\n"
            r"        $|T_{\mathrm{gas}} - T_{\mathrm{rad}}| / "
            r"\max(T_{\mathrm{gas}},\,T_{\mathrm{rad}}) < 10^{-2}$." "\n"
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
            "conservation or interpolation bug in the AMR machinery."
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
            "known analytical box volume to machine precision."
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
            "according to the Lane--Emden equation (polytropic index $n = 1.5$) "
            "and evolves it with self-gravity. The density profile should remain "
            "close to the analytical solution."
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
        "execution": "MPI, 64~CPUs, submitted via SLURM (partition \\texttt{bigrun}, exclusive).",
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
        "id": "mach2_diffusion",
        "title": "Mach~2 Radiative Shock -- Gray Diffusion",
        "description": (
            "A Mach~2 radiative shock with gray (single-group) flux-limited "
            "diffusion. The numerical profiles are compared to the NLTE "
            "analytical solution for density, gas temperature, and radiation "
            "temperature."
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
            "The same Mach~2 radiative shock as the gray diffusion test, but using "
            "multigroup radiation transport with 32 logarithmically spaced energy "
            "groups. In addition to spatial profiles, the radiation spectrum of the "
            "hottest cell is compared to a Planck distribution at the local gas "
            "temperature."
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
    return [
        ("Final $T_{\\mathrm{gas}}$", f"{tgas:.6e}", "---", True),
        ("Final $T_{\\mathrm{rad}}$", f"{trad:.6e}", "---", True),
        ("Relative difference", f"{rel_diff:.6e}", "$ < 10^{-2}$", passed),
    ]


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


def _read_lane_metrics(cases_dir: Path) -> list[MetricRow]:
    kv = _parse_kv_space(cases_dir / "lane_self_gravity" / "lane_gravity_metrics.txt")
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


METRIC_READERS: dict[str, object] = {
    "sod_1d": lambda cd: _read_sod_metrics(cd),
    "sedov_3d_mpi": lambda cd: _read_sedov_metrics(cd),
    "till_compton": lambda cd: _read_till_metrics(cd),
    "amr_random": lambda cd: _read_amr_metrics(cd),
    "voronoi_volume": lambda cd: _read_voronoi_metrics(cd),
    "lane_self_gravity": lambda cd: _read_lane_metrics(cd),
    "mach2_diffusion": lambda cd: _read_mach2_metrics(cd, "mach2_diffusion"),
    "mach2_multigroup": lambda cd: _read_mach2_metrics(cd, "mach2_multigroup"),
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

POSTAMBLE = r"""
\end{document}
"""


def _section_for_test(test: dict, plots_dir: Path, cases_dir: Path) -> str:
    """Return the LaTeX source for one test section."""
    lines = []
    tid = test["id"]
    title = test["title"]

    lines.append(f"\\section{{{title}}}")
    lines.append(f"\\label{{sec:{tid}}}")
    lines.append("")

    # Description
    lines.append("\\subsection*{Description}")
    lines.append(test["description"])
    lines.append("")

    # Initial and boundary conditions
    lines.append("\\subsection*{Initial Conditions}")
    lines.append(test["initial_conditions"])
    lines.append("")

    lines.append("\\subsection*{Boundary Conditions}")
    lines.append(test["boundary_conditions"])
    lines.append("")

    # Mesh movement
    lines.append("\\subsection*{Mesh Movement}")
    lines.append(test["mesh_movement"])
    lines.append("")

    # Execution
    lines.append("\\subsection*{Execution}")
    lines.append(test["execution"])
    lines.append("")

    # Pass criteria
    lines.append("\\subsection*{Pass Criteria}")
    lines.append(test["pass_criteria"])
    lines.append("")

    # Achieved results (metrics table)
    reader = METRIC_READERS.get(tid)
    if reader is not None:
        metric_rows = reader(cases_dir)
        if metric_rows:
            lines.append("\\subsection*{Achieved Results}")
            lines.append(_metrics_table_tex(metric_rows))
            lines.append("")
        else:
            lines.append("\\subsection*{Achieved Results}")
            lines.append(
                "\\textit{No metric output files found --- "
                "run the test suite first.}"
            )
            lines.append("")

    # Plots -- prefer PDF (vector), fall back to PNG (raster)
    plot_files = test.get("plots", [])
    caption = test.get("plot_caption", "")
    if plot_files:
        available = []
        for pf in plot_files:
            stem = Path(pf).stem
            pdf_path = plots_dir / f"{stem}.pdf"
            png_path = plots_dir / pf
            if pdf_path.is_file():
                available.append(pdf_path.resolve())
            elif png_path.is_file():
                available.append(png_path.resolve())
        if available:
            lines.append("\\subsection*{Plots}")
            lines.append("\\begin{figure}[htbp]")
            lines.append("  \\centering")
            for abs_path in available:
                lines.append(
                    f"  \\includegraphics[width=0.95\\textwidth]{{{abs_path}}}"
                )
                lines.append("  \\\\[6pt]")
            if caption:
                lines.append(f"  \\caption{{{caption}}}")
            lines.append(f"  \\label{{fig:{tid}}}")
            lines.append("\\end{figure}")
            lines.append("")
        else:
            lines.append(
                "\\subsection*{Plots}"
            )
            lines.append(
                "\\textit{Plots not available --- "
                "run the test suite and the plot generator first.}"
            )
            lines.append("")
    else:
        lines.append("\\subsection*{Plots}")
        lines.append("\\textit{This test does not produce comparison plots.}")
        lines.append("")

    lines.append("\\newpage")
    lines.append("")
    return "\n".join(lines)


def _summary_table() -> str:
    """Return a LaTeX summary table of all tests."""
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

    rows = [
        ("Sod 1D", "Serial", "1", "Eulerian", "Yes"),
        ("Sedov 3D", "MPI", "128", "Lagrangian", "Yes"),
        ("Till Compton", "Serial", "1", "Lagrangian", "Yes"),
        ("AMR Random", "Serial + MPI", "1 / 64", "Lagrangian + AMR", "No"),
        ("Voronoi Volume", "Serial + MPI", "1 / 64", "Static", "No"),
        ("Lane--Emden", "MPI", "64", "Lagrangian", "Yes"),
        ("Mach 2 Gray", "MPI", "8", "Eulerian", "Yes"),
        ("Mach 2 Multigroup", "MPI", "8", "Eulerian", "Yes"),
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\newpage")
    lines.append("")
    return "\n".join(lines)


def generate_tex(plots_dir: Path, cases_dir: Path) -> str:
    """Return the full LaTeX document as a string."""
    parts = [PREAMBLE, _summary_table()]
    for test in TESTS:
        parts.append(_section_for_test(test, plots_dir, cases_dir))
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
    tex_content = generate_tex(plots_dir, cases_dir)
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
