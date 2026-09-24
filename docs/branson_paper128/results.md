# Branson comparison with the STORM paper replay

Job 10176492 completed 63 cycles to 40 ns on d25g133–140, eight exclusive
nodes and 128 MPI ranks, one core/rank. The reused diagnostic Branson binary
uses strict O2, x86-64-v3, floating-point contraction disabled, and synthetic
scattering disabled. No STORM or upstream Branson source was changed.

## What was matched

Analytic Crooked Pipe geometry, material opacities (2000 / 0.2 cm^-1), volume
heat capacities (1e16 / 1e13 erg cm^-3 keV^-1), 0.05-keV initial material
state, empty initial packet census, 0.5-keV thin-inlet source, vacuum exterior,
0.001 packet energy cutoff, and volume-weighted radius-0.1-cm ring probes.
Time starts at zero with dt=0.01 ns, growth 1.1, cap 10 ns, stopping at 40 ns.
The recorded time/cycle arrays agree to their printed precision.

Branson's nominal global photon budget was raised from 1,858,414 to 13,195,491
per cycle: the rounded mean new-particle count in the optimized STORM paper
replay. Branson uses its native source allocation and census behavior. A matched
nominal mean is not an exact particle-work or uncertainty match.

## What could not be matched with the available implementation

Branson uses the previously tested finer Cartesian grid, 224 x 100 x 100 =
2,240,000 cells, with 0.03125 / 0.04 / 0.04 cm spacings. It cannot reproduce
the paper's curved Voronoi interface refinement (first layer 0.0001 cm), and
this checkout has no DDMC transport implementation. STORM uses 771,345 cells,
DDMC tau threshold 3, and its own emission and census population controls.
Branson's represented inlet area is 0.7744 cm^2 versus the analytic 0.785398;
its thin volume is 14.8032 cm^3 versus analytic 14.922565. No area renormalization
was applied. Static METIS decomposition also differs from STORM's redistribution.

Thus this is the requested physical/time case on Branson's own mesh, not an
identical numerical discretization or a demonstrated equal-accuracy comparison.

## Measured results

Total MPI invocation time, including startup and probe output; one trial each:

| Code/configuration | Total seconds | Seconds per cycle |
|---|---:|---:|
| Original STORM, paper settings | 906.095 | 14.3825 |
| Optimized STORM, paper settings | 794.388 | 12.6093 |
| Branson, finer Cartesian IMC | 163.105 | 2.5890 |

Branson was 4.87 times faster than optimized STORM as run. Its summed transport
time was 157.915 seconds. However, it fails to reproduce the downstream heating:

| Probe at 40 ns | Optimized STORM (keV) | Branson (keV) |
|---|---:|---:|
| P1 | 0.465341 | 0.453061 |
| P2 | 0.295113 | 0.244176 |
| P3 | 0.168312 | 0.0600317 |
| P4 | 0.106217 | 0.0486093 |
| P5 | 0.0471466 | 0.045833 |

Branson P3 and P4 never reached 0.08 keV in this interval; optimized STORM
crossed at 12.924 and 30.640 ns. P5 has not arrived in either run by 40 ns.
Energy conservation alone does not establish spatial accuracy: Branson's maximum
absolute printed radiation/material residuals were 5.55e-16 / 1.15e-13 jerk.

Actual new particles: Branson 775,788,827 versus STORM 831,315,935 (6.68% fewer).
Transported histories including carried census: Branson 788,228,907 versus STORM
2,038,806,401 (61.34% fewer). Event counts and costs also depend on mesh and
transport method. These results do not establish a speed advantage at equal
reference accuracy. Matching the paper wall resolution on a uniform Cartesian
mesh is not practical, and no claim of such a match is made here.

## Reproduction and evidence

- `run.sbatch`: launch script.
- `configuration.json`: choices and input/binary hashes.
- `/home/maorm/RICH/build/branson_paper128/paper_cartesian.xml`: exact input.
- `/home/maorm/RICH/build/branson_paper128/run_10176492/`: logs, probe CSV,
  total timing, hashes, and allocation.
- `analyze.py`, `results.json`, `comparison.png`, `comparison.pdf`.
- STORM paired source results: `../cp_paper_regression/findings.md`.
