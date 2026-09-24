# Mach 2 radiative shock, Lagrangian mesh

Lagrangian variant of `runs/Elad_paper_mach2` (Steinberg & Heizler 2021,
arXiv:2108.13453, Sec. 5.1; Lowrie & Edwards 2008). Same gas, opacity,
initial profile, IMC settings and output format; only the mesh motion, the
time step and the treatment of the wall cells differ.

## Why

With the Eulerian mesh the embedded hydrodynamic shock is spread over 6 to 7
cells at every resolution, so the gas radiates while it is still being heated
and the Zel'dovich temperature spike is clipped (0.261 keV at 1024 cells and
0.265 at 3072 against 0.269 analytic). Steinberg & Heizler used a Lagrangian
hydro code and recovered the spike with 1024 cells. This driver follows the
fluid so the shock is captured over a few cells and nothing advects through the
grid.

## Mesh motion

* `Lagrangian3D` point velocities, regularised by `RoundCells3D`, wrapped in
  `XOnlyMotion3D` so the single row of slab cells only moves along x.
* `RoundCells3D` compares the centre-of-mass offset with the cell radius
  (3V/4pi)^(1/3), which is about 40 slab widths here. The driver rescales its
  `chi` and `eta` so `--round-eta` is the activation threshold in units of the
  initial cell width and `--round-chi` the correction speed in units of
  c_s * offset / dx0.
* Time step: `CourantFriedrichsLewy` (CFL 0.3 times `--cfl-scale`), capped by
  the fixed step the Eulerian run used, so shocked cells (compressed 2.29x) get
  a proportionally smaller step.

## Wall AMR

The box is fixed while the points move, so the cells touching the x walls
stretch (inflow) or get squeezed (outflow; `RoundCells3D` never lets a point
leave the box). `AMR3D` is run every `--amr-interval` cycles with two
boundary-only criteria, both measured in units of the nominal wall-cell mass
(wall state density times the initial cell volume):

* split a wall cell heavier than `--amr-refine-mass` (default 2.0). The new
  point is placed so the dividing face halves the cell, which in steady inflow
  reproduces cells of exactly the initial width;
* merge a wall cell lighter than `--amr-remove-mass` (default 0.4) or whose
  point is closer than `--amr-wall-hugging` dx0 (default 0.05) to the wall.

Two small core changes support this: `AMR3D` accepts absolute new-point
positions (`absolute_refine_positions`), and `RadiationMCStep::afterMeshChange`
relocates the photons and resizes the per-cell counters after the cell list is
rebuilt.

In the default frame (upstream at rest, `mach2_analytic_ic2.dat`) only the left
wall is an inflow; the right wall is static. With the shock-frame table
(`runs/Elad_paper_mach2/mach2_analytic_ic_shock.dat`, upstream on the left)
the left wall is an inflow and the right wall an outflow, and both criteria
are exercised.

## Files and usage

```bash
sbatch build.sh                                  # builds and copies ./rich here
sbatch --dependency=afterok:<build job> run.sh   # 1024 cells on the socket partition
python3 compare.py <profile>_final.txt           # shock-frame comparison figure
```

`run.sh` takes the cell count as an optional argument. Profiles carry a sixth
column with the cell width so the mesh spacing can be inspected. The comparison
script and reference curves are the ones used for the Eulerian study in
`IMC_paper/runs/mach2`.
