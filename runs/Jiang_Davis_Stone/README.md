# Jiang-Davis-Stone G3 IMC analogue

This directory contains a RICH Monte Carlo analogue of simulation G3 from
Jiang, Davis & Stone, *Non-linear Evolution of Rayleigh-Taylor Instability in
a Radiation Supported Atmosphere* (`1212.1742v1.pdf`, section 4.5).

## Scope

The physical setup follows the values stated for G3:

- two-dimensional-equivalent `x-z` slab, represented by one periodic Voronoi
  cell in `y`
- `x` in `[-0.5, 0.5]`, `z` in `[-1, 1]`
- `128 x 1 x 512` cells
- lower density `rho_minus = 1`, upper density `rho_plus = 4`
- downward gravity `g = 0.1`
- pure coherent scattering with `kappa_s = 1` and `kappa_a = 0`
- radiation support fraction `alpha = 1`
- dimensionless light speed `C = 1e4`
- constant gas pressure `P = 1`
- constant dimensionless radiation flux `F_r,z = 0.1`
- radiation energy density `E_r = 1.72` at the bottom, `1.42` at the
  interface, and `0.22` at the top
- the random density perturbation from equation (15), restricted to
  `|z| < 0.5`

The default final time is `28.9`, and snapshots are forced at the paper's G3
comparison times `t = 21.6` and `t = 28.9`.

## Important limitation

This is not an exact reproduction of the Athena calculation. The paper evolves
radiation moments with an implicit two-moment solver and obtains a variable
Eddington tensor from a short-characteristics transfer solve. RICH does not
contain that VET/two-moment algorithm. This implementation instead uses
RICH/STORM implicit Monte Carlo transport with moving scatterers and momentum
coupling.

The paper also does not publish the direction-resolved initial intensity,
the two incoming boundary intensities, the random seed, or the numerical value
of the gas adiabatic index. The implementation therefore uses:

- a positive maximum-entropy angular distribution proportional to
  `exp(beta*mu)`, with `beta` chosen to match the stated local `E_r` and
  constant `F_r,z` exactly (linear P1 would become negative near the top)
- the incoming hemispheres of the same angular distribution at both vertical
  boundaries
- a recorded default random seed
- `gamma = 5/3` by default

These choices are written to the history metadata. They are numerical
substitutions, not claims about the unpublished Athena input deck.

## Build

From the repository root:

```bash
./build_rich.sh gnuReleaseMPI --test_name=Jiang_Davis_Stone --build-subdir=jds_g3 --jobs=16
cp build/gnuReleaseMPI/jds_g3/rich_gnuReleaseMPI runs/Jiang_Davis_Stone/rich
```

## Run

```bash
cd runs/Jiang_Davis_Stone
mpirun -np 8 ./rich \
  --initial-particles 4000 --boundary-particles 4000 --population 4000 \
  --output results/local
```

The physical G3 parameters are frozen. Command-line options control only
resolution for convergence or smoke tests, Monte Carlo populations, output,
the unspecified `gamma` and seed, and MPI transport manager.

`jds_g3_history.csv` is always written, so the comparison plot can always be
made; without `--output` it lands in `jds_g3_output`. VTK and HDF5 dumps,
including `init`/`final`, require `--output`. If `--output` is set without
`--output-cycles`, `--checkpoint-cycles`, or `--vtk-cycles`, the defaults
(50 / 500 / 200) are used. Use `./rich
--help` for the complete list. Driver defaults (80/8/400 packets) are only
for short tests; production packet counts match `submit.sh` (4000/4000/4000).

A small functional test can be run with:

```bash
mpirun -np 2 ./rich \
  --nx 8 --ny 1 --nz 16 \
  --initial-particles 8 --boundary-particles 4 --population 16 \
  --final-time 0.001 --output results/smoke
```

Reducing the resolution or packet populations changes the numerical
experiment and is not suitable for comparison with the paper.

## Slurm

```bash
cd runs/Jiang_Davis_Stone
sbatch submit.sh
```

sbatch submit.sh writes HDF5, VTK, and history to
`/data/shared/maorm/JDS/<job_id>` unless `JDS_OUTPUT_DIR` is set. A restart
without `JDS_OUTPUT_DIR` keeps writing in the restart file's directory.

Environment overrides: `JDS_NX`, `JDS_NZ`, `JDS_INITIAL_PARTICLES`,
`JDS_BOUNDARY_PARTICLES`, `JDS_POPULATION`, `JDS_FINAL_TIME`, `JDS_MANAGER`,
`JDS_SEED`, `JDS_GAMMA`, `JDS_RESTART`, and `JDS_OUTPUT_DIR`. The physical G3
parameters are not overridable.

## Semi-analytic comparison

The Appendix result used here is the classical incompressible rate
`n = sqrt(g k (rho_+-rho_-)/(rho_++rho_-)) = 0.614` (paper quote `0.61`).
G3 is a nonlinear, randomly seeded VET run, so this is a reference, not a
pass/fail criterion. The script also reports hydrostatic `E_r` and the paper
snapshot times `t=21.6` and `t=28.9`.

```bash
python3 compare_jds.py --reference-only
python3 compare_jds.py \
  --input results/local/jds_g3_history.csv \
  --json results/local/jds_g3_metrics.json \
  --plot results/local/jds_g3_comparison.png
```

## Output

- `jds_g3_history.csv`: paper diagnostics and run metadata
- `jds_g3_metrics.json`: fitted growth rate versus the Appendix formula
- `jds_g3_comparison.png`: optional `zmax`, `|zmin|`, and mixing plot
- `init.h5` / `init.pvtu`: first-cycle dump (skipped on restart)
- `final.h5` / `final.pvtu`: dump at the end of the run
- `latest.h5`: restartable checkpoint written every `--output-cycles` (not
  `--checkpoint-cycles`; the latter writes numbered snapshot HDF5 files)
- `snapshots/jds_g3_cycle_*.h5`: archival checkpoints
- `snapshots/jds_g3_cycle_*.pvtu`: visualization output

The history reports a 10-percent density-change displacement (`zmax`,
`zmin`) against the Eq. 15 *perturbed* initial density stored as tracer
`InitialDensity`, not the two-layer base. A volume-weighted mixing fraction,
total radiation energy, a census `Fr,z`, and global packet count are also
written. VTK `Er` is radiation energy density (`ρ E_rad`), matching
`Er_time_avg`.

`compare_jds.py` does not fit `ln(zmax)` while the 10% contour is still the
seeded `|z|<0.5` envelope, and it overlays `0.02 e^{nt}` (the paper linear
amplitude), not `zmax(0) e^{nt}`.

To resume, pass `--restart path/to/latest.h5` and keep `--output` pointing at
the original result directory (or omit `--output` so it is inferred). `submit.sh`
does that when `JDS_RESTART` is set without `JDS_OUTPUT_DIR`.

## Expected comparison

The paper's G3 run develops larger-scale structure, slower displacement
growth, and less mixing than its Eddington-closure G2 run. This implementation
can test whether an anisotropic IMC treatment shows the same qualitative
trend, but quantitative agreement is not an Athena/VET verification.
