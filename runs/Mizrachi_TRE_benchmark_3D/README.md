# TRE-3D with RICH/STORM

This directory implements the canonical genuinely three-dimensional extension
in Section 6 of `../Mizrachi_TRE_benchmark/TRE_Benchmark_Proposal.pdf` using
STORM's gray IMC engine.  The domain is the periodic cube `[0, 2*pi]^3` with
`L0=1 cm`, `sigma_a=1/cm`, `beta=1`, and `U0=1`.  The initial material and
angular radiation fields are the exact superposition of

```
k1=(1,0,0), A1=0.10
k2=(1,1,0), A2=0.07
k3=(1,1,1), A3=0.05
```

The wave vectors span all three spatial directions, so this is not a
one-dimensional extrusion.  Each mode has its own semi-analytic
full-transport decay rate.  Packets are rejection-sampled from the exact
summed anisotropic eigenfunction.  There is no source after initialization;
STORM handles emission, transport, absorption, and material feedback.

The benchmark is an ordinary RICH run: a periodic `Cartesian3D` tessellation
carries `ComputationalCell3D` and `Conserved3D` state, an `IdealGas` with
`f=a/beta` and an energy exponent of four reproduces `u = a T^4 / beta` at
unit density, and a `RadiationMCStep` advances the transport inside a
`Simulation`.  Every rank owns a Hilbert-ordered block of the cube, and the
periodic mesh connects the cube faces through MPI halos, so packets cross
`x=0` into `x=2*pi` exactly as they cross any interior face.

`analyze_tre_3d.py` fits all three material and radiation decay rates,
compares them with the semi-analytic roots, reports normalized RMS errors,
checks total-energy conservation, and writes a four-panel PNG.
By default the rate fit uses the early/mid-time interval from 10% to 50% of
the requested final time.  This keeps the smallest mode above the statistical
noise floor; pass `--rate-window TAU_MIN TAU_MAX` for a convergence study.
The executable's `--output DIR` option creates `DIR/tre_3d_history.csv` every
cycle, overwrites a rolling restart file `DIR/latest.h5`, and overwrites
`DIR/tre_3d_final_profile.csv` with the current spatial profile.  VTK dumps and
optional archival HDF5 snapshots are written to `DIR/snapshots/` at cycle 0, at
the final cycle, and every `--snapshot-every` cycles (100 by default, 0 for
first and last only).  `WriteSimulation` writes one HDF5 file per rank inside a
directory named after the snapshot plus a small master file that links to them,
which is the same format the rest of RICH reads back.  A restart snapshot
stores the whole particle census along with the cells and the mesh, so at
production settings it costs a few gigabytes; `latest.h5` is overwritten in
place so only one restart copy is retained besides any archives.  The decay-rate
analysis needs only the history CSV.

## Build and run

```bash
./build_rich.sh gnuReleaseMPI --test_name=Mizrachi_TRE_benchmark_3D \
  --build-subdir=tre_3d --jobs=16
cp build/gnuReleaseMPI/tre_3d/rich_gnuReleaseMPI runs/Mizrachi_TRE_benchmark_3D/rich
mpirun -n 8 runs/Mizrachi_TRE_benchmark_3D/rich \
  --n 12 --initial-particles 300 --new-photons 4 \
  --population 500 --dtau 0.02 --final-tau 8 \
  --snapshot-every 100 \
  --output runs/Mizrachi_TRE_benchmark_3D/example_output
python3 runs/Mizrachi_TRE_benchmark_3D/analyze_tre_3d.py \
  --input runs/Mizrachi_TRE_benchmark_3D/example_output/tre_3d_history.csv
```

The checked-in `submit.sh` uses the `bigrun` partition and writes each job to
`/data/shared/maorm/TRE3D/<jobid>` by default.  Set `TRE_OUTPUT_ROOT` to use a
different root, or set `TRE_OUTPUT_DIR` to choose an exact output directory.
The executable receives that directory through its `--output` option.  The run
is MPI parallel over the cube, so `--ntasks` scales the transport; packet
count, mesh resolution, and timestep set the accuracy.
The SLURM defaults are the higher-accuracy production settings
`n=16`, `initial-particles=300`, `new-photons=8`, `population=800`,
`dtau=0.01`, and `final-tau=8`.  `TRE_SNAPSHOT_EVERY` (default 100) is passed
as `--snapshot-every`.  Convergence jobs can override them without
editing the script, for example:

```bash
sbatch --export=ALL,TRE_N=24,TRE_FINAL_TAU=4,TRE_SEED=20260833 \
  runs/Mizrachi_TRE_benchmark_3D/submit.sh
```

For independent histories with the same time grid, use
`analyze_tre_3d_ensemble.py`.  It averages the mode histories before fitting
and records every individual fit in its JSON output.  The optional
`--rate-relative-tolerance 0.02` applies a stricter 2% rate requirement.
