# KRTI-S benchmark (Radiative Rayleigh-Taylor)

RICH low-Mach realization of the frozen **KRTI-S-X CGS** problem in
`KRTI_S_X_reference_package/KRTI_S_CGS_Reference_Specification.pdf`.
The incompressible semi-analytic model and compressible ideal-gas RICH model are
not mathematically identical; RICH results must be described as an
incompressible-limit calculation.

## Physics

- `H = 1 cm`, `k = 4 cm^-1`, and `Lx = pi/2 cm`
- `rho_minus = 1 g/cm^3`, `rho_plus = 3 g/cm^3`
- `g = 1e8 cm/s^2`, physical `c = 2.99792458e10 cm/s`
- `kappa_s = 2 cm^2/g`, `kappa_a = 0`
- `Ib = 1.6856350e18 erg/(cm^2 s sr)` at the lower incoming half-space
- `p*(z=0) = 1e12 dyn/cm^2`
- `k eta0 = 1e-3`

The simulation couples `HydroStep` (with constant gravity) and `RadiationMCStep`
(`withHydro=true`) on a periodic-x/y, nonperiodic-z Voronoi mesh. It loads the
stationary anisotropic radiation field and the complete velocity, pressure, and
radiation eigenmode from `KRTI_S_X_reference_package/reference`.

## Build

```bash
cd /path/to/RICH
./build_rich.sh gnuReleaseMPI --test_name=Mizrachi_KRTI --build-subdir=krti --jobs=16
cp build/gnuReleaseMPI/krti/rich_gnuReleaseMPI runs/Mizrachi_KRTI/rich
```

## Run locally

```bash
cd runs/Mizrachi_KRTI
mpirun -np 8 ./rich --case X --cells-per-lambda 24 --output krti_x_output
python3 compare_krti.py --input krti_x_output/krti_history.csv --json krti_x_output/krti_metrics.json
```

## Slurm submission

```bash
cd runs/Mizrachi_KRTI
KRTI_CASE=X sbatch submit.sh
```

Useful environment overrides are `KRTI_CELLS_PER_LAMBDA`,
`KRTI_INITIAL_PARTICLES`, `KRTI_NEW_PHOTONS`, `KRTI_POPULATION`,
`KRTI_FINAL_RT`, and `KRTI_REFERENCE_DIR`. The physical case, pressure,
single-mode perturbation, sharp interface, and periodic boundaries are frozen.

## Growth-rate analysis

`compare_krti.py` imports `krti_s_reference.py`, solves the full-transport
semi-analytic dispersion problem, fits the simulated Fourier amplitude only
below the specified linear-amplitude limit, and reports the relative growth-rate
difference and pass/fail result. Solve the reference without a simulation:

```bash
python3 compare_krti.py --reference-only
```

## Output

These files are written only if `--output` is set. Cycle cadences then use
`--output-cycles` 50, `--archive-cycles` 500, and `--vtk-cycles` 200 unless
those flags are given.

- `krti_history.csv` — fundamental interface amplitude versus time
- `init.h5` / `init.pvtu` — first-cycle dump (skipped on restart)
- `final.h5` / `final.pvtu` — dump at the end of the run
- `krti_metrics.json` — fitted and semi-analytic growth rates
- `krti_comparison.png` — optional amplitude plot

## STORM / RICH prerequisites

Before trusting growth rates, verify:

1. **Momentum coupling** is active (`RadiationMCStep` with `withHydro=true`, pure scattering opacity).
2. **Heat capacity** uses the four-parameter `IdealGas` constructor (required by `RadiationIMC::preStep`).
3. **Physical light speed** — the frozen case uses `c=2.99792458e10 cm/s`; reduced-c runs are different problems.
4. **Static scatterers** — `staticScatterers=true` retains scattering momentum deposition but disables all material-velocity/Doppler terms.
5. **Exact angular initialization** — initial packets sample `I0 + eta0 Re[Ihat exp(ikx)]`, and the lower boundary injects `pi Ib`, not the target net flux.
6. **Constant mass opacity** — STORM receives `chi=rho*kappa`, so the heavy layer has one third the photon mean free path of the light layer.
7. **Constant-intensity lower boundary** — `KRTIRadiationBoundary` samples the required flux-weighted half-space distribution for the frozen `Ib`.
8. **Periodic Voronoi + hydro** — handled in `LinearGauss3D` (face-neighbor resolution for periodic images). Do not manually expand `cells` beyond `tess.GetPointNo()`; that breaks MPI ghost exchange.

The direction-resolved `KRTI-S-X_intensity_eigenfunction.csv` is required.
Regenerate the complete 128x24x16 package with:

```bash
python3 KRTI_S_X_reference_package/krti_s_reference.py \
  --case X --nz 128 --nmu 24 --nphi 16 --outdir reference_regenerated
```
