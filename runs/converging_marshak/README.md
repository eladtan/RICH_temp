# Converging Marshak-wave benchmarks

This directory implements all four spherical benchmarks from Giron et al.,
*Converging Marshak waves in nonhomogeneous media*. The implementation uses
RICH's three-dimensional Voronoi mesh and STORM gray implicit Monte Carlo
transport with hydrodynamics disabled.

The material sphere occupies `r < R`. Two resolved generator layers and a
larger thermostat region surround `r = R`; that region is reset every step to
the paper's time-dependent Marshak bath temperature (Eqs. 63, 66, 69, and 72).
This realizes an isotropic blackbody bath around a genuinely 3D material
sphere while retaining RICH's box-bounded Voronoi tessellation. Only cells
whose generators lie inside the sphere are written and analyzed.

`test.cpp` selects the case using its first positional argument. It uses the
paper's opacity, density, material-energy law, system radius, start time, and
three plotted profile times. Tests 1--3 neglect radiation energy only in the
material EOS, as the paper does; transport radiation remains explicit. Test 4
uses `u_m = a_rad T^4 / 4`, so its total LTE energy is exactly
`5 a_rad T^4 / 4`.

The draft's Table II labels Test 1 with `b=2`, but the physical definition and
the surrounding text give `m=k=0`, hence `b=0`. The implementation follows the
physical opacity and homogeneous-density definition.

## Run

Build this run in the normal RICH configuration so that its executable is
named `rich`, then place or link it in this directory. A small manual run is:

```bash
mpirun -np 4 ./rich 1 \
  --radial-shells 16 --exterior-points 2000 \
  --new-photons 1 --population 8 --steps 40 \
  --output-dir results/smoke/test1

python3 compare_profiles.py \
  --test 1 \
  --input-dir results/smoke/test1 \
  --output-dir results/smoke/test1 \
  --bins 24
```

For production, submit the four-element Slurm array:

```bash
sbatch submit.sh
```

Each array element runs one benchmark on 64 MPI ranks in `bigrun`. Resolution,
packet count, and step count are explicit in `submit.sh`.

`--output` writes restartable HDF5 and VTK dumps: `init.h5`/`init.pvtu` at
cycle 0 (skipped on `--resume`) and `final.h5`/`final.pvtu` at the end, plus
numbered dumps. If `--output` is set without `--hdf5-cycles` or `--vtk-cycles`,
the defaults (500 and 200) are used. Omitting `--output` writes no VTK/HDF5.

## Comparison

`author_table_ii.csv` is a transcription of the exact numerical similarity
profiles in Table II. `compare_profiles.py` volume-averages the unstructured
3D cells into radial shells and compares material temperature and total
energy density with those data. It writes PNG/PDF plots and JSON errors for
all three profile times from the paper.
