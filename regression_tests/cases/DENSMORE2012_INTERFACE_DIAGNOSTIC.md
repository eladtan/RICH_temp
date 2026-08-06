# Densmore exposed IMC-DDMC interface diagnostic

These two cases deliberately remove the ten-cell geometric refinement used by
`desmore2012_mc` and `desmore2012_mc_ddmc`:

- `desmore2012_interface_mc`: pure multigroup IMC.
- `desmore2012_interface_ddmc`: the identical problem and mesh with DDMC on.

The mesh has 100 nominal 0.02-cm thin-material cells and 200 exact 0.005-cm
thick-material cells.  The opacity face is exactly at `x=2 cm`; the first thick
cell is `[2, 2.005] cm`, so there is no tiny IMC-like buffer hiding the
transport-to-DDMC transition.  Because RICH uses an unweighted Voronoi mesh,
the final thin-side generator is shifted to make that face exact.  Only the
last two thin-side widths are consequently distorted, and both cases use the
same distortion.

Example MPI runs with 30 energy groups:

```bash
./build_rich.sh intelReleaseMPI \
  --test_name=regression_tests/cases/desmore2012_interface_mc \
  --energy_groups_num=30
mpirun -np 8 ./build/intelReleaseMPI/rich

./build_rich.sh intelReleaseMPI \
  --test_name=regression_tests/cases/desmore2012_interface_ddmc \
  --energy_groups_num=30
mpirun -np 8 ./build/intelReleaseMPI/rich

python3 regression_tests/cases/compare_densmore2012_interface.py
```

Each case writes:

- `*_profile.txt`: the final temperature profile.
- `*_cells.tsv`: final geometry, material, energy, opacity, Fleck, and DDMC
  eligibility/rate information for every cell.
- `*_interface_history.tsv`: the same diagnostics every ten steps for
  `1.90 <= x <= 2.10 cm`.
- `*_global_diagnostics.tsv`: global material/radiation energy measures, the
  first DDMC cell, and cumulative DDMC/interface event counts.
- `*_rankNNNN_acceleration_debug.txt`: the full existing
  `RadiationIMC::getAccelerationDebugInfo` output around the interface on each
  rank.

The comparison script first verifies that all MC and DDMC cell faces match,
then reports signed, absolute, and volume-weighted temperature differences by
region.  It also writes `densmore2012_interface_comparison.tsv`.
