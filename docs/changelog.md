# Changelog

All notable changes to RICH are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Comprehensive documentation under `docs/`
- GitLab wiki pages under `wiki/`
- Regression test framework (`regression_tests/run_all.sh`)
- 14 regression tests: sod_1d, sedov_3d_mpi, till_compton, amr_random, voronoi_volume, lane_self_gravity, mach2_diffusion, mach2_multigroup, marshak_wave_1-4, gresho_euler, gresho_lagrangian
- Regression result plotting (`regression_tests/plot_results.py`)
- LaTeX test report generation (`regression_tests/generate_test_report.py`)
- Marshak wave benchmarks (Problems 1-4) with self-similar analytical validation
- Gresho vortex tests (Eulerian and Lagrangian mesh)
- Multigroup diffusion radiation transport
- Compton Matrix Monte Carlo (CMMC) module
- `build_rich.sh` canonical build script with change detection
- `--build-subdir` option for parallel test builds

### Changed
- Build system migrated from Make to CMake
- MPI tests submitted via SLURM instead of direct `mpirun`

---

## Version History

### Original Publications

- **Serial RICH**: Yalinewich, Steinberg & Sari (2015), [ApJS 216, 35](http://iopscience.iop.org/0067-0049/216/2/35/)
- **Parallel RICH**: Steinberg, Yalinewich & Sari (2015), [ApJS 216, 14](http://adsabs.harvard.edu/abs/2015ApJS..216...14S)

---

*To add entries: describe the change under the appropriate section (Added, Changed, Fixed, Removed) in `[Unreleased]`. When a version is released, rename `[Unreleased]` to the version number with the date.*
