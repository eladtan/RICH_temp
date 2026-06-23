# AGENTS.md

RICH is a C++17/Fortran HPC code for compressible (astro)physical hydrodynamics on
a moving Voronoi mesh. It builds a single problem-specific executable per run
directory. See `docs/getting-started.md` and `docs/build-system.md` for the
canonical build/run docs.

## Cursor Cloud specific instructions

The VM snapshot already has all system dependencies installed (GNU + gfortran
toolchain, OpenMPI 4.1.6, serial **and** parallel HDF5, VTK 9.1, Boost 1.83,
zlib) plus the git submodules. Do not reinstall these by hand. The only
per-session refresh is `git submodule update --init --recursive` (the update
script handles it). Submodule URLs are SSH in `.gitmodules`; a global
`url."https://github.com/".insteadOf "git@github.com:"` rewrite (already in
`~/.gitconfig`) lets them clone over HTTPS.

### Build environment (already exported from `~/.bashrc`)

RICH's CMake (`config/find_HDF5.cmake`, `config/find_VTK.cmake`,
`config/find_Boost.cmake`) discovers dependencies from environment, not
`apt` default paths. These are set in `~/.bashrc` and are required for every
build:

- `BOOST_DIR=/usr`
- `LD_LIBRARY_PATH=/opt/rich-deps/hdf5-mpi/lib:/opt/rich-deps/vtk/lib`
- `LDFLAGS=-Wl,--no-as-needed`

Non-obvious gotchas:

- **HDF5/VTK discovery is path-pattern based.** CMake scans `LD_LIBRARY_PATH`
  for a dir whose path contains `/hdf5` and one containing `/vtk`, strips a
  trailing `/lib`, and uses `<prefix>/include` + `<prefix>/lib`. The wrapper
  prefixes live under `/opt/rich-deps/` (symlinks into the apt multiarch dirs).
  It **fatal-errors if it finds two distinct `/hdf5` prefixes**, so never add a
  second HDF5 dir to `LD_LIBRARY_PATH`.
- **One prefix serves all configs.** `/opt/rich-deps/hdf5-mpi` points at the
  *parallel* HDF5 build. Because its `H5pubconf.h` defines `H5_HAVE_PARALLEL`,
  CMake auto-adds MPI even for serial (`gnuRelease`) builds, so both
  `gnuRelease*` and `gnuReleaseMPI*` link against it. A serial binary still runs
  fine as a single process. (`/opt/rich-deps/hdf5` is a serial-only prefix kept
  for reference; do not put it in `LD_LIBRARY_PATH` at the same time as the mpi
  one.)
- **`LDFLAGS=-Wl,--no-as-needed` is mandatory.** The default build links HDF5
  statically; Ubuntu's `--as-needed` linker drops `libz` before static
  `libhdf5.a` resolves `inflateEnd`, giving a "DSO missing from command line"
  link error. CMake seeds `CMAKE_EXE_LINKER_FLAGS` from `$LDFLAGS` at configure
  time, so the flag must be present *before* the first CMake run for a config.
  `build_rich.sh` only re-runs CMake when inputs change; if you change `LDFLAGS`,
  delete `build/<config>/` to force reconfigure.

### Building, running, linting

- Build: `./build_rich.sh <config> --test_name=<run_dir>` (e.g.
  `gnuRelease` serial, `gnuReleaseMPI` for MPI). `--test_name` may be a dir under
  `runs/` or a repo-relative path like `regression_tests/cases/sod_1d`. Output
  binary is `build/<config>/rich`.
- Run (serial): `cd <run_dir> && /workspace/build/gnuRelease/rich`.
- Run (MPI): this VM has **4 cores and no SLURM**. Use
  `mpirun --oversubscribe -np <small N> .../build/gnuReleaseMPI/rich`. Do not try
  the SLURM path.
- "Lint" = the strict compiler warnings (`-Wextra -Wshadow -Werror=return-type`,
  etc. in `config/compiler_flags.cmake`); there is no separate linter. A clean
  build is the lint check.

### Regression tests

- Serial suite works directly: `./regression_tests/run_all.sh --mode serial`
  (default config `gnuRelease`). `sod_1d` is a good fast smoke test
  (`--test sod_1d`).
- MPI tests default to **SLURM** submission and many request 128 tasks
  (`SLURM_NTASKS`). With no SLURM and 4 cores, pass `--local` to use `mpirun`,
  but expect heavy oversubscription — prefer running individual small MPI cases
  manually instead of the full MPI suite.

### Python analysis tooling

System `numpy` is 2.x. The apt `python3-matplotlib` (3.6) is ABI-incompatible
with numpy 2; a numpy-2-compatible `matplotlib>=3.9` is pip-installed in
`~/.local`. Most regression checkers (e.g. `check_sod_profile.py`) only need
numpy and work as-is.
