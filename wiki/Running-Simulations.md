# Running Simulations

This guide covers how to run RICH simulations in serial and parallel (MPI) modes.

## Overview

Each RICH simulation is defined by a `main.cpp` (or `test.cpp`) file in a run directory under `runs/`. The build system compiles this file together with the RICH source to produce a single executable.

## Serial Execution

### Build

```bash
./build_rich.sh gnuRelease --test_name=sedov_3d
```

### Run

Navigate to the run directory and execute the binary:

```bash
cd runs/sedov_3d
../../build/gnuRelease/rich
```

Output files (HDF5 snapshots, text profiles) are written to the current working directory.

### With Debug Configuration

```bash
./build_rich.sh gnuDebug --test_name=sedov_3d
cd runs/sedov_3d
../../build/gnuDebug/rich
```

## MPI Execution

### Prerequisites

Load the MPI module before building or running:

```bash
ml openmpi/4.1.6/Intel/OneApi/2024.2.1  # or your site's MPI module
```

### Build

Use an MPI-enabled configuration (any config name containing `MPI`):

```bash
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d
```

### Run via SLURM

On HPC clusters, MPI jobs must be submitted through SLURM:

```bash
cd runs/sedov_3d
sbatch --wait --exclusive --partition=bigrun --ntasks=4 \
  --wrap "mpirun -np 4 ../../build/gnuReleaseMPI/rich"
```

Key SLURM flags:

| Flag | Purpose |
|------|---------|
| `--partition=bigrun` | Required partition for MPI jobs |
| `--exclusive` | No resource contention with other jobs |
| `--ntasks=<N>` | Number of MPI processes (must match `-np`) |
| `--wait` | Block until job finishes |
| `--output=<file>` | Redirect stdout to file |
| `--error=<file>` | Redirect stderr to file |

### Example SLURM Submission Script

For production runs, create a submission script:

```bash
#!/bin/bash
#SBATCH --job-name=sedov3d
#SBATCH --partition=bigrun
#SBATCH --exclusive
#SBATCH --ntasks=128
#SBATCH --output=output_%j
#SBATCH --error=error_%j

# Load required modules
ml restore rich

# Run the simulation
mpirun -np 128 /path/to/RICH/build/gnuReleaseMPI/rich
```

Submit with:

```bash
sbatch submit.sh
```

### MPI Environment Variables

For optimal MPI performance, consider:

```bash
# Use InfiniBand transport
export UCX_TLS=ib
export UCX_IB_REG_METHODS=rcache
export UCX_IB_RCACHE_MAX_REGIONS=1000

# MPI flags
mpirun -x UCX_TLS=ib -mca btl ^openib -np 128 ./rich
```

## Run Directory Structure

A typical run directory contains:

```
runs/my_simulation/
├── main.cpp          # Simulation entry point (or test.cpp)
├── submit.sh         # Optional SLURM submission script
└── (output files generated at runtime)
```

After running, the directory may contain:

```
runs/my_simulation/
├── main.cpp
├── submit.sh
├── snap_0.h5         # HDF5 snapshots
├── snap_100.h5
├── snap_final.h5
├── counter.txt       # Restart counter (if using restart)
├── restart.h5        # Restart file (if using restart)
└── gmon.out          # Profiling data (Prof configs only)
```

## Environment Variables

The RICH binary checks for these at runtime:

| Variable | Purpose |
|----------|---------|
| `RICH_PROFILE_POINTS` | Override default point count in some test cases |

MPI-specific variables are set by the MPI runtime:

| Variable | Purpose |
|----------|---------|
| `RICH_MPI` | Compile-time flag (set by CMake for MPI configs) |

## Restart Capability

Some simulations support restart from HDF5 snapshots. The pattern used in existing runs:

1. The simulation writes `counter.txt` (current cycle number) and `restart.h5` periodically.
2. On restart, the code checks for `counter.txt` and loads the state from `restart.h5`.
3. This is configured per-simulation in the `main.cpp` file.

## Monitoring a Running Simulation

### Check SLURM Job Status

```bash
squeue -u $USER
```

### Watch Output in Real Time

```bash
tail -f output_12345  # where 12345 is the SLURM job ID
```

### Cancel a Job

```bash
scancel <job_id>
```

## Multiple Simultaneous Runs

Use `--build-subdir` to keep multiple executables:

```bash
# Build for different problems
./build_rich.sh gnuReleaseMPI --test_name=sedov_3d --build-subdir=sedov
./build_rich.sh gnuReleaseMPI --test_name=Lane --build-subdir=lane

# Run them independently
cd runs/sedov_3d && ../../build/gnuReleaseMPI/sedov/rich &
cd runs/Lane && ../../build/gnuReleaseMPI/lane/rich &
```

## Performance Tips

1. **Use Release configs** for production runs (`gnuReleaseMPI`, `intelReleaseMPI`).
2. **Intel compiler** often produces faster code for numerical workloads; benchmark with `intelReleaseMPI` vs `gnuReleaseMPI`.
3. **UCX/UCC** can significantly improve MPI performance; ensure they are available.
4. **Node-exclusive** SLURM jobs (`--exclusive`) prevent interference from other jobs.
5. **RoundCells3D** improves mesh quality but adds overhead; balance against accuracy needs.
