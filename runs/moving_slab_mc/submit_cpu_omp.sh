#!/bin/bash
# Moving slab (McClarren & Gentile, 124 groups) on CPU threads: RICH built --with-gpu
# against an OpenMP-only Kokkos, 32 ranks x 9 OpenMP threads on one node.
#SBATCH --account=e-dev-2026d09-248
#SBATCH --partition=booster
#SBATCH --job-name=slab_cpu_omp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=9             # 32 x 9 = 288 cores
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

module --force purge
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 HDF5/1.14.5 VTK/9.4.1 Boost/1.86.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=spread OMP_PLACES=threads
export RICH_OUTPUT_DIR=$SCRATCH_e_dev_2026d09_248/MovingSlab/cpu-omp

srun ../../build/gnuReleaseMPI/slab-omp/rich_gnuReleaseMPI
