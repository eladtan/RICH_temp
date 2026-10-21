#!/bin/bash
# Single-node hohlraum on CPU threads: RICH built --with-gpu against an OpenMP-only Kokkos.
# The rank/thread split is chosen at submission, e.g.
#   sbatch --ntasks-per-node=1   --cpus-per-task=288 submit_kokkos_omp.sh
#   sbatch --ntasks-per-node=288 --cpus-per-task=1   submit_kokkos_omp.sh
#   sbatch --ntasks-per-node=144 --cpus-per-task=2   submit_kokkos_omp.sh
#SBATCH --account=e-dev-2026d09-248
#SBATCH --partition=booster
#SBATCH --job-name=hohlraum_omp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=288
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

module --force purge
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 HDF5/1.14.5 VTK/9.4.1 Boost/1.86.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=spread OMP_PLACES=cores
export RICH_OUTPUT_DIR=$SCRATCH_e_dev_2026d09_248/Hohlraum/${SLURM_NTASKS}x${SLURM_CPUS_PER_TASK}${RUN_TAG:+_$RUN_TAG}
echo "Ranks: $SLURM_NTASKS, OpenMP threads per rank: $OMP_NUM_THREADS, output: $RICH_OUTPUT_DIR"

# Same arguments as run.sh for one node: N_base = 892 per node.
# EXTRA_ARGS adds options, e.g. sbatch --export=ALL,EXTRA_ARGS=--no-output
srun ../../build/gnuReleaseMPI/hohl-omp/rich_gnuReleaseMPI 892 --ibv --hold-small-idle-flushes ${EXTRA_ARGS}
