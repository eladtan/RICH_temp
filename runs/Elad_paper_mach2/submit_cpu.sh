#!/bin/bash
# CPU-only counterpart of submit_gpu.sh: same node type, rank layout and run arguments.
#SBATCH --account=e-dev-2026d09-248
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16          # same rank count as the GPU run
#SBATCH --cpus-per-task=18            # 16 x 18 = 288 cores
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

module --force purge
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 HDF5/1.14.5 VTK/9.4.1 Boost/1.86.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=spread OMP_PLACES=threads
export RICH_OUTPUT_DIR=$SCRATCH_e_dev_2026d09_248/Mach2/cpu

srun ../../build/gnuReleaseMPI/cpu/rich_gnuReleaseMPI 1024 "$RICH_OUTPUT_DIR/mach2"
