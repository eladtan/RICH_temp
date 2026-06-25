#!/bin/bash
#SBATCH --job-name=MovingSlabMC
#SBATCH --output=moving_slab_mc_%j.out
#SBATCH --error=moving_slab_mc_%j.err
#SBATCH --ntasks=48
#SBATCH --nodes=16
#SBATCH --partition=bigrun
#SBATCH --exclusive
#SBATCH --time=04:00:00

# export LD_PRELOAD=/software/x86_64/5.14.0/intel/OneApi/2024.2.1/compiler/2024.2/lib/clang/19/lib/x86_64-unknown-linux-gnu/libclang_rt.asan.so
# export ASAN_OPTIONS=suppressions=${SLURM_SUBMIT_DIR}/asan.supp:detect_leaks=0:abort_on_error=1
# export I_MPI_SHM_LMT=shm

mpirun ./rich
