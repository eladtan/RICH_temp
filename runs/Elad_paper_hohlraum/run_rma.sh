#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Hohlraum3DRMA"
#SBATCH -n 1024
# #SBATCH --exclude=d25g[65-68]
#SBATCH --exclusive
#SBATCH -o hohlraum_%j_slurm.out
#SBATCH -e hohlraum_%j_slurm.err

# srun ulimit -c unlimited

srun --ntasks-per-node=1 systemctl start drop-caches
mpirun ./rich 0.03 --mpi-rma \
    > >(tee "hohlraum_RMA_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out") \
    2> >(tee "hohlraum_RMA_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err" >&2)
