#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Hohlraum3D_P2P"
#SBATCH --exclude=d25g[65-68]
#SBATCH --exclusive
#SBATCH -o hohlraum_p2p_%j_slurm.out
#SBATCH -e hohlraum_p2p_%j_slurm.err

# srun ulimit -c unlimited

srun --ntasks-per-node=1 systemctl start drop-caches
mpirun ./rich 0.03 --p2p \
    > >(tee "hohlraum_P2P_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out") \
    2> >(tee "hohlraum_P2P_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err" >&2)
