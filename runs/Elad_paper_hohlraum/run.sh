#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Hohlraum3D"
#SBATCH -n 1024
#SBATCH --exclusive
#SBATCH -o hohlraum_%j_slurm.out
#SBATCH -e hohlraum_%j_slurm.err

# srun ulimit -c unlimited

# srun --ntasks-per-node=1 systemctl start drop-caches
NBASE=${NBASE:-$(( 892 * SLURM_NNODES ))}
ln -s hohlraum_${SLURM_JOB_ID}_slurm.out hohlraum_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out
ln -s hohlraum_${SLURM_JOB_ID}_slurm.err hohlraum_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err
mpirun ./rich ${NBASE} --ibv --hold-small-idle-flushes
