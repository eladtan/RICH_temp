#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Hohlraum3D"
#SBATCH -n 512
#SBATCH --exclusive
#SBATCH -o hohlraum_%j_slurm.out
#SBATCH -e hohlraum_%j_slurm.err

# srun ulimit -c unlimited

ml restore intel

# srun --ntasks-per-node=1 systemctl start drop-caches
NBASE=${NBASE:-$(( 4000 * SLURM_NNODES ))}
export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Hohlraum/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"
ln -s hohlraum_${SLURM_JOB_ID}_slurm.out hohlraum_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out
ln -s hohlraum_${SLURM_JOB_ID}_slurm.err hohlraum_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err
/home/maorm/utils/memory_access /data/shared/maorm/vtune_results/9 ./rich_intel ${NBASE} --new_ibv
