#!/bin/bash

#SBATCH --job-name=CrookedPipe
#SBATCH --output=CrookedPipe_%j.out
#SBATCH --error=CrookedPipe_%j.err
#SBATCH --ntasks=512
#SBATCH --partition=bigrun
#SBATCH --exclusive
#SBATCH --constraint=d25g

mpirun ./rich 20000 200 --ibv \
    > >(tee "crookedPipe_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out") \
    2> >(tee "crookedPipe_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err" >&2)
