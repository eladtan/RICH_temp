#!/bin/bash

#SBATCH --job-name=CrookedPipeP2P
#SBATCH --output=CrookedPipeP2P_%j.out
#SBATCH --error=CrookedPipeP2P_%j.err
#SBATCH --ntasks=512
#SBATCH --partition=bigrun
#SBATCH --exclusive
#SBATCH --constraint=d25g

mpirun ./rich 20000 200 --p2p \
    > >(tee "crookedPipe_p2p_${SLURM_JOB_ID}_n${SLURM_NTASKS}.out") \
    2> >(tee "crookedPipe_p2p_${SLURM_JOB_ID}_n${SLURM_NTASKS}.err" >&2)
