#!/bin/bash

#SBATCH --job-name=MCRT_MM
#SBATCH --output=MCRT_MM_%j.out
#SBATCH --error=MCRT_MM_%j.err
#SBATCH --ntasks=32
#SBATCH --partition=socket
#SBATCH --exclusive
#SBATCH --nodelist=d24g75 # include d24g75

mpirun ./rich 100000