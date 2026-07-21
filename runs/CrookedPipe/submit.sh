#!/bin/bash

#SBATCH --job-name=CrookedPipe
#SBATCH --output=CrookedPipe_%j.out
#SBATCH --error=CrookedPipe_%j.err
#SBATCH --ntasks=512
#SBATCH --partition=bigrun
#SBATCH --exclusive
#SBATCH --constraint=d24g

mpirun ./rich 20000 100
