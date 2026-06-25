#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Mach2_MC2"
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o mach2_%j.out
#SBATCH -e mach2_%j.err

# ml restore intel
# export MALLOC_CHECK_=3
mpirun ./rich2 1024 mach2_mc2 25 100
