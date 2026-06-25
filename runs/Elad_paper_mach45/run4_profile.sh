#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Mach45_MC_Profile"
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o mach45_%j.out
#SBATCH -e mach45_%j.err

ml restore intel
# mpirun ./rich 2000 output/mach45_mc 25 100 --profile mach45_analytic.dat
mpirun ./rich 2000 output4/mach45_mc 25 10 --profile mach45_analytic.dat
