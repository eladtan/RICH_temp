#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Serial_Mach45_MC"
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -o mach45S_%j.out
#SBATCH -e mach45S_%j.err

# mpirun ./rich 2000 output/mach45_mc 25 100 --profile mach45_analytic.dat
./rich_serial 2000 ~/shared/Mach45/mach45_mc_serial 25 100
