#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Desmore_MC"
#SBATCH -n 16
#SBATCH --exclusive
#SBATCH -o desmore_mc_%j.out
#SBATCH -e desmore_mc_%j.err

ml restore intel
mpirun ./rich 512 output/desmore_step_mc 50 200
