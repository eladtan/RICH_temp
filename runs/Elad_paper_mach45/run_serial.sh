#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Serial_Mach45_MC"
#SBATCH -n 1
#SBATCH --exclusive
#SBATCH -o mach45S_%j.out
#SBATCH -e mach45S_%j.err

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Mach45/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"
# mpirun ./rich 4000 output/mach45_mc 25 100 --profile mach45_analytic.dat
./rich_serial 4000 "${RICH_OUTPUT_DIR}/mach45_serial" 25 100
