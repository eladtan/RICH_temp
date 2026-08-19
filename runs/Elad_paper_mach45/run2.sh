#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Mach45_MC2"
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o mach45_%j.out
#SBATCH -e mach45_%j.err

ml restore intel
export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Mach45/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"
# mpirun ./rich 2000 output/mach45_mc 25 100 --profile mach45_analytic.dat
mpirun ./rich 2000 "${RICH_OUTPUT_DIR}/mach45" 25 100
