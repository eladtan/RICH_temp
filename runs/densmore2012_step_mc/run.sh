#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Densmore_MC"
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o densmore_mc_%j.out
#SBATCH -e densmore_mc_%j.err
#SBATCH --partition=bigrun
#SBATCH --exclude=d25g[133-134]

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/DensmoreStep/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

echo "=== RW ON ==="
mpirun ./rich 256 "${RICH_OUTPUT_DIR}/densmore_step_mc_rw" 50 200 1
echo "=== RW OFF ==="
mpirun ./rich 256 "${RICH_OUTPUT_DIR}/densmore_step_mc_norw" 50 200 0
