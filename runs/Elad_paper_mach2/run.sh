#!/bin/bash
# Mach-2 radiative shock (Steinberg & Heizler 2022, Sec. 5.1).
# Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=Mach2_STORM
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=mach2_storm_%j.out
#SBATCH --error=mach2_storm_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Mach2/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

run_prefix="${RICH_OUTPUT_DIR}/mach2"

exec mpirun -np "${SLURM_NTASKS:-16}" ./rich \
    1024 "$run_prefix" 25 100 \
    --profile mach2_analytic_ic_shock.dat \
    --manager new-rdma-auto
