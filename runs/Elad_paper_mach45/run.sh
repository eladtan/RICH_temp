#!/bin/bash
# Mach-45 radiative shock (Steinberg & Heizler 2022, Sec. 5.2).
# Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=Mach45_IMC
#SBATCH --ntasks=64
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --output=mach45_%j.out
#SBATCH --error=mach45_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Mach45/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

run_prefix="${RICH_OUTPUT_DIR}/mach45"
Np="${NP:-2000}"

exec mpirun -np "${SLURM_NTASKS:-64}" ./rich \
    "$Np" "$run_prefix" 100 400 \
    --profile mach45_analytic.dat \
    --manager new-rdma-auto
