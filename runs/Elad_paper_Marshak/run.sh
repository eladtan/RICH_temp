#!/bin/bash
# Marshak wave (Steinberg & Heizler 2022, Sec. 4.1).
# Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=Marshak_IMC
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=marshak_%j.out
#SBATCH --error=marshak_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Marshak/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

run_prefix="${RICH_OUTPUT_DIR}/marshak"
Nx="${NX:-64}"

exec mpirun -np "${SLURM_NTASKS:-16}" ./rich \
    "$Nx" "$run_prefix" 1 1 mc \
    --manager new-rdma-auto
