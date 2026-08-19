#!/bin/bash
# Crooked pipe with the P2P Monte Carlo manager.
# Submit from this directory:  sbatch submit_p2p.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=CrookedPipeP2P
#SBATCH --ntasks=512
#SBATCH --exclusive
#SBATCH --constraint=d25g
#SBATCH --time=12:00:00
#SBATCH --output=CrookedPipeP2P_%j.out
#SBATCH --error=CrookedPipeP2P_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/CrookedPipe/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

exec mpirun -np "${SLURM_NTASKS:-512}" ./rich \
    20000 200 --output "$RICH_OUTPUT_DIR" --p2p
