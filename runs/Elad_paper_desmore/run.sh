#!/bin/bash
# Densmore 2012 heterogeneous step-opacity (not in the DIMC paper; included in
# the IMC manuscript). Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=Densmore2012
#SBATCH --ntasks=32
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=Densmore2012_%j.out
#SBATCH --error=Densmore2012_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Densmore/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

exec mpirun -np "${SLURM_NTASKS:-32}" ./rich
