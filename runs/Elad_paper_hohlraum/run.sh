#!/bin/bash
# McClarren & Urbatsch cylindrical hohlraum (Steinberg & Heizler 2022, Sec. 4.2).
# Submit from this directory:  sbatch run.sh
# DIMC Fig. 5 is at t = 10 ns; this driver currently stops at 3 ns (see test.cpp).
#SBATCH --partition=bigrun
#SBATCH --job-name=Hohlraum3D
#SBATCH --ntasks=1024
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=hohlraum_%j.out
#SBATCH --error=hohlraum_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/Hohlraum/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

nbase="${NBASE:-$((892 * ${SLURM_NNODES:-1}))}"

exec mpirun -np "${SLURM_NTASKS:-1024}" ./rich \
    "$nbase" --ibv --hold-small-idle-flushes
