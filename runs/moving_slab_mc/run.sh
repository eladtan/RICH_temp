#!/bin/bash
# McClarren & Gentile moving slab (not in the DIMC paper; included in the IMC
# manuscript). Submit from this directory:  sbatch run.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=MovingSlabMC
#SBATCH --ntasks=48
#SBATCH --nodes=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=moving_slab_mc_%j.out
#SBATCH --error=moving_slab_mc_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/MovingSlab/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

exec mpirun -np "${SLURM_NTASKS:-48}" ./rich
