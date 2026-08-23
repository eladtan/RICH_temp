#!/bin/bash
# Graziani crooked pipe (Steinberg & Heizler 2022, Sec. 4.3).
# Submit from this directory:  sbatch submit.sh
#SBATCH --partition=bigrun
#SBATCH --job-name=CrookedPipe
#SBATCH --ntasks=512
#SBATCH --exclusive
#SBATCH --constraint=d25g
#SBATCH --time=12:00:00
#SBATCH --output=CrookedPipe_%j.out
#SBATCH --error=CrookedPipe_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd)" >&2
    exit 1
fi

output_args=()
if [[ -n "${CROOKED_PIPE_OUTPUT:-}" ]]; then
    mkdir -p "$CROOKED_PIPE_OUTPUT"
    output_args=(--output "$CROOKED_PIPE_OUTPUT")
fi

exec mpirun -np "${SLURM_NTASKS:-512}" ./rich \
    20000 100 --cycles 200 --random-walk --manager new-rdma-auto "${output_args[@]}"
