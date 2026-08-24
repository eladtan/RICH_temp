#!/bin/bash
# Graziani crooked pipe (Steinberg & Heizler 2022, Sec. 4.3).
# Submit from this directory:  sbatch submit.sh
#SBATCH --job-name=CrookedPipe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=56
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=00:30:00
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

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0

exec mpirun -np "${SLURM_NTASKS:-112}" ./rich \
    10000 50 --cycles 40 --random-walk --manager new-rdma-auto "${output_args[@]}"
