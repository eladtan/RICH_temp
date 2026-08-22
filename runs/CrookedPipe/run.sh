#!/bin/bash
# Graziani crooked pipe (Steinberg & Heizler 2022, Sec. 4.3).
# Submit from this directory:  sbatch run.sh
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

# Particle queues resize repeatedly. Do not let Open MPI/UCX retain the
# deregistered mappings: the default UCX allowance is 512 MiB per process,
# which can consume 8 GiB on a 16-rank node after several growth cycles.
export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0

exec mpirun -np "${SLURM_NTASKS:-512}" ./rich \
    20000 100 --cycles 200 --random-walk --manager new-rdma-auto
