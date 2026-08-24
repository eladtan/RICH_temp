#!/bin/bash
# Graziani crooked pipe (Steinberg & Heizler 2022, Sec. 4.3).
# Submit from this directory:  sbatch run.sh
#SBATCH --job-name=CrookedPipe
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=CrookedPipe_%j.out
#SBATCH --error=CrookedPipe_%j.err
# BEGIN = started running; END = finished; FAIL = failed after start.
# Cancelled jobs (including pending scancel) do not send FAIL.
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich_cpu ]]; then
    echo "Missing executable ./rich_cpu in $(pwd)" >&2
    exit 1
fi

# Particle queues resize repeatedly. Do not let Open MPI/UCX retain the
# deregistered mappings: the default UCX allowance is 512 MiB per process,
# which can consume 8 GiB on a 16-rank node after several growth cycles.
export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
# Last 32-rank CPU job aborted in libfabric CXI unexpected-match overflow.
export FI_CXI_RX_MATCH_MODE=hybrid

# 4 nodes × 8 ranks. 10000/50/80 is ~10k cells/rank, ~10 M photons/cycle;
# GPU 80-cycle wall was 14.5 min, CPU 80-cycle estimate ~25–30 min.
exec mpirun -np "${SLURM_NTASKS:-32}" ./rich_cpu \
    10000 50 --cycles 80 --no-rw --manager new-rdma-auto
