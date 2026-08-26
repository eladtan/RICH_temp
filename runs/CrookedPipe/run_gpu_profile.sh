#!/bin/bash
# Profile one CrookedPipe GPU cycle with rocprofv3 (kernel + memcpy stats).
# Submit from this directory:  sbatch run_gpu_profile.sh
#SBATCH --job-name=CrookedPipeGPUProf
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=CrookedPipeGPUProf_%j.out
#SBATCH --error=CrookedPipeGPUProf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

if [[ "${1:-}" == --bind ]]; then
    shift
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="${local_rank}"
    export OMP_NUM_THREADS=1

    rocprof="${ROCPROF:-/opt/rocm-6.4.2/bin/rocprofv3}"
    if [[ ! -x "$rocprof" ]]; then
        rocprof="/opt/rocm/bin/rocprofv3"
    fi
    profile_dir="${CROOKEDPIPE_PROFILE_DIR:?CROOKEDPIPE_PROFILE_DIR is not set}"
    rank="${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}"
    mkdir -p "$profile_dir"
    exec "$rocprof" \
        --kernel-trace \
        --memory-copy-trace \
        --stats \
        --summary \
        --kernel-include-regex 'storm_grey_imc' \
        --output-format csv \
        -d "$profile_dir" \
        -o "rank${rank}" \
        -- "$@"
fi

run_dir="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$run_dir"
if [[ ! -x ./rich_gpu ]]; then
    echo "Missing executable ./rich_gpu in $(pwd)" >&2
    exit 1
fi

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid
export CROOKEDPIPE_PROFILE_DIR="${run_dir}/rocprof_${SLURM_JOB_ID:-manual}"
mkdir -p "$CROOKEDPIPE_PROFILE_DIR"

# Same physics as run_gpu.sh; ROCTx in the binary keeps collection off until
# cycle 80 so traces cover one late transport step, not mesh setup.
exec mpirun -np "${SLURM_NTASKS:-32}" --map-by ppr:8:node \
    "$run_dir/run_gpu_profile.sh" --bind "$run_dir/rich_gpu" \
    10000 50 --cycles 80 --profile-cycle 80 --no-rw --manager new-rdma-auto
