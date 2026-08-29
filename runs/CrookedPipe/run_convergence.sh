#!/bin/bash
# Time-step and mesh convergence study for the crooked pipe probes.
# All cases are CPU-only DDMC and stop at the same simulation time, so the
# probe histories can be compared against Fig. 8(a) directly.
#SBATCH --job-name=cp-converge
#SBATCH --partition=batch
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=/ccs/home/maormiz/RICH/validation_runs/cp_converge_%j.out
#SBATCH --error=/ccs/home/maormiz/RICH/validation_runs/cp_converge_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

set -euo pipefail

if [[ "${1:-}" == "--bind" ]]; then
    shift
    local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}"
    unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="$local_rank"
    export OMP_NUM_THREADS=1
    exec "$@"
fi

repo_root="${RICH_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}}"
script_dir="$repo_root/runs/CrookedPipe"
binary="$repo_root/build/gnuReleaseMPI/rich_gnuReleaseMPI"
result_dir="$repo_root/validation_runs/cp_converge_${SLURM_JOB_ID}"
stop_time="${STOP_TIME:-4e-9}"
particles_per_cell="${PARTICLES_PER_CELL:-50}"

if [[ ! -x "$binary" ]]; then
    echo "Missing executable: $binary" >&2
    exit 1
fi

mkdir -p "$result_dir"
sha256sum "$binary" > "$result_dir/executable.sha256"

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

# name  points  max_dt  cycle_cap
cases=(
    "base_p10k_dt1e-9    10000  1e-9   400"
    "dt_p10k_dt1e-10     10000  1e-10  600"
    "dt_p10k_dt3e-11     10000  3e-11  900"
    "mesh_p40k_dt1e-9    40000  1e-9   400"
    "both_p40k_dt1e-10   40000  1e-10  600"
)

{
    echo "binary=$binary"
    echo "nodes=${SLURM_NNODES}"
    echo "tasks=${SLURM_NTASKS}"
    echo "particles_per_cell=$particles_per_cell"
    echo "stop_time=$stop_time"
    echo "mode=ddmc-cpu"
    for case in "${cases[@]}"; do
        echo "case=$case"
    done
} > "$result_dir/config.txt"

for case in "${cases[@]}"; do
    read -r name points max_dt cycle_cap <<< "$case"
    out="$result_dir/$name"
    mkdir -p "$out"
    echo "BEGIN $name points=$points max_dt=$max_dt $(date --iso-8601=seconds)"
    /usr/bin/time -p -o "$out/wall_time.txt" \
        mpirun -np "$SLURM_NTASKS" --map-by ppr:8:node \
        "$script_dir/run_convergence.sh" --bind "$binary" \
        "$points" "$particles_per_cell" \
        --cycles "$cycle_cap" \
        --max-dt "$max_dt" \
        --stop-time "$stop_time" \
        --snapshot-interval 0 \
        --no-snapshots \
        --no-rw \
        --ddmc \
        --ddmc-cpu \
        --manager new-rdma-auto \
        --output "$out" \
        > "$out/stdout.txt" \
        2> "$out/stderr.txt" \
        || echo "FAILED $name"
    echo "END $name $(date --iso-8601=seconds)"
done

python3 "$script_dir/compare_convergence.py" "$result_dir" \
    | tee "$result_dir/convergence.txt" || true

echo "RESULTS $result_dir"
