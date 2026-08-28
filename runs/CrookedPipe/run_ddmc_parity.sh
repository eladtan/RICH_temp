#!/bin/bash
# Paired CPU/GPU DDMC validation in one Frontier allocation.
# Both runs use the same executable, MPI layout, mesh, particles, and cycles.
#SBATCH --job-name=ddmc-parity
#SBATCH --partition=service
#SBATCH --qos=develop
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --account=ast246
#SBATCH --time=01:00:00
#SBATCH --output=/ccs/home/maormiz/RICH/validation_runs/ddmc_parity_%j.out
#SBATCH --error=/ccs/home/maormiz/RICH/validation_runs/ddmc_parity_%j.err
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
result_dir="$repo_root/validation_runs/ddmc_parity_${SLURM_JOB_ID}"
gpu_inner_steps="${GPU_INNER_STEPS:-64}"
gpu_min_launch="${GPU_MIN_LAUNCH:-1024}"
gpu_hold_skips="${GPU_HOLD_SKIPS:-64}"

if [[ ! -x "$binary" ]]; then
    echo "Missing executable: $binary" >&2
    exit 1
fi

mkdir -p "$result_dir/cpu" "$result_dir/gpu" \
    "$result_dir/smoke_cpu" "$result_dir/smoke_gpu"
sha256sum "$binary" > "$result_dir/executable.sha256"
{
    echo "binary=$binary"
    echo "nodes=${SLURM_NNODES}"
    echo "tasks=${SLURM_NTASKS}"
    echo "points=10000"
    echo "particles_per_cell=50"
    echo "cycles=40"
    echo "random_walk=false"
    echo "manager=new-rdma-auto"
    echo "gpu_inner_steps=$gpu_inner_steps"
    echo "gpu_min_launch=$gpu_min_launch"
    echo "gpu_hold_skips=$gpu_hold_skips"
} > "$result_dir/config.txt"

export OMPI_MCA_mpi_leave_pinned=0
export UCX_IB_RCACHE_MAX_UNRELEASED=0
export OMP_NUM_THREADS=1
export FI_CXI_RX_MATCH_MODE=hybrid

common_args=(
    10000 50
    --cycles 40
    --snapshot-interval 0
    --no-snapshots
    --no-rw
    --ddmc
    --gpu-inner-steps "$gpu_inner_steps"
    --gpu-min-launch "$gpu_min_launch"
    --gpu-hold-skips "$gpu_hold_skips"
    --manager new-rdma-auto
)

run_case()
{
    local mode="$1"
    shift
    echo "BEGIN ${mode} $(date --iso-8601=seconds)"
    /usr/bin/time -p -o "$result_dir/${mode}/wall_time.txt" \
        mpirun -np "$SLURM_NTASKS" --map-by ppr:8:node \
        "$script_dir/run_ddmc_parity.sh" --bind "$binary" \
        "${common_args[@]}" --output "$result_dir/${mode}" "$@" \
        > "$result_dir/${mode}/stdout.txt" \
        2> "$result_dir/${mode}/stderr.txt"
    echo "END ${mode} $(date --iso-8601=seconds)"
}

run_smoke()
{
    local mode="$1"
    shift
    mpirun -np "$SLURM_NTASKS" --map-by ppr:8:node \
        "$script_dir/run_ddmc_parity.sh" --bind "$binary" \
        1000 20 --cycles 1 --snapshot-interval 0 --no-snapshots \
        --no-rw --ddmc \
        --gpu-inner-steps "$gpu_inner_steps" \
        --gpu-min-launch "$gpu_min_launch" \
        --gpu-hold-skips "$gpu_hold_skips" \
        --manager new-rdma-auto \
        --output "$result_dir/${mode}" "$@" \
        > "$result_dir/${mode}/stdout.txt" \
        2> "$result_dir/${mode}/stderr.txt"
    test -s "$result_dir/${mode}/crookedpipe_metrics.txt"
}

echo "BEGIN smoke checks $(date --iso-8601=seconds)"
run_smoke smoke_cpu --ddmc-cpu
run_smoke smoke_gpu --ddmc-device
python3 "$script_dir/compare_ddmc_cpu_gpu.py" \
    "$result_dir/smoke_cpu" "$result_dir/smoke_gpu" \
    --report "$result_dir/smoke_correctness.json" \
    > "$result_dir/smoke_correctness.txt"
echo "END smoke checks $(date --iso-8601=seconds)"

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    echo "RESULTS $result_dir"
    exit 0
fi

if [[ "${GPU_ONLY:-0}" == "1" ]]; then
    run_case gpu --ddmc-device
    echo "RESULTS $result_dir"
    exit 0
fi

run_case cpu --ddmc-cpu
run_case gpu --ddmc-device

python3 "$script_dir/compare_ddmc_cpu_gpu.py" \
    "$result_dir/cpu" "$result_dir/gpu" \
    --report "$result_dir/correctness.json" \
    | tee "$result_dir/correctness.txt"

python3 "$script_dir/compare_dimc.py" \
    "$result_dir/cpu/crookedpipe_probes.txt" \
    --output "$result_dir/cpu_vs_reference.png"
python3 "$script_dir/compare_dimc.py" \
    "$result_dir/gpu/crookedpipe_probes.txt" \
    --output "$result_dir/gpu_vs_reference.png"

echo "RESULTS $result_dir"
