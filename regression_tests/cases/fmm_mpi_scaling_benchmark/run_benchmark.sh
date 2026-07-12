#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <rich-binary>" >&2
    exit 2
fi

RICH_BIN="$1"
REPEATS="${FMM_MPI_BENCH_REPEATS:-2}"
FMM_MAX_REMOTE_MIB="${FMM_MPI_BENCH_MAX_REMOTE_MIB:-512}"
SMALL_PARTICLES=1000000
LARGE_PARTICLES=10000000
ALLOCATED_NODES="${SLURM_JOB_NUM_NODES:-0}"
ALLOCATED_TASKS="${SLURM_NTASKS:-0}"

if [[ ! -x "${RICH_BIN}" ]]; then
    echo "Benchmark binary is not executable: ${RICH_BIN}" >&2
    exit 2
fi
if ! [[ "${REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FMM_MPI_BENCH_REPEATS must be a positive integer" >&2
    exit 2
fi
if ! [[ "${FMM_MAX_REMOTE_MIB}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FMM_MPI_BENCH_MAX_REMOTE_MIB must be a positive integer" >&2
    exit 2
fi
if (( ALLOCATED_NODES < 16 )); then
    echo "fmm_mpi_scaling_benchmark requires at least 16 allocated nodes; got ${ALLOCATED_NODES}" >&2
    exit 2
fi
if (( ALLOCATED_TASKS < 16 || ALLOCATED_TASKS % 16 != 0 )); then
    echo "SLURM_NTASKS must be a positive multiple of 16; got ${ALLOCATED_TASKS}" >&2
    exit 2
fi

RANKS_PER_NODE=$((ALLOCATED_TASKS / 16))
if (( RANKS_PER_NODE < 1 )); then
    echo "Could not derive a positive ranks-per-node value" >&2
    exit 2
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The benchmark is single-threaded per MPI rank.  Limit glibc arenas so many
# ranks on one node do not retain independent multi-arena high-water marks.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

rm -f fmm_mpi_scaling_benchmark_*.txt \
      fmm_mpi_scaling_benchmark_metrics.txt

run_case() {
    local particles="$1"
    local nodes="$2"
    local ranks=$((nodes * RANKS_PER_NODE))
    local output="fmm_mpi_scaling_benchmark_${particles}_nodes${nodes}.txt"

    echo "=== particles=${particles}, nodes=${nodes}, ranks=${ranks}, ranks_per_node=${RANKS_PER_NODE}, repeats=${REPEATS}, fmm_max_remote_mib=${FMM_MAX_REMOTE_MIB} ==="
    if ! mpirun -np "${ranks}" \
        --map-by "ppr:${RANKS_PER_NODE}:node" \
        --bind-to core \
        "${RICH_BIN}" \
        --particles "${particles}" \
        --expected-nodes "${nodes}" \
        --expected-ranks-per-node "${RANKS_PER_NODE}" \
        --repeats "${REPEATS}" \
        --fmm-max-remote-mib "${FMM_MAX_REMOTE_MIB}" \
        --output "${output}"; then
        echo "Benchmark MPI subcase failed before producing ${output}." >&2
        echo "Last benchmark_stage line identifies the failing phase." >&2
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then
            echo "Inspect scheduler memory with:" >&2
            echo "  sacct -j ${SLURM_JOB_ID} --format=JobID,State,ExitCode,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocTRES%80" >&2
        fi
        exit 1
    fi

    if [[ ! -s "${output}" ]] || ! grep -q '^pass 1$' "${output}"; then
        echo "Benchmark subcase failed: ${output}" >&2
        exit 1
    fi
}

run_case "${SMALL_PARTICLES}" 8
run_case "${SMALL_PARTICLES}" 16
run_case "${LARGE_PARTICLES}" 8
run_case "${LARGE_PARTICLES}" 16

{
    echo "columns particles expected_nodes ranks unique_nodes ranks_per_node repeats local_particles_min local_particles_max fmm_best_max_seconds fmm_mean_max_seconds quadrupole_best_max_seconds quadrupole_mean_max_seconds fmm_probe_scaled_error quadrupole_probe_scaled_error quadrupole_over_fmm_speedup fmm_particles_per_second quadrupole_particles_per_second fmm_bytes_sent fmm_bytes_received fmm_peak_remote_bytes fmm_peak_process_bytes quadrupole_walk_max_seconds fmm_checksum quadrupole_checksum finite run_pass"
    for particles in "${SMALL_PARTICLES}" "${LARGE_PARTICLES}"; do
        for nodes in 8 16; do
            grep '^row ' "fmm_mpi_scaling_benchmark_${particles}_nodes${nodes}.txt"
        done
    done
} > fmm_mpi_scaling_benchmark_metrics.txt

summary_file="$(mktemp)"
awk -v small="${SMALL_PARTICLES}" -v large="${LARGE_PARTICLES}" '
$1 == "row" {
    count++
    particles = $2
    nodes = $3
    fmm[particles, nodes] = $10
    quad[particles, nodes] = $12
    if (count == 1) ranks_per_node = $6
    if ($6 != ranks_per_node) placement_ok = 0
    run_pass = run_pass && ($27 == 1)
    if (count == 1) {
        run_pass = ($27 == 1)
        placement_ok = 1
    }
}
END {
    complete = count == 4 && placement_ok &&
        ((small SUBSEP 8) in fmm) && ((small SUBSEP 16) in fmm) &&
        ((large SUBSEP 8) in fmm) && ((large SUBSEP 16) in fmm)
    if (!complete) {
        print "row_count", count
        print "pass", 0
        exit 1
    }

    fmm_small_speedup = fmm[small, 8] / fmm[small, 16]
    fmm_large_speedup = fmm[large, 8] / fmm[large, 16]
    quad_small_speedup = quad[small, 8] / quad[small, 16]
    quad_large_speedup = quad[large, 8] / quad[large, 16]

    print "row_count", count
    print "small_particles", small
    print "large_particles", large
    print "ranks_per_node", ranks_per_node
    print "fmm_small_8_to_16_speedup", fmm_small_speedup
    print "fmm_small_8_to_16_efficiency", fmm_small_speedup / 2.0
    print "fmm_large_8_to_16_speedup", fmm_large_speedup
    print "fmm_large_8_to_16_efficiency", fmm_large_speedup / 2.0
    print "quadrupole_small_8_to_16_speedup", quad_small_speedup
    print "quadrupole_small_8_to_16_efficiency", quad_small_speedup / 2.0
    print "quadrupole_large_8_to_16_speedup", quad_large_speedup
    print "quadrupole_large_8_to_16_efficiency", quad_large_speedup / 2.0
    print "pass", (run_pass ? 1 : 0)
    if (!run_pass) exit 1
}
' fmm_mpi_scaling_benchmark_metrics.txt > "${summary_file}"
cat "${summary_file}" >> fmm_mpi_scaling_benchmark_metrics.txt
rm -f "${summary_file}"

cat fmm_mpi_scaling_benchmark_metrics.txt
