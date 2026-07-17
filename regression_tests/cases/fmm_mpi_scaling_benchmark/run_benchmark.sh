#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <rich-binary>" >&2
    exit 2
fi

RICH_BIN="$1"
REPEATS="${FMM_MPI_BENCH_REPEATS:-2}"
FMM_MAX_REMOTE_MIB="${FMM_MPI_BENCH_MAX_REMOTE_MIB:-512}"
FMM_OPERATOR_CACHE_MIB="${FMM_MPI_BENCH_OPERATOR_CACHE_MIB:-64}"
EXTRA_NODES="${FMM_MPI_BENCH_EXTRA_NODES:-}"
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
if ! [[ "${FMM_OPERATOR_CACHE_MIB}" =~ ^[0-9]+$ ]]; then
    echo "FMM_MPI_BENCH_OPERATOR_CACHE_MIB must be a non-negative integer" >&2
    exit 2
fi
if (( ALLOCATED_NODES < 16 )); then
    echo "fmm_mpi_scaling_benchmark requires at least 16 allocated nodes; got ${ALLOCATED_NODES}" >&2
    exit 2
fi
if (( ALLOCATED_TASKS < ALLOCATED_NODES ||
      ALLOCATED_TASKS % ALLOCATED_NODES != 0 )); then
    echo "SLURM_NTASKS must be a positive multiple of the allocated node count; got tasks=${ALLOCATED_TASKS}, nodes=${ALLOCATED_NODES}" >&2
    exit 2
fi

RANKS_PER_NODE=$((ALLOCATED_TASKS / ALLOCATED_NODES))
if (( RANKS_PER_NODE < 1 )); then
    echo "Could not derive a positive ranks-per-node value" >&2
    exit 2
fi

EXTRA_NODE_COUNTS=()
if [[ -n "${EXTRA_NODES//[[:space:]]/}" ]]; then
    read -r -a EXTRA_NODE_COUNTS <<< "${EXTRA_NODES}"
fi
previous_nodes=16
for nodes in "${EXTRA_NODE_COUNTS[@]}"; do
    if ! [[ "${nodes}" =~ ^[1-9][0-9]*$ ]]; then
        echo "FMM_MPI_BENCH_EXTRA_NODES must contain positive integers" >&2
        exit 2
    fi
    if (( nodes <= previous_nodes )); then
        echo "FMM_MPI_BENCH_EXTRA_NODES must be strictly increasing and greater than 16" >&2
        exit 2
    fi
    if (( nodes > ALLOCATED_NODES )); then
        echo "Requested ${nodes} nodes but only ${ALLOCATED_NODES} are allocated" >&2
        exit 2
    fi
    previous_nodes="${nodes}"
done

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
# The benchmark is single-threaded per MPI rank.  Limit glibc arenas so many
# ranks on one node do not retain independent multi-arena high-water marks.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

rm -f fmm_mpi_scaling_benchmark_*.txt \
      fmm_mpi_scaling_benchmark_metrics.txt \
      fmm_mpi_scaling_profile.txt \
      fmm_mpi_extended_scaling_metrics.txt \
      fmm_mpi_extended_scaling_profile.txt

run_case() {
    local particles="$1"
    local nodes="$2"
    local ranks=$((nodes * RANKS_PER_NODE))
    local output="fmm_mpi_scaling_benchmark_${particles}_nodes${nodes}.txt"

    echo "=== particles=${particles}, nodes=${nodes}, ranks=${ranks}, ranks_per_node=${RANKS_PER_NODE}, repeats=${REPEATS}, fmm_max_remote_mib=${FMM_MAX_REMOTE_MIB}, fmm_operator_cache_mib=${FMM_OPERATOR_CACHE_MIB} ==="
    if ! mpirun -np "${ranks}" \
        --map-by "ppr:${RANKS_PER_NODE}:node" \
        --bind-to core \
        "${RICH_BIN}" \
        --particles "${particles}" \
        --expected-nodes "${nodes}" \
        --expected-ranks-per-node "${RANKS_PER_NODE}" \
        --repeats "${REPEATS}" \
        --fmm-max-remote-mib "${FMM_MAX_REMOTE_MIB}" \
        --fmm-operator-cache-mib "${FMM_OPERATOR_CACHE_MIB}" \
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
    elif ! grep -q '^profile_columns ' "${output}" || ! grep -q '^profile ' "${output}"; then
        echo "Benchmark subcase is missing rank-profile output: ${output}" >&2
        exit 1
    fi
}

run_case "${SMALL_PARTICLES}" 8
run_case "${SMALL_PARTICLES}" 16
run_case "${LARGE_PARTICLES}" 8
run_case "${LARGE_PARTICLES}" 16
for nodes in "${EXTRA_NODE_COUNTS[@]}"; do
    run_case "${LARGE_PARTICLES}" "${nodes}"
done

{
    echo "columns particles expected_nodes ranks unique_nodes ranks_per_node repeats local_particles_min local_particles_max fmm_best_max_seconds fmm_mean_max_seconds quadrupole_best_max_seconds quadrupole_mean_max_seconds fmm_probe_scaled_error quadrupole_probe_scaled_error quadrupole_over_fmm_speedup fmm_particles_per_second quadrupole_particles_per_second fmm_bytes_sent fmm_bytes_received fmm_peak_remote_bytes fmm_peak_process_bytes quadrupole_walk_max_seconds fmm_checksum quadrupole_checksum finite run_pass fmm_warm_best_max_seconds fmm_warm_mean_max_seconds fmm_cold_over_warm_speedup fmm_persistent_bytes fmm_local_tree_bytes fmm_local_multipole_bytes fmm_local_local_bytes fmm_let_plan_bytes fmm_operator_cache_bytes fmm_operator_cache_budget_bytes fmm_local_operator_cache_bytes fmm_local_operator_cache_entries fmm_local_operator_cache_max_entries fmm_local_operator_cache_hits fmm_local_operator_cache_misses fmm_local_operator_cache_bypasses fmm_let_operator_cache_bytes fmm_let_operator_cache_entries fmm_let_operator_cache_max_entries fmm_let_operator_cache_hits fmm_let_operator_cache_misses fmm_let_operator_cache_bypasses fmm_process_operator_cache_misses fmm_process_operator_cache_bypasses fmm_topology_reused probe_count fmm_probe_mean_scaled_error quadrupole_probe_mean_scaled_error"
    for particles in "${SMALL_PARTICLES}" "${LARGE_PARTICLES}"; do
        for nodes in 8 16; do
            grep '^row ' "fmm_mpi_scaling_benchmark_${particles}_nodes${nodes}.txt"
        done
    done
} > fmm_mpi_scaling_benchmark_metrics.txt

{
    echo "profile_columns particles expected_nodes mode category metric rank_min rank_mean rank_max max_over_mean"
    for particles in "${SMALL_PARTICLES}" "${LARGE_PARTICLES}"; do
        for nodes in 8 16; do
            output="fmm_mpi_scaling_benchmark_${particles}_nodes${nodes}.txt"
            awk -v particles="${particles}" -v nodes="${nodes}" '
            $1 == "profile" {
                print "profile", particles, nodes, $2, $3, $4, $5, $6, $7, $8
            }
            ' "${output}"
        done
    done
} > fmm_mpi_scaling_profile.txt

summary_file="$(mktemp)"
awk -v small="${SMALL_PARTICLES}" -v large="${LARGE_PARTICLES}" '
$1 == "row" {
    count++
    particles = $2
    nodes = $3
    fmm[particles, nodes] = $10
    quad[particles, nodes] = $12
    warm[particles, nodes] = $28
    cold_to_warm[particles, nodes] = $30
    cache_bytes[particles, nodes] = $36
    cache_bypasses[particles, nodes] = $43 + $49 + $51
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
    warm_small_speedup = warm[small, 8] / warm[small, 16]
    warm_large_speedup = warm[large, 8] / warm[large, 16]

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
    print "fmm_warm_small_8_to_16_speedup", warm_small_speedup
    print "fmm_warm_small_8_to_16_efficiency", warm_small_speedup / 2.0
    print "fmm_warm_large_8_to_16_speedup", warm_large_speedup
    print "fmm_warm_large_8_to_16_efficiency", warm_large_speedup / 2.0
    print "fmm_small_8_cold_to_warm_speedup", cold_to_warm[small, 8]
    print "fmm_small_16_cold_to_warm_speedup", cold_to_warm[small, 16]
    print "fmm_large_8_cold_to_warm_speedup", cold_to_warm[large, 8]
    print "fmm_large_16_cold_to_warm_speedup", cold_to_warm[large, 16]
    print "fmm_operator_cache_bytes_max", \
        (cache_bytes[small,8] > cache_bytes[small,16] ? \
         (cache_bytes[small,8] > cache_bytes[large,8] ? \
          (cache_bytes[small,8] > cache_bytes[large,16] ? cache_bytes[small,8] : cache_bytes[large,16]) : \
          (cache_bytes[large,8] > cache_bytes[large,16] ? cache_bytes[large,8] : cache_bytes[large,16])) : \
         (cache_bytes[small,16] > cache_bytes[large,8] ? \
          (cache_bytes[small,16] > cache_bytes[large,16] ? cache_bytes[small,16] : cache_bytes[large,16]) : \
          (cache_bytes[large,8] > cache_bytes[large,16] ? cache_bytes[large,8] : cache_bytes[large,16])))
    print "fmm_operator_cache_bypasses_large_8", cache_bypasses[large,8]
    print "fmm_operator_cache_bypasses_large_16", cache_bypasses[large,16]
    print "pass", (run_pass ? 1 : 0)
    if (!run_pass) exit 1
}
' fmm_mpi_scaling_benchmark_metrics.txt > "${summary_file}"
cat "${summary_file}" >> fmm_mpi_scaling_benchmark_metrics.txt
rm -f "${summary_file}"


if (( ${#EXTRA_NODE_COUNTS[@]} > 0 )); then
    EXTENDED_NODE_COUNTS=(8 16 "${EXTRA_NODE_COUNTS[@]}")
    {
        echo "columns particles expected_nodes ranks unique_nodes ranks_per_node repeats local_particles_min local_particles_max fmm_best_max_seconds fmm_mean_max_seconds quadrupole_best_max_seconds quadrupole_mean_max_seconds fmm_probe_scaled_error quadrupole_probe_scaled_error quadrupole_over_fmm_speedup fmm_particles_per_second quadrupole_particles_per_second fmm_bytes_sent fmm_bytes_received fmm_peak_remote_bytes fmm_peak_process_bytes quadrupole_walk_max_seconds fmm_checksum quadrupole_checksum finite run_pass fmm_warm_best_max_seconds fmm_warm_mean_max_seconds fmm_cold_over_warm_speedup fmm_persistent_bytes fmm_local_tree_bytes fmm_local_multipole_bytes fmm_local_local_bytes fmm_let_plan_bytes fmm_operator_cache_bytes fmm_operator_cache_budget_bytes fmm_local_operator_cache_bytes fmm_local_operator_cache_entries fmm_local_operator_cache_max_entries fmm_local_operator_cache_hits fmm_local_operator_cache_misses fmm_local_operator_cache_bypasses fmm_let_operator_cache_bytes fmm_let_operator_cache_entries fmm_let_operator_cache_max_entries fmm_let_operator_cache_hits fmm_let_operator_cache_misses fmm_let_operator_cache_bypasses fmm_process_operator_cache_misses fmm_process_operator_cache_bypasses fmm_topology_reused probe_count fmm_probe_mean_scaled_error quadrupole_probe_mean_scaled_error"
        for nodes in "${EXTENDED_NODE_COUNTS[@]}"; do
            grep '^row ' "fmm_mpi_scaling_benchmark_${LARGE_PARTICLES}_nodes${nodes}.txt"
        done
    } > fmm_mpi_extended_scaling_metrics.txt

    {
        echo "profile_columns particles expected_nodes mode category metric rank_min rank_mean rank_max max_over_mean"
        for nodes in "${EXTENDED_NODE_COUNTS[@]}"; do
            output="fmm_mpi_scaling_benchmark_${LARGE_PARTICLES}_nodes${nodes}.txt"
            awk -v particles="${LARGE_PARTICLES}" -v nodes="${nodes}" '
            $1 == "profile" {
                print "profile", particles, nodes, $2, $3, $4, $5, $6, $7, $8
            }
            ' "${output}"
        done
    } > fmm_mpi_extended_scaling_profile.txt

    extended_summary_file="$(mktemp)"
    node_list="${EXTENDED_NODE_COUNTS[*]}"
    awk -v particles="${LARGE_PARTICLES}" -v node_list="${node_list}" '
    BEGIN {
        expected = split(node_list, node, " ")
    }
    $1 == "row" {
        rows++
        n = $3 + 0
        cold[n] = $10 + 0
        quad[n] = $12 + 0
        warm[n] = $28 + 0
        cache_bytes[n] = $36 + 0
        cache_bypasses[n] = $43 + $49 + $51
        if (rows == 1) ranks_per_node = $6 + 0
        if (($6 + 0) != ranks_per_node || ($27 + 0) != 1) bad = 1
    }
    END {
        complete = rows == expected && !bad
        for (i = 1; i <= expected; ++i)
            if (!(node[i] in warm)) complete = 0
        print "extended_row_count", rows
        print "extended_particles", particles
        print "extended_node_counts", node_list
        print "extended_ranks_per_node", ranks_per_node
        for (i = 2; i <= expected; ++i) {
            a = node[i - 1] + 0
            b = node[i] + 0
            ratio = b / a
            print "fmm_cold_" a "_to_" b "_speedup", cold[a] / cold[b]
            print "fmm_cold_" a "_to_" b "_efficiency", (cold[a] / cold[b]) / ratio
            print "fmm_warm_" a "_to_" b "_speedup", warm[a] / warm[b]
            print "fmm_warm_" a "_to_" b "_efficiency", (warm[a] / warm[b]) / ratio
            print "quadrupole_" a "_to_" b "_speedup", quad[a] / quad[b]
            print "quadrupole_" a "_to_" b "_efficiency", (quad[a] / quad[b]) / ratio
        }
        first = node[1] + 0
        last = node[expected] + 0
        ratio = last / first
        print "fmm_warm_" first "_to_" last "_speedup", warm[first] / warm[last]
        print "fmm_warm_" first "_to_" last "_efficiency", (warm[first] / warm[last]) / ratio
        for (i = 1; i <= expected; ++i) {
            n = node[i] + 0
            print "fmm_operator_cache_bytes_" n, cache_bytes[n]
            print "fmm_operator_cache_bypasses_" n, cache_bypasses[n]
        }
        print "extended_pass", (complete ? 1 : 0)
        if (!complete) exit 1
    }
    ' fmm_mpi_extended_scaling_metrics.txt > "${extended_summary_file}"
    cat "${extended_summary_file}" >> fmm_mpi_extended_scaling_metrics.txt
    rm -f "${extended_summary_file}"
fi

echo "=== aggregate scaling metrics ==="
cat fmm_mpi_scaling_benchmark_metrics.txt
echo "=== rank phase/work profile ==="
cat fmm_mpi_scaling_profile.txt
if (( ${#EXTRA_NODE_COUNTS[@]} > 0 )); then
    echo "=== extended 10M-particle scaling metrics ==="
    cat fmm_mpi_extended_scaling_metrics.txt
    echo "Extended rank profile written to fmm_mpi_extended_scaling_profile.txt"
fi
