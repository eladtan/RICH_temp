#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <baseline-rich-binary> <optimized-rich-binary> <output-directory>" >&2
    exit 2
fi

BASELINE_BIN="$1"
OPTIMIZED_BIN="$2"
OUTPUT_DIR="$3"
PARTICLES=30000000
REPEATS="${FMM_AB_REPEATS:-1}"
FMM_MAX_REMOTE_MIB="${FMM_AB_MAX_REMOTE_MIB:-512}"
FMM_OPERATOR_CACHE_MIB="${FMM_AB_OPERATOR_CACHE_MIB:-128}"
NODE_COUNTS=(16 32 64)
ALLOCATED_NODES="${SLURM_JOB_NUM_NODES:-0}"
ALLOCATED_TASKS="${SLURM_NTASKS:-0}"

for binary in "$BASELINE_BIN" "$OPTIMIZED_BIN"; do
    if [[ ! -x "$binary" ]]; then
        echo "Benchmark binary is not executable: $binary" >&2
        exit 2
    fi
done
if ! [[ "$PARTICLES" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$FMM_MAX_REMOTE_MIB" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "$FMM_OPERATOR_CACHE_MIB" =~ ^[0-9]+$ ]]; then
    echo "Invalid FMM A/B benchmark configuration" >&2
    exit 2
fi
if (( ALLOCATED_NODES < 64 )); then
    echo "The 30M A/B benchmark requires a 64-node allocation" >&2
    exit 2
fi
if (( ALLOCATED_TASKS < ALLOCATED_NODES ||
      ALLOCATED_TASKS % ALLOCATED_NODES != 0 )); then
    echo "SLURM_NTASKS must be a positive multiple of allocated nodes" >&2
    exit 2
fi
RANKS_PER_NODE=$((ALLOCATED_TASKS / ALLOCATED_NODES))
if (( RANKS_PER_NODE < 1 )); then
    echo "Could not derive ranks per node" >&2
    exit 2
fi

mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export MALLOC_TRIM_THRESHOLD_="${MALLOC_TRIM_THRESHOLD_:-131072}"

run_variant() {
    local variant="$1"
    local binary="$2"
    local nodes
    for nodes in "${NODE_COUNTS[@]}"; do
        local ranks=$((nodes * RANKS_PER_NODE))
        local output="$OUTPUT_DIR/${variant}_nodes${nodes}.txt"
        echo "=== variant=${variant}, particles=${PARTICLES}, nodes=${nodes}, ranks=${ranks}, ranks_per_node=${RANKS_PER_NODE}, repeats=${REPEATS} ==="
        mpirun -np "$ranks"             --map-by "ppr:${RANKS_PER_NODE}:node"             --bind-to core             "$binary"             --particles "$PARTICLES"             --expected-nodes "$nodes"             --expected-ranks-per-node "$RANKS_PER_NODE"             --repeats "$REPEATS"             --fmm-max-remote-mib "$FMM_MAX_REMOTE_MIB"             --fmm-operator-cache-mib "$FMM_OPERATOR_CACHE_MIB"             --output "$output"
        if [[ ! -s "$output" ]] || ! grep -q '^pass 1$' "$output"; then
            echo "Failed benchmark output: $output" >&2
            exit 1
        fi
        awk '
        $1 == "row" {
            if (NF != 55 || $2 != 30000000 || $53 != 100 ||
                $26 != 1 || $27 != 1 || $52 != 1)
                exit 1
            found = 1
        }
        END { exit !found }
        ' "$output" || {
            echo "Invalid 30M/100-probe row: $output" >&2
            exit 1
        }
    done
}

run_variant baseline "$BASELINE_BIN"
run_variant optimized "$OPTIMIZED_BIN"

PRIMARY="$OUTPUT_DIR/fmm_30m_ab_primary_metrics.txt"
{
    echo "columns variant solver particles nodes ranks ranks_per_node seconds max_scaled_error mean_scaled_error"
    for variant in baseline optimized; do
        for nodes in "${NODE_COUNTS[@]}"; do
            awk -v variant="$variant" '
            $1 == "row" {
                print "row", variant, "fmm", $2, $3, $4, $6, $28, $14, $54
                print "row", variant, "quadrupole", $2, $3, $4, $6, $12, $15, $55
            }
            ' "$OUTPUT_DIR/${variant}_nodes${nodes}.txt"
        done
    done
} > "$PRIMARY"

SUMMARY="$OUTPUT_DIR/fmm_30m_ab_scaling_summary.txt"
awk '
$1 == "row" {
    variant = $2
    solver = $3
    nodes = $5 + 0
    ranks = $6 + 0
    time[variant, solver, nodes] = $8 + 0
    maxerr[variant, solver, nodes] = $9 + 0
    meanerr[variant, solver, nodes] = $10 + 0
    rank_count[nodes] = ranks
}
END {
    print "columns variant solver nodes ranks seconds speedup_from_16 scaling_efficiency max_scaled_error mean_scaled_error"
    variants[1] = "baseline"; variants[2] = "optimized"
    solvers[1] = "fmm"; solvers[2] = "quadrupole"
    nodes_list[1] = 16; nodes_list[2] = 32; nodes_list[3] = 64
    for (v = 1; v <= 2; ++v) {
        for (s = 1; s <= 2; ++s) {
            base = time[variants[v], solvers[s], 16]
            for (n = 1; n <= 3; ++n) {
                node = nodes_list[n]
                speedup = base / time[variants[v], solvers[s], node]
                efficiency = speedup / (node / 16.0)
                printf "scaling %s %s %d %d %.16e %.16e %.16e %.16e %.16e\n",                     variants[v], solvers[s], node, rank_count[node],                     time[variants[v], solvers[s], node], speedup, efficiency,                     maxerr[variants[v], solvers[s], node],                     meanerr[variants[v], solvers[s], node]
            }
        }
    }
    print "columns comparison nodes ranks baseline_seconds optimized_seconds optimization_speedup"
    for (n = 1; n <= 3; ++n) {
        node = nodes_list[n]
        printf "comparison fmm %d %d %.16e %.16e %.16e\n",             node, rank_count[node], time["baseline", "fmm", node],             time["optimized", "fmm", node],             time["baseline", "fmm", node] / time["optimized", "fmm", node]
        printf "comparison quadrupole %d %d %.16e %.16e %.16e\n",             node, rank_count[node], time["baseline", "quadrupole", node],             time["optimized", "quadrupole", node],             time["baseline", "quadrupole", node] /                 time["optimized", "quadrupole", node]
    }
}
' "$PRIMARY" > "$SUMMARY"

REPORT="$OUTPUT_DIR/fmm_30m_ab_results.txt"
{
    echo "=== Configuration ==="
    echo "particles $PARTICLES"
    echo "node_counts 16 32 64"
    echo "ranks_per_node $RANKS_PER_NODE"
    echo "repeats $REPEATS"
    echo "probe_count 100"
    echo "primary_timed_rows 12"
    echo
    cat "$PRIMARY"
    echo
    cat "$SUMMARY"
    echo
    echo "=== Cold FMM diagnostics ==="
    for variant in baseline optimized; do
        for nodes in "${NODE_COUNTS[@]}"; do
            awk -v variant="$variant" '$1 == "row" {
                printf "cold %s nodes=%s ranks=%s seconds=%.16e\n", variant, $3, $4, $10
            }' "$OUTPUT_DIR/${variant}_nodes${nodes}.txt"
        done
    done
} > "$REPORT"

cat "$SUMMARY"
echo "Results: $REPORT"
