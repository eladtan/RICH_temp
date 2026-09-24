#!/bin/bash
# Submit 4x-radial Marshak runs with per-test MPI counts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

chmod +x submit_4x_single.sh

submit_one()
{
    local test="$1"
    local ntasks="$2"
    local extra_export="${3:-}"
    local sbatch_args=(
        --job-name="marshak-4x-t${test}"
        --ntasks="${ntasks}"
        --export=ALL,TEST="${test}",RICH_BIN=./rich_guard${extra_export:+,${extra_export}}
    )
    sbatch "${sbatch_args[@]}" submit_4x_single.sh
}

# Tests 1-3: 2x MPI ranks vs the original 4x launch (64/32/32), plus 3x radial
# shells vs that 4x mesh. Test 4 keeps 192 ranks and the 4x radial mesh.
# Test 1 keeps the graded bath that the 12x radial mesh needs.
submit_one 1 128 BATH_GRADING=1.5
submit_one 2 64
submit_one 3 64
submit_one 4 192

squeue --me -h -O 'JobID:14,Name:25,State:12,NumTasks:6' | grep marshak-4x || true
