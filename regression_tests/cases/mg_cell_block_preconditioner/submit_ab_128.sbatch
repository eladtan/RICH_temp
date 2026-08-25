#!/bin/bash
#SBATCH --job-name=mg-block-ab-128
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=0-08:00:00
#SBATCH --output=slurm-mg-block-ab-128-%j.out

set -uo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the preconditioner test directory}"
root="${RICH_PRECONDITIONER_AB_ROOT:?RICH_PRECONDITIONER_AB_ROOT is required}"
export RICH_AB_POINT_COUNT=524288
export RICH_AB_DT=1.1811759842764439e-4
export RICH_AB_PPR=16

scalar_status=0
block_status=0
compare_status=0
bash "${case_dir}/run_physical_ab_lane.sh" scalar || scalar_status=$?
bash "${case_dir}/run_physical_ab_lane.sh" cell_block || block_status=$?

LD_LIBRARY_PATH=/software/x86_64/5.14.0/python/3.12.1/lib:${LD_LIBRARY_PATH:-} \
PYTHONNOUSERSITE=1 \
/software/x86_64/5.14.0/python/3.12.1/bin/python3 \
  "${case_dir}/compare_physical_ab.py" \
  --scalar-dir "${root}/scalar" \
  --block-dir "${root}/cell_block" \
  --expected-ranks 128 --expected-cells 524288 \
  --output "${root}/comparison.txt" || compare_status=$?

if (( scalar_status != 0 || block_status != 0 || compare_status != 0 )); then
  exit 1
fi
