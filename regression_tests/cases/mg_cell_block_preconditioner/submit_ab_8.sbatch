#!/bin/bash
#SBATCH --job-name=mg-block-ab-8
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=0-04:00:00
#SBATCH --output=slurm-mg-block-ab-8-%j.out

set -uo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the preconditioner test directory}"
root="${RICH_PRECONDITIONER_AB_ROOT:?RICH_PRECONDITIONER_AB_ROOT is required}"
export RICH_AB_POINT_COUNT=32768
export RICH_AB_DT=0.000375
export RICH_AB_PPR=8

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
  --expected-ranks 8 --expected-cells 32768 \
  --output "${root}/comparison.txt" || compare_status=$?

if (( scalar_status != 0 || block_status != 0 || compare_status != 0 )); then
  exit 1
fi
