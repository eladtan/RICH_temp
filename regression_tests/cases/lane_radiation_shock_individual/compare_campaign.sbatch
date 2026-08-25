#!/bin/bash
#SBATCH --job-name=lane-rad-compare
#SBATCH --partition=core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-lane-rad-compare-%j.out

set -uo pipefail
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the benchmark case directory}"
campaign_root="${RICH_CAMPAIGN_ROOT:?RICH_CAMPAIGN_ROOT must name the shared campaign directory}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="/software/x86_64/5.14.0/python/3.12.1/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${campaign_root}/comparison"
/software/x86_64/5.14.0/python/3.12.1/bin/python3 \
  "${case_dir}/compare_campaign.py" \
  --tier production \
  --global-dir "${campaign_root}/global" \
  --full-variable-dir "${campaign_root}/full-variable" \
  --partial-dir "${campaign_root}/partial" \
  --output "${campaign_root}/comparison/verdict.txt"
status=$?
printf '%s\n' "${status}" > "${campaign_root}/comparison/exit_code.txt"
exit "${status}"
