#!/bin/bash
#SBATCH --job-name=marshak-t1-retry
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=marshak-t1-retry-%j.out
#SBATCH --error=marshak-t1-retry-%j.err

# Installs the freshly built binary under a new name, so the still running
# tests 2-4 keep executing the mapped ./rich, and resubmits test 1 with it.
set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${SCRIPT_DIR}"
if [[ ! -x ./rich.new ]]; then
    echo "Expected ${SCRIPT_DIR}/rich.new after the build job" >&2
    exit 1
fi
cp -f ./rich.new ./rich_guard
chmod +x ./rich_guard

sbatch --job-name=marshak-4x-t1 --ntasks=128 \
    --export=ALL,TEST=1,RICH_BIN=./rich_guard,BATH_GRADING="${BATH_GRADING:-1.5}" \
    submit_4x_single.sh
