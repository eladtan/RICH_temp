#!/bin/bash
#SBATCH --job-name=marshak-launch
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=marshak-launch-%j.out
#SBATCH --error=marshak-launch-%j.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${SCRIPT_DIR}"
if [[ ! -x ./rich.new ]]; then
    echo "Expected ${SCRIPT_DIR}/rich.new after the build job" >&2
    exit 1
fi
cp -f ./rich.new ./rich
chmod +x ./rich
bash ./submit_4x_all.sh
