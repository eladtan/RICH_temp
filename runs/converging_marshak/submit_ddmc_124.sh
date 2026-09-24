#!/bin/bash
#SBATCH --job-name=marshak-ddmc-launch
#SBATCH --partition=bigrun
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=marshak-ddmc-launch-%j.out
#SBATCH --error=marshak-ddmc-launch-%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${SCRIPT_DIR}"
if [[ ! -x ./rich.new ]]; then
    echo "Expected ${SCRIPT_DIR}/rich.new after the build job" >&2
    exit 1
fi

BIN=./rich_ddmc_densmore
cp -f ./rich.new "${BIN}"
chmod +x "${BIN}"

MAIL_USER="maor.mizrachi@mail.huji.ac.il"

submit()
{
    local test="$1"
    local nodes="$2"
    local ntasks="$3"
    sbatch \
        --partition=bigrun \
        --nodes="${nodes}" \
        --ntasks="${ntasks}" \
        --ntasks-per-node=16 \
        --job-name="marshak-t${test}-4xyz-ddmc" \
        --output="marshak-t${test}-4xyz-ddmc-%j.out" \
        --error="marshak-t${test}-4xyz-ddmc-%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="${MAIL_USER}" \
        --export=ALL,TEST="${test}",RICH_BIN="${BIN}" \
        submit_ddmc_4xyz.sh
}

# Same 256x32000 mesh as marshak-t2-4xyz. All four on 24 nodes.
submit 1 24 384
submit 2 24 384
submit 3 24 384
submit 4 24 384
