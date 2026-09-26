#!/bin/bash
#SBATCH --job-name=marshak-t34-launch
#SBATCH --partition=core
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --output=marshak-t34-launch-%j.out
#SBATCH --error=marshak-t34-launch-%j.err

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
cd "${SCRIPT_DIR}"

if [[ ! -x ./rich.new ]]; then
    echo "Expected ${SCRIPT_DIR}/rich.new after the build job" >&2
    exit 1
fi

cp -f ./rich.new ./rich_ddmc_rebuild
chmod +x ./rich_ddmc_rebuild

MAIL_USER="maor.mizrachi@mail.huji.ac.il"

submit_one()
{
    local test="$1"
    sbatch \
        --partition=bigrun \
        --nodes=48 \
        --ntasks=768 \
        --ntasks-per-node=16 \
        --time=7-00:00:00 \
        --job-name="marshak-t${test}-4xyz-new" \
        --output="marshak-t${test}-4xyz-new-%j.out" \
        --error="marshak-t${test}-4xyz-new-%j.err" \
        --mail-type=BEGIN,END,FAIL \
        --mail-user="${MAIL_USER}" \
        --export=ALL,TEST="${test}",RICH_BIN=./rich_ddmc_rebuild,RESUME=,CHECKPOINT_DIR="/data/shared/maorm/CovnergingMarshak/Benchmark${test}_rebuild",VTK_CYCLES=200,HDF5_CYCLES=200,LATEST_CYCLES=200 \
        submit_ddmc_4xyz.sh
}

submit_one 3
submit_one 4
squeue --me -n marshak-t3-4xyz-new,marshak-t4-4xyz-new -o '%.10i %.26j %.8T %.6D %R'
