#!/bin/bash
# Fresh DDMC 4x-xyz runs for converging Marshak tests 1-4.
# Same mesh as submit_ddmc_4xyz.sh; VTK/HDF5 go to BenchmarkX_new.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RICH_BIN="${RICH_BIN:-./rich_ddmc_faceavg}"
if [[ ! -x "${RICH_BIN}" ]]; then
    echo "Missing executable: ${SCRIPT_DIR}/${RICH_BIN}" >&2
    exit 1
fi

MAIL_USER="maor.mizrachi@mail.huji.ac.il"

submit_one()
{
    local test="$1"
    sbatch \
        --begin=2026-09-08T00:00:00 \
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
        --export=ALL,TEST="${test}",RICH_BIN="${RICH_BIN}",RESUME=,CHECKPOINT_DIR="/data/shared/maorm/CovnergingMarshak/Benchmark${test}_new",VTK_CYCLES=200,HDF5_CYCLES=200,LATEST_CYCLES=200 \
        submit_ddmc_4xyz.sh
}

for test in 1 2 3 4; do
    submit_one "${test}"
done

squeue --me -h -O 'JobID:14,Name:25,State:12,NumTasks:6,TimeLeft:12' | grep 'marshak-t[1-4]-4xyz-new' || true
