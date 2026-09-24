#!/bin/bash
# Mach-2 radiative shock on a Lagrangian mesh (Steinberg & Heizler 2022, Sec. 5.1).
# Submit from this directory:  sbatch run.sh [cells]
#SBATCH --partition=socket
#SBATCH --job-name=mach2-lagrangian
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=mach2_lag_%j.out
#SBATCH --error=mach2_lag_%j.err

set -euo pipefail

# Lmod is not initialised in non-login shells; the executable needs the
# libfabric/OpenMPI libraries from the restored module collection.
if ! type -t module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module restore
hash -r

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ ! -x ./rich ]]; then
    echo "Missing executable ./rich in $(pwd); run sbatch build.sh first" >&2
    exit 1
fi

cells="${1:-1024}"
run_id="${SLURM_JOB_ID:-local}"
output_dir="/data/shared/maorm/IMC_paper/mach2_lagrangian/${run_id}"
mkdir -p "$output_dir"
prefix="$output_dir/mach2"
echo "Simulation output: ${output_dir}"

{
    printf 'job=%s\ncells=%s\nnodes=%s\ntasks=%s\nshared_output=%s\n' \
        "$run_id" "$cells" "${SLURM_JOB_NUM_NODES:-1}" "${SLURM_NTASKS:-16}" "$output_dir"
    sha256sum ./rich
} > "run_info_${run_id}.txt"

mpirun -np "${SLURM_NTASKS:-16}" ./rich "$cells" "$prefix" 25 100 \
    --profile mach2_analytic_ic2.dat \
    --manager new-rdma-auto

[[ -s "${prefix}_final.txt" && -s "${prefix}_checkpoint.h5" ]] || { echo "Missing final Mach2 profile/checkpoint" >&2; exit 1; }
ln -sfn "$(basename "${prefix}_checkpoint.h5")" "$output_dir/latest.h5"

cp "${prefix}_final.txt" "mach2_lag_${run_id}_final.txt"
python3 compare.py "mach2_lag_${run_id}_final.txt" --output "comparison_${run_id}.png" \
    --title-suffix "Lagrangian mesh + wall AMR; job ${run_id}; ${cells} cells; ${SLURM_NTASKS:-16} ranks"
cp comparison_"${run_id}"*.png comparison_"${run_id}"*.pdf "$output_dir/"
