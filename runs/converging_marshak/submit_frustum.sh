#!/bin/bash
# Converging Marshak in the one-dimensional debug wedge: a narrow frustum of
# 1e-4 sr carrying a single angular cell per radial shell.  Same radial plan,
# same opacities and the same per-cell statistics as the 768-rank sphere run
# (submit_ddmc_4xyz.sh), so a disagreement between the two is an angular or
# geometric effect rather than a physics one.
#SBATCH --job-name=marshak-frustum
#SBATCH --partition=bigrun
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --output=marshak-frustum-%j.out
#SBATCH --error=marshak-frustum-%j.err

set -euo pipefail

# sbatch inherits the submitting shell, and a shell without Lmod initialised
# ships a LD_LIBRARY_PATH that cannot resolve libfabric.so.1, which kills every
# rank at startup.  Restoring the collection here makes the job independent of
# where it was submitted from.
source /etc/profile.d/modules.sh
module use ~/modulefiles/maor
module restore default


TEST="${TEST:-4}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-${PWD}}"
RICH_BIN="${RICH_BIN:-./rich_frustum}"
if [[ ! -x "${SCRIPT_DIR}/${RICH_BIN#./}" ]]; then
    echo "Expected an executable at ${SCRIPT_DIR}/${RICH_BIN#./}" >&2
    exit 1
fi

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/shared/maorm/CovnergingMarshak/Benchmark${TEST}_frustum}"
mkdir -p "${CHECKPOINT_DIR}"

export OMP_NUM_THREADS=1
cd "${SCRIPT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    RESUME_ARGS=(--resume "${RESUME}")
    echo "Resuming test ${TEST} from ${RESUME} on ${SLURM_NTASKS} ranks"
fi

STEPS="${STEPS:-200000}"
VTK_CYCLES="${VTK_CYCLES:-200}"
HDF5_CYCLES="${HDF5_CYCLES:-0}"
LATEST_CYCLES="${LATEST_CYCLES:-200}"

mpirun -np "${SLURM_NTASKS}" "${RICH_BIN}" "${TEST}" \
    --frustum \
    --cone-solid-angle "${CONE_SR:-2.5e-5}" \
    --cells-per-mfp "${CELLS_PER_MFP:-4}" \
    --min-shells "${MIN_SHELLS:-256}" \
    --max-shells "${MAX_SHELLS:-800}" \
    --ddmc true \
    --steps "${STEPS}" \
    --max-dt "${MAX_DT:-1e-11}" \
    --bath-zone "${BATH_ZONE:-0.25}" \
    --output "${CHECKPOINT_DIR}" \
    --hdf5-cycles "${HDF5_CYCLES}" \
    --vtk-cycles "${VTK_CYCLES}" \
    --latest-cycles "${LATEST_CYCLES}" \
    --output-dir "${CHECKPOINT_DIR}" \
    "${RESUME_ARGS[@]}"

if [[ -f "${CHECKPOINT_DIR}/test${TEST}_snapshot3.csv" ]]; then
    python3 ./compare_profiles.py --test "${TEST}" --input-dir "${CHECKPOINT_DIR}" --output-dir "${CHECKPOINT_DIR}" --bins 96 --plot-mode binned
fi

echo "Converging Marshak test ${TEST} frustum (${SLURM_NTASKS} ranks) results: ${CHECKPOINT_DIR}"
