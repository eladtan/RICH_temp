#!/bin/bash
# E3D explicit-sphere IMC benchmark submission.
#
# One SLURM task is one E3D geometry realization.  Submit one realization with:
#
#   sbatch submit_e3d.slurm
#
# The paper/reference initial material temperature is 3 eV (0.003 keV).  It
# can be overridden for a confirmed alternate reference deck with:
#
#   T0_KEV=<value> sbatch submit_e3d.slurm
#
# For the ten-realization paper ensemble, submit a serialized job array so that
# each array task gets its own 32-node allocation:
#
#   sbatch --array=0-9%1 submit_e3d.slurm
#
# The default output directory is on the shared filesystem below this directory;
# override it with E3D_OUTPUT_DIR when the batch filesystem layout requires it.

#SBATCH --job-name=E3D-RICH
#SBATCH --partition=bigrun
#SBATCH --nodes=32
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
# Reserve the requested 8 GiB for every MPI rank.
#SBATCH --mem-per-cpu=8G
#SBATCH --time=03:00:00
#SBATCH --output=e3d_%j.out
#SBATCH --error=e3d_%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SUBMIT_DIR}"

# Standard paper/reference value: 3 eV = 0.003 keV.  Keep this overrideable
# so a later confirmed reference input deck can be used without editing the
# submission script.
T0_KEV="${T0_KEV:-0.003}"
export T0_KEV
echo "Using initial material temperature T0_KEV=${T0_KEV}"

REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
EXECUTABLE="${E3D_EXECUTABLE:-${REPO_ROOT}/build/gnuReleaseMPI/e3d/rich_gnuReleaseMPI}"
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Missing E3D executable: ${EXECUTABLE}" >&2
    echo "Build it with: ./build_rich.sh gnuReleaseMPI --test_name=E3D --build-subdir=e3d" >&2
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

REALIZATION="${SLURM_ARRAY_TASK_ID:-${REALIZATION:-0}}"
GEOMETRY_SEED="${GEOMETRY_SEED:-$((1000 + REALIZATION))}"
TRANSPORT_SEED="${TRANSPORT_SEED:-$((9000 + REALIZATION))}"

MESH_POINTS="${MESH_POINTS:-8000000}"
NEW_PHOTONS_PER_CELL="${NEW_PHOTONS_PER_CELL:-5}"
MAX_PHOTONS_PER_CELL="${MAX_PHOTONS_PER_CELL:-20}"
INITIAL_RADIATION="${INITIAL_RADIATION:-empty}"
INITIAL_PARTICLES_PER_CELL="${INITIAL_PARTICLES_PER_CELL:-0}"
SOURCE_PACKETS_PER_FACE="${SOURCE_PACKETS_PER_FACE:-20}"
DT0="${DT0:-1e-15}"
DT_MAX="${DT_MAX:-1e-10}"
DT_GROWTH="${DT_GROWTH:-1.1}"
T_FINAL="${T_FINAL:-5e-9}"

# VTK snapshots are written only when E3D_OUTPUT_DIR is set; that path is
# passed through as --output DIR.  Tallies stay next to the job otherwise.
OUTPUT_DIR="${E3D_OUTPUT_DIR:-}"
GEOMETRY_DIR="${E3D_GEOMETRY_DIR:-${SUBMIT_DIR}/results/geometry}"
mkdir -p "${GEOMETRY_DIR}"
if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

REALIZATION_TAG="$(printf '%02d' "${REALIZATION}")"
GEOMETRY_FILE="${GEOMETRY_DIR}/e3d_geometry_r${REALIZATION_TAG}.csv"
MPI_RANKS="${SLURM_NTASKS:-256}"

if [[ "${MPI_RANKS}" -ne 256 ]]; then
    echo "WARNING: allocation has ${MPI_RANKS} MPI ranks; requested layout is 256." >&2
fi
if [[ "${SLURM_NNODES:-32}" -ne 32 ]]; then
    echo "WARNING: allocation has ${SLURM_NNODES:-unknown} nodes; requested layout is 32." >&2
fi

ARGS=(
    "${MESH_POINTS}"
    "${NEW_PHOTONS_PER_CELL}"
    "${MAX_PHOTONS_PER_CELL}"
    --realization "${REALIZATION}"
    --geometry-seed "${GEOMETRY_SEED}"
    --transport-seed "${TRANSPORT_SEED}"
    --geometry-file "${GEOMETRY_FILE}"
    --source-packets-per-face "${SOURCE_PACKETS_PER_FACE}"
    --T0-keV "${T0_KEV}"
    --initial-radiation "${INITIAL_RADIATION}"
    --dt0 "${DT0}"
    --dt-max "${DT_MAX}"
    --dt-growth "${DT_GROWTH}"
    --t-final "${T_FINAL}"
    # Modern automatic RDMA manager; stale packet-cell assignments are
    # repaired by the transport path before material-state lookup.
    --manager rdma
)

if [[ "${INITIAL_PARTICLES_PER_CELL}" -gt 0 ]]; then
    ARGS+=(--initial-particles-per-cell "${INITIAL_PARTICLES_PER_CELL}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
    ARGS+=(--output "${OUTPUT_DIR}")
fi
if [[ "${WRITE_VTU:-1}" == "0" ]]; then
    ARGS+=(--no-vtu)
fi

echo "E3D SLURM job ${SLURM_JOB_ID:-local}:"
echo "  nodes=${SLURM_NNODES:-unknown} ranks=${MPI_RANKS} ranks_per_node=${SLURM_NTASKS_PER_NODE:-8}"
echo "  realization=${REALIZATION} geometry_seed=${GEOMETRY_SEED} transport_seed=${TRANSPORT_SEED}"
echo "  mesh=${MESH_POINTS} new_photons_per_cell=${NEW_PHOTONS_PER_CELL} max_photons_per_cell=${MAX_PHOTONS_PER_CELL}"
echo "  output=${OUTPUT_DIR:-<none, no VTK>}"
echo "  manager=rdma (automatic RDMA)"

exec mpirun \
    --map-by ppr:8:node \
    --bind-to core \
    -np "${MPI_RANKS}" \
    "${EXECUTABLE}" \
    "${ARGS[@]}"
