#!/bin/sh
#SBATCH --job-name=sup_ghosts_ms
#SBATCH --output=socket_meshseed_%j.out
#SBATCH --error=socket_meshseed_%j.err
#SBATCH --partition=socket
#SBATCH --ntasks=64
#SBATCH --constraint="ib&d26g"
#SBATCH --time=00:30:00
# Same as run_socket.sh, but against the binary built from the 2026-09-21
# tree (mesh changes) in its own build subdirectory, so the production build
# directory is not disturbed.  Override with RICH_BIN=... if needed.
OPENMPI_BIN=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1/bin
export PATH=${OPENMPI_BIN}:$PATH
RUN_TMP="${PWD}/tmp_${SLURM_JOB_ID}"
mkdir -p "${RUN_TMP}"
export TMPDIR="${RUN_TMP}" TMP="${RUN_TMP}" TEMP="${RUN_TMP}"
export OMPI_MCA_orte_tmpdir_base="${RUN_TMP}"
export UCX_TLS=ib
export RICH_RUNTIME_COLOR=never
RICH_BIN="${RICH_BIN:-/home/elads/RICH-ablation-integration/build/intelReleaseMPI/sup_ghosts/rich_intelReleaseMPI}"
ls -l "${RICH_BIN}"
cp -p suppressed_exchange_ghosts_metrics.txt "suppressed_exchange_ghosts_metrics.before_${SLURM_JOB_ID}.txt" 2>/dev/null
mpirun -mca btl ^openib -np 64 "${RICH_BIN}"
echo "exit=$?"
grep -E "^pass|mismatches" suppressed_exchange_ghosts_metrics.txt
