#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="GrayMarshak"
#SBATCH -n 16
#SBATCH -o gray_marshak_%j.out
#SBATCH -e gray_marshak_%j.err
export UCX_TLS="ib"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_osc="ucx"
ml ucx/1.15.0
which mpirun

export RICH_OUTPUT_DIR="/data/shared/maorm/MC_results/GrayMarshak/$(date +%Y-%m-%d)"
mkdir -p "$RICH_OUTPUT_DIR"
echo "Simulation output: ${RICH_OUTPUT_DIR}"

echo "=== RW ON ==="
mpirun ./rich 128 "${RICH_OUTPUT_DIR}/gray_rw" 50 50 1
echo "=== RW OFF ==="
mpirun ./rich 128 "${RICH_OUTPUT_DIR}/gray_norw" 50 50 0
