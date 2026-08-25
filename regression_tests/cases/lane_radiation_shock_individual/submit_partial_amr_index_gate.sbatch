#!/bin/bash
#SBATCH --job-name=lane-partial-amr-index
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --output=slurm-lane-partial-amr-index-%j.out

set -euo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
openmpi_root=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1
export PATH="${openmpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${openmpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the benchmark case directory}"
bash "${case_dir}/run_partial_amr_index_gate.sh"
