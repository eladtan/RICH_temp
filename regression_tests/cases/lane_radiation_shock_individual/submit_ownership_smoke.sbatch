#!/bin/bash
#SBATCH --job-name=lane-own-smoke
#SBATCH --partition=core
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=00:30:00

set -euo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
openmpi_root=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1
export PATH="${openmpi_root}/bin:${PATH}"
export LD_LIBRARY_PATH="${openmpi_root}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the benchmark case directory}"
lane="${RICH_SMOKE_LANE:?RICH_SMOKE_LANE must be full-variable or partial}"
bash "${case_dir}/run_ownership_smoke.sh" "${lane}"
