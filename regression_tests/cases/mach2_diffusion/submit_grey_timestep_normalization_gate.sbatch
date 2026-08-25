#!/bin/bash
#SBATCH --job-name=grey-dt-gate
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --output=slurm-grey-dt-gate-%j.out

set -euo pipefail
ml openmpi/4.1.6/Intel/OneApi/2024.2.1
case_dir="${RICH_CASE_DIR:?RICH_CASE_DIR must name the grey diffusion case directory}"
bash "${case_dir}/run_grey_timestep_normalization_gate.sh"
