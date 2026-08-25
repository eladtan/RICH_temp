#!/bin/bash
#SBATCH --job-name=fmm-individual-bench
#SBATCH --partition=bigrun
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@300
#SBATCH --output=slurm-fmm-individual-%j.out

set -euo pipefail
set +u
source /etc/profile.d/modules.sh
module purge
module load gcc/12.3.0
module load openmpi/4.1.6/gcc/12.3.0
module load hdf5/1.14.2/gcc/12.3.0_cxx
module load fftw/3.3.7
module load vtk/9.3.0/gcc/12.3.0/with_mesa
module load jsoncpp/1.9.5
module load boost/1.88.0
set -u

case_dir="${RICH_FMM_ITS_CASE_DIR:?RICH_FMM_ITS_CASE_DIR is required}"
bash "${case_dir}/run_campaign.sh"
