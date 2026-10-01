#!/bin/sh
#SBATCH --job-name=sup_ghosts
#SBATCH --output=socket_%j.out
#SBATCH --error=socket_%j.err
#SBATCH --partition=socket
#SBATCH --ntasks=64
#SBATCH --constraint="ib&d26g"
#SBATCH --time=00:30:00
OPENMPI_BIN=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1/bin
export PATH=${OPENMPI_BIN}:$PATH
RUN_TMP="${PWD}/tmp_${SLURM_JOB_ID}"
mkdir -p "${RUN_TMP}"
export TMPDIR="${RUN_TMP}" TMP="${RUN_TMP}" TEMP="${RUN_TMP}"
export OMPI_MCA_orte_tmpdir_base="${RUN_TMP}"
export UCX_TLS=ib
export RICH_RUNTIME_COLOR=never
mpirun -mca btl ^openib -np 64 /home/elads/RICH-ablation-integration/build/intelReleaseMPI/rich_intelReleaseMPI
