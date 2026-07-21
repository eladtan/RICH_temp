#!/bin/sh
#SBATCH --job-name=TDE
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=bigrun
#SBATCH --ntasks=640
#SBATCH --exclusive
##SBATCH --time=820:00:00
#SBATCH --constraint="ib"
##SBATCH --nodelist=l20g[73-114,116-122]
##SBATCH --nodelist=l20g[73-102,104-122]
#SBATCH --exclude=l19g[1-71],l20g[1-51]

# run whatever you need here

#module load intel_parallel_studio_xe/2018_update3
#module load intel/2020u1
#module load gcc/4.8.2
#module load gcc/8.2.0
#ml openmpi/4.0.4/intel/2020u1
echo $PWD
echo $LD_LIBRARY_PATH
#ml restore
ml
#ml slurm/19.05.2
# Prefer OpenMPI mpirun for this run
OPENMPI_BIN=/software/x86_64/5.14.0/openmpi/4.1.6/Intel/OneApi/2024.2.1/bin
export PATH=${OPENMPI_BIN}:$PATH
which mpirun
sinfo -V
lscpu
#mpirun valgrind ./test.exe
#mpirun -genv I_MPI_DEBUG=5 -genv I_MPI_SHM_LMT=shm ./test.exe
#mpirun -genv I_MPI_FABRICS=shm:dapl ./test.exe
export UCX_TLS=ib
mpirun -mca btl ^openib ./rich
# mpirun ./rich



