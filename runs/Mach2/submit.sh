#!/bin/sh
#
#SBATCH --job-name=RT_mpi
#SBATCH --output=output2
#SBATCH --error=error2
#SBATCH --partition=socket
##SBATCH --partition=testing
##SBATCH --partition=serial
#SBATCH --ntasks=32
##SBATCH --ntasks=448
#SBATCH --exclusive
#SBATCH --time=820:00:00
#SBATCH --constraint="ib"
##SBATCH --nodelist=l20g[73-114,116-122]
##SBATCH --nodelist=l20g[73-102,104-122]
#SBATCH --exclude=l06g[1-54]

# run whatever you need here

#module load intel_parallel_studio_xe/2018_update3
#module load intel/2020u1
#module load gcc/4.8.2
#module load gcc/8.2.0
#ml openmpi/4.0.4/intel/2020u1
echo $PWD
echo $LD_LIBRARY_PATH
ml
#ml slurm/19.05.2
which mpirun
ldd ../../build/intelReleaseMPI/rich
sinfo -V
lscpu
#mpirun valgrind ./test.exe
#mpirun -genv I_MPI_DEBUG=5 -genv I_MPI_SHM_LMT=shm ./test.exe
#mpirun -genv I_MPI_FABRICS=shm:dapl ./test.exe
mpiexec -x UCX_TLS=ib -mca btl ^openib ../../build/intelReleaseMPI/rich
#mpiexec ./test.exe
#mpirun ./test.exe
#mpirun ./test.exe
#mpirun -n 384 IMB-MPI1

