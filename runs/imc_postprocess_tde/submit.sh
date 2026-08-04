#!/bin/sh
#SBATCH --job-name=TDE
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=bigrun
#SBATCH --ntasks=960
#SBATCH --exclusive
#SBATCH --distribution=cyclic
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

MPI_TMPDIR="$PWD/mpi_tmp/${SLURM_JOB_ID}"
mkdir -p "$MPI_TMPDIR"
export TMPDIR="$MPI_TMPDIR"
export TMP="$MPI_TMPDIR"
export TEMP="$MPI_TMPDIR"
export OMPI_MCA_orte_tmpdir_base="$MPI_TMPDIR"

echo before
# mpirun -mca btl ^openib ./rich --output.stem luminosity --adaptive.source.enabled true --adaptive.group.quality true --adaptive.group.source-cells true --adaptive.group.frequency-sampling true --adaptive.source.learned-min-photons 100 --adaptive.source.learned-max-photons 5000 --adaptive.source.score-power 2 --adaptive.source.weight-score-fraction 0.85 --transport.source-dt 100 --transport.duration 750000 --transport.photons-per-cell 50 --observer.count 512 --observer.radius 7.5e14 --transport.generations 75 --input.snapshot /data/users/elads/RICH_dutch_restart/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_full_151.h5

mpirun -mca btl ^openib ./rich --output.stem luminosity_F --flux-source.construction-rays 4096 --flux-source.enabled true --flux-source.thermalization-tau 5 --flux-source.ddmc-face-optical-depth 5 --transport.source-dt 100 --transport.duration 750000 --transport.photons-per-cell 50 --observer.count 512 --observer.radius 7.5e14 --transport.generations 75 --input.snapshot /data/users/elads/RICH_dutch_restart/R0.47M0.5BH10000beta1S60n1.5ComptonHiResNewAMR/snap_full_151.h5


echo after
