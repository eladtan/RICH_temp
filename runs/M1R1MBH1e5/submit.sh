#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=384
#SBATCH --partition=genoa
#SBATCH --time=110:18:00
##SBATCH --time=0:10:00
#SBATCH --output=output_%j
#SBATCH --error=error_%j
 
# Load modules for MPI and other parallel libraries
module load 2022
module load foss/2022a
module load CMake/3.24.3-GCCcore-11.3.0
module load Szip/2.1.1-GCCcore-11.3.0
module load OpenMPI/4.1.4-GCC-11.3.0 
module load libfabric/1.15.1-GCCcore-11.3.0
module load UCX/1.12.1-GCCcore-11.3.0
module load VTK/9.2.0.rc2-foss-2022a
module load Boost/1.79.0-GCC-11.3.0 
module load Mesa/22.0.3-GCCcore-11.3.0 
module load gzip/1.12-GCCcore-11.3.0
module load HDF5/1.12.2-gompi-2022a
module load zlib/1.2.12-GCCcore-11.3.0 

# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
 
# Execute the program in parallel on ntasks cores
 
mpirun ./rich 
# srun inspxe-cl -collect mi3 -r result_dir3 -knob stack-depth=24 -- ./rich
#srun vtune -collect hotspots -knob enable-stack-collection=true -trace-mpi -result-dir results ./rich