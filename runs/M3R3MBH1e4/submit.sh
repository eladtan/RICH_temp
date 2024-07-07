#!/bin/bash
#SBATCH --nodes=4
#SBATCH --ntasks=512
#SBATCH --partition=rome
#SBATCH --time=110:18:00
##SBATCH --time=0:10:00
#SBATCH --output=output_%j
#SBATCH --error=error_%j
 
# Load modules for MPI and other parallel libraries
module describe default
module restore default
ml
 
# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
 
# Execute the program in parallel on ntasks cores
 
mpirun ./rich 
# srun inspxe-cl -collect mi3 -r result_dir3 -knob stack-depth=24 -- ./rich
#srun vtune -collect hotspots -knob enable-stack-collection=true -trace-mpi -result-dir results ./rich