#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=256
#SBATCH --partition=rome
#SBATCH --time=00:30:00
#SBATCH --output=output
#SBATCH --error=error
 
# Load modules for MPI and other parallel libraries
ml restore
 
# Create folder and copy input to scratch. This will copy the input file 'input_file' to the shared scratch space
mkdir -p /scratch-shared/$USER
 
# Execute the program in parallel on ntasks cores
 
srun ./rich > output
  