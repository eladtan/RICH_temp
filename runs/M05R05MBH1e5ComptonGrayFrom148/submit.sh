#!/bin/sh
#SBATCH --job-name=TDE
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
#SBATCH --partition=genoa
#SBATCH --nodes=6
#SBATCH --exclusive
#SBATCH --time=01:00:00

# run whatever you need here
ml restore 2024_new
mpirun -x RICH_FMM_GEOM_LOG=1 ../../build/intelReleaseMPI/rich_intelReleaseMPI
