#!/bin/bash
#SBATCH -p bigrun
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --time=00:30:00
#SBATCH -o marshak_run.out
#SBATCH -e marshak_run.err

cd /home/maorm/RICH/runs/Elad_paper_Marshak
mpirun -np 16 ./rich 64 marshak 1 1 mc
