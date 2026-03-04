#!/bin/bash
#SBATCH -p socket
#SBATCH -n 640
#SBATCH --exclusive
#SBATCH -o hohlraum_run.out
#SBATCH -e hohlraum_run.err

cd /home/maorm/RICH/runs/Elad_paper_hohlraum

# delta=0.05 cm  =>  ~28x26x26 = ~19k cells  (coarse test run)
# delta=0.02 cm  =>  ~70x65x65 = ~296k cells (medium)
# delta=0.005 cm  =>  ~140x130x130 = ~2.4M cells (nominal from paper)

mpirun ./rich 0.01 hohlraum 50 200
