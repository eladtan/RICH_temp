#!/bin/bash

#SBATCH --job-name=Densmore2012
#SBATCH --ntasks=32
#SBATCH --partition=bigrun
#SBATCH --out=Densmore2012_%j.out
#SBATCH --err=Densmore2012_%j.err
#SBATCH --exclusive

mpirun ./rich
