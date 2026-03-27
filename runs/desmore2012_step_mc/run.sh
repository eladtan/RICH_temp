#!/bin/bash
#SBATCH -p bigrun
#SBATCH --job-name="Desmore_MC"
#SBATCH -n 32
#SBATCH --exclusive
#SBATCH -o desmore_mc_%j.out
#SBATCH -e desmore_mc_%j.err
#SBATCH --partition=bigrun
#SBATCH --exclude=d25g[133-134]
export UCX_TLS="ib"
export OMPI_MCA_btl="^openib"
export OMPI_MCA_osc="ucx"
which mpirun
echo "=== RW ON ==="
mpirun ./rich 256 output/desmore_step_mc_rw 50 200 1
echo "=== RW OFF ==="
mpirun ./rich 256 output/desmore_step_mc_norw 50 200 0
