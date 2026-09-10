#!/usr/bin/env bash
#SBATCH --job-name=RMTV_lbsteps
#SBATCH --partition=bigrun
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --chdir=/home/maorm/RICH/runs/RMTV
#SBATCH --output=/home/maorm/RICH/runs/RMTV/verification/lbsteps_%j.out
#SBATCH --error=/home/maorm/RICH/runs/RMTV/verification/lbsteps_%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run simulations on gateway nodes}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
rmtv_dir=/home/maorm/RICH/runs/RMTV
rmtv_bin=/home/maorm/RICH/build/gnuReleaseMPI/rich
rmtv_mpirun=/software/x86_64/5.14.0/openmpi/4.1.6/gcc/12.3.0/bin/mpirun
rmtv_output="$rmtv_dir/verification/lbsteps_${SLURM_JOB_ID}"
cd "$rmtv_dir"
printf 'RMTV job=%s nodes=%s tasks=%s output=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_NTASKS" "$rmtv_output"
sha256sum "$rmtv_bin" reference/shape.dat
"$rmtv_mpirun" -np "$SLURM_NTASKS" --bind-to core "$rmtv_bin" \
    --octant --n 48 --quadrature 8 --dt-fraction .005 \
    --photons 16 --initial-photons 16 --max-photons 32 --dump 100000 --max-steps 30 --ddmc-min-tau 3 --lb-interval 1 --cost-step-scale 1 --cost-particle-scale 5 \
    --output "$rmtv_output"
echo "diagnostic run: no analysis"
