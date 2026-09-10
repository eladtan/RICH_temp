#!/usr/bin/env bash
#SBATCH --job-name=RMTV_n48_DDMC
#SBATCH --partition=bigrun
#SBATCH --nodes=4
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --chdir=/home/maorm/RICH/runs/RMTV
#SBATCH --output=/home/maorm/RICH/runs/RMTV/verification/bigrun_n48_%j.out
#SBATCH --error=/home/maorm/RICH/runs/RMTV/verification/bigrun_n48_%j.err

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this script with sbatch; do not run simulations on gateway nodes}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
rmtv_dir=/home/maorm/RICH/runs/RMTV
rmtv_bin=/home/maorm/RICH/build/gnuReleaseMPI/rich
rmtv_mpirun=/software/x86_64/5.14.0/openmpi/4.1.6/gcc/12.3.0/bin/mpirun
rmtv_output="$rmtv_dir/verification/convergence_n48_bigrun_${SLURM_JOB_ID}"
cd "$rmtv_dir"
printf 'RMTV job=%s nodes=%s tasks=%s output=%s\n' "$SLURM_JOB_ID" "$SLURM_JOB_NODELIST" "$SLURM_NTASKS" "$rmtv_output"
sha256sum "$rmtv_bin" reference/shape.dat
"$rmtv_mpirun" -np "$SLURM_NTASKS" --bind-to core "$rmtv_bin" \
    --octant --n 48 --quadrature 8 --dt-fraction .005 \
    --photons 16 --initial-photons 16 --max-photons 32 --dump 100 \
    --output "$rmtv_output"
python3 analyze.py "$rmtv_output" --smoke --require-complete --plot
