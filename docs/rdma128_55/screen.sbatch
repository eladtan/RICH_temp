#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=cp-rdma-55-screen
#SBATCH --output=/home/maorm/RICH/build/rdma128_55/trial_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_55/trial_%j.err
set -euo pipefail
cd /home/maorm/RICH/build/rdma128_20260912
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx
export FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
run_dir="/home/maorm/RICH/build/rdma128_55/trial_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
printf '%s\n' "nodes=$SLURM_JOB_NODELIST"
for label in previous fixed early; do
    binary=/home/maorm/RICH/build/rdma128_o2/candidate_o2
    extra=(--optimized-rdma)
    if [[ "$label" == fixed ]]; then binary=/home/maorm/RICH/build/rdma128_55/fixed_progress; fi
    if [[ "$label" == early ]]; then binary=/home/maorm/RICH/build/rdma128_55/early_rw; fi
    TIMEFORMAT='%R'
    { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "$binary" \
      20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 \
      "${extra[@]}" --output-probes "$run_dir/${label}_probes.txt" > "$run_dir/${label}.out" 2> "$run_dir/${label}.err"; } 2> "$run_dir/${label}.elapsed"
    printf '%s %s seconds\n' "$label" "$(cat "$run_dir/${label}.elapsed")"
done
