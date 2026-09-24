#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=cp-o2-comparison
#SBATCH --output=/home/maorm/RICH/build/rdma128_o2/compare_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_o2/compare_%j.err
set -euo pipefail
cd /home/maorm/RICH/build/rdma128_20260912
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx
export FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
run_dir="/home/maorm/RICH/build/rdma128_o2/compare_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
printf '%s\n' "nodes=$SLURM_JOB_NODELIST"
for backend in ofi mpi; do
  mpirun --map-by ppr:1:node:PE=1 --bind-to core -np 2 /home/maorm/RICH/build/rdma128_o2/protocol_o2 "$backend" > "$run_dir/protocol_${backend}.log" 2>&1
  cat "$run_dir/protocol_${backend}.log"
done
for binary in candidate_o2; do
  mpirun --map-by ppr:2:node:PE=1 --bind-to core -np 4 "/home/maorm/RICH/build/rdma128_o2/$binary" \
    2000 5 10 --wall-layers 1 --wall-thin-layers 1 --wall-tangential 0.1 --boundary-photons 100 \
    --max-steps 24 --energy-ledger --manager rdma --rdma-engine ofi --optimized-rdma \
    --rdma-ring-size 7 --rdma-ring-limit 64 --transport-event-budget 4 --send-max-age-us 1 \
    --output-probes "$run_dir/${binary}_ledger_probes.txt" > "$run_dir/${binary}_ledger.out" 2> "$run_dir/${binary}_ledger.err"
done
for label in baseline1 o3_1 o2_1 o2_2 o3_2 o3_3 o2_3 baseline2; do
    binary=/home/maorm/RICH/build/rdma128_followup/candidate_final
    extra=(--optimized-rdma)
    case "$label" in
      baseline*) binary=./baseline; extra=() ;;
      o2_*) binary=/home/maorm/RICH/build/rdma128_o2/candidate_o2 ;;
    esac
    TIMEFORMAT='%R'
    { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "$binary" \
      20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 \
      "${extra[@]}" --output-probes "$run_dir/${label}_probes.txt" > "$run_dir/${label}.out" 2> "$run_dir/${label}.err"; } 2> "$run_dir/${label}.elapsed"
    printf '%s %s seconds\n' "$label" "$(cat "$run_dir/${label}.elapsed")"
done
