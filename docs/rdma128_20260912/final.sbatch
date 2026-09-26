#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --job-name=cp-rdma128-final
#SBATCH --output=/home/maorm/RICH/build/rdma128_20260912/final_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_20260912/final_%j.err
set -euo pipefail
cd /home/maorm/RICH/build/rdma128_20260912
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx
export FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
run_dir="final_${SLURM_JOB_ID}"
mkdir "$run_dir"
printf '%s\n' "nodes=$SLURM_JOB_NODELIST" "start=$(date -Is)"
sha256sum baseline candidate_final protocol_final
mpirun --map-by ppr:1:node:PE=1 --bind-to core -np 2 ./protocol_final ofi > "$run_dir/protocol_ofi.log" 2>&1
cat "$run_dir/protocol_ofi.log"
mpirun --map-by ppr:1:node:PE=1 --bind-to core -np 4 ./candidate_final 500 --optimized-rdma 1 1 --wall-layers 1 --wall-thin-layers 0 --wall-tangential 0.5 --boundary-photons 2 --max-steps 4 --manager rdma --rdma-engine ofi --rdma-ring-size 7 --rdma-ring-limit 64 --transport-event-budget 4 --send-max-age-us 1 --output-probes "$run_dir/smoke_probes.txt" > "$run_dir/smoke_ofi.log" 2>&1
for label in baseline1 candidate1 candidate2 default1 baseline2; do
    binary=candidate_final
    extra=()
    case "$label" in
      baseline*) binary=baseline ;;
      candidate*) extra=(--optimized-rdma) ;;
    esac
    printf 'begin %s %s\n' "$label" "$(date -Is)"
    TIMEFORMAT='%R'
    { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "./$binary" \
      20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 \
      "${extra[@]}" --output-probes "$run_dir/${label}_probes.txt" > "$run_dir/${label}.out" 2> "$run_dir/${label}.err"; } 2> "$run_dir/${label}.elapsed"
    printf 'end %s wall_seconds=%s\n' "$label" "$(cat "$run_dir/${label}.elapsed")"
    tail -2 "$run_dir/${label}.out"
done
printf 'end=%s\n' "$(date -Is)"
