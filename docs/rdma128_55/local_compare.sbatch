#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:25:00
#SBATCH --job-name=cp-rdma55-local
#SBATCH --output=/home/maorm/RICH/build/rdma128_55/local_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_55/local_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
export STORM_LOCAL_RUN="/tmp/storm_${USER}_rdma55_${SLURM_JOB_ID}"
run_dir="/home/maorm/RICH/build/rdma128_55/local_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
printf 'nodes=%s local=%s\n' "$SLURM_JOB_NODELIST" "$STORM_LOCAL_RUN"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
  mkdir -p "$STORM_LOCAL_RUN"
  cp /home/maorm/RICH/build/rdma128_o2/candidate_o2 "$STORM_LOCAL_RUN/previous"
  cp /home/maorm/RICH/build/rdma128_55/direct_log "$STORM_LOCAL_RUN/candidate"
  cp /home/maorm/RICH/build/rdma128_55/fixed_progress "$STORM_LOCAL_RUN/fixed"
  cp /home/maorm/RICH/build/rdma128_55/debug_final "$STORM_LOCAL_RUN/debug"
'
cd "$STORM_LOCAL_RUN"
/home/maorm/RICH/build/rdma128_55/log_rounding_test > math.log
/home/maorm/RICH/build/rdma128_55/direct_log_test >> math.log
lscpu > hardware.txt
for label in release_ofi release_mpi debug_ofi debug_mpi; do
  binary=./candidate
  backend=${label##*_}
  if [[ "$label" == debug* ]]; then binary=./debug; fi
  mpirun --map-by ppr:2:node:PE=1 --bind-to core -np 4 "$binary" \
    2000 5 10 --wall-layers 1 --wall-thin-layers 1 --wall-tangential 0.1 --boundary-photons 100 \
    --max-steps 24 --energy-ledger --manager rdma --rdma-engine "$backend" --optimized-rdma \
    --rdma-ring-size 7 --rdma-ring-limit 64 --transport-event-budget 4 --send-max-age-us 1 \
    --output-probes "$STORM_LOCAL_RUN/${label}_probes.txt" > "${label}.out" 2> "${label}.err"
  printf 'gate %s passed\n' "$label"
done
for label in previous1 log1 weight2 weighthalf log2 previous2 log3 previous3; do
  binary=./candidate
  extra=(--optimized-rdma)
  case "$label" in
    previous*) binary=./previous ;;
    weight2) binary=./fixed; extra+=(--transport-work-scale 2) ;;
    weighthalf) binary=./fixed; extra+=(--transport-work-scale 0.5) ;;
  esac
  TIMEFORMAT='%R'
  { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "$binary" \
    20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 \
    "${extra[@]}" --output-probes "$STORM_LOCAL_RUN/${label}_probes.txt" > "${label}.out" 2> "${label}.err"; } 2> "${label}.elapsed"
  printf '%s %s seconds\n' "$label" "$(cat "${label}.elapsed")"
done
# Archive only after every timed run; archival is outside the reported timers.
cp ./*.out ./*.err ./*.elapsed ./*_probes.txt ./math.log ./hardware.txt "$run_dir/"
