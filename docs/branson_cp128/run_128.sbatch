#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --job-name=branson-cp128
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --nodelist=d25g[133-140]
#SBATCH --time=00:30:00
#SBATCH --output=/home/maorm/RICH/build/branson_cp128/job_%j.out
#SBATCH --error=/home/maorm/RICH/build/branson_cp128/job_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
export CP_COMPARE_LOCAL="/tmp/storm_${USER}_branson_cp_${SLURM_JOB_ID}"
archive="/home/maorm/RICH/build/branson_cp128/run_${SLURM_JOB_ID}"
mkdir -p "$archive"
printf 'nodes=%s\n' "$SLURM_JOB_NODELIST" > "$archive/allocation.txt"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
 mkdir -p "$CP_COMPARE_LOCAL"
 cp /home/maorm/RICH/build/branson_cp128/branson_o2/BRANSON "$CP_COMPARE_LOCAL/"
 cp /home/maorm/RICH/build/branson_cp128/crooked_pipe.xml "$CP_COMPARE_LOCAL/"
 cp /home/maorm/RICH/build/rdma128_55/direct_log "$CP_COMPARE_LOCAL/storm"
'
cd "$CP_COMPARE_LOCAL"
sha256sum BRANSON storm crooked_pipe.xml > "$archive/inputs.sha256"
cp crooked_pipe.xml "$archive/"
TIMEFORMAT='%R'
rounds="${1:-1}"
for ((round=1;round<=rounds;round++)); do
 if (( round % 2 == 0 )); then order=(storm branson); else order=(branson storm); fi
 for code in "${order[@]}"; do
  label="${code}${round}"
  echo "starting $label at $(date -Is)"
  rc=0
  if [[ "$code" == branson ]]; then
   { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./BRANSON crooked_pipe.xml > "$label.out" 2> "$label.err"; } 2> "$label.elapsed" || rc=$?
  else
   { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./storm 20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 --optimized-rdma --output-probes "$CP_COMPARE_LOCAL/${label}_probes.csv" > "$label.out" 2> "$label.err"; } 2> "$label.elapsed" || rc=$?
  fi
  cp "$label.out" "$label.err" "$label.elapsed" "$archive/"
  if [[ -f "${label}_probes.csv" ]]; then cp "${label}_probes.csv" "$archive/"; fi
  printf '%s rc=%s elapsed=%s\n' "$label" "$rc" "$(cat "$label.elapsed")"
  if [[ "$rc" != 0 ]]; then exit "$rc"; fi
 done
done
