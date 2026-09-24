#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:20:00
#SBATCH --job-name=cp-rdma128-fast
#SBATCH --output=/home/maorm/RICH/build/rdma128_55/accepted_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_55/accepted_%j.err
# Fastest verified O2 configuration for the 120-cycle comparison case.
# Extra script arguments override the case defaults below.
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
export STORM_FAST_LOCAL="/tmp/storm_${USER}_fast_${SLURM_JOB_ID}"
run_dir="/home/maorm/RICH/build/rdma128_55/accepted_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
sha256sum /home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe > "$run_dir/binary.sha256"
printf 'nodes=%s\n' "$SLURM_JOB_NODELIST" > "$run_dir/allocation.txt"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
  mkdir -p "$STORM_FAST_LOCAL"
  cp /home/maorm/RICH/source/monte/examples/crooked_pipe/crooked_pipe "$STORM_FAST_LOCAL/crooked_pipe"
'
cd "$STORM_FAST_LOCAL"
# The timer includes the full MPI invocation; staging and archival are outside it.
TIMEFORMAT='%R'
run_status=0
{ time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./crooked_pipe \
  20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi \
  --max-steps 120 --optimized-rdma --output-probes "$STORM_FAST_LOCAL/probes.csv" \
  "$@" > run.out 2> run.err; } 2> elapsed.seconds || run_status=$?
for output_file in run.out run.err elapsed.seconds probes.csv; do
  if [[ -f "$output_file" ]]; then cp "$output_file" "$run_dir/"; fi
done
printf 'MPI elapsed: %s seconds; exit status: %s; results: %s\n' "$(cat elapsed.seconds)" "$run_status" "$run_dir"
exit "$run_status"
