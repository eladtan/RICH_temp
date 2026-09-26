#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --job-name=cp-diagnose
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --nodelist=d25g[133-140]
#SBATCH --time=00:30:00
#SBATCH --output=/home/maorm/RICH/build/branson_cp128/diagnosis/job_%j.out
#SBATCH --error=/home/maorm/RICH/build/branson_cp128/diagnosis/job_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
export CP_DIAG_LOCAL="/tmp/storm_${USER}_cp_diag_${SLURM_JOB_ID}"
archive="/home/maorm/RICH/build/branson_cp128/diagnosis/run_${SLURM_JOB_ID}"
mkdir -p "$archive"
printf 'nodes=%s\n' "$SLURM_JOB_NODELIST" > "$archive/allocation.txt"
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
 mkdir -p "$CP_DIAG_LOCAL"
 cp /home/maorm/RICH/build/branson_cp128/diagnosis/branson_o2/BRANSON "$CP_DIAG_LOCAL/"
 cp /home/maorm/RICH/build/branson_cp128/diagnosis/*.xml "$CP_DIAG_LOCAL/"
 cp /home/maorm/RICH/build/rdma128_55/direct_log "$CP_DIAG_LOCAL/storm"
'
cd "$CP_DIAG_LOCAL"
sha256sum BRANSON storm *.xml > "$archive/inputs.sha256"
TIMEFORMAT='%R'
for label in coarse_warm coarse_empty fine_empty storm_no_rw; do
 echo "starting $label at $(date -Is)"
 rc=0
 if [[ "$label" == storm_no_rw ]]; then
  { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./storm 20000 2 10 --boundary-photons 100 --manager rdma --rdma-engine ofi --max-steps 120 --optimized-rdma --no-random-walk --output-probes "$CP_DIAG_LOCAL/${label}_probes.csv" > "$label.out" 2> "$label.err"; } 2> "$label.elapsed" || rc=$?
 else
  { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 ./BRANSON "$label.xml" > "$label.out" 2> "$label.err"; } 2> "$label.elapsed" || rc=$?
 fi
 cp "$label.out" "$label.err" "$label.elapsed" "$archive/"
 if [[ -f "${label}_probes.csv" ]]; then cp "${label}_probes.csv" "$archive/"; fi
 printf '%s rc=%s elapsed=%s\n' "$label" "$rc" "$(cat "$label.elapsed")"
 if [[ "$rc" != 0 ]]; then exit "$rc"; fi
done
