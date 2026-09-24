#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --job-name=cp-paper-check
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --exclusive
#SBATCH --nodelist=d25g[133-140]
#SBATCH --time=00:30:00
#SBATCH --output=/home/maorm/RICH/build/cp_paper_regression/job_%j.out
#SBATCH --error=/home/maorm/RICH/build/cp_paper_regression/job_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
archive=/home/maorm/RICH/build/cp_paper_regression/run_${SLURM_JOB_ID}
mkdir -p "$archive"
export CP_PAPER_LOCAL=/tmp/storm_${USER}_paper_${SLURM_JOB_ID}
srun --nodes=8 --ntasks=8 --ntasks-per-node=1 bash -c '
 mkdir -p "$CP_PAPER_LOCAL"
 cp /home/maorm/RICH/build/rdma128_20260912/baseline "$CP_PAPER_LOCAL/baseline"
 cp /home/maorm/RICH/build/rdma128_55/direct_log "$CP_PAPER_LOCAL/optimized"
'
cd "$CP_PAPER_LOCAL"
sha256sum baseline optimized > "$archive/binaries.sha256"
for label in baseline optimized; do
 extra=()
 if [[ "$label" == optimized ]]; then extra=(--optimized-rdma); fi
 echo "Starting $label $(date -Is)"
 TIMEFORMAT='%R'
 rc=0
 { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "./$label" \
  20000 6 6 --channel-points 400000 --probe-radius 0.1 \
  --census-floor 8 --census-cap 2000 --emission-floor 4 \
  --rebalance-interval 5 --transport-work-scale 0.01 \
  --wall-layers 188 --wall-thin-layers 8 --wall-width 1e-4 \
  --wall-tangential 0.06 --wall-growth 2 --boundary-photons 100 \
  --max-dt 1e-8 --final-time 4e-8 --max-steps 170 \
  --ddmc --ddmc-min-tau 3 --manager rdma --rdma-engine ofi \
  "${extra[@]}" --output-probes "$archive/${label}_probes.csv" \
  > "$archive/$label.out" 2> "$archive/$label.err"; } 2> "$archive/$label.elapsed" || rc=$?
 echo "Finished $label rc=$rc $(date -Is)"
 if [[ "$rc" != 0 ]]; then exit "$rc"; fi
done
