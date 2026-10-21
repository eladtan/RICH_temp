#!/bin/bash
# Single-node hohlraum on Frontier CPU threads: RICH built --with-gpu against the
# OpenMP-only Kokkos (build/gnuReleaseMPI/hohl-omp). 56 usable cores per node
# (low-noise mode). Pick the split at submission:
#   sbatch --export=ALL,RANKS=1,THREADS=56  submit_frontier_omp.sh
#   sbatch --export=ALL,RANKS=56,THREADS=1  submit_frontier_omp.sh
# EXTRA_ARGS adds driver options; RICH_BIN overrides the binary; LAUNCH_PREFIX wraps
# the binary (e.g. a profiler).
#SBATCH --job-name=hohlraum_omp
#SBATCH --account=ast246
#SBATCH --partition=service
#SBATCH --qos=develop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=56          # MPI slots; RANKS x THREADS must fit in 56 cores
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=hohlraum_omp_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

RANKS=${RANKS:-56}
THREADS=${THREADS:-1}
run_dir=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$run_dir"
source ../frontier_env.sh omp

RICH_BIN=${RICH_BIN:-../../build/gnuReleaseMPI/hohl-omp/rich_gnuReleaseMPI}
export OMP_NUM_THREADS=$THREADS OMP_PROC_BIND=spread OMP_PLACES=cores
export RICH_OUTPUT_DIR=/lustre/orion/ast246/scratch/$USER/Hohlraum/${RANKS}x${THREADS}${RUN_TAG:+_$RUN_TAG}
mkdir -p "$RICH_OUTPUT_DIR"
echo "Ranks: $RANKS, OpenMP threads per rank: $THREADS, binary $RICH_BIN, output: $RICH_OUTPUT_DIR"

# JUPITER used --ibv (InfiniBand); Slingshot goes through the OFI/MPI RDMA manager.
# N_base = 892 for one node, as in run.sh.
mpirun -np "$RANKS" --map-by "ppr:${RANKS}:node:pe=${THREADS}" --bind-to core --report-bindings \
    ${LAUNCH_PREFIX:-} "$RICH_BIN" 892 --new-rdma --hold-small-idle-flushes ${EXTRA_ARGS:-}
status=$?
echo "mpirun exit status: $status"
exit $status
