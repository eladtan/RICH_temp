#!/bin/bash
# Mach2 on Frontier, GPU or CPU build, 8 ranks per node (one per CCD, 7 cores each).
# GPU: each rank gets the GCD next to its CCD. Submit from this directory:
#   sbatch submit_frontier.sh                      # GPU build (build/gnuReleaseMPI/mach2-hip)
#   sbatch --export=ALL,MODE=cpu submit_frontier.sh # CPU build (build/gnuReleaseMPI/mach2-cpu)
# RICH_BIN overrides the binary, RUN_TAG suffixes the output directory.
#SBATCH --job-name=mach2
#SBATCH --account=ast246
#SBATCH --partition=service
#SBATCH --qos=develop
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=mach2_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

MODE=${MODE:-gpu}
run_dir=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
cd "$run_dir"
source ../frontier_env.sh hip

case $MODE in
    gpu) default_bin=../../build/gnuReleaseMPI/mach2-hip/rich_gnuReleaseMPI
         launch=(../frontier_bind_gcd.sh) ;;
    cpu) default_bin=../../build/gnuReleaseMPI/mach2-cpu/rich_gnuReleaseMPI
         launch=() ;;
    *) echo "MODE must be gpu or cpu" >&2; exit 1 ;;
esac
RICH_BIN=${RICH_BIN:-$default_bin}

export OMP_NUM_THREADS=7 OMP_PROC_BIND=close OMP_PLACES=cores
export RICH_OUTPUT_DIR=/lustre/orion/ast246/scratch/$USER/Mach2/${MODE}${RUN_TAG:+_$RUN_TAG}
export RICH_REPORT_BINDING=1
mkdir -p "$RICH_OUTPUT_DIR"
echo "Mode $MODE, binary $RICH_BIN, ranks $SLURM_NTASKS, output $RICH_OUTPUT_DIR"

# Same arguments as the JUPITER runs (submit_gpu.sh)
mpirun -np "$SLURM_NTASKS" --map-by ppr:1:l3cache:pe=7 --bind-to core \
    "${launch[@]}" "$RICH_BIN" 1024 "$RICH_OUTPUT_DIR/mach2"
status=$?
echo "mpirun exit status: $status"
exit $status
