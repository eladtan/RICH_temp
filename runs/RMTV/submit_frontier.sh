#!/bin/bash
# RMTV on Frontier: octant, Lagrangian mesh (rebuilt every cycle), DDMC. MODE picks the build
# and layout (submit from this directory):
#   cpu      CPU build, 1 thread per rank, e.g. 56 ranks on one node:
#              sbatch -p service --qos=develop -N 1 --ntasks-per-node=56 --export=ALL,MODE=cpu submit_frontier.sh
#   omp      OpenMP-Kokkos build: Monte Carlo and Voronoi subdomains on 7 threads per rank
#   gpu      HIP build: Monte Carlo on the rank's GCD, Voronoi as one domain per rank
#   gpu-sub  HIP build: Monte Carlo on the rank's GCD, Voronoi subdomains on 7 threads
# The defaults below are 8 nodes x 7 ranks; each rank gets one CCD (7 cores) and, with a
# GPU, the GCD next to it. STEPS, N and EXTRA_ARGS override the problem; RICH_BIN the binary.
#SBATCH --job-name=rmtv
#SBATCH --account=ast246
#SBATCH --partition=batch
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=7
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --output=rmtv_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

MODE=${MODE:-gpu}
# sbatch runs a spool copy of this script, so use the submission directory; when the
# script is called from inside another job, use its own directory
run_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/submit_frontier.sh" ]] && run_dir=$SLURM_SUBMIT_DIR
cd "$run_dir"
build=../../build/gnuReleaseMPI
launch=()
case $MODE in
    cpu)     kokkos=hip; threads=1; subdomains=0; bin=$build/rmtv-cpu/rich_gnuReleaseMPI ;;
    omp)     kokkos=omp; threads=7; subdomains=7; bin=$build/rmtv-omp/rich_gnuReleaseMPI ;;
    gpu)     kokkos=hip; threads=7; subdomains=0; bin=$build/rmtv-hip/rich_gnuReleaseMPI; launch=(../frontier_bind_gcd.sh) ;;
    gpu-sub) kokkos=hip; threads=7; subdomains=7; bin=$build/rmtv-hip/rich_gnuReleaseMPI; launch=(../frontier_bind_gcd.sh) ;;
    *) echo "MODE must be cpu, omp, gpu or gpu-sub" >&2; exit 1 ;;
esac
source ../frontier_env.sh "$kokkos"
RICH_BIN=${RICH_BIN:-$bin}
ranks_per_node=${SLURM_NTASKS_PER_NODE:-$(( SLURM_NTASKS / SLURM_JOB_NUM_NODES ))}

export OMP_NUM_THREADS=$threads OMP_PROC_BIND=close OMP_PLACES=cores
export RICH_REPORT_BINDING=1
output=/lustre/orion/ast246/scratch/$USER/RMTV/${MODE}_${SLURM_NTASKS}x${threads}_${SLURM_JOB_ID}
mkdir -p "$output"
echo "Mode $MODE: $SLURM_NTASKS ranks on $SLURM_JOB_NUM_NODES nodes x $threads threads, Voronoi subdomains $subdomains, binary $RICH_BIN, output $output"

# pe=7 takes the 7 usable cores of one CCD (low-noise mode reserves core 0 of each)
mpirun -np "$SLURM_NTASKS" --map-by "ppr:${ranks_per_node}:node:pe=${threads}" --bind-to core \
    "${launch[@]}" "$RICH_BIN" \
    --octant --n "${N:-96}" --quadrature 8 --dt-fraction .005 \
    --photons 16 --initial-photons 16 --max-photons 32 --dump 100000 --max-steps "${STEPS:-30}" \
    --ddmc-min-tau 3 --lb-interval 1 --motion lagrangian --jitter 1e-6 --lb-tolerance 3 \
    --voronoi-subdomains "$subdomains" --output "$output" ${EXTRA_ARGS:-}
status=$?
echo "mpirun exit status: $status"
exit $status
