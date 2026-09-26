#!/bin/bash
#SBATCH --partition=bigrun
#SBATCH --nodes=8
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --job-name=cp-o2-profile-p2p
#SBATCH --output=/home/maorm/RICH/build/rdma128_next/profile_%j.out
#SBATCH --error=/home/maorm/RICH/build/rdma128_next/profile_%j.err
set -euo pipefail
cd /home/maorm/RICH/build/rdma128_20260912
export OMP_NUM_THREADS=1 KOKKOS_NUM_THREADS=1
export OMPI_MCA_btl=^openib OMPI_MCA_osc=ucx
export FI_PROVIDER=verbs
unset CP_BRANSON_WEIGHTS
run_dir="/home/maorm/RICH/build/rdma128_next/profile_${SLURM_JOB_ID}"
mkdir -p "$run_dir"
for label in rdma_profile p2p_profile p2p_default p2p_fair; do
    extra=(--optimized-rdma)
    manager=p2p
    sample=()
    case "$label" in
      rdma_profile) manager=rdma ;;
      p2p_default) extra=() ;;
    esac
    case "$label" in
      *_profile)
        export STORM_SAMPLE_DIR="$run_dir/$label"
        mkdir -p "$STORM_SAMPLE_DIR"
        sample=(-x LD_PRELOAD=/home/maorm/RICH/build/rdma128_followup/sampler.so -x STORM_SAMPLE_DIR) ;;
    esac
    TIMEFORMAT='%R'
    { time mpirun --map-by ppr:16:node:PE=1 --bind-to core -np 128 "${sample[@]}" \
      /home/maorm/RICH/build/rdma128_o2/candidate_o2 \
      20000 2 10 --boundary-photons 100 --manager "$manager" --rdma-engine ofi --max-steps 120 \
      "${extra[@]}" --output-probes "$run_dir/${label}_probes.txt" > "$run_dir/${label}.out" 2> "$run_dir/${label}.err"; } 2> "$run_dir/${label}.elapsed"
    printf '%s %s seconds\n' "$label" "$(cat "$run_dir/${label}.elapsed")"
done
