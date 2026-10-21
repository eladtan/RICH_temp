#!/bin/bash
# Moving slab (McClarren & Gentile, 124 groups) on the GPUs: RICH built --with-gpu
# against a CUDA Kokkos, 32 ranks x 9 cores, 8 ranks per GPU sharing it through MPS.
#SBATCH --account=e-dev-2026d09-248
#SBATCH --partition=booster
#SBATCH --job-name=slab_gpu
#SBATCH --nodes=1                     # MPS below is started once, in the batch step: single node only
#SBATCH --ntasks-per-node=32          # 8 ranks per GPU
#SBATCH --cpus-per-task=9             # 32 x 9 = 288 cores
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=maor.mizrachi@mail.huji.ac.il

module --force purge
module load Stages/2025 GCC/13.3.0 OpenMPI/5.0.5 HDF5/1.14.5 VTK/9.4.1 Boost/1.86.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK} OMP_PROC_BIND=spread OMP_PLACES=threads
export RICH_OUTPUT_DIR=$SCRATCH_e_dev_2026d09_248/MovingSlab/gpu

# One MPS server for the node, started in the batch step so it lives for the whole job.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/mps_pipe_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/mps_log_$SLURM_JOB_ID
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d || { echo "Failed to start the MPS control daemon" >&2; exit 1; }
stop_mps() {
    echo quit | nvidia-cuda-mps-control
    rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
}
trap stop_mps EXIT

srun --gpu-bind=closest ../../build/gnuReleaseMPI/slab-gpu/rich_gnuReleaseMPI
status=$?

# MPS logs one "NEW CLIENT" line per process that connected through it
echo "MPS clients served: $(cat "$CUDA_MPS_LOG_DIRECTORY/control.log" 2>/dev/null | grep -c 'NEW CLIENT')"
exit $status
