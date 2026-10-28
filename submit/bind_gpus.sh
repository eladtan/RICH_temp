#!/bin/bash
# Give one MPI rank its GPUs, then run the program:
#   mpirun ... submit/bind_gpus.sh ./rich ...
#
# Set by the generated job script:
#   RICH_SUBMIT_GPUS_PER_RANK  GPUs each rank sees
#   RICH_SUBMIT_RANKS_PER_GPU  ranks sharing one GPU (oversubscription; GPUs per rank is then 1)
#   RICH_SUBMIT_GPU_ENV        ROCR_VISIBLE_DEVICES (AMD) or CUDA_VISIBLE_DEVICES (NVIDIA)
#   RICH_SUBMIT_GPU_ORDER      optional: GPU ids in the order of the cores they sit next to, e.g.
#                              "4 5 2 3 6 7 0 1" on Frontier (GCD of CCD 0, 1, ...). Ranks are
#                              placed on consecutive cores, so local rank order follows it.
# The visible GPUs are renumbered from 0 inside the process, so other masks are cleared.
set -euo pipefail

per_rank=${RICH_SUBMIT_GPUS_PER_RANK:-1}
per_gpu=${RICH_SUBMIT_RANKS_PER_GPU:-1}
variable=${RICH_SUBMIT_GPU_ENV:-CUDA_VISIBLE_DEVICES}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-${MPI_LOCALRANKID:-${PMI_LOCAL_RANK:-0}}}}
read -r -a order <<< "${RICH_SUBMIT_GPU_ORDER:-}"

if (( per_gpu > 1 )); then
    slots=($(( local_rank / per_gpu )))
else
    slots=()
    for (( i = 0; i < per_rank; i++ )); do
        slots+=($(( local_rank * per_rank + i )))
    done
fi
devices=()
for slot in "${slots[@]}"; do
    if (( ${#order[@]} )); then
        devices+=("${order[slot % ${#order[@]}]}")
    else
        devices+=("$slot")
    fi
done

unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES
export "$variable=$(IFS=,; echo "${devices[*]}")"

if [[ -n "${RICH_REPORT_BINDING:-}" ]]; then
    rank=${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}
    echo "rank $rank (local $local_rank) on $(hostname): cores" \
        "$(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status), $variable=${!variable}" >&2
fi
exec "$@"
