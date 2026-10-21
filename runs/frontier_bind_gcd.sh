#!/bin/bash
# Run one MPI rank on the Frontier GCD closest to the cores it is bound to:
#   mpirun --map-by ppr:1:l3cache:pe=7 --bind-to core runs/frontier_bind_gcd.sh ./rich ...
#
# A Frontier node has 8 CCDs (8 cores and one L3 each; core 0 of each is reserved
# in low-noise mode) and 8 GCDs. The GCD attached to CCD c is GCD_OF_CCD[c]
# (OLCF Frontier user guide, node diagram). ROCR_VISIBLE_DEVICES leaves the rank
# one GCD, which HIP renumbers to device 0, so HIP/CUDA masks are cleared.
set -euo pipefail

GCD_OF_CCD=(4 5 2 3 6 7 0 1)
first_core=$(awk '/^Cpus_allowed_list/ {split($2, a, /[-,]/); print a[1]}' /proc/self/status)
ccd=$(( first_core / 8 ))
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=${GCD_OF_CCD[$ccd]}

if [[ -n "${RICH_REPORT_BINDING:-}" ]]; then
    rank=${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}
    echo "rank $rank on $(hostname): cores $(awk '/^Cpus_allowed_list/ {print $2}' /proc/self/status), GCD $ROCR_VISIBLE_DEVICES" >&2
fi
exec "$@"
