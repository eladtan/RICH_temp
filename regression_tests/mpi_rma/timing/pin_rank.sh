#!/usr/bin/env bash
set -euo pipefail
# Identical OS CPU IDs with Open MPI's SSH launch and MPICH's Slurm launch.
# Override the comma-separated list for another allocated machine.
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${SLURM_LOCALID:-}}}
if [[ -z "$local_rank" ]]; then
    printf 'Cannot identify the local MPI rank\n' >&2
    exit 1
fi
IFS=, read -r -a cores <<< "${BENCHMARK_CPUS:-0,2}"
if (( local_rank >= ${#cores[@]} )); then
    printf 'Not enough entries in BENCHMARK_CPUS\n' >&2
    exit 1
fi
exec taskset --cpu-list "${cores[local_rank]}" "$@"
