#!/bin/bash

# asan_wrap.sh
rank=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}
export ASAN_OPTIONS="${ASAN_OPTIONS}:log_path=asan.r${rank}"
exec "$@"
