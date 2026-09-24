#!/bin/bash
set -eu
rank=${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-unknown}}
exec perf stat -x, -e cycles:u,instructions:u,assists.sse_avx_mix:u,assists.fp:u -o "${PROFILE_PREFIX}.rank${rank}.stat.csv" -- "$@"
