#!/bin/bash
set -eu
rank=${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-unknown}}
exec perf record -q -e cycles:u -F 199 --call-graph dwarf,8192 -o "${PROFILE_PREFIX}.rank${rank}.data" -- "$@"
