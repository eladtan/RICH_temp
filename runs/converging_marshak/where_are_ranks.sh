#!/bin/bash
# Sample every local rich rank and report the application-level frame it is stuck in.
pids=$(pgrep rich_ddmc_densm)
for p in ${pids}; do
    perf record -F 99 -g -p "${p}" -o "/tmp/mm_${p}.data" -- sleep 4 >/dev/null 2>&1 &
done
wait
for p in ${pids}; do
    sym=$(perf report -i "/tmp/mm_${p}.data" --stdio --no-children -g graph,0.5,caller --percent-limit 8 2>/dev/null \
        | grep -oE 'RMAFactory::Initialize|RadiationMCStep::RadiationMCStep|RDMAMonteCarloManager|Voronoi3D::[A-Za-z_]+|RadiationIMC::[A-Za-z_]+|PMPI_[A-Za-z_]+|ompi_coll_base_[a-z_]+' \
        | sort -u | tr '\n' ',')
    echo "$(hostname) ${p} ${sym:-NOSYM}"
done
rm -f /tmp/mm_*.data
