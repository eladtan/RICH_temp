#ifndef INDIVIDUAL_GHOST_SOURCES_HPP
#define INDIVIDUAL_GHOST_SOURCES_HPP

#include "mpi_commands.hpp"
#include "../3D/tessellation/Tessellation3D.hpp"
#include <algorithm>
#include <limits>
#include <map>
#include <vector>

// Ghost-send lists use canonical indices, not the compact partial target order.
// Only targets have complete geometry for owner-computed slopes or source fields.
inline std::vector<std::vector<std::size_t> > CollectIndividualGhostSourceIndices(
    Tessellation3D const& tess)
{
    const std::size_t owned_count = tess.GetPointNo();
    auto const& peers = tess.GetDuplicatedProcs();
    auto const& duplicated_all_points = tess.GetDuplicatedAllPointIndices();
    auto const& local_to_canonical = tess.GetIndicesInAllPoints();
    bool metadata_valid = peers.size() == duplicated_all_points.size() &&
        local_to_canonical.size() >= owned_count;
    std::map<std::size_t, std::size_t> canonical_to_local;
    for(auto const& mapping : local_to_canonical)
    {
        if(mapping.first >= owned_count)
            continue;
        if(mapping.second == std::numeric_limits<std::size_t>::max())
        {
            metadata_valid = false;
            continue;
        }
        auto const inserted = canonical_to_local.emplace(mapping.second, mapping.first);
        if(!inserted.second && inserted.first->second != mapping.first)
            metadata_valid = false;
    }

    std::vector<std::vector<std::size_t> > sources(peers.size());
    for(std::size_t peer = 0;
        peer < duplicated_all_points.size() && peer < sources.size(); ++peer)
    {
        for(std::size_t all_point : duplicated_all_points[peer])
        {
            if(all_point >= tess.GetAllPointsNo())
            {
                metadata_valid = false;
                continue;
            }
            std::size_t const canonical = tess.GetInputIndexForAllPoint(all_point);
            auto const local = canonical_to_local.find(canonical);
            if(local != canonical_to_local.end())
                sources[peer].push_back(local->second);
        }
        std::sort(sources[peer].begin(), sources[peer].end());
        sources[peer].erase(std::unique(sources[peer].begin(), sources[peer].end()),
            sources[peer].end());
    }

    int metadata_valid_int = metadata_valid ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, &metadata_valid_int, 1, MPI_INT, MPI_MIN,
        MPI_COMM_WORLD);
    if(metadata_valid_int == 0)
        throw UniversalError("Individual ghost source metadata is invalid");
    return sources;
}

#endif
