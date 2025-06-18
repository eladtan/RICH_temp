#ifdef RICH_MPI

#include "ExchangeFaces.hpp"

boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ExchangeFaces(const Tessellation3D &tess, const std::vector<size_t> &facesList)
{
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> faces_map;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ghosts_map = ExchangeGhosts(tess);

    boost::container::flat_map<rank_t, size_t> ranksDupIndices;
    for(size_t i = 0; i < tess.GetDuplicatedProcs().size(); ++i)
    {
        ranksDupIndices.insert({tess.GetDuplicatedProcs()[i], i});
    }

    struct NeighborsInfo
    {
        size_t localFaceIndex;
        size_t remoteLocalGhostIndex;
        size_t neighborIndexInGhostsArray;
    };

    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t N = tess.GetPointNo();
    std::vector<std::vector<NeighborsInfo>> neighbors(size);
    for(const size_t &faceIdx : facesList)
    {
        const std::pair<size_t, size_t> &faceNeighbors = tess.GetFaceNeighbors(faceIdx);
        if(faceNeighbors.first < N and faceNeighbors.second < N)
        {
            // local neighbors
            faces_map.insert({faceIdx, {rank, faceIdx}});
            continue;
        }

        size_t ghostIndex = (faceNeighbors.first < N)? faceNeighbors.second : faceNeighbors.first;
        const auto &[otherRank, remoteLocalGhostIndex] = ghosts_map.at(ghostIndex);

        size_t localIndex = (faceNeighbors.first < N)? faceNeighbors.first : faceNeighbors.second;
        
        size_t rankIdx = ranksDupIndices.at(otherRank);

        // search `localIndex` in `DupPoints[rankIdx]`
        const std::vector<size_t> &dupPoints = tess.GetDuplicatedPoints()[rankIdx];
        auto it = std::find(dupPoints.cbegin(), dupPoints.cend(), localIndex);
        if(it == dupPoints.cend())
        {
            // should not happen
            UniversalError eo("Local index not found in duplicated points");
            eo.addEntry("Local Index", localIndex);
            eo.addEntry("Rank", rank);
            eo.addEntry("Other Rank", otherRank);
            eo.addEntry("Ghost Index", ghostIndex);
            eo.addEntry("Face Index", faceIdx);
            throw eo;
        }

        size_t neighborIndexInGhostsArray = std::distance(dupPoints.cbegin(), it);
        // translate my local point to the language of the peer
        neighbors[otherRank].push_back({faceIdx, remoteLocalGhostIndex, neighborIndexInGhostsArray});
    }

    // now exchange the information about neighbors
    std::vector<std::vector<NeighborsInfo>> exchangedNeighbors = MPI_Exchange_all_to_all(neighbors, MPI_COMM_WORLD);

    struct FacesMatch
    {
        size_t localFaceIndex;
        size_t remoteFaceIndex;
    };

    std::vector<std::vector<FacesMatch>> facesMatches(size);

    for(rank_t otherRank = 0; otherRank < size; ++otherRank)
    {
        size_t rankIndexInGhost = ranksDupIndices.at(otherRank);
        const std::vector<size_t> &ghostsPoints = tess.GetGhostIndeces()[rankIndexInGhost];
        for(const NeighborsInfo &info : exchangedNeighbors[otherRank])
        {
            size_t onePoint = info.remoteLocalGhostIndex;
            size_t secondPoint = ghostsPoints[info.neighborIndexInGhostsArray];

            bool found = false;
            // find the matching face
            for(const size_t &faceIdx : tess.GetCellFaces(onePoint))
            {
                const std::pair<size_t, size_t> &faceNeighbors = tess.GetFaceNeighbors(faceIdx);
                if(faceNeighbors.first == secondPoint or faceNeighbors.second == secondPoint)
                {
                    // found a match
                    facesMatches[otherRank].push_back({info.localFaceIndex, faceIdx});
                    found = true;
                    break; // no need to check other faces
                }
            }
            if(not found)
            {
                UniversalError eo("Matching face not found for ghost point");
                eo.addEntry("Ghost Point", secondPoint);
                eo.addEntry("Remote Local Ghost Index", info.remoteLocalGhostIndex);
                eo.addEntry("Neighbor Index In Ghosts Array", info.neighborIndexInGhostsArray);
                eo.addEntry("Rank", rank);
                eo.addEntry("Other Rank", otherRank);
                eo.addEntry("Face Index", info.localFaceIndex);
                throw eo;
            }
        }
    }

    std::vector<std::vector<FacesMatch>> exchangedFacesMatches = MPI_Exchange_all_to_all(facesMatches, MPI_COMM_WORLD);

    for(rank_t otherRank = 0; otherRank < size; otherRank++)
    {
        for(const FacesMatch &match : exchangedFacesMatches[otherRank])
        {
            faces_map.insert({match.localFaceIndex, {otherRank, match.remoteFaceIndex}});
        }
    }
    return faces_map;
}

#endif // RICH_MPI