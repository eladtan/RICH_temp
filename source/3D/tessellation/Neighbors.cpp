#include "Neighbors.hpp"
#include <algorithm>

#ifdef RICH_MPI
    struct Request : public Serializable
    {
        size_t askingPointIdx;
        size_t remotePointIdxInGhost;

        Request(size_t _askingPointIdx = std::numeric_limits<size_t>::max(), size_t _remotePointIdxInGhost = std::numeric_limits<size_t>::max())
            : askingPointIdx(_askingPointIdx), remotePointIdxInGhost(_remotePointIdxInGhost){};

        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->askingPointIdx);
            bytes += serializer->insert(this->remotePointIdxInGhost);
            return bytes;
        };

        force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytesRead = 0;
            bytesRead += serializer->extract(this->askingPointIdx, byteOffset);
            bytesRead += serializer->extract(this->remotePointIdxInGhost, byteOffset + bytesRead);
            return bytesRead;
        };
    };

    struct Response : public Serializable
    {
        size_t requestID;
        RemotePoint neighbor;

        Response(size_t _requestID, const RemotePoint &_neighbor) : requestID(_requestID), neighbor(_neighbor) {};

        Response(): Response(std::numeric_limits<size_t>::max(), RemotePoint()) {};
        
        force_inline size_t dump(Serializer *serializer) const override
        {
            size_t bytes = 0;
            bytes += serializer->insert(this->requestID);
            bytes += serializer->insert(this->neighbor.rank);
            bytes += serializer->insert(this->neighbor.indexOnRank);
            bytes += serializer->insert(this->neighbor.distance);
            return bytes;
        };

        force_inline size_t load(const Serializer *serializer, size_t byteOffset) override
        {
            size_t bytesRead = 0;
            bytesRead += serializer->extract(this->requestID, byteOffset);
            bytesRead += serializer->extract(this->neighbor.rank, byteOffset + bytesRead);
            bytesRead += serializer->extract(this->neighbor.indexOnRank, byteOffset + bytesRead);
            bytesRead += serializer->extract(this->neighbor.distance, byteOffset + bytesRead);
            return bytesRead;
        };
    };

    std::ostream &operator<<(std::ostream &stream, const Request &request)
    {
        return stream << "Request(asking " << request.askingPointIdx << " ghost " << request.remotePointIdxInGhost << ")";
    };

    std::ostream &operator<<(std::ostream &stream, const Response &response)
    {
        return stream << "Response(ID " << response.requestID << ", Neighbor " << response.neighbor.indexOnRank << " in " << response.neighbor.rank << ")";
    };

#endif // RICH_MPI

void addPointToSet(boost::container::flat_set<RemotePoint> &set, const RemotePoint &point)
{
    auto it = set.find(point);
    if(it != set.end())
    {
        // change distance to 0
        (*it).distance = std::min<size_t>((*it).distance, point.distance);
    }
    else
    {
        set.insert(point);
    }
}

#ifdef RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost, const MPI_Comm &comm)
#else // RICH_MPI
    PointsToNeighborsMap GetKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost)
#endif // RICH_MPI
{
    PointsToNeighborsMap result;

    #ifdef RICH_MPI
        int rank = 0, size = 1;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    #endif // RICH_MPI

    if(order == 0)
    {
        // base case: return the points themselves
        for(const size_t &pointIdx : points)
        {
            #ifdef RICH_MPI
                result[pointIdx] = boost::container::flat_set<RemotePoint>({RemotePoint(rank, pointIdx, 0)});
            #else // RICH_MPI
                result[pointIdx] = boost::container::flat_set<RemotePoint>({RemotePoint(pointIdx, 0)});
            #endif // RICH_MPI
        }
        return result;
    }

    #ifdef RICH_MPI
        // send requests
        std::vector<std::vector<Request>> requests(size);
    #else // RICH_MPI
        std::vector<size_t> requestPoints;
    #endif// RICH_MPI
    std::vector<size_t> nextSearch;

    #ifdef RICH_MPI
        std::vector<std::pair<int, size_t>> indicesInGhosts;
        const std::vector<int> &dupProcs = tess.GetDuplicatedProcs();
    #endif // RICH_MPI

    std::vector<size_t> neighbor_buf;
    for(const size_t &pointIdx : points)
    {
        tess.GetNeighbors(pointIdx, neighbor_buf);
        for(const size_t &neighbor : neighbor_buf)
        {
            if(tess.IsPointOutsideBox(neighbor))
            {
                continue; // skip mirrored points
            }
            #ifdef RICH_MPI
                int _rank = tess.GetOwner(tess.GetMeshPoint(neighbor));
                if(_rank == rank)
                {
                    requests[rank].push_back({pointIdx, neighbor});
                    continue;
                }
                // otherwise, belongs to a remote
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                const std::vector<size_t> &ghostsOfRank = tess.GetGhostIndeces()[rankIndexInGhostIndices];
                size_t ghostIndexInGhost = std::distance(ghostsOfRank.cbegin(), std::find(ghostsOfRank.cbegin(), ghostsOfRank.cend(), neighbor));
                requests[_rank].push_back({pointIdx, ghostIndexInGhost});
            #else // RICH_MPI
                requestPoints.push_back(pointIdx);
                nextSearch.push_back(neighbor);
            #endif // RICH_MPI
        }
        result[pointIdx] = boost::container::flat_set<RemotePoint>();
        if(atMost)
        {
            #ifdef RICH_MPI
                result[pointIdx].insert(RemotePoint(rank, pointIdx, 0));
            #else // RICH_MPI
                result[pointIdx].insert(RemotePoint(pointIdx, 0));
            #endif // RICH_MPI
        }
    }

    #ifdef RICH_MPI
        std::vector<std::vector<Request>> incomingRequests = MPI_Exchange_all_to_all(requests, comm);

        // calculate what local points should continue the search
        for(int _rank = 0; _rank < size; _rank++)
        {
            if(incomingRequests[_rank].empty())
            {
                continue;
            }
            if(_rank != rank)
            {
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                assert(rankIndexInGhostIndices != dupProcs.size()); // validate rank indeed exists
                const std::vector<size_t> &dupPointsOfRank = tess.GetDuplicatedPoints()[rankIndexInGhostIndices];
                for(const Request &indexInGhostArray : incomingRequests[_rank])
                {
                    size_t localIndex = dupPointsOfRank[indexInGhostArray.remotePointIdxInGhost];
                    nextSearch.push_back(localIndex);
                }
            }
            else
            {
                for(const Request &indexInGhostArray : incomingRequests[_rank])
                {
                    size_t localIndex = indexInGhostArray.remotePointIdxInGhost;
                    nextSearch.push_back(localIndex);
                }
            }
        }
    #endif // RICH_MPI
    
    // Multiple queried points often share first-order neighbors. Recursing on
    // duplicates repeats the same MPI requests and graph traversal while the
    // result map ultimately keeps only one entry per point.
    std::vector<size_t> uniqueNextSearch = nextSearch;
    std::sort(uniqueNextSearch.begin(), uniqueNextSearch.end());
    uniqueNextSearch.erase(std::unique(uniqueNextSearch.begin(), uniqueNextSearch.end()),
                           uniqueNextSearch.end());

    // recurse - get neighbors of neighbors
    #ifdef RICH_MPI
        PointsToNeighborsMap neighborPoints = GetKOrderNeighbors(tess, uniqueNextSearch, order - 1, atMost, comm);
    #else // RICH_MPI
        PointsToNeighborsMap neighborPoints = GetKOrderNeighbors(tess, uniqueNextSearch, order - 1, atMost);
    #endif // RICH_MPI

    #ifdef RICH_MPI
        // now, answer back the results
        std::vector<std::vector<Response>> responses(size); // will be used to exchange_all_to_all
        
        // calculate what local points should continue the search
        for(int _rank = 0; _rank < size; _rank++)
        {
            const std::vector<Request> &requestsOfRank = incomingRequests[_rank];
            if(requestsOfRank.empty())
            {
                continue;
            }
            if(_rank != rank)
            {
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                assert(rankIndexInGhostIndices != dupProcs.size()); // validate rank indeed exists
                const std::vector<size_t> &dupPointsOfRank = tess.GetDuplicatedPoints()[rankIndexInGhostIndices];
                for(size_t requestNumber = 0; requestNumber < requestsOfRank.size(); requestNumber++) 
                {
                    const Request &indexInGhostArray = requestsOfRank[requestNumber];
                    size_t localIndex = dupPointsOfRank[indexInGhostArray.remotePointIdxInGhost];
                    for(const RemotePoint &neighbor : neighborPoints[localIndex])
                    {
                        responses[_rank].emplace_back(requestNumber, RemotePoint(neighbor.rank, neighbor.indexOnRank, neighbor.distance + 1)); // increase the distance by 1
                    }
                }
            }
            else
            {
                for(size_t requestNumber = 0; requestNumber < requestsOfRank.size(); requestNumber++) 
                {
                    const Request &index = requestsOfRank[requestNumber];
                    size_t localIndex = index.remotePointIdxInGhost;
                    for(const RemotePoint &neighbor : neighborPoints[localIndex])
                    {
                        responses[_rank].emplace_back(requestNumber, RemotePoint(neighbor.rank, neighbor.indexOnRank, neighbor.distance + 1)); 
                    }
                }
            }
        }

        std::vector<std::vector<Response>> incomingResponses = MPI_Exchange_all_to_all(responses, comm);

        // translate the results to the original point
        for(int _rank = 0; _rank < size; _rank++)
        {
            const std::vector<Request> &rankRequests = requests[_rank];
            const std::vector<Response> &rankResponse = incomingResponses[_rank];
            for(const Response &rankResponse : rankResponse)
            {
                size_t askedPointID = rankRequests[rankResponse.requestID].askingPointIdx;
                addPointToSet(result[askedPointID], rankResponse.neighbor);
            }
        }
    #else // RICH_MPI
        for(size_t requestNumber = 0; requestNumber < requestPoints.size(); requestNumber++)
        {
            size_t pointIdx = requestPoints[requestNumber];
            size_t neighborIndex = nextSearch[requestNumber];
            for(const RemotePoint &neighbor : neighborPoints[neighborIndex])
            {
                addPointToSet(result[pointIdx], neighbor);
            }
        }
    #endif // RICH_MPI

    return result;
}

#ifdef RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost, const MPI_Comm &comm)
#else // RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, const std::vector<size_t> &points, size_t order, bool atMost)
#endif // RICH_MPI
{
    NeighborsList result;

    int rank = 0, size = 1;
    #ifdef RICH_MPI
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    #endif // RICH_MPI

    if(order == 0)
    {
        // base case: return the points themselves
        for(const size_t &pointIdx : points)
        {
            #ifdef RICH_MPI
                result.insert(RemotePoint(rank, pointIdx, 0));
            #else // RICH_MPI
                result.insert(RemotePoint(pointIdx, 0));
            #endif // RICH_MPI
        }
        return result;
    }

    #ifdef RICH_MPI
        // send requests
        std::vector<std::vector<Request>> requests(size);
    #else // RICH_MPI
        std::vector<size_t> requestPoints;
    #endif// RICH_MPI
    std::vector<size_t> nextSearch;

    #ifdef RICH_MPI
        std::vector<std::pair<int, size_t>> indicesInGhosts;
        const std::vector<int> &dupProcs = tess.GetDuplicatedProcs();
    #endif // RICH_MPI

    std::vector<size_t> neighbor_buf;
    for(const size_t &pointIdx : points)
    {
        tess.GetNeighbors(pointIdx, neighbor_buf);
        for(const size_t &neighbor : neighbor_buf)
        {
            if(tess.IsPointOutsideBox(neighbor))
            {
                continue; // skip mirrored points
            }
            #ifdef RICH_MPI
                int _rank = tess.GetOwner(tess.GetMeshPoint(neighbor));
                if(_rank == rank)
                {
                    requests[rank].push_back({pointIdx, neighbor});
                    continue;
                }
                // otherwise, belongs to a remote
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                const std::vector<size_t> &ghostsOfRank = tess.GetGhostIndeces()[rankIndexInGhostIndices];
                size_t ghostIndexInGhost = std::distance(ghostsOfRank.cbegin(), std::find(ghostsOfRank.cbegin(), ghostsOfRank.cend(), neighbor));
                requests[_rank].push_back({pointIdx, ghostIndexInGhost});
            #else // RICH_MPI
                requestPoints.push_back(pointIdx);
                nextSearch.push_back(neighbor);
            #endif // RICH_MPI
        }
        result = boost::container::flat_set<RemotePoint>();
        if(atMost)
        {
            #ifdef RICH_MPI
                result.insert(RemotePoint(rank, pointIdx, 0));
            #else // RICH_MPI
                result.insert(RemotePoint(pointIdx, 0));
            #endif // RICH_MPI
        }
    }

    #ifdef RICH_MPI
        std::vector<std::vector<Request>> incomingRequests = MPI_Exchange_all_to_all(requests, comm);

        // calculate what local points should continue the search
        for(int _rank = 0; _rank < size; _rank++)
        {
            if(incomingRequests[_rank].empty())
            {
                continue;
            }
            if(_rank != rank)
            {
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                assert(rankIndexInGhostIndices != dupProcs.size()); // validate rank indeed exists
                const std::vector<size_t> &dupPointsOfRank = tess.GetDuplicatedPoints()[rankIndexInGhostIndices];
                for(const Request &indexInGhostArray : incomingRequests[_rank])
                {
                    size_t localIndex = dupPointsOfRank[indexInGhostArray.remotePointIdxInGhost];
                    nextSearch.push_back(localIndex);
                }
            }
            else
            {
                for(const Request &indexInGhostArray : incomingRequests[_rank])
                {
                    size_t localIndex = indexInGhostArray.remotePointIdxInGhost;
                    nextSearch.push_back(localIndex);
                }
            }
        }
    #endif // RICH_MPI
    
    // recurse - get neighbors of neighbors
    #ifdef RICH_MPI
        PointsToNeighborsMap neighborPoints = GetKOrderNeighbors(tess, nextSearch, order - 1, atMost, comm);
    #else // RICH_MPI
        PointsToNeighborsMap neighborPoints = GetKOrderNeighbors(tess, nextSearch, order - 1, atMost);
    #endif // RICH_MPI

    #ifdef RICH_MPI
        // now, answer back the results
        std::vector<std::vector<Response>> responses(size); // will be used to exchange_all_to_all
        
        // calculate what local points should continue the search
        for(int _rank = 0; _rank < size; _rank++)
        {
            const std::vector<Request> &requestsOfRank = incomingRequests[_rank];
            if(requestsOfRank.empty())
            {
                continue;
            }
            if(_rank != rank)
            {
                size_t rankIndexInGhostIndices = std::distance(dupProcs.cbegin(), std::find(dupProcs.cbegin(), dupProcs.cend(), _rank));
                assert(rankIndexInGhostIndices != dupProcs.size()); // validate rank indeed exists
                const std::vector<size_t> &dupPointsOfRank = tess.GetDuplicatedPoints()[rankIndexInGhostIndices];
                for(size_t requestNumber = 0; requestNumber < requestsOfRank.size(); requestNumber++) 
                {
                    const Request &indexInGhostArray = requestsOfRank[requestNumber];
                    size_t localIndex = dupPointsOfRank[indexInGhostArray.remotePointIdxInGhost];
                    for(const RemotePoint &neighbor : neighborPoints[localIndex])
                    {
                        responses[_rank].emplace_back(requestNumber, RemotePoint(neighbor.rank, neighbor.indexOnRank, neighbor.distance + 1)); // increase the distance by 1
                    }
                }
            }
            else
            {
                for(size_t requestNumber = 0; requestNumber < requestsOfRank.size(); requestNumber++) 
                {
                    const Request &index = requestsOfRank[requestNumber];
                    size_t localIndex = index.remotePointIdxInGhost;
                    for(const RemotePoint &neighbor : neighborPoints[localIndex])
                    {
                        responses[_rank].emplace_back(requestNumber, RemotePoint(neighbor.rank, neighbor.indexOnRank, neighbor.distance + 1)); 
                    }
                }
            }
        }

        std::vector<std::vector<Response>> incomingResponses = MPI_Exchange_all_to_all(responses, comm);

        // translate the results to the original point
        for(int _rank = 0; _rank < size; _rank++)
        {
            const std::vector<Request> &rankRequests = requests[_rank];
            const std::vector<Response> &rankResponse = incomingResponses[_rank];
            for(const Response &rankResponse : rankResponse)
            {
                addPointToSet(result, rankResponse.neighbor);
            }
        }
    #else // RICH_MPI
        for(size_t requestNumber = 0; requestNumber < requestPoints.size(); requestNumber++)
        {
            // size_t pointIdx = requestPoints[requestNumber];
            size_t neighborIndex = nextSearch[requestNumber];
            for(const RemotePoint &neighbor : neighborPoints[neighborIndex])
            {
                addPointToSet(result, neighbor);
            }
        }
    #endif // RICH_MPI

    return result;
}

#ifdef RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, size_t order, bool atMost, const MPI_Comm &comm)
#else // RICH_MPI
    NeighborsList GetAllKOrderNeighbors(const Tessellation3D &tess, size_t order, bool atMost)
#endif // RICH_MPI
{
    std::vector<size_t> points(tess.GetPointNo());
    std::iota(points.begin(), points.end(), 0);
    #ifdef RICH_MPI
        return GetAllKOrderNeighbors(tess, points, order, atMost, comm);
    #else // RICH_MPI
        return GetAllKOrderNeighbors(tess, points, order, atMost);
    #endif // RICH_MPI
}

#ifdef RICH_MPI

std::vector<std::vector<std::pair<ComputationalCell3D, Vector3D>>> GetNeighborCellsPoints(const Tessellation3D &tess, const std::vector<size_t> &points, const std::vector<ComputationalCell3D> &cells, size_t order)
{
    PointsToNeighborsMap map = GetKOrderNeighbors(tess, points, order, true);
    rank_t size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<boost::container::flat_set<size_t>> alreadySent(size);
    std::vector<std::vector<ComputationalCell3DVector3D>> toSend(size);
    for(size_t pointIdx : points)
    {
        auto const &neighborsSet = map[pointIdx];
        for(const RemotePoint &p : neighborsSet)
        {
            // p is a remote neighbor. Send the rank of p myself (the point `pointIdx`)
            if(p.rank == rank)
            {
                continue; // don't send to myself
            }
            if(alreadySent[p.rank].find(pointIdx) != alreadySent[p.rank].cend())
            {
                continue; // already sent `pointIdx` to the rank, skip
            }
            alreadySent[p.rank].insert(pointIdx);
            toSend[p.rank].emplace_back(cells[pointIdx], tess.GetMeshPoint(pointIdx), pointIdx);
        }
    }

    std::vector<std::vector<ComputationalCell3DVector3D>> toRecv = MPI_Exchange_all_to_all(toSend, MPI_COMM_WORLD);
    for(auto &vec : toRecv)
    {
        std::sort(vec.begin(), vec.end(), [](const ComputationalCell3DVector3D &a, const ComputationalCell3DVector3D &b){return a.local_index < b.local_index;});
    }

    size_t const N = points.size();
    std::vector<std::vector<std::pair<ComputationalCell3D, Vector3D>>> result(N);
    ComputationalCell3D cdummy;
    ComputationalCell3DVector3D dummy_for_search(cdummy, Vector3D(), 0);
    for(size_t i = 0; i < N; i++)
    {
        const size_t &pointIdx = points[i];
        const auto &neighborsSet = map[pointIdx];
        for(const RemotePoint &p : neighborsSet)
        {
            // p is a neighbor
            if(p.rank == rank)
            {
                continue;
            }
            dummy_for_search.local_index = p.indexOnRank;
            // from the points that were received by the rank of p, find where p.rank sent the point
            auto it = std::lower_bound(toRecv[p.rank].begin(), toRecv[p.rank].end(), dummy_for_search, [](const ComputationalCell3DVector3D &a, const ComputationalCell3DVector3D &b){return a.local_index < b.local_index;});
            if(it == toRecv[p.rank].end() or (*it).local_index != p.indexOnRank)
            {
                continue;
            }
            result[i].push_back(std::make_pair((*it).cell, (*it).point));
        }
    }
    return result;
}

#endif // RICH_MPI
