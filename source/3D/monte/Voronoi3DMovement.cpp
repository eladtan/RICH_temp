#include <cassert>
#include "Voronoi3DMovement.hpp"
#include "3D/elementary/Vector3D.hpp"
#include <MeshDecomposer3D/environment/EnvironmentAgent.hpp>
#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#include <MeshDecomposer3D/load_balancing/LoadBalancer.hpp>
#include "misc/universal_error.hpp"
#include "misc/utils.hpp"
#include "mpi/mpi_commands.hpp"
#include <bits/chrono.h>

#define RADIUSES_FACTOR 2

namespace
{
    constexpr double UPDATE_NEW_CELLS_BOX_EPS_FACTOR = 16.0;
}

#ifdef RICH_MPI

// Move according to sent points
void InternalMovements(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<size_t> &cellIDs)
{
    START_TIMER("Internal Movement");
    size_t count = 0;
    boost::container::flat_map<size_t, size_t> cellIDtoIndex;
    size_t N = cellIDs.size();
    for(size_t i = 0; i < N; i++)
    {
        cellIDtoIndex[cellIDs.at(i)] = i;
    }

    for(Particle3D &particle : particles)
    {
        size_t cellID = particle.cellID;

        auto it = cellIDtoIndex.find(cellID);
        if(it == cellIDtoIndex.cend())
        {
            continue;
        }

        count++;
        size_t newCellIndex = (*it).second;
        if(particle.cellIndex != newCellIndex)
        {
            particle.cellIndex = newCellIndex;
        }
    }
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank == 0)
        {
            std::cout << "Internal movements (determined by cell ID) counted for " << count << " particles" << std::endl;
        }
    #endif // RICH_MPI
}

#endif // RICH_MPI

void AssertLocations(const Tessellation3D &tess, const std::vector<Particle3D> &particles)
{
    START_TIMER("Assert Locations");
    size_t N = tess.GetPointNo();
    size_t Nparticles = particles.size();
    // check if the particles are inside the cells by checking the scalar products
    for(size_t i = 0; i < Nparticles; i++)
    {
        const Particle3D &particle = particles[i];
        size_t cellIndex = particle.cellIndex; 
        if(cellIndex >= tess.GetPointNo())
        {
            UniversalError eo("AssertLocations: Particle cell index is out of range");
            eo.addEntry("Particle", particle);
            eo.addEntry("Particle Index", i);
            eo.addEntry("Cell Index", cellIndex);
            eo.addEntry("N", N);
            throw eo;
        }
        if(not tess.IsPointInCell(particle.location, cellIndex))
        {
            UniversalError eo("AssertLocations: Particle is not in its cell");
            eo.addEntry("Particle", particle);
            eo.addEntry("Particle Index", i);
            eo.addEntry("Cell Index", cellIndex);
            eo.addEntry("Cell Point", tess.GetMeshPoint(cellIndex));
            throw eo;
        }
    }
}

#ifdef RICH_MPI

void TransferParticlesWithTranslationMap(const Tessellation3D &tess, std::vector<Particle3D> &particles, const boost::container::flat_map<size_t, std::pair<rank_t, size_t>> &cellsTranslation)
{
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    START_TIMER("Prepare Transfer Data");

    std::vector<Serializer> senders(size);
    std::vector<Particle3D> selfParticles;
    size_t sentCounter = 0;
    for(Particle3D &p : particles)
    {
        size_t particleCellIdx = p.cellIndex;
        auto [newRank, newCellIdx] = cellsTranslation.at(particleCellIdx);
        p.cellIndex = newCellIdx;
        if(newRank == rank)
        {
            selfParticles.push_back(p);
        }
        else
        {
            senders[newRank].insert(p);
            sentCounter++;
        }
    }

    particles.clear();

    std::vector<std::vector<Particle3D>> allNewParticles;
    {
        START_TIMER_PREEMPTIVE("Particles Exchange");
        allNewParticles = MPI_Exchange_all_to_all_serializers<Particle3D>(senders, MPI_COMM_WORLD);
    }

    size_t receivedCounter = 0;
    particles = std::move(selfParticles);
    std::for_each(allNewParticles.cbegin(), allNewParticles.cend(), [&particles, &receivedCounter](const std::vector<Particle3D> &procParticles)
    {
        receivedCounter += procParticles.size();
        particles.insert(particles.end(), procParticles.cbegin(), procParticles.cend());
    });

    MPI_Reduce((rank == 0)? MPI_IN_PLACE : &sentCounter, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce((rank == 0)? MPI_IN_PLACE : &receivedCounter, &receivedCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    conditional_shrink(particles);
    MPI_Barrier(MPI_COMM_WORLD);
}

void UpdateNewCellsAfterExchange(const Tessellation3D &tess, std::vector<Particle3D> &particles, const ExchangeChain &chain)
{
    TransferParticlesWithTranslationMap(tess, particles, chain.GetTranslationMap());
    AssertLocations(tess, particles);
}

void UpdateNewCellsPullback(const Tessellation3D &tess, std::vector<Particle3D> &particles, const ExchangeChain &chain)
{
    TransferParticlesWithTranslationMap(tess, particles, chain.GetReversedTranslationMap());
    AssertLocations(tess, particles);
}

void UpdateNewCellsAfterExchange(const Tessellation3D &tess, std::vector<Particle3D> &particles)
{
    ExchangeChain chain;
    size_t N = tess.GetSelfIndex().size();
    for(const std::vector<size_t> &sent : tess.GetSentPoints())
    {
        N += sent.size();
    }
    chain.Reset(N);
    chain.Exchange(tess.GetSentProcs(), tess.GetSentPoints(), tess.GetSelfIndex());
    UpdateNewCellsAfterExchange(tess, particles, chain);
}

static size_t ResolveRemainingParticles(const Tessellation3D &tess, std::vector<Particle3D> &particles, const OctTree<IndexedVector3D> &octTree)
{
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    size_t N = tess.GetPointNo();

    auto [tess_ll, tess_ur] = tess.GetBoxCoordinates();
    Vector3D tess_boxsize = tess_ur - tess_ll;
    tess_ll -= UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
    tess_ur += UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
    OctTree<IndexedVector3D> wideTree(IndexedVector3D(tess_ll, std::numeric_limits<size_t>::max()), IndexedVector3D(tess_ur, std::numeric_limits<size_t>::max()));
    for(size_t i = 0; i < N; i++)
        wideTree.insert(IndexedVector3D(tess.GetMeshPoint(i), i));
    DistributedOctTree<IndexedVector3D> distributedOctTree(&wideTree);

    double avgCellSize = 0;
    if(N > 0)
    {
        for(size_t i = 0; i < N; i++)
            avgCellSize += tess.GetWidth(i);
        avgCellSize /= N;
    }
    double avgOfAvgCellSize = avgCellSize;
    MPI_Allreduce(MPI_IN_PLACE, &avgOfAvgCellSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    avgOfAvgCellSize /= size;
    double initialRadius = avgCellSize * RADIUSES_FACTOR;
    if(N == 0)
        initialRadius = avgOfAvgCellSize * RADIUSES_FACTOR;

    std::vector<Particle3D> resolvedParticles;
    boost::container::flat_set<size_t> particlesLeft;
    std::vector<boost::container::flat_set<rank_t>> ranksTested;
    for(size_t i = 0; i < particles.size(); i++)
    {
        Particle3D &p = particles[i];
        size_t closestCell = std::numeric_limits<size_t>::max();
        if(N > 0)
        {
            closestCell = tess.GetContainingCell(p.location);
        }
        if(closestCell < N and tess.IsPointInCell(p.location, closestCell))
        {
            p.cellIndex = closestCell;
            resolvedParticles.push_back(p);
            ranksTested.push_back({});
        }
        else
        {
            particlesLeft.insert(i);
            ranksTested.push_back({rank});
        }
    }
    {
        size_t numParticlesLeft = particlesLeft.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesLeft, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0)
            std::cout << "Number of particles left to determination: " << numParticlesLeft << std::endl;
    }

    std::vector<double> radiuses(particles.size(), initialRadius);
    std::vector<std::vector<Particle3D>> sendValues(size);
    std::vector<std::vector<size_t>> sendIndicesCpy(size);
    std::vector<std::vector<size_t>> acknowledgementValues(size);

    size_t iterations = 0;

    while(true)
    {
        iterations++;
        size_t localLeftParticles = particlesLeft.size();
        size_t globalLeftParticles;
        MPI_Allreduce(&localLeftParticles, &globalLeftParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(globalLeftParticles == 0)
            break;

        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            sendValues[_rank].clear();
            sendIndicesCpy[_rank].clear();
            acknowledgementValues[_rank].clear();
        }
        
        for(size_t i : particlesLeft)
        {
            Particle3D &p = particles[i];
            bool atLeastOneNew = false;
            while(not atLeastOneNew)
            {
                if(ranksTested[i].size() == size)
                {
                    UniversalError eo("UpdateNewCells: All ranks were tested for particle");
                    eo.addEntry("Particle", p);
                    eo.addEntry("Cell Index", p.cellIndex);
                    eo.addEntry("N", N);
                    size_t closestPointIdx = tess.GetContainingCell(p.location);
                    eo.addEntry("Local Closest Local Point", closestPointIdx);
                    eo.addEntry("Local Closest Point Value", tess.GetMeshPoint(closestPointIdx));
                    eo.addEntry("Distance to closest local point", abs(tess.GetMeshPoint(closestPointIdx) - p.location));
                    try
                    {
                        tess.IsPointInCell(p.location, closestPointIdx, true);
                        eo.addEntry("Point In Cell?", true);
                    }
                    catch(const UniversalError &eo2)
                    {
                        eo.addEntry("Point In Cell?", false);
                        eo.join(eo2);
                    }
                    eo.addEntry("Rank", rank);
                    throw eo;
                }

                auto intersectingRanks = distributedOctTree.getIntersectingRanks(p.location, radiuses[i]);
                for(rank_t _rank : intersectingRanks)
                {
                    if(ranksTested[i].find(_rank) == ranksTested[i].end())
                    {
                        atLeastOneNew = true; 
                        sendValues[_rank].push_back(p);
                        sendIndicesCpy[_rank].push_back(i);
                        ranksTested[i].insert(_rank);
                    }
                }
                radiuses[i] *= RADIUSES_FACTOR;
            }
        }

        std::vector<std::vector<Particle3D>> receiveValues = MPI_Exchange_all_to_all_sparse(sendValues, MPI_COMM_WORLD);
        assert(receiveValues.size() == size);

        if(octTree.getSize() > 0)
        {
            for(rank_t _rank = 0; _rank < size; _rank++)
            {
                std::vector<Particle3D> &particlesFromRank = receiveValues[_rank];
                size_t Np = particlesFromRank.size();
                for(size_t i = 0; i < Np; i++)
                {
                    Particle3D &p = particlesFromRank[i];
                    // in case the particle is on a face, we need to consider two points
                    auto candidates = octTree.getKClosestPoints(p.location, 2);
                    for(const auto &[cell, dist] : candidates)
                    {
                        size_t index = cell.getIndex();
                        if(index < N and tess.IsPointInCell(p.location, index))
                        {
                            p.cellIndex = index;
                            resolvedParticles.push_back(p);
                            acknowledgementValues[_rank].push_back(i);
                            break;
                        }
                    }
                }
            }
        }

        std::vector<std::vector<size_t>> acknowledgements = MPI_Exchange_all_to_all_sparse(acknowledgementValues, MPI_COMM_WORLD);
        assert(acknowledgements.size() == size);

        std::vector<size_t> toErase;
        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            for(size_t i : acknowledgements[_rank])
            {
                toErase.push_back(sendIndicesCpy[_rank][i]);
            }
        }
        std::sort(toErase.begin(), toErase.end());
        toErase.erase(std::unique(toErase.begin(), toErase.end()), toErase.end());

        std::vector<size_t> remainingVec;
        remainingVec.reserve(particlesLeft.size());
        std::set_difference(particlesLeft.begin(), particlesLeft.end(), toErase.begin(), toErase.end(), std::back_inserter(remainingVec));
        particlesLeft = boost::container::flat_set<size_t>(boost::container::ordered_unique_range_t{}, remainingVec.begin(), remainingVec.end());
    }

    particles = std::move(resolvedParticles);
    return iterations;
}

void FirstInaccurateMovements(const Tessellation3D &tess, std::vector<Particle3D> &particles)
{
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Distribute(particles, MPI_COMM_WORLD);

    const std::shared_ptr<EnvironmentAgent<Vector3D>> &envAgent = tess.GetEnvironmentAgent();
    std::vector<Particle3D> newParticles;
    std::vector<Serializer> senders(size);

    auto start = std::chrono::high_resolution_clock::now();
    size_t sentCounter = 0;
    for(Particle3D &p : particles)
    {
        rank_t approxOwner = envAgent->getOwner(p.location);
        if(approxOwner == rank)
        {
            newParticles.push_back(p);
        }
        else
        {
            senders[approxOwner].insert(p);
            sentCounter++;
            // if(sentCounter <= 10)
            // {
            //     const std::shared_ptr<LoadBalancer<Vector3D>> lb = tess.GetLoadBalancer();
            //     const HilbertLoadBalancer<Vector3D> *hlb = dynamic_cast<const HilbertLoadBalancer<Vector3D>*>(lb.get());
            //     size_t b1 = (approxOwner == 0)? 0 : hlb->boundaries[std::min(approxOwner - 1, size - 1)];
            //     size_t b2 = hlb->boundaries[std::min(approxOwner, size - 1)];
            //     Vector3D b1_xyz = hlb->convertor->d2xyz(b1);
            //     Vector3D b2_xyz = hlb->convertor->d2xyz(b2);
            //     std::cout << "Rank " << rank << " wants to ask rank " << approxOwner << "(boundaries: b1: " << b1 << ", b1_xyz: " << b1_xyz << ", b2: " << b2 << ", b2_xyz: " << b2_xyz << "), particle location is " << p.location << ", its d is " << hlb->convertor->xyz2d(p.location) << std::endl;
            // }
        }
    }

    size_t Nparticles = particles.size();
    particles.clear();
    auto end = std::chrono::high_resolution_clock::now();
    double timeInLoop1 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "[First movements] Rank " << rank << ", time in loop 1: " << timeInLoop1 << " (had " << Nparticles << " particles)" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Particle3D>> receiveValues = MPI_Exchange_all_to_all_serializers<Particle3D>(senders, MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    double timeInExchange = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "[First movements] Rank " << rank << ", time in exchange: " << timeInExchange << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    start = std::chrono::high_resolution_clock::now();
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        const std::vector<Particle3D> &particlesFromRank = receiveValues[_rank];
        newParticles.insert(newParticles.end(), particlesFromRank.cbegin(), particlesFromRank.cend());
    }
    particles = std::move(newParticles);

    MPI_Allreduce(MPI_IN_PLACE, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    end = std::chrono::high_resolution_clock::now();
    double timeInLoop2 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "[First movements] Rank " << rank << ", time in loop 2: " << timeInLoop2 << std::endl;

    if(rank == 0)
    {
        std::cout << "First inaccurate movements sent for " << sentCounter << " particles" << std::endl;
    }
}
#endif // RICH_MPI

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<size_t> &cellIDs)
{
    if(not dynamic_cast<const Voronoi3D*>(&tess))
    {
        throw UniversalError("UpdateNewCells: Tessellation is not a Voronoi3D");
    }
    START_TIMER("Update New Cells");

    try
    {
        size_t N = tess.GetPointNo();
    #ifndef RICH_MPI
        if(N == 0)
        {
            if(particles.empty())
                return;

            UniversalError eo("UpdateNewCells: particles remain on a rank with no local cells");
            eo.addEntry("Particle count", particles.size());
            throw eo;
        }
        for(Particle3D &p : particles)
        {
            if(p.cellIndex < N)
            {
                if(tess.IsPointInCell(p.location, p.cellIndex))
                {
                    // the point is inside my domain, correct location
                    continue; // done!
                }               
            }
            p.cellIndex = tess.GetContainingCell(p.location);
        }
    #else // RICH_MPI
        rank_t rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        InternalMovements(tess, particles, cellIDs);

        START_TIMER_PREEMPTIVE("Local Trees Construction");
        
        auto [tess_ll, tess_ur] = tess.GetBoxCoordinates();
        Vector3D ll = tess_ll;
        Vector3D ur = tess_ur;
        if(N > 0)
        {
            ll = Vector3D(std::numeric_limits<double>::max());
            ur = Vector3D(std::numeric_limits<double>::lowest());

            const std::vector<Vector3D> &vertices = tess.GetFacePoints();
            for(const point_vec &vec : tess.GetAllPointsInFace())
            {
                for(size_t pointIdx : vec)
                {
                    const Vector3D &p = vertices[pointIdx];
                    ll.x = std::min(ll.x, p.x);
                    ll.y = std::min(ll.y, p.y);
                    ll.z = std::min(ll.z, p.z);
                    ur.x = std::max(ur.x, p.x);
                    ur.y = std::max(ur.y, p.y);
                    ur.z = std::max(ur.z, p.z);
                }
            }

            Vector3D boxsize = ur - ll;
            ll -= EPSILON * boxsize;
            ur += EPSILON * boxsize;
        }

        Vector3D tess_boxsize = tess_ur - tess_ll;
        tess_ll -= UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
        tess_ur += UPDATE_NEW_CELLS_BOX_EPS_FACTOR * EPSILON * tess_boxsize;
        BoundingBox<Vector3D> bb(tess_ll, tess_ur);
        BoundingBox<Vector3D> subBox(ll, ur);

        if(not bb.contains(subBox))
        {
            UniversalError eo("UpdateNewCells: Sub-box is not contained within the main bounding box");
            eo.addEntry("Voronoi box", bb);
            eo.addEntry("Sub-box", subBox);
            for(const Particle3D &p : particles)
            {
                if(not bb.contains(p.location))
                {
                    eo.addEntry("Particle " + std::to_string(p.id) + " is out of box", p);
                }
            }
            throw eo;
        }

        OctTree<IndexedVector3D> octTree(IndexedVector3D(ll, std::numeric_limits<size_t>::max()), IndexedVector3D(ur, std::numeric_limits<size_t>::max()));

        for(size_t i = 0; i < N; i++)
            octTree.insert(IndexedVector3D(tess.GetMeshPoint(i), i));
        assert(octTree.getSize() == N);
    
        std::vector<Particle3D> myParticles;
        std::vector<Particle3D> shouldExchangeParticles;

        START_TIMER_PREEMPTIVE("Self Update");
        if(octTree.getSize() > 0)
        {
            for(size_t i = 0; i < particles.size(); i++)
            {
                // first, test whether the point is inside my domain
                Particle3D &p = particles[i];
                if(p.cellIndex < N)
                {
                    // first, test whether the point is inside the cell it declares it is in
                    if(tess.IsPointInCell(p.location, p.cellIndex))
                    {
                        // the point is inside my domain, correct location
                        myParticles.push_back(p);
                        continue;
                    }
                }

                if(tess.IsPointOutsideBox(p.location))
                {
                    UniversalError eo("Particle location is outside of the bounding box");
                    eo.addEntry("Particle", p);
                    eo.addEntry("Bounding Box", bb);
                    throw eo;
                }

                auto twoClosest = octTree.getKClosestPoints(p.location, 2);
                bool found = false;
                for(const auto &[cell, dist] : twoClosest)
                {
                    size_t index = cell.getIndex();
                    if(tess.IsPointInCell(p.location, index))
                    {
                        p.cellIndex = index;
                        myParticles.push_back(p);
                        found = true;
                        break;
                    }
                }
                if(not found)
                {
                    shouldExchangeParticles.push_back(p);
                }
            }
        }
        else
        {
            shouldExchangeParticles = std::move(particles);
        }

        particles.clear();

        FirstInaccurateMovements(tess, shouldExchangeParticles);
        MPI_Barrier(MPI_COMM_WORLD);
        
        particles = std::move(shouldExchangeParticles);

        START_TIMER_PREEMPTIVE("Main Loop");

        size_t iterations = ResolveRemainingParticles(tess, particles, octTree);

        if(rank == 0)
        {
            std::cout << "Rank " << rank << ", done UpdateNewCells, iterations is " << iterations << std::endl;
        }

        particles.insert(particles.end(), myParticles.begin(), myParticles.end());
    #endif // RICH_MPI
        AssertLocations(tess, particles);
    }
    catch(const UniversalError &eo)
    {
        throw;
    }
}

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<ComputationalCell3D> &cells)
{
    size_t N = tess.GetPointNo();
    std::vector<size_t> cellIDs;
    cellIDs.reserve(N);

    for(size_t i = 0; i < N; i++)
    {
        cellIDs.push_back(cells[i].ID);
    }

    UpdateNewCells(tess, particles, cellIDs);

    // Keep the persistent cell ID synchronized with the resolved local index.
    // MC transport updates particle.cellIndex during cell crossings, but the
    // cellID field is used by InternalMovements() after a later mesh exchange.
    // Leaving it stale can make the first remap phase trust the wrong cell.
    for(Particle3D &p : particles)
        if(p.cellIndex < N)
            p.cellID = cells[p.cellIndex].ID;
}
