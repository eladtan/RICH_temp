#include "Voronoi3DMovement.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/environment/EnvironmentAgent.h"
#include "3D/tessellation/loadBalancing/HilbertLoadBalancer.hpp"
#include "3D/tessellation/loadBalancing/LoadBalancer.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "misc/universal_error.hpp"
#include "mpi/mpi_commands.hpp"
#include <bits/chrono.h>

#define RADIUSES_FACTOR 2

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
    particles.shrink_to_fit();

    std::vector<std::vector<Particle3D>> allNewParticles;
    {
        START_TIMER_PREEMPTIVE("Particles Exchange");
        allNewParticles = MPI_Iexchange_all_to_all_serializers<Particle3D>(senders, MPI_COMM_WORLD);
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

#endif // RICH_MPI

#ifdef RICH_MPI
void FirstInaccurateMovements(const Tessellation3D &tess, std::vector<Particle3D> &particles)
{
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // MPI_Distribute(particles, MPI_COMM_WORLD);

    const std::shared_ptr<EnvironmentAgent> &envAgent = tess.GetEnvironmentAgent();
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
            //     const std::shared_ptr<LoadBalancer> lb = tess.GetLoadBalancer();
            //     const HilbertLoadBalancer *hlb = dynamic_cast<const HilbertLoadBalancer*>(lb.get());
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
    particles.shrink_to_fit();
    auto end = std::chrono::high_resolution_clock::now();
    double timeInLoop1 = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    // std::cout << "[First movements] Rank " << rank << ", time in loop 1: " << timeInLoop1 << " (had " << Nparticles << " particles)" << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<Particle3D>> receiveValues = MPI_Iexchange_all_to_all_serializers<Particle3D>(senders, MPI_COMM_WORLD);
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

void FirstMovementsDistributed(
    const Tessellation3D &tess,
    std::vector<Particle3D> &particles,
    DistributedOctTree<IndexedVector3D> &distributedOctTree)
{
    if(distributedOctTree.getOctTree() == nullptr)
    {
        FirstInaccurateMovements(tess, particles);
        return;
    }

    // MPI_Distribute(particles, MPI_COMM_WORLD);

    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::vector<Particle3D> newParticles;
    std::vector<Serializer> senders(size);
    size_t sentCounter = 0;
    size_t localN = tess.GetPointNo();

    // Gather all mesh points globally for nearest-point routing (Voronoi property)
    int localCount = static_cast<int>(localN);
    std::vector<int> allCounts(size);
    MPI_Allgather(&localCount, 1, MPI_INT, allCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int totalPoints = 0;
    std::vector<int> displacements(size);
    for(rank_t r = 0; r < size; r++)
    {
        displacements[r] = totalPoints * 3;
        totalPoints += allCounts[r];
        allCounts[r] *= 3;
    }

    std::vector<double> localMeshPts(localN * 3);
    for(size_t i = 0; i < localN; i++)
    {
        const Vector3D &pt = tess.GetMeshPoint(i);
        localMeshPts[i * 3]     = pt.x;
        localMeshPts[i * 3 + 1] = pt.y;
        localMeshPts[i * 3 + 2] = pt.z;
    }

    std::vector<double> allMeshPts(totalPoints * 3);
    MPI_Allgatherv(localMeshPts.data(), static_cast<int>(localN * 3), MPI_DOUBLE,
                   allMeshPts.data(), allCounts.data(), displacements.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Build rank lookup: for each global mesh point index, which rank owns it
    std::vector<rank_t> pointRank(totalPoints);
    {
        int idx = 0;
        for(rank_t r = 0; r < size; r++)
        {
            int n = allCounts[r] / 3;
            for(int j = 0; j < n; j++)
                pointRank[idx++] = r;
        }
    }

    // Route each particle to the rank owning its nearest mesh point
    for(Particle3D &p : particles)
    {
        rank_t target = rank;
        double bestDist2 = std::numeric_limits<double>::max();
        for(int i = 0; i < totalPoints; i++)
        {
            double dx = p.location.x - allMeshPts[i * 3];
            double dy = p.location.y - allMeshPts[i * 3 + 1];
            double dz = p.location.z - allMeshPts[i * 3 + 2];
            double d2 = dx * dx + dy * dy + dz * dz;
            if(d2 < bestDist2)
            {
                bestDist2 = d2;
                target = pointRank[i];
            }
        }
        if(target == rank)
        {
            newParticles.push_back(p);
        }
        else
        {
            senders[target].insert(p);
            sentCounter++;
        }
    }

    particles.clear();
    particles.shrink_to_fit();

    std::vector<std::vector<Particle3D>> receiveValues = MPI_Iexchange_all_to_all_serializers<Particle3D>(senders, MPI_COMM_WORLD);

    for(rank_t r = 0; r < size; r++)
    {
        newParticles.insert(newParticles.end(), receiveValues[r].cbegin(), receiveValues[r].cend());
    }
    particles = std::move(newParticles);

    MPI_Allreduce(MPI_IN_PLACE, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "First distributed movements sent for " << sentCounter << " particles" << std::endl;
    }
}
#endif // RICH_MPI

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<size_t> &cellIDs)
{
    if(not dynamic_cast<const Voronoi3D*>(&tess))
    {
        throw UniversalError("UpdateNewCells: Tessellation is not a Voronoi3D");
    }
    bool verbose = true;
    START_TIMER("Update New Cells");

    try
    {
        size_t N = tess.GetPointNo();
    #ifndef RICH_MPI
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
        
        Vector3D ll(std::numeric_limits<double>::max());
        Vector3D ur(std::numeric_limits<double>::lowest());

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

        auto [tess_ll, tess_ur] = tess.GetBoxCoordinates();
        tess_ll -= EPSILON * (tess_ur - tess_ll);
        tess_ur += EPSILON * (tess_ur - tess_ll);
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
        OctTree<IndexedVector3D> octTree2(IndexedVector3D(tess_ll, std::numeric_limits<size_t>::max()), IndexedVector3D(tess_ur, std::numeric_limits<size_t>::max()));

        for(size_t i = 0; i < N; i++)
        {
            const Vector3D &point = tess.GetMeshPoint(i);
            octTree.insert(IndexedVector3D(point, i));
            octTree2.insert(IndexedVector3D(point, i));
        }
        assert(octTree.getSize() == N);
        assert(octTree2.getSize() == N);
        
        START_TIMER_PREEMPTIVE("Distributed OctTree Construction");
        DistributedOctTree<IndexedVector3D> distributedOctTree(&octTree2);
    
        START_TIMER_PREEMPTIVE("Calculating Radiuses");
        double avgCellSize = 0;
        if(N > 0)
        {
            for(size_t i = 0; i < N; i++)
            {
                avgCellSize += tess.GetWidth(i);
            }
            avgCellSize /= N;
        }
        double avgOfAvgCellSize = avgCellSize;
        MPI_Allreduce(MPI_IN_PLACE, &avgOfAvgCellSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        avgOfAvgCellSize /= size;
        double initialRadius = avgCellSize * RADIUSES_FACTOR;
        if(N == 0)
        {
            initialRadius = avgOfAvgCellSize * RADIUSES_FACTOR;
        }
        
        boost::container::flat_set<size_t> particlesLeft;
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
        particles.shrink_to_fit();

        auto start4 = std::chrono::high_resolution_clock::now();
        FirstInaccurateMovements(tess, shouldExchangeParticles);
        // FirstMovementsDistributed(tess, shouldExchangeParticles, distributedOctTree);
        auto end4 = std::chrono::high_resolution_clock::now();
        double timeFirstMovements = std::chrono::duration_cast<std::chrono::duration<double>>(end4 - start4).count();
        // std::cout << "Rank " << rank << ", time first movements: " << timeFirstMovements << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        std::vector<Particle3D> newParticles = std::move(myParticles);

        particles = std::move(shouldExchangeParticles);
        std::vector<boost::container::flat_set<rank_t>> ranksTested;

        auto start3 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < particles.size(); i++)
        {
            Particle3D &p = particles[i];

            size_t closestCell = std::numeric_limits<size_t>::max();
            if(N > 0)
            {
                closestCell = octTree.closestPoint(p.location).getIndex();
            }

            if(closestCell < N and tess.IsPointInCell(p.location, closestCell))
            {
                p.cellIndex = closestCell;
                newParticles.push_back(p);
                ranksTested.push_back({});
            }
            else
            {
                particlesLeft.insert(i);
                ranksTested.push_back({rank});
            }
        }
        size_t numParticlesLeft = particlesLeft.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesLeft, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(verbose and rank == 0)
        {
            std::cout << "Number of particles left to determination: " << numParticlesLeft << std::endl;
        }

        auto end3 = std::chrono::high_resolution_clock::now();
        double timeRightBeforeLoop = std::chrono::duration_cast<std::chrono::duration<double>>(end3 - start3).count();
        // std::cout << "Rank " << rank << ", time right before loop: " << timeRightBeforeLoop << std::endl;
        MPI_Barrier(MPI_COMM_WORLD);
        
        std::vector<double> radiuses(particles.size(), initialRadius);
        std::vector<std::vector<Particle3D>> sendValues(size);
        std::vector<std::vector<size_t>> sendIndicesCpy(size);
        std::vector<std::vector<size_t>> acknowledgementValues(size);

        size_t iterations = 0;
        rank_t maxRanksTested = 0;

        START_TIMER_PREEMPTIVE("Main Loop");

        auto loop_start = std::chrono::high_resolution_clock::now();
        // std::cout << "Rank " << rank << " has " << particlesLeft.size() << " particles left to determine over " << N << " cells, with initial radius " << initialRadius << std::endl;
        while(true)
        {
            double timeInTree = 0;
            double timeInAlltoall1 = 0;
            double timeInAlltoall2 = 0;
            double timeInPreparation = 0;
            double timeInMainLoop = 0;
            double timeInEndLoop = 0;
            size_t maxSendSize = 0;
            bool stopPreparationLoop = false;

            iterations++;
            size_t localLeftParticles = particlesLeft.size();
            size_t globalLeftParticles;
            MPI_Allreduce(&localLeftParticles, &globalLeftParticles, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            if(globalLeftParticles == 0)
            {
                break;
            }
            for(rank_t _rank = 0; _rank < size; _rank++)
            {
                sendValues[_rank].clear();
                sendValues[_rank].shrink_to_fit();
                sendIndicesCpy[_rank].clear();
                acknowledgementValues[_rank].clear();
            }
            
            auto start2 = std::chrono::high_resolution_clock::now();
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
                        size_t closestPointIdx = octTree.closestPoint(p.location).getIndex();
                        eo.addEntry("Local Closest Local Point", closestPointIdx);
                        eo.addEntry("Local Closest Point Value", tess.GetMeshPoint(closestPointIdx));
                        eo.addEntry("Is Point in Cell", tess.IsPointInCell(p.location, closestPointIdx));
                        eo.addEntry("Rank", rank);
                        throw eo;
                    }

                    auto start = std::chrono::high_resolution_clock::now();
                    auto intersectingRanks = distributedOctTree.getIntersectingRanks(p.location, radiuses[i]);
                    auto end = std::chrono::high_resolution_clock::now();
                    timeInTree += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
                    for(rank_t _rank : intersectingRanks)
                    {
                        if(ranksTested[i].find(_rank) == ranksTested[i].end())
                        {
                            // not tested yet
                            atLeastOneNew = true; 
                            sendValues[_rank].push_back(p);
                            maxSendSize = std::max(maxSendSize, sendValues[_rank].size());
                            if(maxSendSize > 1e5)
                            {
                                // stopPreparationLoop = true;
                            }
                            sendIndicesCpy[_rank].push_back(i);
                            ranksTested[i].insert(_rank);
                        }
                    }
                    maxRanksTested = std::max(maxRanksTested, static_cast<rank_t>(ranksTested[i].size()));
                    radiuses[i] *= RADIUSES_FACTOR;
                }
                if(stopPreparationLoop)
                {
                    break;
                }
            }
            auto end2 = std::chrono::high_resolution_clock::now();
            timeInPreparation += std::chrono::duration_cast<std::chrono::duration<double>>(end2 - start2).count();

            // measure time of alltoall

            auto start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<Particle3D>> receiveValues = MPI_Iexchange_all_to_all(sendValues, MPI_COMM_WORLD);
            auto end = std::chrono::high_resolution_clock::now();
            timeInAlltoall1 += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            assert(receiveValues.size() == size);

            start = std::chrono::high_resolution_clock::now();
            if(octTree.getSize() > 0)
            {
                for(rank_t _rank = 0; _rank < size; _rank++)
                {
                    std::vector<Particle3D> &particlesFromRank = receiveValues[_rank];
                    size_t Np = particlesFromRank.size();
                    for(size_t i = 0; i < Np; i++)
                    {
                        Particle3D &p = particlesFromRank[i];
                        size_t containingCell = octTree.closestPoint(p.location).getIndex();
                        if(containingCell < N and tess.IsPointInCell(p.location, containingCell))
                        {
                            p.cellIndex = containingCell;
                            newParticles.push_back(p);
                            acknowledgementValues[_rank].push_back(i);
                        }
                    }
                }
            }
            end = std::chrono::high_resolution_clock::now();
            timeInMainLoop += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            start = std::chrono::high_resolution_clock::now();
            std::vector<std::vector<size_t>> acknowledgements = MPI_Iexchange_all_to_all(acknowledgementValues, MPI_COMM_WORLD);
            end = std::chrono::high_resolution_clock::now();
            timeInAlltoall2 += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

            assert(acknowledgements.size() == size);

            start = std::chrono::high_resolution_clock::now();
            std::vector<size_t> toErase;
            for(rank_t _rank = 0; _rank < size; _rank++)
            {
                const std::vector<size_t> &rankAcknowledgements = acknowledgements[_rank];
                for(size_t i : rankAcknowledgements)
                {
                    toErase.push_back(sendIndicesCpy[_rank][i]);
                }
            }
            std::sort(toErase.begin(), toErase.end());
            toErase.erase(std::unique(toErase.begin(), toErase.end()), toErase.end());

            std::vector<size_t> remainingVec;
            remainingVec.reserve(particlesLeft.size());
            std::set_difference(particlesLeft.begin(), particlesLeft.end(),
                        toErase.begin(), toErase.end(),
                        std::back_inserter(remainingVec));
            size_t prevSize = particlesLeft.size();
            particlesLeft = boost::container::flat_set<size_t>(
                boost::container::ordered_unique_range_t{},
                remainingVec.begin(), remainingVec.end());

            end = std::chrono::high_resolution_clock::now();
            timeInEndLoop += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
            // std::cout << "Rank " << rank << " end loop: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count()
            //           << "s, particlesLeft=" << particlesLeft.size()
            //           << ", erased=" << (prevSize - particlesLeft.size()) << std::endl;

            // MPI_Barrier(MPI_COMM_WORLD);
            // std::cout << "Rank " << rank << " in iteration " << iterations << ", time in preparation: " << timeInPreparation << " (out of that, time in tree: " << timeInTree << "), time in alltoall1: " << timeInAlltoall1 << ", time in main loop: " << timeInMainLoop << 
            //     ", time in alltoall2: " << timeInAlltoall2 << ", time in end loop: " << timeInEndLoop << std::endl;
        }

        auto loop_end = std::chrono::high_resolution_clock::now();
        double timeInLoop = std::chrono::duration_cast<std::chrono::duration<double>>(loop_end - loop_start).count();
        // std::cout << "Rank " << rank << ", time in loop: " << timeInLoop << std::endl;

        if(rank == 0)
        {
            std::cout << "Rank " << rank << ", done UpdateNewCells, iterations is " << iterations << std::endl;
        }

        particles = std::move(newParticles);
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
}