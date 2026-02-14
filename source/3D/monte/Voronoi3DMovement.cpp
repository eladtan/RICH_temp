#include "Voronoi3DMovement.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/environment/EnvironmentAgent.h"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"
#include "misc/universal_error.hpp"

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
    size_t Nparticles = particles.size();
    // check if the particles are inside the cells by checking the scalar products
    for(size_t i = 0; i < Nparticles; i++)
    {
        const Particle3D &particle = particles[i];
        size_t cellIndex = particle.cellIndex; 
        assert(cellIndex < tess.GetPointNo());
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

    std::vector<std::vector<Particle3D>> particlesToProcessors(size);
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
            particlesToProcessors[newRank].push_back(p);
            sentCounter++;
        }
    }

    std::vector<std::vector<Particle3D>> allNewParticles;
    {
        START_TIMER_PREEMPTIVE("Particles Exchange");
        allNewParticles = MPI_Iexchange_all_to_all(particlesToProcessors, MPI_COMM_WORLD);
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

    const std::shared_ptr<EnvironmentAgent> &envAgent = tess.GetEnvironmentAgent();
    std::vector<Particle3D> newParticles;
    std::vector<std::vector<Particle3D>> sendValues(size);

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
            sendValues[approxOwner].push_back(p);
            sentCounter++;
        }
    }
    
    std::vector<std::vector<Particle3D>> receiveValues = MPI_Iexchange_all_to_all(sendValues, MPI_COMM_WORLD);
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        const std::vector<Particle3D> &particlesFromRank = receiveValues[_rank];
        newParticles.insert(newParticles.end(), particlesFromRank.cbegin(), particlesFromRank.cend());
    }
    particles = std::move(newParticles);

    MPI_Allreduce(MPI_IN_PLACE, &sentCounter, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(rank == 0)
    {
        std::cout << "First inaccurate movements sent for " << sentCounter << " particles" << std::endl;
    }
}
#endif // RICH_MPI

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles, const std::vector<size_t> &cellIDs)
{
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
        else
        {
            avgCellSize = abs(tess_ur - tess_ll);
        }
        double initialRadius = avgCellSize;

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

                size_t closestCell = octTree.closestPoint(p.location).getIndex();
                if(tess.IsPointInCell(p.location, closestCell))
                {
                    // the point is inside my domain, new location
                    p.cellIndex = closestCell;
                    myParticles.push_back(p);
                    // done!
                }
                else
                {
                    shouldExchangeParticles.push_back(p);
                }
            }
        }
        else
        {
            for(size_t i = 0; i < particles.size(); i++)
            {
                shouldExchangeParticles.push_back(particles[i]);
            }
        }

        size_t numParticlesLeft = particlesLeft.size();
        MPI_Allreduce(MPI_IN_PLACE, &numParticlesLeft, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        if(verbose and rank == 0)
        {
            std::cout << "Number of particles left to determination: " << numParticlesLeft << std::endl;
        }

        // first guesses
        FirstInaccurateMovements(tess, shouldExchangeParticles);

        particles = myParticles;
        std::vector<Particle3D> newParticles = std::move(myParticles);
        std::vector<boost::container::flat_set<rank_t>> ranksTested(particles.size());

        for(size_t i = 0; i < shouldExchangeParticles.size(); i++)
        {
            Particle3D &p = shouldExchangeParticles[i];
            size_t closestCell = octTree.closestPoint(p.location).getIndex();

            if(tess.IsPointInCell(p.location, closestCell))
            {
                p.cellIndex = closestCell;
                newParticles.push_back(p);
                ranksTested.push_back({});
            }
            else
            {
                size_t idx = particles.size();
                particlesLeft.insert(idx);
                ranksTested.push_back({rank});
            }
            particles.push_back(p);
        }

        std::vector<double> radiuses(particles.size(), initialRadius);
        std::vector<std::vector<Particle3D>> sendValues(size);
        std::vector<std::vector<size_t>> sendIndicesCpy(size);
        std::vector<std::vector<size_t>> acknowledgementValues(size);

        size_t iterations = 0;
        rank_t maxRanksTested = 0;

        START_TIMER_PREEMPTIVE("Main Loop");

        while(true)
        {
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
                        size_t closestPointIdx = octTree.closestPoint(p.location).getIndex();
                        eo.addEntry("Closest Local Point", closestPointIdx);
                        eo.addEntry("Closest Point Value", tess.GetMeshPoint(closestPointIdx));
                        eo.addEntry("Is Point in Cell", tess.IsPointInCell(p.location, closestPointIdx));
                        eo.addEntry("Rank", rank);
                        throw eo;
                    }
                    auto intersectingRanks = distributedOctTree.getIntersectingRanks(p.location, radiuses[i]);
                    for(rank_t _rank : intersectingRanks)
                    {
                        if(ranksTested[i].find(_rank) == ranksTested[i].end())
                        {
                            // not tested yet
                            atLeastOneNew = true; 
                            sendValues[_rank].push_back(p);
                            sendIndicesCpy[_rank].push_back(i);
                            ranksTested[i].insert(_rank);
                        }
                    }
                    maxRanksTested = std::max(maxRanksTested, static_cast<rank_t>(ranksTested[i].size()));
                    radiuses[i] *= RADIUSES_FACTOR;
                }
            }

            std::vector<std::vector<Particle3D>> receiveValues = MPI_Iexchange_all_to_all(sendValues, MPI_COMM_WORLD);
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

            std::vector<std::vector<size_t>> acknowledgements = MPI_Iexchange_all_to_all(acknowledgementValues, MPI_COMM_WORLD);
            assert(acknowledgements.size() == size);

            for(rank_t _rank = 0; _rank < size; _rank++)
            {
                const std::vector<size_t> &rankAcknowledgements = acknowledgements[_rank];
                for(size_t i : rankAcknowledgements)
                {
                    particlesLeft.erase(sendIndicesCpy[_rank][i]);
                }
            }
        }

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