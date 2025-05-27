#include "Voronoi3DMovement.hpp"
#include "ds/DistributedOctTree/DistributedOctTree.hpp"

#define RADIUSES_FACTOR 2

void AssertLocations(const Tessellation3D &tess, const std::vector<Particle3D> &particles)
{
    size_t Nparticles = particles.size();
    // check if the particles are inside the cells by checking the scalar products
    for(size_t i = 0; i < Nparticles; i++)
    {
        const Particle3D &particle = particles[i];
        size_t cellIndex = particle.cellIndex; 
        assert(cellIndex < tess.GetPointNo());
        if(not tess.IsPointInCell(particle.location, cellIndex))
        {
            std::cerr << "Voronoi3DMovement: new particle's location is wrong" << std::endl;
            std::cerr << "Particle " << i << " (" << particle.location << ", cell " << cellIndex << ": " << tess.GetMeshPoint(cellIndex) << ") is outside the cell" << std::endl; 
            exit(1); // todo: ok?
        }
    }
}

void UpdateNewCells(const Tessellation3D &tess, std::vector<Particle3D> &particles)
{
#ifndef RICH_MPI
    for(Particle3D &p : particles)
    {
        p.cellIndex = tess.GetContainingCell(p.location);
    }
#else // RICH_MPI
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto [ll, ur] = tess.GetBoxCoordinates();
    
    OctTree<IndexedVector3D> octTree(IndexedVector3D(ll, std::numeric_limits<size_t>::max()), IndexedVector3D(ur, std::numeric_limits<size_t>::max()));
    
    size_t N = tess.GetPointNo();
    for(size_t i = 0; i < N; i++)
    {
        const Vector3D &point = tess.GetMeshPoint(i);
        octTree.insert(IndexedVector3D(point, i));
    }
    assert(octTree.getSize() == N);
    
    DistributedOctTree<IndexedVector3D> distributedOctTree(&octTree);

    std::vector<Particle3D> newParticles;

    double avgCellSize = 0;
    for(size_t i = 0; i < N; i++)
    {
        avgCellSize += tess.GetWidth(i);
    }
    avgCellSize /= N;
    double initialRadius = avgCellSize;
    std::vector<double> radiuses(particles.size(), initialRadius);
    std::vector<boost::container::flat_set<rank_t>> ranksTested(particles.size());

    boost::container::flat_set<size_t> particlesLeft;

    for(size_t i = 0; i < particles.size(); i++)
    {
        // first, test whether the point is inside my domain
        Particle3D &p = particles[i];
        size_t closestCell = octTree.closestPoint(p.location).getIndex();
        if(tess.IsPointInCell(p.location, closestCell))
        {
            // the point is inside my domain
            p.cellIndex = closestCell;
            newParticles.push_back(p);
            // done!
        }
        else
        {
            particlesLeft.insert(i);
            ranksTested[i].insert(rank);
        }
    }

    std::vector<std::vector<Particle3D>> sendValues(size);
    std::vector<std::vector<size_t>> sendIndicesCpy(size);
    std::vector<std::vector<size_t>> acknowledgementValues(size);

    size_t iterations = 0;
    rank_t maxRanksTested = 0;

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

        std::vector<std::vector<Particle3D>> receiveValues = MPI_Exchange_all_to_all(sendValues, MPI_COMM_WORLD);
        assert(receiveValues.size() == size);

        for(rank_t _rank = 0; _rank < size; _rank++)
        {
            std::vector<Particle3D> &particlesFromRank = receiveValues[_rank];
            size_t Np = particlesFromRank.size();
            for(size_t i = 0; i < Np; i++)
            {
                Particle3D &p = particlesFromRank[i];
                size_t containingCell = octTree.closestPoint(p.location).getIndex();
                if(tess.IsPointInCell(p.location, containingCell))
                {
                    p.cellIndex = containingCell;
                    newParticles.push_back(p);
                    acknowledgementValues[_rank].push_back(i);
                }
            }
        }

        std::vector<std::vector<size_t>> acknowledgements = MPI_Exchange_all_to_all(acknowledgementValues, MPI_COMM_WORLD);
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

std::vector<Particle3D> MoveParticlesAlongMesh(const Tessellation3D &tess, const std::vector<Particle3D> &particles)
{
    return {}; // TODO
}