#ifndef MONTE_CARLO_PARTICLE_HPP
#define MONTE_CARLO_PARTICLE_HPP

#include <vector>
#include <limits>
#ifdef RICH_MPI
    #include <mpi.h>
    #include <functional>
    #include "mpi/mpi_commands.hpp"
    #include "misc/serializable.hpp"
#endif // RICH_MPI
#include "misc/universal_error.hpp"

#define EPSILON 1e-12

using dt_t = double;
using distance_t = double;

template<typename T, typename Grid>
struct MonteCarloParticle
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
    #ifdef RICH_MPI
        rank_t rank = -1;
        #ifdef MONTECARLO_DEBUG
            size_t cellIndexInPrevRank = std::numeric_limits<size_t>::max();
            T previousLocation = T(std::numeric_limits<double>::max());
            size_t particleTHInLastRank = std::numeric_limits<size_t>::max();
            size_t particleIndexInLastRank = std::numeric_limits<size_t>::max();
            bool checkedHere = true; // reset checked here flag
            size_t ghostIndex = std::numeric_limits<size_t>::max();
            T newCellValue = T(std::numeric_limits<double>::max());
            rank_t nextRank = std::numeric_limits<rank_t>::max();
            rank_t sentByRank = std::numeric_limits<rank_t>::max();
            bool removedFromRank = false;
            size_t lastSeen = 0;
            rank_t lastSeenRank = std::numeric_limits<rank_t>::max();
            rank_t lastSeenRankBuf = std::numeric_limits<rank_t>::max();
            size_t lastSeenIndex = std::numeric_limits<size_t>::max();
        #endif // MONTECARLO_DEBUG
    #endif // RICH_MPI
    size_t id = std::numeric_limits<size_t>::max();
    size_t cellID = std::numeric_limits<size_t>::max();
    T location = T(std::numeric_limits<typename T::value_type>::max());
    T velocity = T(std::numeric_limits<typename T::value_type>::max());
    size_t cellIndex = std::numeric_limits<size_t>::max();
    dt_t timeLeft = std::numeric_limits<dt_t>::max();
    double energy = std::numeric_limits<double>::max();
    double weight = std::numeric_limits<double>::max();
    double initialWeight = std::numeric_limits<double>::max();
    size_t steps = 0;
    bool on_track = false;
    bool sent = false;

    explicit MonteCarloParticle(size_t id_ = std::numeric_limits<size_t>::max(), const T &location_ = T(std::numeric_limits<double>::max()), const T &velocity_ = T(std::numeric_limits<double>::max()), dt_t timeLeft_ = dt_t(std::numeric_limits<double>::max())):
        id(id_), location(location_), velocity(velocity_), cellIndex(std::numeric_limits<size_t>::max()), timeLeft(timeLeft_), energy(0), weight(0), initialWeight(0), steps(0), on_track(false)
    {
        #ifdef MONTECARLO_DEBUG
        this->checkedHere = true;
        this->ghostIndex = std::numeric_limits<size_t>::max();
        this->newCellValue = T(std::numeric_limits<double>::max());
        this->nextRank = std::numeric_limits<rank_t>::max();
        this->removedFromRank = false;
        this->sentByRank = std::numeric_limits<rank_t>::max();
        #endif // MONTECARLO_DEBUG
    };

    std::pair<size_t, distance_t> distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const;

    friend inline std::ostream &operator<<(std::ostream &stream, const MonteCarloParticle &particle)
    {
        #ifdef RICH_MPI
                return stream << "Particle(ID " << particle.id << " of rank " << particle.rank << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps << ")";
        #else // RICH_MPI
                return stream << "Particle(ID " << particle.id << ", location " << particle.location << " in cell " << particle.cellIndex << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ", steps " << particle.steps << ")";
        #endif // RICH_MPI
    }

    inline bool operator==(const MonteCarloParticle &other) const
    {
        #ifdef RICH_MPI
            return this->id == other.id and this->rank == other.rank;
        #else
            return this->id == other.id;
        #endif
    }

    #ifdef RICH_MPI
        size_t dump(Serializer *serializer) const override;

        size_t load(const Serializer *serializer, size_t byteOffset) override;
    #endif // RICH_MPI
};

template<typename T, typename Grid>
std::pair<size_t, dt_t> MonteCarloParticle<T, Grid>::distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const
{
    std::pair<size_t, dt_t> best = {std::numeric_limits<size_t>::max(), std::numeric_limits<dt_t>::max()};
    size_t &min_face = best.first;
    dt_t &min_alpha = best.second;

    // #ifdef MONTECARLO_DEBUG
    // if(not grid.IsPointInCell(this->location, this->cellIndex))
    // {
    //     const T &declaredCell = grid.GetMeshPoint(this->cellIndex);
    //     size_t containingIdx = grid.GetContainingCell(this->location);
    //     const T &containingCell = grid.GetMeshPoint(containingIdx);
    //     UniversalError eo("MonteCarloParticle<T, Grid>::distanceToNearestFace: Particle is outside its cell");
    //     eo.addEntry("Particle", *this);
    //     eo.addEntry("Declared Cell Index", this->cellIndex);
    //     eo.addEntry("Declared Cell", declaredCell);
    //     eo.addEntry("Declared Cell - Distance", abs(declaredCell - this->location));
    //     eo.addEntry("Real Containing Cell Index", containingIdx);
    //     eo.addEntry("Real Containing Cell", containingCell);
    //     eo.addEntry("Real Cell - Distance", abs(containingCell - this->location));
    //     throw eo;
    // }
    // #endif // MONTECARLO_DEBUG

    const double velocityAbs = EPSILON * fastabs(this->velocity);
    const auto &faces = grid.GetCellFaces(this->cellIndex);
    size_t Nfaces = faces.size();

    bool verbose = false; //  (this->rank == 93 and (this->id == 2449725 or this->id == 2432353));
    bool crash = false;
    if(verbose)
    {
        const std::pair<size_t, size_t> &neighbors = grid.GetFaceNeighbors(min_face);
        // size_t otherNeighbor = (neighbors.first == this->cellIndex)? neighbors.second : neighbors.first;
        if(not grid.IsPointInCell(this->location, this->cellIndex))
        {
            const T &declaredCell = grid.GetMeshPoint(this->cellIndex);
            size_t containingIdx = grid.GetContainingCell(this->location);
            const T &containingCell = grid.GetMeshPoint(containingIdx);
            std::cout << "Particle " << (*this) << ", illegal location (declared " << this->cellIndex << " (" << declaredCell << ") - distance " << abs(declaredCell - this->location) <<
                            ", actual: " << containingIdx << " (" << containingCell << ") - distance " << abs(containingCell - this->location) << "). steps: " << this->steps << std::endl;
            crash = true;
        }
    }

    for(size_t i = 0; i < Nfaces; ++i)
    {
        // std::cout << "Thread " << omp_get_thread_num() << std::endl;

        const size_t &faceIdx  = faces[i];
        // const T &normal = grid.Normal(faceIdx); // normal to face
        const T &normal = normalsOfCell[i]; // normal to face
        // std::cout << "Normal of face " << faceIdx << " is " << normal << std::endl;
        // const std::pair<size_t, size_t> &sides = grid.GetFaceNeighbors(faceIdx);
        
        if(crash)
        {
            const T &onPlane = grid.FaceCM(faceIdx);
            double distance = std::abs(ScalarProd(normal, this->location - onPlane)) / abs(normal);
            const std::pair<size_t, size_t> &neighbors = grid.GetFaceNeighbors(faceIdx);
            size_t otherNeighbor = (neighbors.first == this->cellIndex)? neighbors.second : neighbors.first;
            #ifdef RICH_MPI
                std::cout << "Particle " << this->id << " of rank " << this->rank << ", distance from face " << faceIdx << " (other neighbor: " << otherNeighbor << ") is " << distance << std::endl;
            #else // RICH_MPI
                std::cout << "Particle " << this->id << ", distance from face " << faceIdx << " (other neighbor: " << otherNeighbor << ") is " << distance << std::endl;
            #endif // RICH_MPI
        }

        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(BOOST_UNLIKELY(normalVelocityScalarProd >= -velocityAbs)) // positive
        {
            continue;
        }
        const T &pointOnFace = pointsOnFaces[i]; // (grid.GetMeshPoint(sides.first) + grid.GetMeshPoint(sides.second)) / 2;
        
        dt_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;

        __builtin_prefetch(&Nfaces, 0, 0);

        if(i < Nfaces - 1)
        {
            const T *nextNormal = &normalsOfCell[i + 1];
            __builtin_prefetch(nextNormal, 0, 2);
        }

        // if(verbose)
        // {
        //     const std::pair<size_t, size_t> &neighbors = grid.GetFaceNeighbors(faceIdx);
        //     size_t otherNeighbor = (neighbors.first == this->cellIndex)? neighbors.second : neighbors.first;
        //     std::cout << "Particle " << (*this) << ", face " << faceIdx << " (other neighbor: " << otherNeighbor << ") alpha " << alpha << ", current min " << min_alpha << std::endl;
        // } 
        __builtin_prefetch(&min_alpha, 1, 3);
        __builtin_prefetch(&min_face, 1, 3);
        if(BOOST_UNLIKELY(alpha < min_alpha))
        {
            if(alpha > 0 /* EPSILON */)
            {
                min_alpha = alpha;
                min_face = faceIdx;
            }
        }
    }

    if(crash)
    {
        exit(1);
    }

    if(min_alpha != std::numeric_limits<distance_t>::max())
    {
        return best;
    }


    // should not reach here
    
    // check if point is inside domain
    if(grid.IsPointOutsideBox(this->location))
    {
        UniversalError eo("Particle is outside the domain, but still considered");
        #ifdef RICH_MPI
            rank_t rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            eo.addEntry("Rank", rank);
        #endif // RICH_MPI
        eo.addEntry("Particle", *this);
        throw eo;
    }
    // assert the point is inside this cell
    size_t realContainingCell = grid.GetContainingCell(this->location);
    if(realContainingCell != this->cellIndex)
    {
        UniversalError eo("MonteCarloParticle::distanceToNearestFace: the containing cellIndex is incorrect");
        eo.addEntry("Real containing cell index", realContainingCell);
        eo.addEntry("Real containing point", grid.GetMeshPoint(realContainingCell));
        eo.addEntry("Distance to real", abs(grid.GetMeshPoint(realContainingCell) - this->location));
        eo.addEntry("Declared containing cell", this->cellIndex);
        eo.addEntry("Declared containing point", grid.GetMeshPoint(this->cellIndex));
        eo.addEntry("Distance to declared", abs(grid.GetMeshPoint(this->cellIndex) - this->location));
        
        for(size_t i = 0; i < Nfaces; ++i)
        {
            const size_t &faceIdx = faces[i];
            const T &normal = grid.Normal(faceIdx);
            const std::pair<size_t, size_t> &sides = grid.GetFaceNeighbors(faceIdx);
            // size_t otherNeighbor = (sides.first == this->cellIndex)? sides.second : sides.first; // todo remove
            double distanceFromFace = std::abs(ScalarProd(normal, grid.FaceCM(faceIdx) - this->location)) / abs(normal);
            eo.addEntry("Face " + std::to_string(i) + " index", faceIdx);
            eo.addEntry("Distance from face " + std::to_string(i), distanceFromFace);
        }
        eo.addEntry("Norg", grid.GetPointNo());
        eo.addEntry("Particle", (*this));
        throw eo;
    }

    UniversalError eo("MonteCarloParticle::distanceToNearestFace: no face intersection found");
    #ifdef RICH_MPI
    rank_t rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        eo.addEntry("Rank", rank);
    #endif // RICH_MPI
    eo.addEntry("Particle", *this);
    eo.addEntry("Cell Index", this->cellIndex);
    eo.addEntry("Cell Point", grid.GetMeshPoint(this->cellIndex));
    eo.addEntry("Distance to cell point", abs(grid.GetMeshPoint(this->cellIndex) - this->location));
    eo.addEntry("Faces Indices", grid.GetCellFaces(this->cellIndex));
    for(const size_t &faceIdx : grid.GetCellFaces(this->cellIndex))
    {
        const T &normal = grid.Normal(faceIdx); // normal to face
        const std::pair<size_t, size_t> &sides = grid.GetFaceNeighbors(faceIdx);
        const T &pointOnFace = (grid.GetMeshPoint(sides.first) + grid.GetMeshPoint(sides.second)) / 2;
        size_t otherNeighbor = (sides.first == this->cellIndex)? sides.second : sides.first; // todo remove
        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        dt_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;
        // if(verbose) std::cout << "For ID " << this->id << " of cell " << cellIndex << ", face " << faceIdx << " with neighbor " << otherNeighbor << ", distance is " << alpha << " (current min: " << min_alpha << ") and point will be " << this->location + alpha * this->velocity << std::endl;
        eo.addEntry("Face " + std::to_string(faceIdx) + " distance to face", alpha);
        const T &pointOtherSide = grid.GetMeshPoint(otherNeighbor);
        eo.addEntry("Face " + std::to_string(faceIdx) + " distance from other neighbor", abs(pointOnFace - pointOtherSide));
    }
    throw eo;
}

#ifdef RICH_MPI
template<typename T, typename Grid>
size_t MonteCarloParticle<T, Grid>::dump(Serializer *serializer) const
{
    size_t bytes = 0;
    bytes += serializer->insert(this->rank);
    bytes += serializer->insert(this->id);
    bytes += serializer->insert(this->cellID);
    bytes += serializer->insert(this->location);
    bytes += serializer->insert(this->velocity);
    bytes += serializer->insert(this->cellIndex);
    bytes += serializer->insert(this->timeLeft);
    bytes += serializer->insert(this->energy);
    bytes += serializer->insert(this->weight);
    bytes += serializer->insert(this->initialWeight);
    bytes += serializer->insert(this->steps);
    bytes += serializer->insert(this->on_track);
    bytes += serializer->insert(this->sent);
    #ifdef MONTECARLO_DEBUG
    bytes += serializer->insert(this->checkedHere);
    bytes += serializer->insert(this->ghostIndex);
    bytes += serializer->insert(this->newCellValue);
    bytes += serializer->insert(this->nextRank);
    bytes += serializer->insert(this->removedFromRank);
    bytes += serializer->insert(this->sentByRank);
    bytes += serializer->insert(this->lastSeen);
    bytes += serializer->insert(this->lastSeenRank);
    bytes += serializer->insert(this->lastSeenRankBuf);
    bytes += serializer->insert(this->lastSeenIndex);
    #endif // MONTECARLO_DEBUG
    return bytes;
}

template<typename T, typename Grid>
size_t MonteCarloParticle<T, Grid>::load(const Serializer *serializer, size_t byteOffset)
{
    size_t bytes = 0;
    bytes += serializer->extract(this->rank, byteOffset);
    bytes += serializer->extract(this->id, byteOffset + bytes);
    bytes += serializer->extract(this->cellID, byteOffset + bytes);
    bytes += serializer->extract(this->location, byteOffset + bytes);
    bytes += serializer->extract(this->velocity, byteOffset + bytes);
    bytes += serializer->extract(this->cellIndex, byteOffset + bytes);
    bytes += serializer->extract(this->timeLeft, byteOffset + bytes);
    bytes += serializer->extract(this->energy, byteOffset + bytes);
    bytes += serializer->extract(this->weight, byteOffset + bytes);
    bytes += serializer->extract(this->initialWeight, byteOffset + bytes);
    bytes += serializer->extract(this->steps, byteOffset + bytes);
    bytes += serializer->extract(this->on_track, byteOffset + bytes);
    bytes += serializer->extract(this->sent, byteOffset + bytes);
    #ifdef MONTECARLO_DEBUG
    bytes += serializer->extract(this->checkedHere, byteOffset + bytes);
    bytes += serializer->extract(this->ghostIndex, byteOffset + bytes);
    bytes += serializer->extract(this->newCellValue, byteOffset + bytes);
    bytes += serializer->extract(this->nextRank, byteOffset + bytes);
    bytes += serializer->extract(this->removedFromRank, byteOffset + bytes);
    bytes += serializer->extract(this->sentByRank, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeen, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenRank, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenRankBuf, byteOffset + bytes);
    bytes += serializer->extract(this->lastSeenIndex, byteOffset + bytes);
    #endif // MONTECARLO_DEBUG
    return bytes;
}
#endif // RICH_MPI

#endif // MONTE_CARLO_PARTICLE_HPP