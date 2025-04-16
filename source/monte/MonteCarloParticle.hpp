#ifndef MONTE_CARLO_PARTICLE_HPP
#define MONTE_CARLO_PARTICLE_HPP

#include <vector>
#ifdef RICH_MPI
    // #include <mpi.h>
    // #include <functional>
    // #include "mpi/mpi_commands.hpp"
    // #include "misc/serializable.hpp"
#endif // RICH_MPI

using dt_t = double;
using distance_t = double;

template<typename T, typename Grid>
struct MonteCarloParticle
                    // #ifdef RICH_MPI
                    //     : public Serializable
                    // #endif // RICH_MPI
{
    size_t id;
    T location;
    T velocity;
    size_t cellIndex;
    dt_t timeLeft;
    double energy;
    double weight;

    explicit MonteCarloParticle(size_t id_ = std::numeric_limits<size_t>::max(), const T &location_ = T(), const T &velocity_ = T(), dt_t timeLeft_ = dt_t()):
         location(location_), velocity(velocity_), timeLeft(timeLeft_), cellIndex(std::numeric_limits<size_t>::max())
    {};

    std::pair<size_t, distance_t> distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const;

    friend inline std::ostream &operator<<(std::ostream &stream, const MonteCarloParticle &particle)
    {
        return stream << "Particle(ID " << particle.id << ", location " << particle.location << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ")";
        // return stream << "Particle(location " << particle.location << ", velocity " << particle.velocity << ", time " << particle.timeLeft << ")";
    }
};

template<typename T, typename Grid>
std::pair<size_t, dt_t> MonteCarloParticle<T, Grid>::distanceToNearestFace(const Grid &grid, const std::vector<T> &normalsOfCell, const std::vector<T> &pointsOnFaces) const
{
    std::pair<size_t, dt_t> best = {std::numeric_limits<size_t>::max(), std::numeric_limits<dt_t>::max()};
    size_t &min_face = best.first;
    dt_t &min_alpha = best.second;

    // if(this->id == 6804436)
    // {
    //     T loc = this->location;
    //     double distanceToCenter = abs(grid.GetMeshPoint(this->cellIndex) - loc);
    //     // for(size_t i = 0; i < grid.GetPointNo(); i++)
    //     // {
    //     //     double distance = abs(grid.GetMeshPoint(i) - this->location);
    //     //     assert(distanceToCenter <= distance + EPSILON);
    //     // }
        
    //     // // TOOD: remove!
    //     size_t realContaining = grid.GetContainingCell(this->location);
    //     double distanceToReal = abs(grid.GetMeshPoint(realContaining) - loc);
    //     if(realContaining != this->cellIndex and fabs(distanceToCenter - distanceToReal) < EPSILON)
    //     {
    //         std::cout << (*this) << " written in " << this->cellIndex << " (distance: " << distanceToCenter << ") but real containing cell is " << realContaining << " (distance: " << abs(grid.GetMeshPoint(realContaining) - this->location) << ")" << std::endl;
    //         exit(1);
    //     }
    //     assert(realContaining == this->cellIndex); // the particle is inside the cell
    // }

    const double velocityAbs = EPSILON * fastabs(this->velocity);
    const auto &faces = grid.GetCellFaces(this->cellIndex);
    size_t Nfaces = faces.size();

    for(size_t i = 0; i < Nfaces; ++i)
    {
        // std::cout << "Thread " << omp_get_thread_num() << std::endl;

        const size_t &faceIdx  = faces[i];
        // const T &normal = grid.Normal(faceIdx); // normal to face
        const T &normal = normalsOfCell[i]; // normal to face
        // std::cout << "Normal of face " << faceIdx << " is " << normal << std::endl;
        // const std::pair<size_t, size_t> &sides = grid.GetFaceNeighbors(faceIdx);
        
        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(BOOST_UNLIKELY(abs(normalVelocityScalarProd) < velocityAbs)) // zero
        {
            continue;
        }
        const T &pointOnFace = pointsOnFaces[i]; // (grid.GetMeshPoint(sides.first) + grid.GetMeshPoint(sides.second)) / 2;
        dt_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;

        __builtin_prefetch(&Nfaces, 0, 0);

        if(i < Nfaces - 1)
        {
            const Vector3D *nextNormal = &normalsOfCell[i + 1];
            __builtin_prefetch(nextNormal, 0, 2);
        }

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

    if(min_alpha != std::numeric_limits<distance_t>::max())
    {
        return best;
    }

    // should not reach here
    
    // check if point is inside domain
    if(grid.IsPointOutsideBox(this->location))
    {
        UniversalError eo("Particle is outside the domain, but still considered");
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
        eo.addEntry("Norg", grid.GetPointNo());
        eo.addEntry("Particle", (*this));
        throw eo;
    }

    UniversalError eo("MonteCarloParticle::distanceToNearestFace: no face intersection found");
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    eo.addEntry("Rank", rank);
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

#endif // MONTE_CARLO_PARTICLE_HPP