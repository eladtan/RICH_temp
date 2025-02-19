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
    rank_t sender;

    explicit MonteCarloParticle(size_t id_ = std::numeric_limits<size_t>::max(), const T &location_ = T(), const T &velocity_ = T(), dt_t timeLeft_ = dt_t()):
         id(id_), location(location_), velocity(velocity_), timeLeft(timeLeft_), cellIndex(std::numeric_limits<size_t>::max()), sender(-1)
    {};

    std::pair<size_t, distance_t> distanceToNearestFace(const Grid &grid) const;

    // #ifdef RICH_MPI
    //     inline size_t getChunkSize() const override{return this->location.getChunkSize() + this->velocity.getChunkSize() + 1;};

    //     inline std::vector<double> serialize() const override
    //     {
    //         std::vector<double> location_ser = this->location.serialize();
    //         std::vector<double> velocity_ser = this->velocity.serialize();
    //         location_ser.insert(location_ser.end(), velocity_ser.cbegin(), velocity_ser.cend());
    //         location_ser.push_back(static_cast<double>(this->distanceLeft));
    //         return location_ser;
    //     }

    //     inline void unserialize(const std::vector<double> &data) override
    //     {
    //         std::vector<double> location_data = std::vector<double>(data.cbegin(), data.cbegin() + this->location.getChunkSize());
    //         std::vector<double> velocity_data = std::vector<double>(data.cbegin() + this->location.getChunkSize(), data.cbegin() + this->location.getChunkSize() + this->velocity.getChunkSize());
    //         this->location.unserialize(location_data);
    //         this->velocity.unserialize(velocity_data);
    //         this->distanceLeft = data[this->location.getChunkSize() + this->velocity.getChunkSize()];
    //     }
    // #endif // RICH_MPI

    friend inline std::ostream &operator<<(std::ostream &stream, const MonteCarloParticle &particle)
    {
        return stream << "Particle(ID " << particle.id << ", location " << particle.location << ", velocity " << particle.velocity << ")";
    }
};

template<typename T, typename Grid>
std::pair<size_t, dt_t> MonteCarloParticle<T, Grid>::distanceToNearestFace(const Grid &grid) const
{
    dt_t min_alpha = std::numeric_limits<distance_t>::max();
    size_t min_face = std::numeric_limits<size_t>::max();

    for(const size_t &faceIdx : grid.GetCellFaces(this->cellIndex))
    {
        const T &normal = grid.Normal(faceIdx); // normal to face
        const std::pair<size_t, size_t> &sides = grid.GetFaceNeighbors(faceIdx);
        const T &pointOnFace = (grid.GetMeshPoint(sides.first) + grid.GetMeshPoint(sides.second)) / 2;
        size_t otherNeighbor = (sides.first == this->cellIndex)? sides.second : sides.first; // todo remove
        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(abs(normalVelocityScalarProd) < EPSILON) // zero
        {
            continue;
        }
        dt_t alpha = ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;
        if(alpha < min_alpha and alpha > 0)
        {
            min_alpha = alpha;
            min_face = faceIdx;
        }
        // if(verbose) std::cout << "For ID " << this->id << " of cell " << cellIndex << ", face " << faceIdx << " with neighbor " << otherNeighbor << ", distance is " << alpha << " (current min: " << min_alpha << ") and point will be " << this->location + alpha * this->velocity << std::endl;
    }

    if(min_alpha != std::numeric_limits<distance_t>::max())
    {
        return {min_face, min_alpha};
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