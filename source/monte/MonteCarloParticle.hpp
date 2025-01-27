#ifndef MONTE_CARLO_PARTICLE_HPP
#define MONTE_CARLO_PARTICLE_HPP

#include <vector>
#ifdef RICH_MPI
    #include <mpi.h>
    #include <functional>
    #include "mpi/mpi_commands.hpp"
    #include "misc/serializable.hpp"
#endif // RICH_MPI

using dt_t = double;
using distance_t = double;

template<typename T, typename Grid>
class MonteCarloParticle
                    #ifdef RICH_MPI
                        : public Serializable
                    #endif // RICH_MPI
{
public:
    T location;
    T velocity;
    distance_t distanceLeft;

    explicit MonteCarloParticle(const T &location_ = T(), const T &velocity_ = T()): location(location_), velocity(velocity_), distanceLeft(0){};

    void step();

    void move(dt_t dt);

    std::pair<size_t, distance_t> distanceToNearestFace(const Grid &grid, size_t cellIndex) const;

    inline bool isDone() const{return this->distanceLeft < EPSILON;}; // TODO: EPS?

    #ifdef RICH_MPI
        inline size_t getChunkSize() const override{return this->location.getChunkSize() + this->velocity.getChunkSize() + 1;};

        inline std::vector<double> serialize() const override
        {
            std::vector<double> location_ser = this->location.serialize();
            std::vector<double> velocity_ser = this->velocity.serialize();
            location_ser.insert(location_ser.end(), velocity_ser.cbegin(), velocity_ser.cend());
            location_ser.push_back(static_cast<double>(this->distanceLeft));
            return location_ser;
        }

        inline void unserialize(const std::vector<double> &data) override
        {
            std::vector<double> location_data = std::vector<double>(data.cbegin(), data.cbegin() + this->location.getChunkSize());
            std::vector<double> velocity_data = std::vector<double>(data.cbegin() + this->location.getChunkSize(), data.cbegin() + this->location.getChunkSize() + this->velocity.getChunkSize());
            this->location.unserialize(location_data);
            this->velocity.unserialize(velocity_data);
            this->distanceLeft = data[this->location.getChunkSize() + this->velocity.getChunkSize()];
        }
    #endif // RICH_MPI

    friend inline std::ostream &operator<<(std::ostream &stream, const MonteCarloParticle &particle)
    {
        return stream << "Particle(location " << particle.location << ", velocity " << particle.velocity << ")";
    }
};

template<typename T, typename Grid>
void MonteCarloParticle<T, Grid>::step()
{
    // TODO: wrong implementation
    distance_t d_scatter = random(); // TODO
    distance_t d_cell = this->distanceToNearestFace(tess, cellIndex); // TODO: ??
    distance_t distance = 0;
    this->move(dt); // todo: what's `dt`?
}

template<typename T, typename Grid>
std::pair<size_t, distance_t> MonteCarloParticle<T, Grid>::distanceToNearestFace(const Grid &grid, size_t cellIndex) const
{
    distance_t min_alpha = std::numeric_limits<distance_t>::max();
    size_t min_face = std::numeric_limits<size_t>::max();

    for(const size_t &faceIdx : grid.GetCellFaces(cellIndex))
    {
        const T &normal = grid.Normal(faceIdx); // normal to face
        const T &pointOnFace = grid.GetFacePoints(faceIdx)[0];
        double normalVelocityScalarProd = ScalarProd(normal, this->velocity);
        if(normalVelocityScalarProd < EPSILON) // negative
        {
            continue;
        }
        distance_t alpha = -1 * ScalarProd((pointOnFace - this->location), normal) / normalVelocityScalarProd;
        if(alpha < min_alpha)
        {
            min_alpha = alpha;
            min_face = faceIdx;
        }
    }

    if(min_alpha != std::numeric_limits<distance_t>::max())
    {
        return {min_face, min_alpha};
    }

    UniversalError eo("MonteCarloParticle::distanceToNearestFace: no face intersection found");
    eo.addEntry("Particle", *this);
    eo.addEntry("Cell Index", cellIndex);
    eo.addEntry("Faces Indices", grid.GetCellFaces(cellIndex));
    throw eo;
}

#ifdef RICH_MPI
    template<typename T, typename Grid, typename OwnerFunction = std::function<rank_t(const MonteCarloParticle<T, Grid>&)>>
    void MonteCarloTimestep(std::vector<MonteCarloParticle<T, Grid>> &particles, const Grid &grid, dt_t dt, const OwnerFunction &ownership)
    {
        using Particle = MonteCarloParticle<T, Grid>;
        // initialize
        for(Particle &p : particles)
        {
            p.distanceLeft = dt * abs(p.velocity); // todo: correct?
        }

        std::vector<size_t> leftParticlesIndices;

        while(true)
        {
            // get left particles
            leftParticlesIndices.clear();
            size_t N = particles.size();

            for(size_t particleIdx = 0; particleIdx < N; particleIdx++)
            {
                if(not particles[particleIdx].isDone())
                {
                    leftParticlesIndices.push_back(particleIdx);
                }
            }
            size_t leftParticlesNum = leftParticlesIndices.size();
            MPI_Allreduce(MPI_IN_PLACE, &leftParticlesNum, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

            if(leftParticlesNum == 0)
            {
                break; // all is good
            }

            for(const size_t &leftParticleIdx : leftParticlesIndices)
            {
                // advance
                particles[leftParticleIdx].step();
            }

            particles = MPI_Exchange_by_ownership(particles, ownership, MPI_COMM_WORLD);
        }
    }

    void MonteCarloTimestep(std::vector<MonteCarloParticle<Vector3D, Tessellation3D>> &particles, const Tessellation3D &tess, const dt_t dt)
    {
        using Particle = MonteCarloParticle<Vector3D, Tessellation3D>;
        MonteCarloTimestep<Vector3D, Tessellation3D>(particles, tess, dt, [&tess](const Particle &particle){return tess.GetOwner(particle.location);});
    }
#endif // RICH_MPI

#endif // MONTE_CARLO_PARTICLE_HPP