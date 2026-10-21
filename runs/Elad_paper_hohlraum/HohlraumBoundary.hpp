#ifndef HOHLRAUM_BOUNDARY_HPP
#define HOHLRAUM_BOUNDARY_HPP

#include <boost/math/special_functions/pow.hpp>
#include "monte/boundary/BoundaryCondition.hpp"
#include "CMMC/src/units/units.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include "monte/utils/CounterRNG.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#endif
#include "newtonian/three_dimensional/computational_cell.hpp"

// Boundary condition for the hohlraum problem:
//  - x=Lx face: blackbody source at given temperature
//  - ALL boundaries: vacuum (particles escape / are removed)
template<typename T, typename Grid>
class HohlraumBoundary : public BoundaryCondition<T, Grid>
{
public:
    HohlraumBoundary(const Grid &grid, const std::vector<ComputationalCell3D> &cells,
                     double temperature, size_t Npercell);

    MonteCarloParticleStatus apply(MonteCarloParticle<T> &particle) override;

    std::vector<MonteCarloParticle<T>> generateNewBoundaryParticles(double fullDt) override;

private:
    const std::vector<ComputationalCell3D> &cells;
    double temperature;
    size_t Npercell;
    std::uint64_t generationCall_ = 0;
};

template<typename T, typename Grid>
HohlraumBoundary<T, Grid>::HohlraumBoundary(const Grid &grid,
                                             const std::vector<ComputationalCell3D> &cells,
                                             double temperature, size_t Npercell)
    : BoundaryCondition<T, Grid>(grid), cells(cells),
      temperature(temperature), Npercell(Npercell)
{}

template<typename T, typename Grid>
MonteCarloParticleStatus HohlraumBoundary<T, Grid>::apply(MonteCarloParticle<T> &particle)
{
    // All boundaries are vacuum: remove every particle that reaches the boundary
    return MonteCarloParticleStatus::REMOVE;
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T>> HohlraumBoundary<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    // Photons are generated in parallel. Every random draw comes from STORM's
    // counter-based RNG keyed by (call, rank, photon slot), so the result does not
    // depend on the number of threads or on scheduling.
    const double T4 = boost::math::pow<4>(this->temperature);
    const size_t N = this->grid.GetPointNo();
    std::uint64_t rank = 0;
#ifdef RICH_MPI
    int mpiRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    rank = static_cast<std::uint64_t>(mpiRank);
#endif
    const std::uint64_t callSeed = STORM::CounterRNG::mix(0x486f686c7261756dULL ^ this->generationCall_++);

    // Emitting faces (x=Lx side, r < 0.65), each split into a triangle fan for
    // uniform sampling by area.
    struct SourceFace
    {
        size_t cellIndex;
        double energyPerPhoton;
        std::vector<double> cumulativeAreas;
        std::vector<std::array<T, 3>> triangles;
    };
    std::vector<SourceFace> faces;
    const auto &facePoints = this->grid.GetFacePoints();
    for(size_t i = 0; i < N; i++)
    {
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if(neighborIdx >= N && this->grid.IsPointOutsideBox(neighborIdx))
            {
                T normal = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                // Emit only from the x=Lx face (normal pointing in +x direction)
                if(normal.x < -0.99 && std::sqrt(point.y * point.y + point.z * point.z) < 0.65)
                {
                    SourceFace face{i, units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell, {}, {}};
                    const auto &vertices = this->grid.GetPointsInFace(faceIdx);
                    double totalArea = 0;
                    for(size_t v = 1; v + 1 < vertices.size(); v++)
                    {
                        const T &a = facePoints[vertices[0]];
                        const T &b = facePoints[vertices[v]];
                        const T &c = facePoints[vertices[v + 1]];
                        totalArea += abs(CrossProduct(b - a, c - a));
                        face.cumulativeAreas.push_back(totalArea);
                        face.triangles.push_back({a, b, c});
                    }
                    faces.push_back(std::move(face));
                }
            }
        }
    }

    const size_t photonsPerFace = this->Npercell;
    std::vector<MonteCarloParticle<T>> newParticles(faces.size() * photonsPerFace);
    const long long total = static_cast<long long>(newParticles.size());
#pragma omp parallel for schedule(static)
    for(long long slot = 0; slot < total; slot++)
    {
        const SourceFace &face = faces[static_cast<size_t>(slot) / photonsPerFace];
        const std::uint64_t key = STORM::CounterRNG::makeKey(callSeed, rank, static_cast<std::uint64_t>(slot));
        MonteCarloParticle<T> &newParticle = newParticles[static_cast<size_t>(slot)];

        // Uniform point on the face: pick a triangle by area, then a point in it.
        const double target = STORM::CounterRNG::unitOpen(key, 0) * face.cumulativeAreas.back();
        size_t triangle = static_cast<size_t>(std::lower_bound(face.cumulativeAreas.begin(), face.cumulativeAreas.end(), target) - face.cumulativeAreas.begin());
        triangle = std::min(triangle, face.triangles.size() - 1);
        double r1 = STORM::CounterRNG::unitOpen(key, 1);
        double r2 = STORM::CounterRNG::unitOpen(key, 2);
        if(r1 + r2 > 1)
        {
            r1 = 1 - r1;
            r2 = 1 - r2;
        }
        const auto &tri = face.triangles[triangle];
        newParticle.location = (1 - r1 - r2) * tri[0] + r1 * tri[1] + r2 * tri[2];

        // Lambert emission into -x direction (into the hohlraum)
        double mu = std::sqrt(STORM::CounterRNG::unitOpen(key, 3));
        newParticle.velocity.x = mu;
        double _1mmu = std::sqrt(1 - mu * mu);
        double theta = 2 * M_PI * STORM::CounterRNG::unitOpen(key, 4);
        newParticle.velocity.y = _1mmu * std::cos(theta);
        newParticle.velocity.z = _1mmu * std::sin(theta);
        newParticle.velocity *= units::clight;
        newParticle.frequency = 0;
        newParticle.weight = face.energyPerPhoton;
        newParticle.initialWeight = newParticle.weight;
        newParticle.timeLeft = fullDt * STORM::CounterRNG::unitOpen(key, 5);
        newParticle.cellIndex = face.cellIndex;

        if(this->grid.IsPointOutsideBox(newParticle.location))
        {
            const Vector3D original = newParticle.location;
            const Vector3D direction = this->grid.GetMeshPoint(newParticle.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(newParticle.location) && t < 1.0)
            {
                newParticle.location = original + t * direction;
                t *= 2;
            }
        }
    }

    return newParticles;
}

#endif // HOHLRAUM_BOUNDARY_HPP
