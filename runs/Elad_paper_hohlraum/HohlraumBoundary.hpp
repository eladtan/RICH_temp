#ifndef HOHLRAUM_BOUNDARY_HPP
#define HOHLRAUM_BOUNDARY_HPP

#include <boost/math/special_functions/pow.hpp>
#include "monte/boundary/BoundaryCondition.hpp"
#include "CMMC/src/units/units.hpp"
#include "monte/utils/RandomOnFace.hpp"
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

    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) override;

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

private:
    const std::vector<ComputationalCell3D> &cells;
    double temperature;
    size_t Npercell;
};

template<typename T, typename Grid>
HohlraumBoundary<T, Grid>::HohlraumBoundary(const Grid &grid,
                                             const std::vector<ComputationalCell3D> &cells,
                                             double temperature, size_t Npercell)
    : BoundaryCondition<T, Grid>(grid), cells(cells),
      temperature(temperature), Npercell(Npercell)
{}

template<typename T, typename Grid>
MonteCarloParticleStatus HohlraumBoundary<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
{
    // All boundaries are vacuum: remove every particle that reaches the boundary
    return MonteCarloParticleStatus::REMOVE;
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> HohlraumBoundary<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    const double T4 = boost::math::pow<4>(this->temperature);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re(0);

    std::vector<MonteCarloParticle<T, Grid>> newParticles;
    size_t N = this->grid.GetPointNo();
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();

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
                    double energyToProduce = units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
                    for(size_t j = 0; j < this->Npercell; j++)
                    {
                        newParticles.emplace_back();
                        MonteCarloParticle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = STORM::RandomPointOnFace<T, Grid>(this->grid, faceIdx);
                        double mu = std::sqrt(unif(re));
                        // Lambert emission into -x direction (into the hohlraum)
                        newParticle.velocity.x = mu;
                        double _1mmu = std::sqrt(1 - mu * mu);
                        double theta = 2 * M_PI * unif(re);
                        newParticle.velocity.y = _1mmu * std::cos(theta);
                        newParticle.velocity.z = _1mmu * std::sin(theta);
                        newParticle.velocity *= units::clight;
                        newParticle.frequency = 0;
                        newParticle.weight = energyToProduce;
                        newParticle.initialWeight = newParticle.weight;
                        newParticle.timeLeft = fullDt * unif(re);
                        newParticle.cellIndex = i;
                    }
                }
            }
        }
    }


    for(auto &p : newParticles)
    {
        if(this->grid.IsPointOutsideBox(p.location))
        {
            const Vector3D original = p.location;
            const Vector3D direction = this->grid.GetMeshPoint(p.cellIndex) - original;
            double t = 1e-6;
            while(this->grid.IsPointOutsideBox(p.location) && t < 1.0)
            {
                p.location = original + t * direction;
                t *= 2;
            }
        }
    }

    return newParticles;
}

#endif // HOHLRAUM_BOUNDARY_HPP
