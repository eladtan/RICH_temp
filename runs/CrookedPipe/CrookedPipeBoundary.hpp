#ifndef CROOKED_PIPE_BOUNDARY_HPP
#define CROOKED_PIPE_BOUNDARY_HPP

#include <boost/math/special_functions/pow.hpp>
#include "monte/boundary/BoundaryCondition.hpp"
#include "CMMC/src/units/units.hpp"
#include "3D/tessellation/utils/RandomOnFace.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#define MONTECARLO_EPS 1e-8

template<typename T, typename Grid>
class CrookedPipeBoundaryCondition : public BoundaryCondition<T, Grid>
{
public:
    CrookedPipeBoundaryCondition(const Grid &grid, const std::vector<ComputationalCell3D> &cells);
    
    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) override;

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

private:
    const std::vector<ComputationalCell3D> &cells;
 };

template<typename T, typename Grid>
CrookedPipeBoundaryCondition<T, Grid>::CrookedPipeBoundaryCondition(const Grid &grid, const std::vector<ComputationalCell3D> &cells):
    BoundaryCondition<T, Grid>(grid), cells(cells)
{}

template<typename T, typename Grid>
MonteCarloParticleStatus CrookedPipeBoundaryCondition<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    MonteCarloParticleStatus status;
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    for(const typename Grid::Face_T &face : faces)
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        T normal = CrossProduct(u, v);
        double absU = abs(u);
        if(std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
        {
            // intersects this face
            normal /= abs(normal);
            if(std::abs(normal.x) > 0.99 && this->cells[particle.cellIndex].tracers[1] > 0.5)
            {
                return MonteCarloParticleStatus::REMOVE;
            }
            // Reflect the particle
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
            particle.location = particle.location * (1 - MONTECARLO_EPS) + MONTECARLO_EPS * this->grid.GetMeshPoint(particle.cellIndex);
            status = MonteCarloParticleStatus::REFLECT;
            return status;
        }
    }

    // should not reach here
    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> CrookedPipeBoundaryCondition<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    static const double T4 = boost::math::pow<4>(0.5 * units::kev_kelvin);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re(0);

    // re.seed(0); // todo: remove

    std::vector<MonteCarloParticle<T, Grid>> newParticles;
    size_t N = this->grid.GetPointNo();
    for(size_t i = 0; i < N; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i)? neighbors.second : neighbors.first;
            if(neighborIdx >= N and this->grid.IsPointOutsideBox(neighborIdx))
            {
                T normal = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if(normal.x < -0.99 && cell.tracers[1] > 0.5)
                { // outside the box
                    // std::cout << "Generating 100 new particles on cell " << i << " (" << this->grid.GetMeshPoint(i) << "), face " << faceIdx << ", with normal " << normal << ", cell tracers[1] " << cell.tracers[1] << std::endl;
                    double energyToProduce;
                    energyToProduce = units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / 100;
                    for(size_t j = 0; j < 100; j++)
                    {
                        newParticles.emplace_back();
                        MonteCarloParticle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = RandomPointOnFace(this->grid, faceIdx);
                        newParticle.location = newParticle.location * (1 - MONTECARLO_EPS) + MONTECARLO_EPS * this->grid.GetMeshPoint(i);
                        if(newParticle.location.x < 1e-12)
                        {
                            UniversalError eo("CrookedPipeBoundaryCondition::generateNewBoundaryParticles: new particle location is too close to zero");
                            eo.addEntry("New particle location", newParticle.location);
                            throw eo;
                        }
                        double mu = std::sqrt(unif(re));
                        // Lambert Emission Law
                        newParticle.velocity.x = (normal.x > 0)? -mu : mu;
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
    return newParticles;
}

#endif // CROOKED_PIPE_BOUNDARY_HPP