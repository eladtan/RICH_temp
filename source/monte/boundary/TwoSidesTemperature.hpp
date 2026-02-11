#ifndef TWO_SIDES_TEMPERATURE_HPP
#define TWO_SIDES_TEMPERATURE_HPP

#include <boost/math/special_functions/pow.hpp>
#include "BoundaryCondition.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/tessellation/utils/RandomOnFace.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

template<typename T, typename Grid>
class TwoSidesTemperature : public BoundaryCondition<T, Grid>
{
public:
    TwoSidesTemperature(const Grid &grid, const std::vector<ComputationalCell3D> &cells, double temperatureLeft, double temperatureRight, size_t Npercell, bool withHydro = false);
    
    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) override;

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

private:
    const std::vector<ComputationalCell3D> &cells;
    double temperatureLeft;
    double temperatureRight;
    size_t Npercell;
    bool withHydro;
};

template<typename T, typename Grid>
TwoSidesTemperature<T, Grid>::TwoSidesTemperature(const Grid &grid, const std::vector<ComputationalCell3D> &cells, double temperatureLeft, double temperatureRight, size_t Npercell, bool withHydro):
    BoundaryCondition<T, Grid>(grid), cells(cells), temperatureLeft(temperatureLeft), temperatureRight(temperatureRight), Npercell(Npercell), withHydro(withHydro)
{}

template<typename T, typename Grid>
MonteCarloParticleStatus TwoSidesTemperature<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
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
            if(std::abs(normal.x) > 0.99)
            {
                return MonteCarloParticleStatus::REMOVE;
            }
            // Reflect the particle
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
            status = MonteCarloParticleStatus::REFLECT;
            return status;
        }
    }

    // should not reach here
    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> TwoSidesTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    static const double T4_L = boost::math::pow<4>(this->temperatureLeft);
    static const double T4_R = boost::math::pow<4>(this->temperatureRight);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re(0);
    double gamma;

    std::vector<MonteCarloParticle<T, Grid>> newParticles;
    size_t N = this->grid.GetPointNo();
    for(size_t i = 0; i < N; i++)
    {
        const ComputationalCell3D &cell = this->cells[i];
        bool calculatedGamma = false;
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i)? neighbors.second : neighbors.first;
            if(neighborIdx >= N and this->grid.IsPointOutsideBox(neighborIdx))
            {
                T normal = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if(std::abs(normal.x) > 0.99)
                { // outside the box
                    if(not calculatedGamma)
                    {
                        gamma = (this->withHydro)? 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity) * units::inv_clight2) : 1;
                        calculatedGamma = true;
                    }
                    double energyToProduce;
                    if(normal.x > 0)
                    {
                        energyToProduce = units::sigma_sb * T4_R * this->grid.GetArea(faceIdx) * fullDt * gamma / this->Npercell;
                    }
                    else
                    {
                        energyToProduce = units::sigma_sb * T4_L * this->grid.GetArea(faceIdx) * fullDt * gamma / this->Npercell;
                    }
                    for(size_t j = 0; j < this->Npercell; j++)
                    {
                        newParticles.emplace_back();
                        MonteCarloParticle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = RandomPointOnFace(this->grid, faceIdx);
                        double mu = std::sqrt(unif(re));
                        // Lambert Emission Law
                        newParticle.velocity.x = (normal.x > 0)? -mu : mu;
                        double _1mmu = std::sqrt(1 - mu * mu);
                        double theta = 2 * M_PI * unif(re);
                        newParticle.velocity.y = _1mmu * std::cos(theta);
                        newParticle.velocity.z = _1mmu * std::sin(theta);
                        newParticle.velocity *= units::clight;
                        newParticle.energy = 0;
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

#endif // TWO_SIDES_TEMPERATURE_HPP