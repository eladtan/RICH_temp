#ifndef SIDE_TEMPERATURE_HPP
#define SIDE_TEMPERATURE_HPP

#include <cmath>
#include <boost/math/special_functions/pow.hpp>
#include "BoundaryCondition.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/tessellation/utils/RandomOnFace.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "misc/utils.hpp"

template<typename T, typename Grid>
class SideTemperature : public BoundaryCondition<T, Grid>
{
public:
    SideTemperature(const Grid &grid, const std::vector<ComputationalCell3D> &cells, double temperature, size_t Npercell, bool multigroup = false);
    
    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) override;

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        T nOut;
        if (!this->getDDMCOrientedOutwardNormal(
                faceIdx, insideCellIndex, outsidePointIndex, nOut))
            return DDMCBoundaryFaceBehavior::Unsupported;

        // Left x boundary: thermal source / removal face.
        if (nOut.x < -0.99)
            return DDMCBoundaryFaceBehavior::Unsupported;

        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    void SetTemperature(double temp) { temperature = temp; }

private:
    const std::vector<ComputationalCell3D> &cells;
    double temperature;
    size_t Npercell;
    std::array<double, ENERGY_GROUPS_NUM + 1> cumulativePlanckFunction;
    bool multigroup;
};

template<typename T, typename Grid>
SideTemperature<T, Grid>::SideTemperature(const Grid &grid, const std::vector<ComputationalCell3D> &cells, double temperature, size_t Npercell, bool multigroup):
    BoundaryCondition<T, Grid>(grid), cells(cells), temperature(temperature), Npercell(Npercell), multigroup(multigroup)
{
    if(this->multigroup)
    {
        double const kT = units::k_boltz * temperature;
        this->cumulativePlanckFunction[0] = 0.0;
    
        for(size_t g = 1; g < (ENERGY_GROUPS_NUM + 1); g++)
        {
            double const a = ComputationalCell3D::energyBoundaries[g-1] / kT;
            double const b = ComputationalCell3D::energyBoundaries[g] / kT;
            this->cumulativePlanckFunction[g] = planck_integral::planck_integral(a, b);
            this->cumulativePlanckFunction[g] += this->cumulativePlanckFunction[g-1];
        }
        if(std::abs(this->cumulativePlanckFunction.back() - 1.0) > 1e-8)
        {
            UniversalError eo("Cumulative Planck function does not sum to 1");
            eo.addEntry("Cumulative Planck function", this->cumulativePlanckFunction);
            throw eo;
        }
    }
    else
    {
        std::fill(this->cumulativePlanckFunction.begin(), this->cumulativePlanckFunction.end(), std::numeric_limits<double>::quiet_NaN());
    }
}

template<typename T, typename Grid>
MonteCarloParticleStatus SideTemperature<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    MonteCarloParticleStatus status = MonteCarloParticleStatus::DONE;
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    for(const typename Grid::Face_T &face : faces)
    {
        T normal;
        double faceScale = 0.0;
        if(this->getInwardBoxFaceNormalIfClose(face, particle.location, normal, faceScale))
        {
            if(std::abs(normal.x) > 0.99)
            {
                if(std::abs(particle.location.x - ll.x) < std::abs(ur.x - particle.location.x))
                {
                    return MonteCarloParticleStatus::REMOVE;
                }
            }
            if(this->reflectParticleOnBoxFace(particle, face))
                status = MonteCarloParticleStatus::REFLECT;
        }
    }
    if(status == MonteCarloParticleStatus::REFLECT)
        return status;

    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> SideTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    static const double T4 = boost::math::pow<4>(this->temperature);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re(0);

    std::vector<MonteCarloParticle<T, Grid>> newParticles;
    size_t N = this->grid.GetPointNo();
    for(size_t i = 0; i < N; i++)
    {
        const T &point = this->grid.GetMeshPoint(i);
        for(const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i)? neighbors.second : neighbors.first;
            if(neighborIdx >= N and this->grid.IsPointOutsideBox(neighborIdx))
            {
                T normal = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if(normal.x < -0.99)
                { // outside the box
                    double energyToProduce = units::sigma_sb * T4 * this->grid.GetArea(faceIdx) * fullDt / this->Npercell;
                    for(size_t j = 0; j < this->Npercell; j++)
                    {
                        newParticles.emplace_back();
                        MonteCarloParticle<T, Grid> &newParticle = newParticles.back();
                        newParticle.location = RandomPointOnFace(this->grid, faceIdx);
                        // Boundary-face vertices are obtained from floating-point
                        // clipping and can lie a few ulps outside the nominal box.
                        // Move the emitted packet into the owning convex Voronoi
                        // cell before the manager performs its strict box check.
                        constexpr double inwardNudge = 1e-10;
                        newParticle.location =
                            (1.0 - inwardNudge) * newParticle.location +
                            inwardNudge * point;
                        double mu = std::sqrt(unif(re));
                        // Lambert Emission Law
                        newParticle.velocity.x = mu;
                        double _1mmu = std::sqrt(1 - mu * mu);
                        double theta = 2 * M_PI * unif(re);
                        newParticle.velocity.y = _1mmu * std::cos(theta);
                        newParticle.velocity.z = _1mmu * std::sin(theta);
                        newParticle.velocity *= units::clight;
                        newParticle.frequency = 0;
                        if(this->multigroup)
                        {
                            newParticle.frequency = LinearInterpolation(this->cumulativePlanckFunction, ComputationalCell3D::energyBoundaries, unif(re));
                        }
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

#endif // SIDE_TEMPERATURE_HPP
