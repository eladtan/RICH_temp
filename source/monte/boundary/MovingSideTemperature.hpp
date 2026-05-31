#ifndef MOVING_SIDE_TEMPERATURE_HPP
#define MOVING_SIDE_TEMPERATURE_HPP

#include <cmath>
#include <boost/math/special_functions/pow.hpp>
#include "BoundaryCondition.hpp"
#include "Radiation/CMMC/src/planck_integral/planck_integral.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/tessellation/utils/RandomOnFace.hpp"
#include "3D/radiation/LorentzTransformation.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "misc/utils.hpp"
#ifdef RICH_MPI
#include <mpi.h>
#endif

template<typename T, typename Grid>
class MovingSideTemperature : public BoundaryCondition<T, Grid>
{
public:
    MovingSideTemperature(const Grid &grid, double temperature, size_t Npercell, bool multigroup = false);

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

        // Left x moving thermal source. Not implemented as a DDMC boundary yet.
        if (nOut.x < -0.99)
            return DDMCBoundaryFaceBehavior::Unsupported;

        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }

    void SetTemperature(double temperature);

private:
    double temperature_;
    size_t Npercell_;
    bool multigroup_;
    std::array<double, ENERGY_GROUPS_NUM + 1> cumulativePlanckFunction_;

    T leftFaceVelocity_;
    double prevLeftX_;

    void RecomputePlanckCDF();
    void updateLeftFaceVelocityFromBox(double fullDt);
};

template<typename T, typename Grid>
MovingSideTemperature<T, Grid>::MovingSideTemperature(
    const Grid &grid, double temperature, size_t Npercell, bool multigroup)
    : BoundaryCondition<T, Grid>(grid),
      temperature_(temperature), Npercell_(Npercell), multigroup_(multigroup),
      leftFaceVelocity_(0, 0, 0)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    prevLeftX_ = ll.x;

    if (multigroup_)
        RecomputePlanckCDF();
    else
        std::fill(cumulativePlanckFunction_.begin(), cumulativePlanckFunction_.end(),
                  std::numeric_limits<double>::quiet_NaN());
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::SetTemperature(double temperature)
{
    temperature_ = temperature;
    if (multigroup_)
        RecomputePlanckCDF();
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::RecomputePlanckCDF()
{
    double const kT = units::k_boltz * temperature_;
    cumulativePlanckFunction_[0] = 0.0;
    for (size_t g = 1; g < (ENERGY_GROUPS_NUM + 1); ++g)
    {
        double const a = ComputationalCell3D::energyBoundaries[g - 1] / kT;
        double const b = ComputationalCell3D::energyBoundaries[g] / kT;
        cumulativePlanckFunction_[g] = cumulativePlanckFunction_[g - 1]
                                     + planck_integral::planck_integral(a, b);
    }
    double const total = cumulativePlanckFunction_.back();
    if (!(total > 0.0) || !std::isfinite(total))
    {
        UniversalError eo("MovingSideTemperature: invalid Planck CDF");
        eo.addEntry("temperature", temperature_);
        eo.addEntry("total", total);
        throw eo;
    }
    for (double &x : cumulativePlanckFunction_)
        x /= total;
    cumulativePlanckFunction_.back() = 1.0;
}

template<typename T, typename Grid>
void MovingSideTemperature<T, Grid>::updateLeftFaceVelocityFromBox(double fullDt)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    double currentLeftX = ll.x;
    if (fullDt > 0.0)
        leftFaceVelocity_ = T((currentLeftX - prevLeftX_) / fullDt, 0, 0);
    prevLeftX_ = currentLeftX;
}

template<typename T, typename Grid>
MonteCarloParticleStatus MovingSideTemperature<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    MonteCarloParticleStatus result = MonteCarloParticleStatus::DONE;
    for (const typename Grid::Face_T &face : faces)
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        T normal = CrossProduct(u, v);
        double absU = abs(u);
        if (std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
        {
            normal /= abs(normal);
            if (std::abs(normal.x) > 0.99)
            {
                if (std::abs(particle.location.x - ll.x) < std::abs(ur.x - particle.location.x))
                    return MonteCarloParticleStatus::REMOVE;
            }
            const double unsignedDistance = std::abs(ScalarProd(particle.location - onFace, normal));
            particle.location -= 2 * unsignedDistance * normal;
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
            const T &center = this->grid.GetMeshPoint(particle.cellIndex);
            constexpr double nudge = 1e-6;
            particle.location = particle.location * (1 - nudge) + nudge * center;
            result = MonteCarloParticleStatus::REFLECT;
        }
    }
    if(result == MonteCarloParticleStatus::REFLECT)
        return result;
    std::cerr << "MovingSideTemperature: particle is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>>
MovingSideTemperature<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    updateLeftFaceVelocityFromBox(fullDt);

    double const T4 = boost::math::pow<4>(temperature_);
    std::uniform_real_distribution<double> unif(0, 1);
    static std::mt19937_64 re([](){
        int seed = 0;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &seed);
#endif
        return static_cast<std::mt19937_64::result_type>(seed);
    }());

    std::vector<MonteCarloParticle<T, Grid>> newParticles;
    size_t const N = this->grid.GetPointNo();

    for (size_t i = 0; i < N; ++i)
    {
        const T &point = this->grid.GetMeshPoint(i);
        for (const size_t &faceIdx : this->grid.GetCellFaces(i))
        {
            const std::pair<size_t, size_t> &neighbors = this->grid.GetFaceNeighbors(faceIdx);
            size_t neighborIdx = (neighbors.first == i) ? neighbors.second : neighbors.first;
            if (neighborIdx >= N && this->grid.IsPointOutsideBox(neighborIdx))
            {
                T nOut = normalize(this->grid.GetMeshPoint(neighborIdx) - point);
                if (nOut.x < -0.99)
                {
                    std::cout<<"Left velocity: "<<leftFaceVelocity_<<", nOut: "<<nOut<<std::endl;
                    double const area = this->grid.GetArea(faceIdx);

                    double const v2 = ScalarProd(leftFaceVelocity_, leftFaceVelocity_);
                
                    double const gammaFace = 1.0 / std::sqrt(1.0 - v2 / (units::clight * units::clight));
                
                    // T_bath is defined in the comoving/fluid frame.
                    // This is therefore the proper-frame emitted energy per packet.
                    double const dtFace = fullDt / gammaFace;
                    // double const w_n = ScalarProd(leftFaceVelocity_, nOut);
                    // double const sweptFactor = (w_n > 0.0) ? 1.0 + 4.0 * w_n / units::clight : 1.0;
                    double const packetEnergyFace = units::sigma_sb * T4 * area * dtFace / Npercell_;
                    double const fluidEnergy = packetEnergyFace * Npercell_ * gammaFace;

                    T e1(0, 1, 0);
                    T e2(0, 0, 1);
                    double totalWeight = 0.0;
                    for (size_t j = 0; j < Npercell_; ++j)
                    {
                        MonteCarloParticle<T, Grid> p;
                        p.location = RandomPointOnFace(this->grid, faceIdx);
                        p.weight = packetEnergyFace;
                        p.timeLeft = fullDt * unif(re);
                        p.cellIndex = i;
                        p.frequency = 0;
                        if (multigroup_)
                            p.frequency = LinearInterpolation(cumulativePlanckFunction_,
                    ComputationalCell3D::energyBoundaries, unif(re));
                    
                    do {
                        double const mu = std::sqrt(unif(re));
                        double const sinTheta = std::sqrt(1.0 - mu * mu);
                        double const phi = 2.0 * M_PI * unif(re);
                        
                        T dirFace = (-mu) * nOut
                        + sinTheta * std::cos(phi) * e1
                        + sinTheta * std::sin(phi) * e2;
                        dirFace = normalize(dirFace);
                        
                        p.velocity = units::clight * dirFace;
                        LorentzTransformation(p, -1.0 * leftFaceVelocity_);
                        p.initialWeight = p.weight;
                        } while (p.velocity.x < 0);
                        totalWeight += p.weight;
                        if(j == 0)
                            std::cout<<"Fluid energy: "<<packetEnergyFace<<" end weight: "<<p.weight<<std::endl;
                        newParticles.push_back(p);
                    }
                    std::cout<<"Total weight: "<<totalWeight<<", fluidEnergy: "<<fluidEnergy<<" expected lab weight: "<<fluidEnergy * (1 + 2 * leftFaceVelocity_.x / (3 * units::clight)) <<std::endl;
                }
            }
        }
    }
    return newParticles;
}

#endif // MOVING_SIDE_TEMPERATURE_HPP
