#ifndef HOHLRAUM_OPACITY_HPP
#define HOHLRAUM_OPACITY_HPP

#include <random>
#include "3D/radiation/RadiationOpacity.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/radiation/LorentzTransformation.hpp"

class HohlraumOpacity : public RadiationOpacity
{
public:
    HohlraumOpacity();

    inline ~HohlraumOpacity() override = default;

    double getPlanckOpacity(const ComputationalCell3D &cell) const override;

    double getScatteringOpacity(const ComputationalCell3D &cell) const override;

    Vector3D getRandomVelocity(const ComputationalCell3D &cell) const override;

    Vector3D getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const override;

private:
    mutable std::mt19937_64 rng;
};

#endif // HOHLRAUM_OPACITY_HPP
