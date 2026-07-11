#ifndef FAST_MULTIPOLE_ACCELERATION_3D_HPP
#define FAST_MULTIPOLE_ACCELERATION_3D_HPP

#include <vector>

#include "3D/gravity/fmm/SerialFmmGravityCalculator.hpp"
#include "newtonian/three_dimensional/ConservativeForce3D.hpp"

class FastMultipoleAcceleration3D : public Acceleration3D
{
public:
    FastMultipoleAcceleration3D(FmmGravityOptions options = FmmGravityOptions(), double G = 1.0);

    void operator()(const Tessellation3D& tess,
                    const vector<ComputationalCell3D>& cells,
                    const vector<Conserved3D>& fluxes,
                    const double time,
                    vector<Vector3D>& acc) const override;

    const FmmSolveStats& getLastStats() const noexcept;

private:
    double G_;
    mutable SerialFmmGravityCalculator calculator_;
    mutable std::vector<Vector3D> points_;
    mutable std::vector<double> masses_;
};

#endif // FAST_MULTIPOLE_ACCELERATION_3D_HPP
