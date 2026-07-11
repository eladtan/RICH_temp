#ifndef FAST_MULTIPOLE_ACCELERATION_3D_HPP
#define FAST_MULTIPOLE_ACCELERATION_3D_HPP

#include <cstdint>
#include <vector>

#ifdef RICH_MPI
#include "3D/gravity/fmm/mpi/DistributedFmmGravityCalculator.hpp"
#else
#include "3D/gravity/fmm/SerialFmmGravityCalculator.hpp"
#endif
#include "newtonian/three_dimensional/ConservativeForce3D.hpp"

class FastMultipoleAcceleration3D : public Acceleration3D
{
public:
    FastMultipoleAcceleration3D(FmmGravityOptions options = FmmGravityOptions(),
                                double G = 1.0);
#ifdef RICH_MPI
    FastMultipoleAcceleration3D(FmmGravityOptions options,
                                FmmDistributedOptions distributedOptions,
                                double G = 1.0);
#endif

    void operator()(const Tessellation3D& tess,
                    const vector<ComputationalCell3D>& cells,
                    const vector<Conserved3D>& fluxes,
                    const double time,
                    vector<Vector3D>& acc) const override;

    const FmmSolveStats& getLastStats() const noexcept;

private:
    double G_;
#ifdef RICH_MPI
    mutable DistributedFmmGravityCalculator calculator_;
#else
    mutable SerialFmmGravityCalculator calculator_;
#endif
    mutable std::vector<Vector3D> points_;
    mutable std::vector<double> masses_;
#ifdef RICH_MPI
    mutable std::vector<std::uint64_t> cellIds_;
#endif
};

#endif // FAST_MULTIPOLE_ACCELERATION_3D_HPP
