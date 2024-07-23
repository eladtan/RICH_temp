#ifndef TIME_ADVANCE2_HPP
#define TIME_ADVANCE2_HPP 1

#include "HydroTimeAdvance.hpp"
#include "../TimeStepUtils.hpp"

class TimeAdvance2 : public HydroTimeAdvance
{
public:
    TimeAdvance2(Tessellation3D& tess, std::vector<ComputationalCell3D> &cells, vector<Conserved3D> &extensive, const EquationOfState& eos,
                        const PointMotion3D& pm, const FluxCalculator3D& fc, const CellUpdater3D& cu,
                        const ExtensiveUpdater3D& eu, const SourceTerm3D& source);

    void beforeAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt) override;

    void afterAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt) override;

private:
    std::vector<Conserved3D> fluxes;
    std::vector<Conserved3D> mid_extensives;
    std::vector<ComputationalCell3D> activeCells;
    std::vector<Conserved3D> activeExtensive;
    std::vector<Conserved3D> activeMidExtensive;
    std::vector<Vector3D> point_vel;
    std::vector<Vector3D> activePointVel;
    std::vector<Vector3D> activeFaceVelocities;

    void UpdateVelocities(const std::vector<ComputationalCell3D> &activeCells, const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt);
};

#endif // TIME_ADVANCE2_HPP