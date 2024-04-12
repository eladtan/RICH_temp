#ifndef TIME_ADVANCE2_HPP
#define TIME_ADVANCE2_HPP 1

#include "HydroTimeAdvance.hpp"
#include "../TimeStepUtils.hpp"

class TimeAdvance2 : public HydroTimeAdvance
{
public:
    TimeAdvance2(Tessellation3D& tess, std::vector<ComputationalCell3D> &cells, vector<Conserved3D> &extensive, const EquationOfState& eos,
                        const FluxCalculator3D& fc, const CellUpdater3D& cu,
                        const ExtensiveUpdater3D& eu, const SourceTerm3D& source);

    void beforeAdvance(dt_t currentTime, dt_t dt, std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel) override;

    void afterAdvance(dt_t currentTime, dt_t dt, std::vector<Vector3D> &point_vel, std::vector<Vector3D> &face_vel) override;

private:
    std::vector<Conserved3D> fluxes;
    std::vector<Conserved3D> mid_extensives;
};

#endif // TIME_ADVANCE2_HPP