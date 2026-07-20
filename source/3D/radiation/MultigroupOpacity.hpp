#ifndef MULTIGROUP_OPACITY_HPP
#define MULTIGROUP_OPACITY_HPP

#include "RadiationOpacity.hpp"
#include <memory>
#include "CMMC/src/units/units.hpp"
#include "CMMC/src/planck_integral/planck_integral.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "misc/utils.hpp"

class MultigroupOpacity
{
public:
    MultigroupOpacity(std::shared_ptr<OpacityCalculator> opacity);
    
    void GetCummulativeOpacity(const ComputationalCell3D &cell);

    double GetThermalEnergy(const ComputationalCell3D &cell, double random);

    inline const std::array<double, ENERGY_GROUPS_NUM> &getEnergyCenters(void) const {return this->energyCenters;}

    inline const std::array<double, ENERGY_GROUPS_NUM + 1> &getCummulativeOpacity(void) const {return this->cummulativeOpacity;}

    inline void ResetCummulativeOpacityCellID(void) {this->cummulativeOpacityCellID = std::numeric_limits<size_t>::max();}

    // Physical thermal emission probability per energy group for this cell.
    // Zero-width or zero-source groups return zero probability.
    std::array<double, ENERGY_GROUPS_NUM> GetThermalGroupPdf(const ComputationalCell3D &cell);

    // Sample thermal energy within one physical group. random is clamped to [0, 1).
    // Empty groups return the precomputed group energy center.
    double SampleThermalEnergyInGroup(const ComputationalCell3D &cell, size_t group, double random);

private:
    std::array<double, ENERGY_GROUPS_NUM> energyCenters;
    std::array<double, ENERGY_GROUPS_NUM + 1> cummulativeOpacity;
    size_t cummulativeOpacityCellID;
    std::shared_ptr<OpacityCalculator> opacity;
};

#endif // MULTIGROUP_OPACITY_HPP
