#ifndef HYDRO_TIME_ADVANCE_HPP
#define HYDRO_TIME_ADVANCE_HPP

#include "newtonian/three_dimensional/computational_cell.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "newtonian/three_dimensional/point_motion_3d.hpp"
#include "newtonian/three_dimensional/time_step_function3D.hpp"
#include "newtonian/three_dimensional/flux_calculator_3d.hpp"
#include "newtonian/three_dimensional/cell_updater_3d.hpp"
#include "newtonian/three_dimensional/extensive_updater3d.hpp"
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "Radiation/conj_grad_solve.hpp"
#include "mpi/mpi_commands.hpp"
#include "../../timesteps.h"
#include "HydroAdvanceUtils.hpp"

class HydroTimeAdvance
{
public:
    HydroTimeAdvance(Tessellation3D& tess, std::vector<ComputationalCell3D> &cells, vector<Conserved3D> &extensive, const EquationOfState& eos,
                        const PointMotion3D& pm, const FluxCalculator3D& fc, const CellUpdater3D& cu,
                        const ExtensiveUpdater3D& eu, const SourceTerm3D& source):
                        tess(tess), cells(cells), extensive(extensive), eos(eos), pm(pm), fc(fc), cu(cu), eu(eu), source(source)
    {}

    virtual ~HydroTimeAdvance() = default;

    virtual void beforeAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt) = 0;

    virtual void afterAdvance(const std::vector<size_t> &participatingIndices, dt_t currentTime, dt_t dt) = 0;

    inline Tessellation3D &getTessellation(){return this->tess;}

    inline std::vector<ComputationalCell3D> &getCells(){return this->cells;}

    inline std::vector<Conserved3D> &getExtensive(){return this->extensive;}

    inline const EquationOfState &getEOS() const{return this->eos;}

    inline const SourceTerm3D &getSourceTerm() const{return this->source;}

    inline const Tessellation3D &getTessellation() const{return this->tess;}

    inline const std::vector<ComputationalCell3D> &getCells() const{return this->cells;}

    inline const std::vector<Conserved3D> &getExtensive() const{return this->extensive;}

protected:
    Tessellation3D& tess;
    std::vector<ComputationalCell3D>& cells;
    vector<Conserved3D> &extensive;
    const EquationOfState& eos;
    const PointMotion3D& pm;
    const FluxCalculator3D& fc;
    const CellUpdater3D& cu;
    const ExtensiveUpdater3D& eu;
    const SourceTerm3D& source;
};

#endif // HYDRO_TIME_ADVANCE_HPP