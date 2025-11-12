
#ifndef NUMERICAL_SCHEME_HELPER_HPP
#define NUMERICAL_SCHEME_HELPER_HPP

#include "problem_config_3d.hpp"
#include "newtonian/common/equation_of_state.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "newtonian/three_dimensional/Lagrangian3D.hpp"
#include "newtonian/three_dimensional/eulerian_3d.hpp"
#include "newtonian/three_dimensional/RoundCells3D.hpp"
#include "newtonian/three_dimensional/Hllc3D.hpp"
#include "newtonian/three_dimensional/SpatialReconstruction3D.hpp"
#include "newtonian/three_dimensional/LinearGauss3D.hpp"
#include "newtonian/three_dimensional/PCM3D.hpp"
#include "newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "newtonian/three_dimensional/default_cell_updater.hpp"
#include "newtonian/three_dimensional/default_extensive_updater.hpp"
#include "newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include <vector>
#include <memory>
#include <stdexcept>

namespace rich3d {

using std::pair;
using std::vector;

using FluxConditionAction = pair<const ::ConditionActionFlux1::Condition3D*, const ::ConditionActionFlux1::Action3D*>;

struct NumericalSchemeComponents {
    // Reconstruction schemes
    ::LinearGauss3D reconstruction_lg;
    ::PCM3D reconstruction_pcm;
    ::SpatialReconstruction3D* reconstruction_ptr;

    // Riemann solver
    ::Hllc3D riemann_hllc;

    // Flux calculation
    ::IsBulkFace3D isbulk;
    ::IsBoundaryFace3D isboundary;
    ::RegularFlux3D normal_flux;
    ::FreeFlowFlux3D boundary_flux;
    std::unique_ptr<::ConditionActionFlux1> flux_calc;

    // Updaters
    ::ConditionExtensiveUpdater3D extensive_updater;
    ::DefaultCellUpdater cell_updater;

    // Timestep
    ::CourantFriedrichsLewy timestep;

    // Point motion
    ::Lagrangian3D motion_lagrangian;
    ::Eulerian3D motion_eulerian;
    ::RoundCells3D motion_round;
    ::PointMotion3D* point_motion_ptr;

    NumericalSchemeComponents(const NumericalConfig& config, const ::EquationOfState& eos, ::Ghost3D& ghost,
                              ::SourceTerm3D& force)
        : reconstruction_lg(eos, ghost, true, 0.2, 0.25, 0.75), reconstruction_pcm(ghost), reconstruction_ptr(nullptr),
          normal_flux(riemann_hllc), boundary_flux(riemann_hllc), extensive_updater({}),
          timestep(config.cfl, 1.0, force), motion_round(motion_lagrangian, eos), point_motion_ptr(nullptr) {
        // Set reconstruction pointer
        switch (config.reconstruction) {
            case NumericalConfig::Reconstruction::LINEAR_GAUSS:
                reconstruction_ptr = &reconstruction_lg;
                break;
            case NumericalConfig::Reconstruction::PCM:
                reconstruction_ptr = &reconstruction_pcm;
                break;
        }

        // Validate Riemann solver
        if (config.riemann_solver != NumericalConfig::RiemannSolver::HLLC) {
            throw std::runtime_error("Only HLLC Riemann solver is currently implemented");
        }

        // Initialize flux calculator
        vector<FluxConditionAction> flux_sequence = {{&isboundary, &boundary_flux}, {&isbulk, &normal_flux}};
        flux_calc = std::make_unique<::ConditionActionFlux1>(flux_sequence, *reconstruction_ptr);

        // Set point motion pointer
        switch (config.grid_motion) {
            case NumericalConfig::GridMotion::LAGRANGIAN:
                point_motion_ptr = &motion_lagrangian;
                break;
            case NumericalConfig::GridMotion::EULERIAN:
                point_motion_ptr = &motion_eulerian;
                break;
            case NumericalConfig::GridMotion::ROUND_CELLS:
                point_motion_ptr = &motion_round;
                break;
        }
    }
};

} // namespace rich3d

#endif // NUMERICAL_SCHEME_HELPER_HPP
