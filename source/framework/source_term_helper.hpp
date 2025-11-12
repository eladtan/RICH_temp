
#ifndef SOURCE_TERM_HELPER_HPP
#define SOURCE_TERM_HELPER_HPP

#include "problem_config_3d.hpp"
#include "newtonian/three_dimensional/SourceTerm3D.hpp"
#include "newtonian/three_dimensional/SeveralSources3D.hpp"
#include "newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "Radiation/Diffusion.hpp"
#include "Radiation/DiffusionForce.hpp"
#include <vector>
#include <memory>
#include <stdexcept>
#include <iostream>

namespace rich3d {

using std::vector;

struct SourceTermComponents {
    vector<std::shared_ptr<::SourceTerm3D>> all_sources;
    std::unique_ptr<::DiffusionOpenBoundary> diffusion_boundary;
    std::unique_ptr<::Diffusion> diffusion_solver;
    std::shared_ptr<::DiffusionForce> radiation_force;
    ::ZeroForce3D zero_force;
    ::SeveralSources3D combined_sources;
    ::SourceTerm3D* force_ptr;

    SourceTermComponents(vector<std::shared_ptr<::SourceTerm3D>> sources)
        : all_sources(std::move(sources)), combined_sources(all_sources), force_ptr(nullptr) {}
};

inline SourceTermComponents setup_source_terms(const Problem3DConfig& config) {
    vector<std::shared_ptr<::SourceTerm3D>> all_sources;

    // Add accelerations as conservative forces
    for (const auto& acc : config.sources.accelerations) {
        all_sources.push_back(std::make_shared<::ConservativeForce3D>(*acc, config.sources.conservative_mass_flux));
    }

    // Add custom SourceTerm3D objects directly
    for (const auto& src : config.sources.source_terms) {
        all_sources.push_back(src);
    }

    SourceTermComponents comp(std::move(all_sources));

    // Radiation diffusion
    if (config.radiation.enabled) {
        if (!config.radiation.opacity) {
            throw std::runtime_error("Radiation enabled but no opacity provided");
        }

        auto opacity_calc = dynamic_cast<::DiffusionCoefficientCalculator*>(config.radiation.opacity.get());
        if (!opacity_calc) {
            throw std::runtime_error("Opacity must implement DiffusionCoefficientCalculator interface");
        }

        comp.diffusion_boundary = std::make_unique<::DiffusionOpenBoundary>();
        comp.diffusion_solver = std::make_unique<::Diffusion>(
            *opacity_calc, *comp.diffusion_boundary, *config.physics.eos, std::vector<std::string>(),
            config.radiation.flux_limiter_on, true, config.radiation.compton_cooling);

        comp.radiation_force = std::make_shared<::DiffusionForce>(*comp.diffusion_solver, *config.physics.eos,
                                                                  config.radiation.flux_limiter_on);

        comp.all_sources.push_back(comp.radiation_force);
        // Rebuild combined_sources with new sources
        comp.combined_sources = ::SeveralSources3D(comp.all_sources);
    }

    // Set force pointer
    if (comp.all_sources.empty()) {
        comp.force_ptr = &comp.zero_force;
    } else {
        comp.force_ptr = &comp.combined_sources;
    }

    return comp;
}

} // namespace rich3d

#endif // SOURCE_TERM_HELPER_HPP
