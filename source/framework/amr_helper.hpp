
#ifndef AMR_HELPER_HPP
#define AMR_HELPER_HPP

#include "problem_config_3d.hpp"
#include "mesh_helper.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"
#include "newtonian/three_dimensional/AMR3D.hpp"
#include "newtonian/three_dimensional/SpatialReconstruction3D.hpp"
#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace rich3d {

using ::ComputationalCell3D;
using ::Tessellation3D;
using ::Vector3D;
using std::pair;
using std::vector;

using RefinementResult = pair<vector<size_t>, vector<Vector3D>>;
using RemovalResult = pair<vector<size_t>, vector<double>>;

class NeverRefine : public ::CellsToRefine3D {
public:
    RefinementResult ToRefine(Tessellation3D const&, vector<ComputationalCell3D> const&, double) const override {
        return {{}, {}};
    }
};

class OutOfDomainRemover3D : public ::CellsToRemove3D {
private:
    Vector3D lower_bound_;
    Vector3D upper_bound_;
    double margin_;

public:
    OutOfDomainRemover3D(const Vector3D& lower, const Vector3D& upper, double margin = 0.0)
        : lower_bound_(lower), upper_bound_(upper), margin_(margin) {}

    RemovalResult ToRemove(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
                           double time) const override {
        vector<size_t> to_remove;
        vector<double> merits;

        for (size_t i = 0; i < tess.GetPointNo(); ++i) {
            const Vector3D& pos = tess.GetMeshPoint(i);

            // Check if point is outside domain (with margin)
            if (!is_point_in_domain(pos, lower_bound_, upper_bound_, margin_)) {
                to_remove.push_back(i);
                merits.push_back(1.0); // TODO: based on distance?
            }
        }

        return {to_remove, merits};
    }
};

struct AMRComponents {
    // Out-of-domain removal components
    std::unique_ptr<OutOfDomainRemover3D> out_of_domain_remover;
    std::unique_ptr<::CellsToRefine3D> no_refine;
    std::unique_ptr<::AMR3D> out_of_domain_amr;

    // User AMR components
    std::unique_ptr<::AMR3D> amr_ptr;
};

inline bool has_free_flow_boundary(const BoundaryConfig& boundary) {
    using Type = BoundaryConfig::Type;
    return boundary.x_lower == Type::FREE_FLOW || boundary.x_upper == Type::FREE_FLOW ||
           boundary.y_lower == Type::FREE_FLOW || boundary.y_upper == Type::FREE_FLOW ||
           boundary.z_lower == Type::FREE_FLOW || boundary.z_upper == Type::FREE_FLOW;
}

inline AMRComponents setup_amr(const Problem3DConfig& config, ::SpatialReconstruction3D* reconstruction_ptr) {
    AMRComponents amr_comp;

    // Out of domain cell removal
    bool has_free_flow = has_free_flow_boundary(config.boundary);

    if (has_free_flow && !config.domain.dynamic.enabled) {
        // Need to remove cells that escape the domain
        amr_comp.out_of_domain_remover =
            std::make_unique<OutOfDomainRemover3D>(config.domain.lower_bound, config.domain.upper_bound, 0.1);

        // Create a dummy refiner that never refines
        amr_comp.no_refine = std::make_unique<NeverRefine>();

        amr_comp.out_of_domain_amr = std::make_unique<::AMR3D>(*config.physics.eos, *amr_comp.no_refine,
                                                               *amr_comp.out_of_domain_remover, *reconstruction_ptr);

        std::cout << "  Out-of-domain removal: Enabled (FREE_FLOW boundaries detected)\n";
    }

    // User AMR setup
    if (config.amr.enabled) {
        if (!config.amr.custom_refine || !config.amr.custom_remove) {
            throw std::runtime_error("AMR enabled but custom refinement/removal criteria not provided");
        }

        amr_comp.amr_ptr = std::make_unique<::AMR3D>(*config.physics.eos, *config.amr.custom_refine,
                                                     *config.amr.custom_remove, *reconstruction_ptr);
    }

    return amr_comp;
}

} // namespace rich3d

#endif // AMR_HELPER_HPP
