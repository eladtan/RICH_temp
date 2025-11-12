
#ifndef BOUNDARY_HELPER_HPP
#define BOUNDARY_HELPER_HPP

#include "problem_config_3d.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include <memory>
#include <map>

namespace rich3d {

// Import legacy RICH types from global namespace
using ::FreeFlowGenerator3D;
using ::Ghost3D;
using ::RigidWallGenerator3D;
using ::SeveralGhostGenerator3D;
using ::Tessellation3D;
using ::Vector3D;
using GhostBoundaryResult = pair<Ghost3D*, std::shared_ptr<void>>;

class FaceIndexChooser : public SeveralGhostGenerator3D::GhostCriteria3D {
private:
    std::array<size_t, 6> face_to_ghost_; // [x-, x+, y-, y+, z-, z+]
    Vector3D lower_, upper_;

public:
    FaceIndexChooser(const std::array<size_t, 6>& indices, const Vector3D& lower, const Vector3D& upper)
        : face_to_ghost_(indices), lower_(lower), upper_(upper) {}

    size_t GhostChoose(Tessellation3D const& tess, size_t index) const override {
        const Vector3D& p = tess.GetMeshPoint(index);
        const double eps = 1e-10;

        if (std::abs(p.x - lower_.x) < eps)
            return face_to_ghost_[0]; // x-
        if (std::abs(p.x - upper_.x) < eps)
            return face_to_ghost_[1]; // x+
        if (std::abs(p.y - lower_.y) < eps)
            return face_to_ghost_[2]; // y-
        if (std::abs(p.y - upper_.y) < eps)
            return face_to_ghost_[3]; // y+
        if (std::abs(p.z - lower_.z) < eps)
            return face_to_ghost_[4]; // z-
        if (std::abs(p.z - upper_.z) < eps)
            return face_to_ghost_[5]; // z+

        return 0; // Fallback
    }
};

struct BoundaryPackage {
    std::map<BoundaryConfig::Type, std::unique_ptr<Ghost3D>> standard_ghosts;
    std::unique_ptr<FaceIndexChooser> chooser;
    std::unique_ptr<SeveralGhostGenerator3D> several;
};

inline std::string get_boundary_type_name(BoundaryConfig::Type type) {
    switch (type) {
        case BoundaryConfig::Type::RIGID_WALL:
            return "RIGID_WALL";
        case BoundaryConfig::Type::FREE_FLOW:
            return "FREE_FLOW";
        case BoundaryConfig::Type::PERIODIC:
            return "PERIODIC";
        case BoundaryConfig::Type::CUSTOM:
            return "CUSTOM";
        default:
            return "UNKNOWN";
    }
}

inline GhostBoundaryResult create_boundary_conditions(const BoundaryConfig& bc, const Vector3D& lower,
                                                      const Vector3D& upper) {
    auto package = std::make_shared<BoundaryPackage>();

    // Standard ghosts at fixed indices: [0]=FREE_FLOW, [1]=RIGID_WALL
    package->standard_ghosts[BoundaryConfig::Type::FREE_FLOW] = std::make_unique<FreeFlowGenerator3D>();
    package->standard_ghosts[BoundaryConfig::Type::RIGID_WALL] = std::make_unique<RigidWallGenerator3D>();

    std::vector<Ghost3D*> ghost_list = {
        package->standard_ghosts[BoundaryConfig::Type::FREE_FLOW].get(), // index 0
        package->standard_ghosts[BoundaryConfig::Type::RIGID_WALL].get() // index 1
    };

    // Helper to get index for a face
    auto add_face = [&](BoundaryConfig::Type type, const std::shared_ptr<Ghost3D>& custom) -> size_t {
        if (custom) {
            ghost_list.push_back(custom.get());
            return ghost_list.size() - 1;
        }
        // PERIODIC is handled by tessellation, use FREE_FLOW as placeholder
        if (type == BoundaryConfig::Type::RIGID_WALL)
            return 1;
        return 0; // FREE_FLOW or PERIODIC -> index 0
    };

    std::array<size_t, 6> face_indices = {
        add_face(bc.x_lower, bc.custom_x_lower), add_face(bc.x_upper, bc.custom_x_upper),
        add_face(bc.y_lower, bc.custom_y_lower), add_face(bc.y_upper, bc.custom_y_upper),
        add_face(bc.z_lower, bc.custom_z_lower), add_face(bc.z_upper, bc.custom_z_upper)};

    package->chooser = std::make_unique<FaceIndexChooser>(face_indices, lower, upper);
    package->several = std::make_unique<SeveralGhostGenerator3D>(ghost_list, *package->chooser);

    return {package->several.get(), package};
}

} // namespace rich3d

#endif
