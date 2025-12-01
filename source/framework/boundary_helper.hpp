#ifndef BOUNDARY_HELPER_HPP
#define BOUNDARY_HELPER_HPP

#include "problem_config_3d.hpp"
#include "newtonian/three_dimensional/Ghost3D.hpp"
#include <memory>
#include <array>

namespace rich3d {

// Import legacy RICH types
using ::FreeFlowGenerator3D;
using ::Ghost3D;
using ::RigidWallGenerator3D;
using ::SeveralGhostGenerator3D;
using ::Tessellation3D;
using ::Vector3D;

// Face selector for mixed boundaries
class FaceSelector : public SeveralGhostGenerator3D::GhostCriteria3D {
    std::array<size_t, 6> indices_;
    Vector3D lower_, upper_;

public:
    FaceSelector(const std::array<size_t, 6>& idx, const Vector3D& lo, const Vector3D& hi)
        : indices_(idx), lower_(lo), upper_(hi) {}

    size_t GhostChoose(Tessellation3D const& tess, size_t index) const override {
        const Vector3D& p = tess.GetMeshPoint(index);
        if (p.x < lower_.x) return indices_[0];
        if (p.x > upper_.x) return indices_[1];
        if (p.y < lower_.y) return indices_[2];
        if (p.y > upper_.y) return indices_[3];
        if (p.z < lower_.z) return indices_[4];
        if (p.z > upper_.z) return indices_[5];
        return indices_[0];
    }
};

// Boundary package - keeps all ghost generators alive with stable addresses
struct BoundaryPackage {
    // Value members for base ghosts - stable addresses guaranteed
    FreeFlowGenerator3D free_flow;
    RigidWallGenerator3D rigid_wall;

    // For mixed boundaries
    std::vector<std::shared_ptr<Ghost3D>> custom_ghosts;
    std::shared_ptr<FaceSelector> selector;
    std::shared_ptr<SeveralGhostGenerator3D> several;

    // Single ghost for uniform case
    std::shared_ptr<Ghost3D> ghost;

    BoundaryPackage(const std::shared_ptr<Ghost3D>& g) : ghost(g) {}

    Ghost3D* get_ghost_ptr() {
        return several ? several.get() : ghost.get();
    }
};

// Factory
inline std::unique_ptr<BoundaryPackage> create_boundary_conditions(
    const BoundaryConfig& bc, const Vector3D& lower, const Vector3D& upper) {

    // Check if all boundaries are the same type (no custom)
    bool all_same = (bc.x_lower == bc.x_upper && bc.x_lower == bc.y_lower &&
                     bc.x_lower == bc.y_upper && bc.x_lower == bc.z_lower && bc.x_lower == bc.z_upper);

    if (all_same && !bc.custom_x_lower && !bc.custom_x_upper &&
        !bc.custom_y_lower && !bc.custom_y_upper &&
        !bc.custom_z_lower && !bc.custom_z_upper) {

        // Simple uniform case - use single ghost
        std::shared_ptr<Ghost3D> ghost;
        if (bc.x_lower == BoundaryConfig::Type::RIGID_WALL) {
            ghost = std::make_shared<RigidWallGenerator3D>();
        } else {
            ghost = std::make_shared<FreeFlowGenerator3D>();
        }
        return std::make_unique<BoundaryPackage>(ghost);
    }

    // Mixed boundaries - use SeveralGhostGenerator3D
    auto pkg = std::make_unique<BoundaryPackage>(nullptr);

    // Build ghost pointer list using VALUE member addresses
    std::vector<Ghost3D*> ghost_ptrs = {&pkg->free_flow, &pkg->rigid_wall};
    std::array<size_t, 6> face_idx;

    auto add_face = [&](BoundaryConfig::Type type, const std::shared_ptr<Ghost3D>& custom) -> size_t {
        if (custom) {
            pkg->custom_ghosts.push_back(custom);
            ghost_ptrs.push_back(custom.get());
            return ghost_ptrs.size() - 1;
        }
        return (type == BoundaryConfig::Type::RIGID_WALL) ? 1 : 0;
    };

    face_idx[0] = add_face(bc.x_lower, bc.custom_x_lower);
    face_idx[1] = add_face(bc.x_upper, bc.custom_x_upper);
    face_idx[2] = add_face(bc.y_lower, bc.custom_y_lower);
    face_idx[3] = add_face(bc.y_upper, bc.custom_y_upper);
    face_idx[4] = add_face(bc.z_lower, bc.custom_z_lower);
    face_idx[5] = add_face(bc.z_upper, bc.custom_z_upper);

    pkg->selector = std::make_shared<FaceSelector>(face_idx, lower, upper);
    pkg->several = std::make_shared<SeveralGhostGenerator3D>(ghost_ptrs, *pkg->selector);

    return pkg;
}

} // namespace rich3d

#endif
