#ifndef RIGID_BOUNDARY_CONDITION_HPP
#define RIGID_BOUNDARY_CONDITION_HPP

#include "BoundaryCondition.hpp"
#include "misc/universal_error.hpp"

template<typename T, typename Grid>
class RigidBoundaryCondition : public BoundaryCondition<T, Grid>
{
public:
    RigidBoundaryCondition(const Grid &grid);

    ~RigidBoundaryCondition() override;

    MonteCarloParticleStatus apply(MonteCarloParticle<T, Grid> &particle) override;

    std::vector<MonteCarloParticle<T, Grid>> generateNewBoundaryParticles(double fullDt) override;

    DDMCBoundaryFaceBehavior getDDMCBoundaryFaceBehavior(
        size_t faceIdx,
        size_t insideCellIndex,
        size_t outsidePointIndex) const override
    {
        (void)faceIdx;
        (void)insideCellIndex;
        (void)outsidePointIndex;
        return DDMCBoundaryFaceBehavior::ReflectingRigid;
    }
};

template<typename T, typename Grid>
RigidBoundaryCondition<T, Grid>::RigidBoundaryCondition(const Grid &grid)
    : BoundaryCondition<T, Grid>(grid)
{
}

template<typename T, typename Grid>
RigidBoundaryCondition<T, Grid>::~RigidBoundaryCondition()
{}

template<typename T, typename Grid>
MonteCarloParticleStatus RigidBoundaryCondition<T, Grid>::apply(MonteCarloParticle<T, Grid> &particle)
{
    const auto &[ll, ur] = this->grid.GetBoxCoordinates();
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    MonteCarloParticleStatus status;
    for(const typename Grid::Face_T &face : faces)
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        T normal = CrossProduct(u, v);
        double absU = abs(u);
        if(std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
        {
            normal /= abs(normal);
            const double unsignedDistance = std::abs(ScalarProd(particle.location - onFace, normal));
            particle.location -= 2 * unsignedDistance * normal;
            T boxCenter = 0.5 * (ll + ur);
            constexpr double nudge = 1e-6;
            particle.location = particle.location * (1 - nudge) + nudge * boxCenter;
            // Box face normals point inward (see BuildBox in Voronoi3D.cpp).
            // vn > 0 means velocity is already directed inward — no reflection needed.
            double vn = ScalarProd(particle.velocity, normal);
            if (vn <= 0)
                particle.velocity -= 2 * vn * normal;
            status = MonteCarloParticleStatus::REFLECT;
        }
    }
    if(status == MonteCarloParticleStatus::REFLECT)
        return status;
    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> RigidBoundaryCondition<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    (void)fullDt;
    return {};
}

#endif // RIGID_BOUNDARY_CONDITION_HPP