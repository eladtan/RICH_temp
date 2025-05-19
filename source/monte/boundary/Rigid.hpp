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
    bool verbose = false; // change when necessary
    MonteCarloParticleStatus status;
    const std::vector<typename Grid::Face_T> &faces = this->grid.GetBoxFaces();
    if(verbose) std::cout << "Particle " << particle << " is on the boundary" << std::endl;
    for(const typename Grid::Face_T &face : faces)
    {
        const T &onFace = face.vertices[0];
        T u = face.vertices[1] - face.vertices[0];
        T v = face.vertices[2] - face.vertices[0];
        T normal = CrossProduct(u, v);
        double absU = abs(u);
        if(verbose) std::cout << "Face normal is " << normal << ", point on face: " << onFace << ", scalar prod " << ScalarProd(normal, particle.location - onFace) << ", absU = " << absU << std::endl;
        if(std::fabs(ScalarProd(normal, particle.location - onFace)) < EPSILON * absU * absU * absU)
        {
            if(verbose) std::cout << "Inside" << std::endl;
            // Reflect the particle
            normal /= abs(normal);
            particle.velocity -= 2 * ScalarProd(particle.velocity, normal) * normal;
            status = MonteCarloParticleStatus::REFLECT;
            if(verbose) std::cout << "Setting particle velocity to " << particle.velocity << std::endl;
            return status;
        }
    }

    // should not reach here
    std::cerr << "Particle " << particle << " is not on any boundary" << std::endl;
    exit(1);
    // UniversalError eo("Particle is not on any boundary");
    // eo.addEntry("Particle", particle);
    // eo.addEntry("Grid Number of Faces", faces.size());
    // for(size_t i = 0; i < faces.size(); i++)
    // {
    //     const typename Grid::Face &face = faces[i];
    //     const T &onFace = face.vertices[0];
    //     T u = face.vertices[1] - face.vertices[0];
    //     T v = face.vertices[2] - face.vertices[0];
    //     T normal = CrossProduct(u, v);        
    //     double absU = abs(u);
    //     eo.addEntry("Face " + std::to_string(i) + " Spanning Vector 1", u);
    //     eo.addEntry("Face " + std::to_string(i) + " Spanning Vector 2", v);
    //     eo.addEntry("Face " + std::to_string(i) + " Point on Face", onFace);
    //     eo.addEntry("Face " + std::to_string(i) + " Normal", normal);
    //     eo.addEntry("Face " + std::to_string(i) + " absU", absU);
    //     eo.addEntry("Face " + std::to_string(i) + " Scalar Product with Particle Location (abs)", std::fabs(ScalarProd(normal, particle.location - onFace)));
    // }
    // throw eo;
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> RigidBoundaryCondition<T, Grid>::generateNewBoundaryParticles(double fullDt)
{
    return {};
}

#endif // RIGID_BOUNDARY_CONDITION_HPP