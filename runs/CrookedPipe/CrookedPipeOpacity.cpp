#include "CrookedPipeOpacity.hpp"

CrookedPipeOpacity::CrookedPipeOpacity()
{
    this->rng = std::mt19937_64(0); // seed 0
}

double CrookedPipeOpacity::getPlanckOpacity(const ComputationalCell3D &cell) const
{
    if(cell.tracers[1] > 0.5)
    {
        // if tracer 1 is present, return a very high opacity
        return 0.2; // arbitrary large value
    }
    else
    {
        // otherwise, return a small opacity
        return 2000; // arbitrary small value
    }
}

double CrookedPipeOpacity::getScatteringOpacity(const ComputationalCell3D &cell) const
{
    return 0;
}

Vector3D CrookedPipeOpacity::getRandomVelocity(const ComputationalCell3D &cell) const
{
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);

    // // set rng's seed to cell.ID
    // this->rng.seed(cell.ID);

    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}

Vector3D CrookedPipeOpacity::getNewScatterVelocity(const ComputationalCell3D &cell, const MCParticle &particle) const
{
    static std::uniform_real_distribution<double> dist(-1 + EPSILON, 1 - EPSILON);

    // // set rng's seed to cell.ID
    // this->rng.seed(cell.ID);
    
    double x = dist(this->rng);
    double y = dist(this->rng);
    double z = dist(this->rng);
    Vector3D direction = normalize(Vector3D(x, y, z));
    return direction * units::clight;
}