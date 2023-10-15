#ifndef GRAVITY_ACC_3D_HPP
#define GRAVITY_ACC_3D_HPP

#include "3D/gravity/GravityAgent.hpp"
#include "newtonian/three_dimensional/ConservativeForce3D.hpp"

class GravityAcceleration3D : public Acceleration3D
{
public:
    GravityAcceleration3D(double theta, bool quadrupole = false, double G = 1): theta(theta), quadrupole(quadrupole), G(G){};

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
    {
        std::vector<Vector3D> points = tess.getMeshPoints();
        points.resize(tess.GetPointNo());
        std::vector<gravity_result_t> masses;
        masses.reserve(points.size());
        for(size_t cellIdx = 0; cellIdx < points.size(); cellIdx++)
        {
            masses.push_back((cells[cellIdx].density) * (tess.GetVolume(cellIdx)));
        }

        std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
        GravityAgent agent(points, masses, boundaries.first, boundaries.second, this->theta, this->quadrupole);
        acc = std::move(agent.getForces(points, masses));
        size_t const N = tess.GetPointNo();
        for(size_t i = 0; i < N; ++i)
            acc[i] *= this->G;
    }

private:
    double theta, G;
    bool quadrupole;
};

#endif // GRAVITY_ACC_3D_HPP