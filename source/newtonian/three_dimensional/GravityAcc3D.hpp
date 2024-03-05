#ifndef GRAVITY_ACC_3D_HPP
#define GRAVITY_ACC_3D_HPP

#ifdef RICH_MPI
    #include "3D/gravity/DistributedGravityCalculator.hpp"
#else // RICH_MPI
    #include "3D/gravity/GravityTree.hpp"
#endif // RICH_MPI

#include "newtonian/three_dimensional/ConservativeForce3D.hpp"

class GravityAcceleration3D : public Acceleration3D
{
public:
    GravityAcceleration3D(double theta, bool quadrupole = false, double G = 1): theta(theta), quadrupole(quadrupole), G(G){};

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
    {
        std::vector<Vector3D> points = tess.GetAllCM();
        points.resize(tess.GetPointNo());
        std::vector<gravity_result_t> masses;
        masses.reserve(points.size());
        for(size_t cellIdx = 0; cellIdx < points.size(); cellIdx++)
        {
            masses.push_back((cells[cellIdx].density) * (tess.GetVolume(cellIdx)));
        }
        std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();

        size_t N = tess.GetPointNo();

        #ifdef RICH_MPI
            DistributedGravityCalculator agent(tess, masses, this->theta, this->quadrupole);
            acc = agent.getAcceleration(points);
        #else // RICH_MPI
            GravityTree<Vector3D> gravTree(boundaries.first, boundaries.second, this->theta, this->quadrupole);
            std::vector<MassedPoint<Vector3D>> massedPoints;
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                massedPoints.emplace_back(MassedPoint<Vector3D>(points[pointIdx], masses[pointIdx]));
            }
            gravTree.build(massedPoints);
            
            acc.clear();
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                acc.push_back(gravTree.gravity(point));
            }
        #endif // RICH_MPI

        for(size_t i = 0; i < N; ++i)
            acc[i] *= this->G;
    }

private:
    double theta, G;
    bool quadrupole;
};

#endif // GRAVITY_ACC_3D_HPP