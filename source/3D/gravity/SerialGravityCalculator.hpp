#ifndef SERIAL_GRAVITY_CALCULATOR_HPP
#define SERIAL_GRAVITY_CALCULATOR_HPP

#include <vector>
#include "3D/tessellation/Tessellation3D.hpp"
#include "GravityTree.hpp"

class SerialGravityCalculator
{
public:
    SerialGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_ = false);

    std::vector<Vector3D> getAcceleration(const std::vector<Vector3D> &points) const;

    inline ~SerialGravityCalculator()
    {
        delete this->gravityTree;
    }

private:
    const Tessellation3D &tess;
    double theta;
    double thetaSquared;
    bool quadrupole;
    GravityTree<Vector3D> *gravityTree;
};

SerialGravityCalculator::SerialGravityCalculator(const Tessellation3D &tess_, const std::vector<gravity_result_t> &masses_, double theta_, bool quadrupole_):
    tess(tess_), theta(theta_), thetaSquared(theta_ * theta_), quadrupole(quadrupole_)
{
    auto [ll, ur] = this->tess.GetBoxCoordinates();
    GravityTree<Vector3D> *gravTree = new GravityTree<Vector3D>(ll, ur, this->theta, this->quadrupole);
    std::vector<MassedPoint<Vector3D>> massedPoints;
    size_t N = this->tess.GetPointNo();
    massedPoints.reserve(N);
    for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
    {
        massedPoints.emplace_back(MassedPoint<Vector3D>(this->tess.GetCellCM(pointIdx), masses_[pointIdx]));
    }
    gravTree->build(massedPoints);
    this->gravityTree = gravTree;
}

std::vector<Vector3D> SerialGravityCalculator::getAcceleration(const std::vector<Vector3D> &points) const
{
    // calculate the results, locally
    std::vector<Vector3D> results;
    for(const Vector3D &point : points)
    {
        results.emplace_back(this->gravityTree->gravity(point));
    }
    return results;
}

#endif // SERIAL_GRAVITY_CALCULATOR_HPP