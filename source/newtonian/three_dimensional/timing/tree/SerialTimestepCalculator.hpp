#ifndef SERIAL_TIMESTEP_CALCULATOR_HPP
#define SERIAL_TIMESTEP_CALCULATOR_HPP

#include "TimingTree.hpp"

class SerialTimestepCalculator
{
public:
    using NodeData = TimingTree<Vector3D>::NodeData;

public:
    SerialTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities);

    SerialTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, TimingTree<Vector3D> *timingTree);
   
    ~SerialTimestepCalculator();

    inline std::vector<dt_t> calculateTimes() const
    {
        std::vector<size_t> allPoints(this->tess.GetPointNo());
        std::iota(allPoints.begin(), allPoints.end(), 0);
        return this->calculateTimesForIndices(allPoints);
    }

    std::vector<dt_t> calculateTimesForIndices(const std::vector<size_t> &cellsIndices) const;

private:
    TimingTree<Vector3D> *timingTree;
    bool createdTimingTree; // if the timing tree should be deleted at the end (avoid memory leaks)
    const Tessellation3D &tess; // if the timing tree should be deleted at the end
    const std::vector<Vector3D> &faceVelocities;
    std::vector<ComputationalCell3D> &cells;
};

#endif // SERIAL_TIMESTEP_CALCULATOR_HPP