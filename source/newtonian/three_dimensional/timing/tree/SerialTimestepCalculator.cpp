#include "SerialTimestepCalculator.hpp"

SerialTimestepCalculator::SerialTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities):
        tess(tess), cells(cells), faceVelocities(faceVelocities), createdTimingTree(true)
{
    TimingTree<Vector3D> *timingTree = new TimingTree<Vector3D>(ll, ur);
    timingTree->build(tess, cells, faceVelocities);
    this->timingTree = timingTree;
}

SerialTimestepCalculator::SerialTimestepCalculator(const Vector3D &ll, const Vector3D &ur, const Tessellation3D &tess, std::vector<ComputationalCell3D> &cells, const std::vector<Vector3D> &faceVelocities, TimingTree<Vector3D> *timingTree):
    tess(tess), cells(cells), faceVelocities(faceVelocities), createdTimingTree(false), timingTree(timingTree)
{}
   
SerialTimestepCalculator::~SerialTimestepCalculator()
{
    if(this->createdTimingTree)
    {
        delete this->timingTree;
    }
}

std::vector<dt_t> SerialTimestepCalculator::calculateTimesForIndices(const std::vector<size_t> &cellsIndices) const
{
    // check all the cells are mine
    for(const size_t &cellIndex : cellsIndices)
    {
        Vector3D point = this->tess.GetMeshPoint(cellIndex);
        if(not this->timingTree->getOctTree()->find(point))
        {
            UniversalError eo("SerialTimestepCalculator: A point does not appear in the tessellation");
            eo.addEntry("Index", cellIndex);
            eo.addEntry("Point", point);
            throw eo;
        }
    }
    
    // calculate the results
    std::vector<dt_t> times;
    for(const size_t &cellIndex : cellsIndices)
    {
        dt_t cellTime = this->timingTree->time(cellIndex);
        this->cells[cellIndex].dt = cellTime;
        times.push_back(cellTime);
    }
    return times;
}