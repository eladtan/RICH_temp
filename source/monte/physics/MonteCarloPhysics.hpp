#ifndef MONTE_CARLO_PHYSICS_HPP
#define MONTE_CARLO_PHYSICS_HPP

#include <tuple>
#include "monte/MonteCarloParticle.hpp"
#include "monte/MonteCarloFunctionality.hpp"

template<typename T, typename Grid>
class MonteCarloPhysics
{
public:
    using MCParticle = MonteCarloParticle<T, Grid>;

    MonteCarloPhysics(const Grid &grid);

    virtual ~MonteCarloPhysics() = default;

    void updateGridData(void);

    virtual void preStep(double fullDt) = 0;

    virtual MonteCarloFunctionality<T, Grid> step(MCParticle &particle) = 0;

    virtual void postStep(const std::vector<MCParticle> &particles) = 0;

protected:
    const Grid &grid;
    
    std::tuple<size_t, dt_t, size_t> getIntersectionDetails(MCParticle &particle);

    struct
    {
        std::vector<std::vector<T>> normalsOfCells;
        std::vector<std::vector<T>> pointsOnFaces;
    } gridData;
};

template<typename T, typename Grid>
MonteCarloPhysics<T, Grid>::MonteCarloPhysics(const Grid &grid)
    : grid(grid)
{}

template<typename T, typename Grid>
void MonteCarloPhysics<T, Grid>::updateGridData(void)
{
    size_t Ncells = this->grid.GetPointNo();

    this->gridData.normalsOfCells = std::vector<std::vector<T>>(Ncells);
    this->gridData.pointsOnFaces = std::vector<std::vector<T>>(Ncells);

    for(size_t i = 0; i < Ncells; i++)
    {
        std::vector<T> &normals = this->gridData.normalsOfCells[i];
        std::vector<T> &onFaces = this->gridData.pointsOnFaces[i];
        const face_vec &faces = this->grid.GetCellFaces(i);
        for(const size_t &faceIdx : faces)
        {
            normals.push_back(this->grid.Normal(faceIdx));
            onFaces.push_back(this->grid.FaceCM(faceIdx));
        }
    }
}

template<typename T, typename Grid>
inline std::tuple<size_t, dt_t, size_t> MonteCarloPhysics<T, Grid>::getIntersectionDetails(MCParticle &particle)
{
    size_t cellIndex = particle.cellIndex;
    const std::vector<Vector3D> &normalsOfFaces = this->gridData.normalsOfCells[cellIndex];
    const std::vector<Vector3D> &pointsOnFaces = this->gridData.pointsOnFaces[cellIndex];
    auto [faceIntersect, timeIntersect] = particle.distanceToNearestFace(this->grid, normalsOfFaces, pointsOnFaces);
    // std::cout << "faceIntersect is " << faceIntersect << " and timeIntersect is " << timeIntersect << std::endl;
    assert(faceIntersect < this->grid.GetTotalFacesNumber());
    assert(timeIntersect >= 0);
    const std::pair<size_t, size_t> &cellNeighbors = this->grid.GetFaceNeighbors(faceIntersect);
    assert(particle.cellIndex == cellNeighbors.first or particle.cellIndex == cellNeighbors.second);
    size_t nextCellIndex = (cellNeighbors.first == particle.cellIndex)? cellNeighbors.second : cellNeighbors.first;
    // std::cout << "nextCellIndex is " << nextCellIndex << std::endl;
    return std::make_tuple(faceIntersect, timeIntersect, nextCellIndex);
}

#endif // MONTE_CARLO_PHYSICS_HPP