#include "MonteCarloPhysics3D.hpp"

MonteCarloRadiationPhysics3D::MonteCarloRadiationPhysics3D(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<RadiationOpacity> opacity)
    : MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary), cells(cells), conserved(conserved), eos(std::move(eos)), opacity(std::move(opacity))
{   
    this->dist = std::uniform_real_distribution<double>(std::numeric_limits<double>::epsilon(), 1 - std::numeric_limits<double>::epsilon());
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI
    this->re = std::mt19937_64(rank);
}
