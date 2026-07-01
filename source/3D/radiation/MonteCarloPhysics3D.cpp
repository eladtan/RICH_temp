#include "MonteCarloPhysics3D.hpp"

MonteCarloRadiationPhysics3D::MonteCarloRadiationPhysics3D(Tessellation3D &grid, const std::shared_ptr<BoundaryCond> &boundary, std::vector<ComputationalCell3D> &cells, std::vector<Conserved3D> &conserved, std::shared_ptr<EquationOfState> eos, std::shared_ptr<OpacityCalculator> opacity)
    : STORM::MonteCarloPhysics<Vector3D, Tessellation3D>(grid, boundary), cells(cells), conserved(conserved), eos(std::move(eos)), opacity(std::move(opacity))
{   
    this->dist = std::uniform_real_distribution<double>(std::numeric_limits<double>::epsilon(), 1 - std::numeric_limits<double>::epsilon());
    int rank = 0;
    #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI
    uint64_t baseSeed = static_cast<uint64_t>(rank) * 3;
    this->re = std::mt19937_64(baseSeed);
    this->opacity->rng_.seed(baseSeed + 1);
    ReseedRandomInCell(baseSeed + 2);
    size_t N = this->grid.GetPointNo();
    this->Erad_time_avg.resize(N, 0);
    for(size_t i = 0; i < N; i++)
    {
        this->Erad_time_avg[i] = cells[i].Erad;
    }
}
