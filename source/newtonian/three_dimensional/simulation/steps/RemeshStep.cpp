#include "RemeshStep.hpp"

RemeshStep::RemeshStep(Tessellation3D &tess,
                       std::vector<ComputationalCell3D> &cells,
                       std::vector<Conserved3D> &extensives,
                       PointGenerator generator
                       #ifdef RICH_MPI
                       , std::shared_ptr<CostCalculator3D> cost
                       #endif // RICH_MPI
                       )
    : tess(tess), cells(cells), extensives(extensives), generator(std::move(generator))
    #ifdef RICH_MPI
    , exchangeChain(MPI_COMM_WORLD), cost(cost)
    #endif // RICH_MPI
{}

void RemeshStep::step(double /*dt*/)
{
    std::vector<Vector3D> newPoints = this->generator(this->tess, 0);

#ifdef RICH_MPI
    this->exchangeChain.Reset(this->tess.GetPointNo());
    this->tess.BuildParallel(newPoints);
    this->exchangeChain.Exchange(this->tess.GetSentProcs(),
                                 this->tess.GetSentPoints(),
                                 this->tess.GetSelfIndex());

    MPI_exchange_data(this->tess, this->extensives, false);
    MPI_exchange_data(this->tess, this->cells, false);
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#else
    this->tess.Build(newPoints);
#endif // RICH_MPI

    if (this->postRebuild)
        this->postRebuild();
}

double RemeshStep::suggestTimeStep(void) const
{
    return 1e200;
}

#ifdef RICH_MPI
    bool RemeshStep::allowRebalance(void)
    {
        return true;
    }

    std::string RemeshStep::getRequiredLB(void) const
    {
        return "remesh";
    }

    std::vector<double> RemeshStep::getLoadBalanceWeights(void)
    {
        if (this->cost)
            return this->cost->CalculateCost(this->tess, this->cells);
        return std::vector<double>(this->tess.GetPointNo(), 1.0);
    }

    ExchangeChain RemeshStep::GetExchangeChain(void)
    {
        return this->exchangeChain;
    }
#endif // RICH_MPI
