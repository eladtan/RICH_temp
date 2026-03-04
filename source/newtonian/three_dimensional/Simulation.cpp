#include "Simulation.hpp"

Simulation::Simulation(Tessellation3D &tess_, const std::vector<ComputationalCell3D> &cells_, EquationOfState &eos_) : tess(tess_), cells(cells_), extensives(cells_.size()), eos(eos_)
{
    #ifdef RICH_MPI
        this->currentLoad = nullptr;
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    #endif // RICH_MPI
}

void Simulation::addPhysics(std::shared_ptr<PhysicsStep> physicsStep)
{
    this->physics.push_back(physicsStep);
}

double Simulation::GetTime(void) const
{
    return this->tracker.getTime();
}

size_t Simulation::GetCycle(void) const
{
    return static_cast<size_t>(this->tracker.getCycle());
}

void Simulation::SetCycle(size_t cycle)
{
    this->tracker.cycle = cycle;
}

void Simulation::SetTime(double t)
{
    this->tracker.time = t;
}

void Simulation::SetTimeStep(double dt)
{
    this->tsc->SetTimeStep(dt);
}

double Simulation::GetTimeStep(void) const
{
    return this->tsc->GetTimeStep();
}

#ifdef RICH_MPI
    void Simulation::buildDataTransfer(void)
    {
        MPI_exchange_data(this->tess, this->extensives, false);
        MPI_exchange_data(this->tess, this->cells, false);
        ComputationalCell3D cdummy;
        MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transfer();
        }
    }

    void Simulation::buildDataTransfer(const ExchangeChain &chain)
    {
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transferChain(chain);
        }
    }
#endif // RICH_MPI

void Simulation::step(void)
{
    double next_time_step = std::numeric_limits<double>::max();
    // double dt = std::numeric_limits<double>::max();

    for(std::shared_ptr<PhysicsStep> physics : this->physics)
    {
        std::string name = physics->getName();
        if(this->rank == 0) std::cout << "Running physics: " << name << std::endl;

        #ifdef RICH_MPI
            std::string LB = physics->getRequiredLB();

            if(this->currentLB != LB)
            {
                if(this->rank == 0) std::cout << "Changing load balance to " << LB << " (from " << this->currentLB << ")" << std::endl;
                std::shared_ptr<LoadBalancer> load;
                auto it = this->loads.find(LB);
                if(it != this->loads.cend())
                {
                    if(this->rank == 0) std::cout << "Load balance restored" << std::endl;
                    // load found, restore
                    load = it->second;
                    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
                    this->tess.SetLoadBalancer(load);
                    std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
                    if(this->rank == 0)
                    {
                        std::cout << "Restoring lb time: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << std::endl;
                    }
                }
                else
                {
                    if(this->rank == 0) std::cout << "Load balance generated for first time" << std::endl;
                    // Tessellation3D already has a load balance, use it
                    std::vector<double> weights = physics->getLoadBalanceWeights();
                    std::vector<Vector3D> points = this->tess.getMeshPoints();
                    points.resize(this->tess.GetPointNo());
                    this->tess.BuildParallel(points, weights, true /* don't allow rebalance */);
                }
                this->buildDataTransfer();

                std::vector<Vector3D> points = this->tess.getMeshPoints();
                points.resize(this->tess.GetPointNo());

                if(this->currentLoad != nullptr)
                {
                    physics->uponLBChange();
                }
            }

            if(physics->allowRebalance())
            {
                if(this->rank == 0) std::cout << "allowRebalance=true, computing weights..." << std::endl;
                std::vector<double> weights = physics->getLoadBalanceWeights();
                if(this->rank == 0) std::cout << "Weights computed (" << weights.size() << "), checking ShouldRebalance..." << std::endl;
                bool shouldRebalance = this->tess.ShouldRebalance(weights);
                if(this->rank == 0)
                {
                    std::cout << "Should Rebalance: " << shouldRebalance << std::endl;
                }
                if(shouldRebalance)
                {
                    if(this->rank == 0) std::cout << "Doing rebalance on LB " << LB << std::endl;
                    // std::cout << "Rank " << this->rank << " weights: " << weights << std::endl;

                    physics->beforeLB();
                    if(this->rank == 0) std::cout << "beforeLB done, calling Rebalance..." << std::endl;
                    // have new build
                    this->tess.Rebalance(weights);
                    if(this->rank == 0) std::cout << "Rebalance done, calling buildDataTransfer..." << std::endl;
                    this->buildDataTransfer();
                    if(this->rank == 0) std::cout << "buildDataTransfer done, calling afterLB..." << std::endl;
                    physics->afterLB();
                    if(this->rank == 0) std::cout << "afterLB done" << std::endl;
                 }
            }

            std::shared_ptr<LoadBalancer> load = this->tess.GetLoadBalancer();
            this->loads[LB] = load;
            this->currentLoad = load;
            this->currentLB = LB;
        #endif // RICH_MPI

        // if(std::numeric_limits<double>::max())
        double dt = this->tsc->GetTimeStep();
        if(this->rank == 0) std::cout << "Running physics: " << name << "with dt " << dt << std::endl;

        physics->step(dt);

        #ifdef RICH_MPI
            this->buildDataTransfer(physics->GetExchangeChain());
        #endif // RICH_MPI

        double dt_suggest = physics->suggestTimeStep();
        next_time_step = std::min(next_time_step, dt_suggest);
        // if(this->rank == 0) std::cout << "Suggested " << next_time_step << ", dt_suggest " << dt_suggest << std::endl;

        if(this->rank == 0)
        {
            std::cout << name << " suggested " << dt_suggest << " for dt " << std::endl;
            std::cout << std::endl;
        }
    }
    

    this->tsc->SetTimeStep(next_time_step);
    // if(this->rank == 0) std::cout << "Time step will be " << this->tsc.GetTimeStep() << std::endl;

    this->tracker.updateCycle();
}
