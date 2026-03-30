#include "Simulation.hpp"
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"

Simulation::Simulation(Tessellation3D &tess_, const std::vector<ComputationalCell3D> &cells_, EquationOfState &eos_, bool new_start) : tess(tess_), cells(cells_), extensives(cells_.size()), eos(eos_), Max_ID_(0)
{
    #ifdef RICH_MPI
        this->currentLoad = nullptr;
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    #else // RICH_MPI
        this->rank = 0;
        this->size = 1;
    #endif // RICH_MPI

    if (new_start)
        initializeCellIDs();
    else
        recomputeMaxID();

#ifdef RICH_MPI
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#endif

    size_t N = this->tess.GetPointNo();
    for (size_t i = 0; i < N; ++i)
        PrimitiveToConserved(this->cells[i], this->tess.GetVolume(i), this->extensives[i]);
}

void Simulation::initializeCellIDs(void)
{
    size_t N = this->cells.size();
    size_t nstart = 0;
#ifdef RICH_MPI
    std::vector<size_t> nrecv(static_cast<size_t>(this->size), 0);
    size_t nsend = N;
    MPI_Allgather(&nsend, 1, MPI_UNSIGNED_LONG_LONG, &nrecv[0], 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
    for (int i = 0; i < this->rank; ++i)
        nstart += nrecv[static_cast<size_t>(i)];
#endif
    for (size_t i = 0; i < N; ++i)
        this->cells[i].ID = nstart + i;
    this->Max_ID_ = nstart + N - 1;
#ifdef RICH_MPI
    for (int i = this->rank + 1; i < this->size; ++i)
        this->Max_ID_ += nrecv[static_cast<size_t>(i)];
#endif
}

void Simulation::recomputeMaxID(void)
{
    size_t N = this->cells.size();
    size_t maxid = 0;
    for (size_t i = 0; i < N; ++i)
        maxid = std::max(maxid, this->cells[i].ID);
#ifdef RICH_MPI
    MPI_Allreduce(&maxid, &this->Max_ID_, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    this->Max_ID_ = maxid;
#endif
}

size_t& Simulation::GetMaxID(void)
{
    return this->Max_ID_;
}

const size_t& Simulation::GetMaxID(void) const
{
    return this->Max_ID_;
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
    return this->tracker.getCycle();
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
    MEMORY_DEBUG_PRINT("Simulation::step START cycle=" + std::to_string(this->tracker.getCycle()));
    double next_time_step = std::numeric_limits<double>::max();
    // double dt = std::numeric_limits<double>::max();
    #ifdef RICH_MPI
        if(this->rank == 0)
    #endif // RICH_MPI
    {
        std::cout << "Cycle " << this->tracker.getCycle() << " at time " << this->tracker.getTime() << std::endl;
    }

    for(std::shared_ptr<PhysicsStep> physics : this->physics)
    {
        std::string name = physics->getName();
        if(this->rank == 0) std::cout << "Running physics: " << name << std::endl;

        #ifdef RICH_MPI
            std::string LB = physics->getRequiredLB();
            bool firstTime = false;

            if(this->currentLB != LB)
            {
                if(this->rank == 0) std::cout << "Changing load balance to " << LB << " (from " << this->currentLB << ")" << std::endl;
                auto it = this->loads.find(LB);
                if(it != this->loads.cend())
                {
                    if(this->rank == 0) std::cout << "Load balance restored" << std::endl;
                    this->setCurrentLoadBalance(LB);
                }
                else
                {
                    if(this->rank == 0) std::cout << "Load balance generated for first time" << std::endl;
                    std::vector<double> weights = physics->getLoadBalanceWeights();
                    std::vector<Vector3D> points = this->tess.getMeshPoints();
                    points.resize(this->tess.GetPointNo());
                    this->tess.BuildParallel(points, weights, true);
                    firstTime = true;
                    this->buildDataTransfer();
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

        double dt = this->tsc->GetTimeStep();
        if(this->rank == 0) std::cout << "Running " << name << " with dt " << dt << std::endl;
        std::cout.flush();

        MEMORY_DEBUG_PRINT("Before " + name);
        physics->step(dt);
        MEMORY_DEBUG_PRINT("After " + name);

        double dt_suggest = physics->suggestTimeStep();
        next_time_step = std::min(next_time_step, dt_suggest);
        // if(this->rank == 0) std::cout << "Suggested " << next_time_step << ", dt_suggest " << dt_suggest << std::endl;
        
        #ifdef RICH_MPI
            this->buildDataTransfer(physics->GetExchangeChain());
            
            if(firstTime)
            {
                std::vector<double> weights = physics->getLoadBalanceWeights();
                this->tess.Rebalance(weights);
                this->buildDataTransfer();
            }
        #endif // RICH_MPI

        if(this->rank == 0)
        {
            std::cout << name << " suggested " << dt_suggest << " for dt " << std::endl;
            std::cout << std::endl;
        }
    }
    
    this->tracker.updateCycle();
    this->tracker.updateTime(this->tsc->GetTimeStep());

    this->tsc->SetTimeStep(next_time_step);
    // if(this->rank == 0) std::cout << "Time step will be " << this->tsc.GetTimeStep() << std::endl;
}

#ifdef RICH_MPI
void Simulation::storeLoadBalance(const std::string &name, std::shared_ptr<LoadBalancer> lb)
{
    this->loads[name] = lb;
}

void Simulation::PresetLoadBalance(const std::string &name)
{
    this->currentLoad = this->tess.GetLoadBalancer();
    this->loads[name] = this->currentLoad;
    this->currentLB = name;
}

void Simulation::setCurrentLoadBalance(const std::string &name)
{
    if(name.empty())
    {
        return;
    }
    auto it = this->loads.find(name);
    if(it == this->loads.end())
    {
        UniversalError eo("setCurrentLoadBalance: unknown load balance");
        eo.addEntry("Name", name);
        throw eo;
    }

    size_t Ntotal = this->tess.GetPointNo();
    #ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &Ntotal, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    #endif // RICH_MPI

    if(Ntotal == 0)
    {
        this->tess.PresetLoadBalancer(it->second);
    }
    else
    {
        this->tess.SetLoadBalancer(it->second);
        this->buildDataTransfer();
    }

    std::shared_ptr<LoadBalancer> load = this->tess.GetLoadBalancer();
    this->loads[name] = load;
    this->currentLoad = load;
    this->currentLB = name;
}

std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer>>> Simulation::GetLoads(void) const
{
    return std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer>>>(this->loads.begin(), this->loads.end());
}
#endif // RICH_MPI
