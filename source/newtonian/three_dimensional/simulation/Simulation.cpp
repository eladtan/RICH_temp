#include "Simulation.hpp"
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include <malloc.h>

Simulation::Simulation(Tessellation3D &tess_, const std::vector<ComputationalCell3D> &cells_, EquationOfState &eos_, bool new_start) :
     tess(tess_), cells(cells_), extensives(cells_.size()), eos(eos_), Max_ID(0), wallclockTime(0)
#ifdef RICH_MPI
     , currentBox(tess_.GetBoxCoordinates())
#endif // RICH_MPI
{
    #ifdef RICH_MPI
        this->currentLoad = nullptr;
        MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
        MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    #else // RICH_MPI
        this->rank = 0;
        this->size = 1;
    #endif // RICH_MPI

    if(new_start)
    {
        this->initializeCellIDs();
    }
    else
    {
        this->recomputeMaxID();
    }

#ifdef RICH_MPI
    ComputationalCell3D cdummy;
    MPI_exchange_data(this->tess, this->cells, true, 1, &cdummy);
#endif

    size_t N = this->tess.GetPointNo();
    for(size_t i = 0; i < N; ++i)
    {
        PrimitiveToConserved(this->cells[i], this->tess.GetVolume(i), this->extensives[i]);
    }
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
    this->Max_ID = nstart + N - 1;
#ifdef RICH_MPI
    for (int i = this->rank + 1; i < this->size; ++i)
        this->Max_ID += nrecv[static_cast<size_t>(i)];
#endif
}

void Simulation::recomputeMaxID(void)
{
    size_t N = this->cells.size();
    size_t maxid = 0;
    for (size_t i = 0; i < N; ++i)
        maxid = std::max(maxid, this->cells[i].ID);
#ifdef RICH_MPI
    MPI_Allreduce(&maxid, &this->Max_ID, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
#else
    this->Max_ID = maxid;
#endif
}

size_t &Simulation::GetMaxID(void)
{
    return this->Max_ID;
}

const size_t &Simulation::GetMaxID(void) const
{
    return this->Max_ID;
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
        if (chain.GetNorg() == 0)
            return;
        for(MigrationBuffer &buff : this->migrationBuffers)
        {
            buff.transferChain(chain);
        }
    }
#endif // RICH_MPI

double Simulation::GetWallclockTime(void) const
{
    return this->wallclockTime;
}

void Simulation::SetWallclockTime(double t)
{
    this->wallclockTime = t;
}

void Simulation::step(void)
{
    MEMORY_DEBUG_PRINT("Simulation::step START cycle=" + std::to_string(this->tracker.getCycle()));
    this->lastPhysicsTimes.clear();
    auto stepWallStart = std::chrono::high_resolution_clock::now();
    double next_time_step = std::numeric_limits<double>::max();
    // double dt = std::numeric_limits<double>::max();
    #ifdef RICH_MPI
        if(this->rank == 0)
    #endif // RICH_MPI
    {
        std::cout << "\nCycle " << this->tracker.getCycle() << " at time " << this->tracker.getTime() << std::endl;
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
                    // std::vector<double> weights = physics->getLoadBalanceWeights();
                    // std::vector<Vector3D> points = this->tess.getMeshPoints();
                    // points.resize(this->tess.GetPointNo());
                    // this->tess.BuildParallel(points, weights, true);
                    firstTime = true;
                    // this->buildDataTransfer();
                }
            }

            bool forceRebalance = this->forceRebalanceSteps > 0 && this->tracker.getCycle() < this->forceRebalanceSteps;
            double rebalanceTime = 0;
            if(physics->allowRebalance() || forceRebalance)
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
                    auto rebalanceStart = std::chrono::high_resolution_clock::now();

                    physics->beforeLB();
                    this->tess.Rebalance(weights);
                    if(this->rank == 0)
                    {
                        std::cout << "Did rebalanced - load balance:" << std::endl;
                        auto lb = this->tess.GetLoadBalancer();
                        if (lb) lb->printInfo();
                    }                
                    this->buildDataTransfer();
                    physics->afterLB();

                    MPI_Barrier(MPI_COMM_WORLD);
                    rebalanceTime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - rebalanceStart).count();
                    if(this->rank == 0) std::cout << "Rebalance time: " << rebalanceTime << "s" << std::endl;
                }
                else
                {
                    if(this->rank == 0) std::cout << LB << " is already rebalanced" << std::endl;
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
        double dt_before = dt;

        malloc_trim(0);
        MEMORY_DEBUG_PRINT("Before " + name);
        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI
        auto start = std::chrono::high_resolution_clock::now();
        physics->step(dt);

        double dt_actual = this->tsc->GetTimeStep();
        if(this->rank == 0 && dt_actual != dt_before)
            std::cout << "Hydro dt actually used: " << dt_actual << " (requested: " << dt_before << ")" << std::endl;

        double localTime = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - start).count();

        #ifdef RICH_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif // RICH_MPI

        MEMORY_DEBUG_PRINT("After " + name);
        auto end = std::chrono::high_resolution_clock::now();
        double physicsTime = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

        #ifdef RICH_MPI
        double totalPhysicsTime = physicsTime + rebalanceTime;
        if(this->rank == 0) std::cout << "Physics " << name << " time: " << totalPhysicsTime << " (step=" << physicsTime << "s, rebalance=" << rebalanceTime << "s)" << std::endl;
        this->lastPhysicsTimes[name] = totalPhysicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #else
        if(this->rank == 0) std::cout << "Physics " << name << " time: " << physicsTime << std::endl;
        this->lastPhysicsTimes[name] = physicsTime;
        this->lastLocalPhysicsTimes[name] = localTime;
        #endif

        double dt_suggest = physics->suggestTimeStep();
        next_time_step = std::min(next_time_step, dt_suggest);
        // if(this->rank == 0) std::cout << "Suggested " << next_time_step << ", dt_suggest " << dt_suggest << std::endl;
        
        #ifdef RICH_MPI
            this->buildDataTransfer(physics->GetExchangeChain());
            
            if(firstTime)
            {
                physics->beforeLB();
                std::vector<double> weights = physics->getLoadBalanceWeights();
                this->tess.Rebalance(weights);
                if(this->rank == 0)
                {
                    std::cout << "Rebalanced first time - load balance:" << std::endl;
                    auto lb = this->tess.GetLoadBalancer();
                    if (lb) lb->printInfo();
                }            
                this->buildDataTransfer();
                physics->afterLB();
            }
        #endif // RICH_MPI

        #ifdef RICH_MPI
            std::pair<Vector3D, Vector3D> newBox = this->tess.GetBoxCoordinates();
            if(newBox != this->currentBox)
            {
                this->currentBox = newBox;
                for(auto [loadName, load] : this->loads)
                {
                    if(loadName == this->currentLB)
                    {
                        continue;
                    }
                    load->changeBox(this->currentBox);
                }
            }
        #endif // RICH_MPI

        if(this->rank == 0)
        {
            std::cout << name << " suggested " << dt_suggest << " for dt " << std::endl;
            std::cout << std::endl;
        }
    }
    
    this->tracker.updateCycle();
    double dt_used = this->tsc->GetTimeStep();
    if(this->rank == 0)
        std::cout << "Advancing time by dt=" << dt_used << ", next suggested dt=" << next_time_step << std::endl;
    this->tracker.updateTime(dt_used);

    this->tsc->SetTimeStep(next_time_step);

    double stepWallSec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - stepWallStart).count();
    this->wallclockTime += stepWallSec;
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

    if(this->rank == 0)
    {
        std::cout << "Changed load balance:" << std::endl;
        this->currentLoad->printInfo();
    }
}

std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer>>> Simulation::GetLoads(void) const
{
    return std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer>>>(this->loads.begin(), this->loads.end());
}
#endif // RICH_MPI
