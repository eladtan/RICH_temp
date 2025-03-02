#ifndef MONTE_CARLO_TIMESTEP
#define MONTE_CARLO_TIMESTEP

#include <limits> // temporary
#include <chrono> // temporary
#include "utils/debug/vtune.h"
#include "MonteCarloParticle.hpp"
#include "MonteCarloManager.hpp"

template<typename T, typename Grid>
class MonteCarloTimestep
{
    using Particle = MonteCarloParticle<T, Grid>;

public:
    MonteCarloTimestep(const Grid &grid, const std::vector<Particle> &particles);

    ~MonteCarloTimestep();

    std::vector<Particle> step(dt_t dt);

private:
    rank_t rank, size;
    volatile int finish;
    MPI_Win finish_win;
    boost::container::flat_map<size_t, std::pair<rank_t, size_t>> ranks_ghost_map;
    MonteCarloManager<T, Grid> pointsManager;
    const Grid &grid;

    void MarkFinish();

boost::container::flat_map<size_t, std::pair<rank_t, size_t>> GetGhostMap();
};

template<typename T, typename Grid>
MonteCarloTimestep<T, Grid>::MonteCarloTimestep(const Grid &grid, const std::vector<Particle> &particles): grid(grid), pointsManager(grid, particles, particles.size() * 10)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    this->finish = 0;
    MPI_Win_create((void*)&this->finish, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &this->finish_win);
    this->ranks_ghost_map = this->GetGhostMap();
}

template<typename T, typename Grid>
MonteCarloTimestep<T, Grid>::~MonteCarloTimestep()
{
    MPI_Win_free(&this->finish_win);
}

template<typename T, typename Grid>
void MonteCarloTimestep<T, Grid>::MarkFinish()
{
    std::cout << "Broadcasting FINISH to everyone" << std::endl;
    int done = 1;
    MPI_Win_lock_all(0, this->finish_win);
    for(rank_t _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Put(&done, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->finish_win);
    }
    MPI_Win_unlock_all(this->finish_win);
}

template<typename T, typename Grid>
boost::container::flat_map<size_t, std::pair<rank_t, size_t>> MonteCarloTimestep<T, Grid>::GetGhostMap()
{
    std::vector<std::vector<size_t>> incoming = MPI_exchange_data(grid.GetDuplicatedProcs(), grid.GetDuplicatedPoints());
    const std::vector<std::vector<size_t>> &ghosts = grid.GetGhostIndeces();
    for(size_t i = 0; i < incoming.size(); i++)
    {
        int _rank = grid.GetDuplicatedProcs()[i];
        for(size_t j = 0; j < incoming[i].size(); j++)
        {
            assert(incoming[i].size() == ghosts[i].size());
            ranks_ghost_map[ghosts[i][j]] = {_rank, incoming[i][j]};
        }
    }
    return ranks_ghost_map;
}

template<typename T, typename Grid>
std::vector<MonteCarloParticle<T, Grid>> MonteCarloTimestep<T, Grid>::step(dt_t fullDt)
{
    int length = *this->pointsManager.to_handle_list_length;
    for(size_t i = 0; i < length; i++)
    {
        Particle &p = this->pointsManager.GetParticle(i);
        p.cellIndex = this->grid.GetContainingCell(p.location);

        // if(not this->grid.PointInMyDomain(this->grid.GetMeshPoint(p.cellIndex)))
        // {
        //     std::cout << "Particle " << p << " is not in my domain, rank " << this->rank << std::endl;
        //     exit(1);
        // }
        // // assert(this->grid.PointInMyDomain(p.location));
        // std::cout << "Setting particle " << p << "'s cell index to " << p.cellIndex << std::endl;
    }
    
    std::cout << "Rank " << this->rank << ", average particles per cell: " << ((double) length) / this->grid.GetPointNo() << std::endl;
    size_t Ncells = grid.GetPointNo();
    std::vector<Particle> newParticles;

    MPI_Barrier(MPI_COMM_WORLD);

    vtune_start();

    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Rank " << this->rank << ", starting step() with " << *this->pointsManager.th_list_length << "!" << std::endl;
    volatile int &tohandle = *this->pointsManager.th_list_length;
    while(not this->finish)
    {
        length = tohandle;
        assert(length >= 0);
        int particleChange = 0;
        if(length == 0)
        {
            usleep(20);
            continue;
        }
        for(int i = 0; i < length; i++)
        {
            size_t particleIndex = this->pointsManager.th_list[i];
            if(particleIndex == EMPTY)
            {
                continue;
            }
            assert(i >= 0); // i changes inside the loop as well
            // std::cout << i << " / " << *this->pointsManager.to_handle_list_length << std::endl;
            Particle &particle = this->pointsManager.GetParticle(i);
            // std::cout << "Rank " << this->rank << " iterates over particle toHandleIndex " << i << " (this is particle of index " << this->pointsManager.to_handle_list[i] << ") (" << particle << ") out of " << length << " active" << std::endl;
            auto [faceIntersect, timeIntersect] = particle.distanceToNearestFace(grid);
            timeIntersect *= (1 + EPSILON);
            dt_t timeScattering = std::numeric_limits<dt_t>::max(); // TODO
            dt_t timeLeft = fullDt - particle.timeLeft;

            dt_t dt = std::min(timeLeft, std::min(timeScattering, timeIntersect));
            assert(dt >= 0);

            if(dt == timeIntersect)
            {
                const std::pair<size_t, size_t> &cellNeighbors = grid.GetFaceNeighbors(faceIntersect);
                assert(particle.cellIndex == cellNeighbors.first or particle.cellIndex == cellNeighbors.second);
                size_t nextCellIndex = (cellNeighbors.first == particle.cellIndex)? cellNeighbors.second : cellNeighbors.first;
                assert(nextCellIndex != particle.cellIndex);
                T previousLocation = particle.location;
                particle.timeLeft -= dt;
                particle.location += particle.velocity * dt;
                if(nextCellIndex < Ncells)
                {
                    particle.cellIndex = nextCellIndex;
                    // assert(realContainingCell == nextCellIndex);
                }
                else
                {
                    // a ghost point, check rank and index in rank
                    auto it = ranks_ghost_map.find(nextCellIndex);
                    if(it == ranks_ghost_map.end())
                    {
                        // std::cout << "Rank " << this->rank << " is done with particle " << particle << ", since it leaves the domain." << std::endl;
                        this->pointsManager.ClearParticle(i); // remove it
                        i -= 1; length -= 1;
                        particleChange += 1; // decrement particles num later
                        // std::cout << "Num particles left: " << left << std::endl;
                        continue;
                    }
                    auto [otherRank, neighborIndexInRank] = it->second;
                    particle.cellIndex = neighborIndexInRank;
                    particle.sender = this->rank;
                    // TODO: velocity?
                    // std::cout << "Moving particle " << particle << " to rank " << otherRank << std::endl;
                    this->pointsManager.MoveParticle(i, otherRank);
                    i -= 1; length -= 1; // repeat the particle in this location, as it is a new one replacing the leaving particle
                    continue;
                }
            }
            else if(dt == timeScattering)
            {
            }
            else if(dt == timeLeft)
            {
                newParticles.push_back(particle);
                // remove particle from current list
                this->pointsManager.RemoveFromToHandleList(i); // remove it
                i -= 1; length -= 1;
                particleChange += 1; // decrement particles num later
                continue;
            }
        }
        if(particleChange > 0)
        {
            int left = this->pointsManager.DecrementParticlesNum(particleChange);
            if(left == 0)
            {
                this->MarkFinish();
            }
        }
    }
    
    assert(*this->pointsManager.to_handle_list_length == 0);
    std::cout << "Rank " << this->rank << ", ended step() with " << *this->pointsManager.to_handle_list_length << "!" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    if(this->rank == 0)
    {
        // print time in double (seconds)
        std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count() << " seconds" << std::endl;    
    }

    vtune_stop();
    return newParticles;
}

#endif // MONTE_CARLO_TIMESTEP