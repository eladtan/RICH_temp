#ifndef MONTE_CARLO_MANAGER
#define MONTE_CARLO_MANAGER

#include "3D/tesselation/voronoi/Voronoi3D.hpp"
#include "MonteCarloParticle.hpp"
#include "mpi/mpi_commands.hpp"

template<typename T, typename Grid>
class MonteCarloTimestep;

template<typename T, typename Grid>
class MonteCarloManager
{
public:
    using Particle = MonteCarloParticle<T, Grid>;

    MonteCarloManager(const Grid &tess, const std::vector<Particle> &particles, size_t list_length);
    
    ~MonteCarloManager();

    void MoveParticle(size_t index, rank_t rank);

    void RemoveFromToHandleList(size_t indexInToHandle);
    
    int IncrementParticlesNum(int n = 1);
    
    int DecrementParticlesNum(int n = 1){return this->IncrementParticlesNum(-n);};

    Particle &GetParticle(size_t indexInToHandle)
    {
        if(indexInToHandle >= *this->to_handle_list_length)
        {
            UniversalError eo("Trying to reach an illegal particle");
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            eo.addEntry("Rank", rank);
            eo.addEntry("indexInToHandle", indexInToHandle);
            eo.addEntry("toHandleList length", *this->to_handle_list_length);
            throw eo;
        }
        assert(indexInToHandle < *this->to_handle_list_length);
        return this->particles[this->to_handle_list[indexInToHandle]];
    };

    template<typename T2, typename Grid2>
    friend class MonteCarloTimestep;

private:
    rank_t rank, size;
    size_t buffer_size;
    volatile int *num_particles;
    const Grid &tess;
    volatile int *available_list_length;
    volatile int *to_handle_list_length;
    volatile int *editing_available_list_mutex;
    size_t *available_list;
    size_t *to_handle_list;
    Particle *particles;
    MPI_Win editing_available_list_mutex_win;
    MPI_Win num_particles_win;
    MPI_Win to_handle_list_win;
    MPI_Win to_handle_list_length_win;
    MPI_Win available_list_win;
    MPI_Win available_list_length_win;
    MPI_Win particles_win;

    void AcquireAvailableListMutex(int rank);

    void ReleaseAvailableListMutex(int rank);
};

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::MonteCarloManager(const Grid &tess, const std::vector<Particle> &particles, size_t list_length): tess(tess)
{    
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &this->size);
    MPI_Win_allocate(sizeof(int) * 1, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &this->available_list_length, &this->available_list_length_win);
    MPI_Win_set_errhandler(this->available_list_length_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(int) * 1, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &this->to_handle_list_length, &this->to_handle_list_length_win);
    MPI_Win_set_errhandler(this->to_handle_list_length_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(int) * 1, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &this->editing_available_list_mutex, &this->editing_available_list_mutex_win);
    MPI_Win_set_errhandler(this->editing_available_list_mutex_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(size_t) * list_length, sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD, &this->available_list, &this->available_list_win);
    MPI_Win_set_errhandler(this->available_list_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(size_t) * list_length, sizeof(size_t), MPI_INFO_NULL, MPI_COMM_WORLD, &this->to_handle_list, &this->to_handle_list_win);
    MPI_Win_set_errhandler(this->to_handle_list_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate(sizeof(Particle) * list_length, sizeof(Particle), MPI_INFO_NULL, MPI_COMM_WORLD, &this->particles, &this->particles_win);
    MPI_Win_set_errhandler(this->particles_win, MPI_ERRORS_RETURN);
    MPI_Win_allocate((this->rank == 0)? sizeof(int) : 0, sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &this->num_particles, &this->num_particles_win);
    MPI_Win_set_errhandler(this->num_particles_win, MPI_ERRORS_RETURN);

    this->buffer_size = list_length;

    *this->editing_available_list_mutex = 0;

    int my_length = particles.size();
    *this->to_handle_list_length = my_length;
    MPI_Reduce(&my_length, (void*)this->num_particles, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    *this->available_list_length = list_length - my_length;
    // copy particles by bytes
    std::memcpy(this->particles, particles.data(), my_length * sizeof(Particle));
    std::memset(this->to_handle_list, std::numeric_limits<size_t>::max(), list_length);
    std::memset(this->available_list, std::numeric_limits<size_t>::max(), list_length);

    size_t len = *this->to_handle_list_length;
    size_t avail_len = *this->available_list_length;
    for(size_t i = 0; i < len; i++)
    {
        this->to_handle_list[i] = i;
        assert(this->to_handle_list[i] < particles.size());
    }

    for(size_t i = 0; i < avail_len; i++)
    {
        this->available_list[i] = len + i;
        assert(this->available_list[i] < list_length);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T, typename Grid>
MonteCarloManager<T, Grid>::~MonteCarloManager()
{
    MPI_Win_free(&this->editing_available_list_mutex_win);
    MPI_Win_free(&this->num_particles_win);
    MPI_Win_free(&this->available_list_length_win);
    MPI_Win_free(&this->available_list_win);
    MPI_Win_free(&this->available_list_length_win);
    MPI_Win_free(&this->available_list_win);
    MPI_Win_free(&this->particles_win);
}

template<typename T, typename Grid>
int MonteCarloManager<T, Grid>::IncrementParticlesNum(int n)
{
    int num_particles;
    int retval = MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, this->num_particles_win);
    assert(retval == 0);
    MPI_Fetch_and_op(&n, &num_particles, MPI_INT, 0, 0, MPI_SUM, this->num_particles_win);
    MPI_Win_unlock(0, this->num_particles_win);
    return num_particles + n;
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::RemoveFromToHandleList(size_t indexInToHandle)
{
    static int minus_one = -1;
    this->AcquireAvailableListMutex(this->rank);
    // remove from self to handle list by swapping with last element and popping
    int retval = MPI_Win_lock(MPI_LOCK_SHARED, this->rank, 0, this->to_handle_list_length_win);
    assert(retval == 0);
    int handle_list_length = *this->to_handle_list_length;
    int tmpIdx = handle_list_length - 1;
    // std::cout << "Rank " << this->rank << " removes indexToHandle " << indexInToHandle << ", which is particle " << this->to_handle_list[indexInToHandle];
    // std::cout << " (" << this->GetParticle(indexInToHandle) << "), replacing it with idxToHandle " << tmpIdx << ", which is particle " << this->to_handle_list[tmpIdx];
    // std::cout << " (" << this->GetParticle(tmpIdx) << ") by moving it to the end (list length is updated to " << (*this->to_handle_list_length)-1 << ")" << std::endl;
    // if(indexInToHandle != tmpIdx and this->GetParticle(indexInToHandle).id == this->GetParticle(tmpIdx).id)
    // {
    //     UniversalError eo("Can not, somehow, remove particle: duplication");
    //     eo.addEntry("indexInToHandle", indexInToHandle);
    //     eo.addEntry("toHandle[" + std::to_string(indexInToHandle) + "]", this->to_handle_list[indexInToHandle]);
    //     eo.addEntry("tmpIdx", tmpIdx);
    //     eo.addEntry("toHandle[" + std::to_string(tmpIdx) + "]", this->to_handle_list[tmpIdx]);
    //     eo.addEntry("List length", handle_list_length);
    //     eo.addEntry("shard ID", this->GetParticle(indexInToHandle).id);
    //     eo.addEntry("Particle", this->GetParticle(indexInToHandle));
    //     throw eo;
    // }
    this->to_handle_list[indexInToHandle] = this->to_handle_list[tmpIdx];
    // std::cout << "Rank " << this->rank << ", particle in indexToHandle " << indexInToHandle << " (which is " << this->to_handle_list[indexInToHandle] << ") is now " << this->GetParticle(indexInToHandle) << " and list length is " << (*this->to_handle_list_length - 1) << " (instead of " << handle_list_length << ")" << std::endl;
    MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->to_handle_list_length_win);
    MPI_Win_unlock(this->rank, this->to_handle_list_length_win);
    this->ReleaseAvailableListMutex(this->rank);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::AcquireAvailableListMutex(int rank)
{
    static int plus_one = 1;
    static int minus_one = -1;

    int retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->editing_available_list_mutex_win);
    assert(retval == 0);
    int mutex_value = -1;
    do
    {
        MPI_Fetch_and_op(&plus_one, &mutex_value, MPI_INT, rank, 0, MPI_SUM, this->editing_available_list_mutex_win);
        MPI_Win_flush(rank, this->editing_available_list_mutex_win);
        assert(mutex_value >= 0);
        if(mutex_value != 0)
        {
            // mutex is occupied, cancel
            MPI_Accumulate(&minus_one, 1, MPI_INT, rank, 0, 1, MPI_INT, MPI_SUM, this->editing_available_list_mutex_win);
            usleep(10); // wait a little bit before trying again
        }
    } while(mutex_value != 0);
    MPI_Win_unlock(rank, this->editing_available_list_mutex_win);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::ReleaseAvailableListMutex(int rank)
{
    static int minus_one = -1;
    int retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->editing_available_list_mutex_win);
    assert(retval == 0);
    // MPI_Fetch_and_op(&minus_one, &mutex_value, MPI_INT, rank, 0, MPI_SUM, this->editing_available_list_mutex_win);
    MPI_Accumulate(&minus_one, 1, MPI_INT, rank, 0, 1, MPI_INT, MPI_SUM, this->editing_available_list_mutex_win);
    MPI_Win_unlock(rank, this->editing_available_list_mutex_win);
}

template<typename T, typename Grid>
void MonteCarloManager<T, Grid>::MoveParticle(size_t indexInToHandle, rank_t rank)
{
    static const int minus_one = -1;
    static const int plus_one = 1;

    assert(indexInToHandle < *this->to_handle_list_length);
    size_t index = this->to_handle_list[indexInToHandle];
    assert(index < this->buffer_size);

    this->AcquireAvailableListMutex(rank);

    int retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->available_list_length_win);
    assert(retval == 0);
    // pop an element from remote's available_list
    int availableListIdx = -1;
    MPI_Fetch_and_op(&minus_one, &availableListIdx, MPI_INT, rank, 0, MPI_SUM, this->available_list_length_win);
    // MPI_Win_flush(rank, this->available_list_length_win);
    MPI_Win_unlock(rank, this->available_list_length_win);
    if(availableListIdx > this->buffer_size)
    {
        std::cerr << "Error! available list index is " << availableListIdx << ", while buffer size is " << this->buffer_size << std::endl;
    }
    assert(availableListIdx <= this->buffer_size);
    
    // Get last element of available list
    availableListIdx -= 1;      
    retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->available_list_win);
    assert(retval == 0);
    size_t availableIndex = -1;
    MPI_Get(&availableIndex, 1, MPI_UNSIGNED_LONG_LONG, rank, availableListIdx, 1, MPI_UNSIGNED_LONG_LONG, this->available_list_win);    
    MPI_Win_unlock(rank, this->available_list_win);
    if(availableIndex >= this->buffer_size)
    {
        std::cerr << "ERROR!!!! available list length is " << availableListIdx << ", availableIndex is " << availableIndex << ", while buffer size is " << this->buffer_size << std::endl;
        exit(1);
    }
    assert(availableIndex < this->buffer_size);

    // set the particle in remote's `availableIndex`
    // TODO: this part should not be inside the mutex
    retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->particles_win);
    assert(retval == 0);
    MPI_Put(&this->particles[index], sizeof(Particle), MPI_BYTE, rank, availableIndex, sizeof(Particle), MPI_BYTE, this->particles_win);
    MPI_Win_unlock(rank, this->particles_win);

    retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->to_handle_list_length_win);
    assert(retval == 0);
    // add the index to the list of 'to_handle' particles
    int toHandleListIdx;
    MPI_Get(&toHandleListIdx, 1, MPI_INT, rank, 0, 1, MPI_INT, this->to_handle_list_length_win);
    MPI_Win_flush_local(rank, this->to_handle_list_length_win);

    retval = MPI_Win_lock(MPI_LOCK_SHARED, rank, 0, this->to_handle_list_win);
    MPI_Put(&availableIndex, 1, MPI_UNSIGNED_LONG_LONG, rank, toHandleListIdx, 1, MPI_UNSIGNED_LONG_LONG, this->to_handle_list_win);
    // std::cout << "Rank " << this->rank << " writes to rank " << rank << " in handleListIdx " << toHandleListIdx << " particle index " << availableIndex << ", which is particle " << this->GetParticle(indexInToHandle) << std::endl; 
    MPI_Win_unlock(rank, this->to_handle_list_win);

    MPI_Accumulate(&plus_one, 1, MPI_INT, rank, 0, 1, MPI_INT, MPI_SUM, this->to_handle_list_length_win);
    MPI_Win_unlock(rank, this->to_handle_list_length_win);
    
    // release mutex
    this->ReleaseAvailableListMutex(rank);

    this->RemoveFromToHandleList(indexInToHandle);

    this->AcquireAvailableListMutex(this->rank);
    // add place to self available_list by adding index to the available list
    retval = MPI_Win_lock(MPI_LOCK_SHARED, this->rank, 0, this->available_list_length_win);
    assert(retval == 0);

    // pop an element from remote's available_list
    int localAvailableListIdx;
    MPI_Fetch_and_op(&plus_one, &localAvailableListIdx, MPI_INT, this->rank, 0, MPI_SUM, this->available_list_length_win);
    // MPI_Win_flush(rank, this->available_list_length_win);
    retval = MPI_Win_lock(MPI_LOCK_SHARED, this->rank, 0, this->available_list_win);
    assert(retval == 0);

    assert(localAvailableListIdx < this->buffer_size);
    assert(index < this->buffer_size);
    // std::cout << "Rank " << this->rank << " Puts to self availalble list " << index << ", on index " << localAvailableListIdx << std::endl;
    MPI_Put(&index, 1, MPI_UNSIGNED_LONG_LONG, this->rank, localAvailableListIdx, 1, MPI_UNSIGNED_LONG_LONG, this->available_list_win);
    MPI_Win_unlock(this->rank, this->available_list_win);
    MPI_Win_unlock(this->rank, this->available_list_length_win);
    this->ReleaseAvailableListMutex(this->rank);
}

#endif // MONTE_CARLO_MANAGER