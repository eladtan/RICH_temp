#include "DistributedMutex.hpp"

#ifdef RICH_MPI

DistributedMutex::DistributedMutex(const MPI_Comm &comm, rank_t rank):
    comm(comm), rank(rank), value(nullptr), destroyed(false)
{
    assert(this->comm != MPI_COMM_NULL);
    rank_t my_rank, size;
    MPI_Comm_rank(this->comm, &my_rank);
    MPI_Comm_size(this->comm, &size);
    assert(size > 1);
    
    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    int err = MPI_Win_allocate((my_rank == rank)? sizeof(int) : 0, sizeof(int), info, this->comm, &this->value, &this->win);
    if(err != MPI_SUCCESS)
    {
        char msg[MPI_MAX_ERROR_STRING];
        int msg_len;
        MPI_Error_string(err, msg, &msg_len);
        std::cerr << "Error: DistributedMutex MPI_Win_allocate failed with error code " << err << ": " << msg << std::endl;
        exit(1);
    }
    MPI_Win_set_errhandler(this->win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    int *model = nullptr;
    int flag = 0;
    MPI_Win_get_attr(this->win, MPI_WIN_MODEL, &model, &flag);
    if(flag && model && *model == MPI_WIN_SEPARATE)
    {
        std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
        exit(1);
    }

    if(my_rank == rank && this->value != nullptr)
    {
        *static_cast<int*>(this->value) = 0;
    }

    MPI_Barrier(this->comm);
}

void DistributedMutex::Destroy()
{
    MPI_Win_free(&this->win);
    this->destroyed = true;
}

DistributedMutex::~DistributedMutex()
{
    if(not std::uncaught_exceptions())
    {
        if(not this->destroyed)
        {
            this->Destroy();
        }
    }
}

void DistributedMutex::Sync(void)
{
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
    MPI_Win_sync(this->win);
    MPI_Win_unlock(this->rank, this->win);
}

void DistributedMutex::Lock(void)
{
    // static int plus_one = 1;
    // static int minus_one = -1;

    // int val = -1;
    // MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
    // do
    // {
    //     val = -1;
    //     int retval = MPI_Fetch_and_op(&plus_one, &val, MPI_INT, this->rank, 0, MPI_SUM, this->win);
    //     assert(retval == MPI_SUCCESS);
    //     MPI_Win_flush(this->rank, this->win);
    //     assert(val >= 0);
    //     if(val > 0)
    //     {
    //         // std::cout << "Failed to lock mutex, trying again" << std::endl;
    //         // failure, decrement
    //         MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
    //         MPI_Win_flush(this->rank, this->win);
    //         usleep(10); // sleep a while and try again
    //     }
    // }
    // while(val > 0);
    // MPI_Win_unlock_all(this->win);
    // // MPI_Win_unlock(this->rank, this->win);
    // assert(val <= 0);
    static int plus_one = 1;
    static int minus_one = -1;
    static int zero = 0;

    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->win);
    int old = 0;
    while(true)
    {
        MPI_Compare_and_swap(&plus_one, &zero, &old, MPI_INT, this->rank, 0, this->win);
        MPI_Win_flush(this->rank, this->win);
        if(old == 0)
        {
            break;
        }

        // int retval = MPI_Fetch_and_op(&plus_one, &val, MPI_INT, this->rank, 0, MPI_SUM, this->win);
        // assert(retval == MPI_SUCCESS);
        // MPI_Win_flush(this->rank, this->win);
        // assert(val >= 0);
        // if(val > 0)
        // {
        //     // std::cout << "Failed to lock mutex, trying again" << std::endl;
        //     // failure, decrement
        //     MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
        //     MPI_Win_flush(this->rank, this->win);
        //     usleep(10); // sleep a while and try again
        // }
    }
    // while(val > 0);
    MPI_Win_unlock_all(this->win);
    // MPI_Win_unlock(this->rank, this->win);
}

void DistributedMutex::Unlock(void)
{
    static int minus_one = -1;
    static int zero = 0;
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->win);
    MPI_Put(&zero, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->win);
    // MPI_Accumulate(&minus_one, 1, MPI_INT, this->rank, 0, 1, MPI_INT, MPI_SUM, this->win);
    // MPI_Win_sync(this->win);
    MPI_Win_flush(this->rank, this->win);
    MPI_Win_unlock(this->rank, this->win);
}

#endif // RICH_MPI