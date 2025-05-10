#include "ProgressCounter.hpp"

#ifdef RICH_MPI

ProgressCounter::ProgressCounter(const MPI_Comm &comm, int myNumParticles)
{
    MPI_Comm_rank(comm, &this->rank);
    MPI_Comm_size(comm, &this->size);

    this->master_rank = 0;

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "accumulate_ordering", "none"); // No strict ordering
    MPI_Info_set(info, "accumulate_ops", "same_op");
    MPI_Info_set(info, "same_disp_unit", "true");
    MPI_Win_allocate(sizeof(int), sizeof(int), info, comm, &this->is_done, &this->is_done_win);
    MPI_Win_set_errhandler(this->is_done_win, MPI_ERRORS_RETURN);
    MPI_Info_free(&info);

    int *model, flag;
    MPI_Win_get_attr(this->is_done_win, MPI_WIN_MODEL, &model, &flag);
    if(*model == MPI_WIN_SEPARATE)
    {
        std::cout << "MPI is using WIN_SEPARATE (" << MPI_WIN_SEPARATE << "). Can not continue" << std::endl;
        exit(1);
    }
    
    MPI_Win_lock(MPI_LOCK_SHARED, this->rank, MPI_MODE_NOCHECK, this->is_done_win);
    int zero = 0;
    MPI_Put(&zero, 1, MPI_INT, this->rank, 0, 1, MPI_INT, this->is_done_win);
    MPI_Win_unlock(this->rank, this->is_done_win);

    int totalNumParticles = 0;
    MPI_Reduce(&myNumParticles, (void*)&totalNumParticles, 1, MPI_INT, MPI_SUM, this->master_rank, comm);
    this->counter = std::make_shared<GlobalCounter>(comm, totalNumParticles);

    MPI_Barrier(comm);
}

ProgressCounter::~ProgressCounter()
{
    MPI_Win_free(&this->is_done_win);
}

int ProgressCounter::Increment(int n)
{
    int result = this->counter->Increment(n);
    int currValue = result + n;
    if(currValue == 0)
    {
        this->MarkDone();
    }
    return currValue;
}

void ProgressCounter::MarkDone(void)
{
    static int plus_one = 1;
    MPI_Win_lock_all(MPI_MODE_NOCHECK, this->is_done_win);
    for(rank_t _rank = 0; _rank < this->size; _rank++)
    {
        MPI_Put(&plus_one, 1, MPI_INT, _rank, 0, 1, MPI_INT, this->is_done_win);
    }
    MPI_Win_unlock_all(this->is_done_win);
}

#endif // RICH_MPI