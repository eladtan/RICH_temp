#include "utils_for_tests.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace utils_for_tests {
    int get_mpi_rank() {
        int rank = 0;
        
        #ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif

        return rank;
    }

    int get_mpi_world_size(){
        int ws = 1;

        #ifdef RICH_MPI
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        #endif

        return ws;
    }
}