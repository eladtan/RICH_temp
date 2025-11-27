#include "utils_for_tests.hpp"
#include <catch2/catch_test_macros.hpp>

namespace utils_for_tests::mpi {
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

    void mpi_barrier(){
        #ifdef RICH_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
    }

    RichBasicTestFixture::RichBasicTestFixture() 
    :   rank{get_mpi_rank()},
        comm_size{get_mpi_world_size()} 
    {}

    RichNoMpiTestFixture::RichNoMpiTestFixture() : RichBasicTestFixture() {
        if(rank != rank_root){
            SKIP("Test has no MPI, running only in root");
        }
    }


    RichMpiFixture::RichMpiFixture() : RichBasicTestFixture(){}
}