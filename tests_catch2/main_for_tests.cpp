#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>
#include <iostream>
#include <fstream>
#include "config_tests.hpp"
#include "utils_for_tests.hpp"
#include <chrono>

#ifdef RICH_MPI
#include <mpi.h>
#endif

int main(int argc, char* argv[]) {
#ifdef RICH_MPI
    MPI_Init(&argc, &argv);
#endif

    tests_config::assert_order_arg_is_given(argc, argv);
    tests_config::parseTestsConfigArguments(argc, argv);

    auto const rank = utils_for_tests::mpi::get_mpi_rank();
    if(rank == utils_for_tests::mpi::rank_root) { 
        std::cout << "Tests Config: " << tests_config::TestsConfig::instance().repr() << std::endl; 
    }
    
    using clock = std::chrono::steady_clock;

    auto const start = clock::now();
    // There must be only one call to Catch::Session....
    int result = Catch::Session().run(argc, argv);

    auto const elapsed = std::chrono::duration<double>(clock::now() - start); 
    
    if(rank == utils_for_tests::mpi::rank_root){
        std::cout << "Total Time Elapsed for Root: " << elapsed.count() << " seconds." << std::endl;
    }

#ifdef RICH_MPI
    MPI_Finalize();
#endif

    return result;
}