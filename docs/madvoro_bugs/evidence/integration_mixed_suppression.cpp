#include <algorithm>
#include <iostream>
#include <mpi.h>
#include "regression_tests/lib/voronoi_test_common.hpp"

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2) { MPI_Finalize(); return 2; }
    const bool mixed = argc <= 1 || std::string(argv[1]) != "all_suppress";
    const Vector3D ll(0,0,0), ur(1,1,1);
    auto points = rank == 0
        ? MadVoro::regression_tests::RandRectangular(128, ll, ur, 761231ULL)
        : std::vector<Vector3D>{};
    MadVoro::regression_tests::VoronoiGrid grid(ll, ur);
    points = grid.BuildParallel(points);
    const auto countBefore = points.size();
    points.emplace_back(rank == 0 ? 0.31 : 0.73, 0.391239, 0.712341);
    const std::vector<double> validWeights(points.size(), 1.0);
    const bool suppressHere = !mixed || rank == 0;
    std::cout << "rank=" << rank << " count_before=" << countBefore
              << " input_count=" << points.size() << " weight_count=" << validWeights.size()
              << " suppressExchange=" << suppressHere << " entering_second_build" << std::endl;
    points = grid.BuildParallel(points, validWeights, true, suppressHere);
    std::cout << "rank=" << rank << " second_build_returned count=" << points.size() << std::endl;
    MPI_Finalize();
    return 0;
}
