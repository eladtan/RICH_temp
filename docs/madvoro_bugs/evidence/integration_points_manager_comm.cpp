#include <iostream>
#include <mpi.h>
#include "examples/Vector3D.hpp"
#include <MeshDecomposer3D/points_manager/HilbertPointsManager.hpp>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const Vector3D ll(0,0,0), ur(1,1,1);
    HilbertPointsManager<Vector3D> manager(ll, ur, MPI_COMM_SELF);
    std::vector<Vector3D> points{
        Vector3D(rank == 0 ? 0.15 : 0.75, 0.21, 0.31),
        Vector3D(rank == 0 ? 0.16 : 0.76, 0.22, 0.32)};
    std::vector<double> weights(points.size(), 1.0);
    std::vector<EmptyPayload> payloads(points.size());
    std::vector<size_t> active{0,1};
    std::cout << "world_rank=" << rank << " entering MPI_COMM_SELF manager update" << std::endl;
    auto result = manager.update(points, weights, payloads, active, false, true);
    std::cout << "world_rank=" << rank << " result_count=" << result.newPoints.size() << std::endl;
    MPI_Finalize();
    return result.newPoints.size() == points.size() ? 0 : 3;
}
