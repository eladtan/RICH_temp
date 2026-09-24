#include <iostream>
#include <mpi.h>
#include "examples/Vector3D.hpp"
#include <MeshDecomposer3D/points_manager/HilbertPointsManager.hpp>
#include <MeshDecomposer3D/load_balancing/OneDimensionalLoadBalancer.hpp>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    const std::string mode = argc > 1 ? argv[1] : "kernel";
    const Vector3D ll(0,0,0), ur(1,1,1);
    std::vector<Vector3D> points{Vector3D(0.15,0.21,0.31), Vector3D(0.76,0.62,0.72)};
    std::vector<double> weights(points.size(), 1.0);
    try
    {
        if(mode == "onedim")
        {
            OneDimensionalLoadBalancer<Vector3D> lb(ll, ur, Axis::X);
            lb.rebalance(points, weights);
            std::cout << "bins_after_rebalance=" << lb.GetBins().size() << std::endl;
            auto ranks = lb.getIntersectingRanks(Vector3D(0.5,0.5,0.5), 0.1);
            std::cout << "intersecting_count=" << ranks.size() << std::endl;
        }
        else
        {
            HilbertPointsManager<Vector3D> pm(ll,ur);
            std::vector<EmptyPayload> payloads(points.size());
            std::vector<size_t> active{0,1};
            auto result = pm.update(points, weights, payloads, active, false, true);
            points = result.newPoints;
            weights = result.newWeights;
            std::cout << "first_update_succeeded changing_to_identity_kernel" << std::endl;
            pm.setIndexing(std::make_shared<const Kernelization3D::Identity<Vector3D>>());
            result = pm.update(points, weights, payloads, active, false, true);
            std::cout << "second_update_succeeded" << std::endl;
        }
    }
    catch(const std::exception &err)
    {
        std::cout << "caught_exception=" << err.what() << std::endl;
        MPI_Finalize();
        return 3;
    }
    MPI_Finalize();
    return 0;
}
