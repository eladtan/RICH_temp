#include <MeshDecomposer3D/load_balancing/HilbertLoadBalancer.hpp>
#include "3D/tessellation/Voronoi3D.hpp"
#include "misc/mesh_generator3D.hpp"
#include "utils/printing/print.hpp"
#include "3D/output/write3D.hpp"

std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<size_t>> &v)
{
    os << "{";
    for(size_t i = 0; i < v.size(); i++)
    {
        os << v[i];
        os << " (size: " << v[i].size() << ")";
        if(i < v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "}";
    return os;
}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <number of points>" << std::endl;
        return 1;
    }
    size_t N = std::stoul(argv[1]);

    MPI_Init(&argc, &argv);
    rank_t rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Vector3D ll(-1, -1, -1), ur(1, 1, 1);
    
    std::vector<Vector3D> points;
    if(rank == 0)
    {
        points = RandRectangular(N, ll, ur);
    }
    points = MPI_Spread(points, 0, MPI_COMM_WORLD);
    
    Voronoi3D tess(ll, ur);
    try
    {
        tess.BuildParallel(points);
    }
    catch(const UniversalError &eo)
    {
        reportError(eo);
        throw;
    }
    std::cout << "Rank " << rank << " has " << tess.GetPointNo() << " points currently" << std::endl;

    WriteVoronoiParallel(tess, "voronoi_old.h5");

    std::shared_ptr<LoadBalancer<Vector3D>> loadBalancer = tess.GetLoadBalancer();
    HilbertLoadBalancer<Vector3D> *plb = dynamic_cast<HilbertLoadBalancer<Vector3D>*>(loadBalancer.get());
    if(plb == nullptr)
    {
        throw UniversalError("LoadBalancer is not a HilbertLoadBalancer");
    }
    std::cerr << "LoadBalancer is a HilbertLoadBalancer with boundaries: " << plb->boundaries << std::endl;

    std::vector<size_t> newBoundaries;
    for(rank_t _rank = 0; _rank < size; _rank++)
    {
        newBoundaries.push_back(plb->boundaries[2 * (_rank / 2)]);
    }
    newBoundaries.back() = plb->boundaries.back();

    plb->boundaries = newBoundaries;
    tess.SetLoadBalancer(loadBalancer);
    // std::cout << "Rank " << rank << " has " << tess.GetPointNo() << " points currently" << std::endl;

    // std::cout << "===== " << rank << " ===== DupProcs " << tess.GetDuplicatedProcs() << std::endl;
    // std::cout << "===== " << rank << " ===== DupPoints " << tess.GetDuplicatedPoints() << std::endl;
    // std::cout << "===== " << rank << " ===== Ghosts " << tess.GetGhostIndeces() << std::endl;

    std::cout << "Rank " << rank << " has " << tess.GetPointNo() << " points currently" << std::endl;
    double volume = 0;
    for(size_t i = 0; i < tess.GetPointNo(); i++)
    {
        volume += tess.GetVolume(i);
    }
    MPI_Allreduce(MPI_IN_PLACE, &volume, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double expectedVolume = (ur.x - ll.x) * (ur.y - ll.y) * (ur.z - ll.z);
    if(rank == 0)
    {
        std::cout << "Total volume: " << volume << ", expected: " << expectedVolume << std::endl;
        std::cout << "Error: " << std::abs(volume - expectedVolume) / expectedVolume << std::endl;
    }

    WriteVoronoiParallel(tess, "voronoi_new.h5");
    MPI_Finalize();
}