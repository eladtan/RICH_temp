#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <vector>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

int main()
{
    int rank = 0;
    int world_size = 1;
#ifdef RICH_MPI
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif

    try {
        Vector3D ll(0.0, 0.0, 0.0);
        Vector3D ur(1.0, 1.0, 1.0);
        const std::size_t Np = (world_size > 1)
            ? static_cast<std::size_t>(1e6)
            : static_cast<std::size_t>(1e4);

        std::vector<Vector3D> points;
        if (rank == 0) {
            points = RandRectangular(Np, ll, ur);
        }
#ifdef RICH_MPI
        points = MPI_Spread(points, 0, MPI_COMM_WORLD);
#endif

        Voronoi3D tess(ll, ur);
#ifdef RICH_MPI
        tess.BuildParallel(points);
#else
        tess.Build(points);
#endif

        // Sum local cell volumes (indices [0, GetPointNo()) are real cells)
        double local_vol = 0.0;
        const std::size_t n_local = tess.GetPointNo();
        for (std::size_t i = 0; i < n_local; ++i) {
            local_vol += tess.GetVolume(i);
        }

        double total_vol = local_vol;
#ifdef RICH_MPI
        MPI_Allreduce(&local_vol, &total_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

        const double box_vol = (ur.x - ll.x) * (ur.y - ll.y) * (ur.z - ll.z);
        const double rel_err = std::abs(total_vol - box_vol) / box_vol;
        const int passed = (rel_err < 1e-10) ? 1 : 0;

        if (rank == 0) {
            std::cout << "total_volume = " << total_vol << "\n"
                      << "box_volume   = " << box_vol << "\n"
                      << "rel_error    = " << rel_err << "\n"
                      << "pass         = " << passed << std::endl;

            std::ofstream out("voronoi_volume_metrics.txt");
            out.setf(std::ios::scientific);
            out.precision(16);
            out << "mode " << (world_size > 1 ? "mpi" : "serial") << "\n";
            out << "num_points " << Np << "\n";
            out << "total_volume " << total_vol << "\n";
            out << "box_volume " << box_vol << "\n";
            out << "rel_error " << rel_err << "\n";
            out << "pass " << passed << "\n";
            out.close();
        }

#ifdef RICH_MPI
        MPI_Finalize();
#endif
        return passed ? 0 : 1;
    }
    catch (UniversalError const& e) {
        reportError(e);
#ifdef RICH_MPI
        MPI_Abort(MPI_COMM_WORLD, 2);
#endif
        return 2;
    }
}
