#include "read3D.hpp"

std::vector<Vector3D> ReadVoronoiPointsHelper(const HDF5Reader &reader)
{
    std::vector<Vector3D> res;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    reader.ReadElement("/mesh_point_x", x);
    reader.ReadElement("/mesh_point_y", y);
    reader.ReadElement("/mesh_point_z", z);
    size_t const N = x.size();
    for(size_t i = 0; i < N; ++i)
    {
        res.push_back(Vector3D(x[i], y[i], z[i]));
    }
    return res;
}

std::vector<Vector3D> ReadVoronoiPoints(const std::string &filename)
{
    HDF5Reader reader(filename);
    return ReadVoronoiPointsHelper(reader);
}

#ifdef RICH_MPI
    std::vector<Vector3D> ReadVoronoiPointsParallel(const std::string &filename)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::filesystem::path input_directory = std::filesystem::path(filename).replace_extension("");
        std::string rank_file = (input_directory / std::to_string(rank) / ".h5").string();
        HDF5Reader reader(rank_file);
        return ReadVoronoiPointsHelper(reader);
    }
#endif // RICH_MPI