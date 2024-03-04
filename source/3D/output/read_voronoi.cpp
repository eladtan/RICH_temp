#include "read3D.hpp"

std::vector<Vector3D> ReadVoronoiPointsHelper(H5File &file)
{
    std::vector<Vector3D> res;
    Group read_location = file.openGroup("/");
    const vector<double> x = read_double_vector_from_hdf5(read_location, "mesh_point_x");
    const vector<double> y = read_double_vector_from_hdf5(read_location, "mesh_point_y");
    const vector<double> z = read_double_vector_from_hdf5(read_location, "mesh_point_z");
    size_t const N = x.size();
    for(size_t i = 0; i < N; ++i)
    {
        res.push_back(Vector3D(x[i], y[i], z[i]));
    }
    return res;
}
std::vector<Vector3D> ReadVoronoiPoints(const std::string &filename)
{
    H5File file(filename, H5F_ACC_RDONLY);
    std::vector<Vector3D> res = ReadVoronoiPointsHelper(file);
    file.close();
    return res;
}

#ifdef RICH_MPI
    std::vector<Vector3D> ReadVoronoiPointsParallel(const std::string &filename)
    {
        std::filesystem::path input_directory = std::filesystem::path(filename).replace_extension("");
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        H5File file(input_directory / std::to_string(rank) / ".h5", H5F_ACC_RDONLY);
        std::vector<Vector3D> res = ReadVoronoiPointsHelper(file);
        file.close();
        return res;
    }
#endif // RICH_MPI