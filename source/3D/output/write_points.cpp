#include "write3D.hpp"

namespace
{
void WritePointsToWriter(HDF5Writer &writer, const std::string &basePath,
                         const std::vector<Vector3D> &points,
                         const std::vector<std::vector<double>> &data,
                         const std::vector<std::string> &names,
                         int mpi_rank)
{
    size_t const Ncells = points.size();

    std::vector<double> x(Ncells), y(Ncells), z(Ncells);
    for(size_t i = 0; i < Ncells; ++i)
    {
        x[i] = points[i].x;
        y[i] = points[i].y;
        z[i] = points[i].z;
    }

    std::string const prefix = basePath.empty() ? "" : (basePath + "/");
    writer.WriteElement(prefix + "X", x);
    writer.WriteElement(prefix + "Y", y);
    writer.WriteElement(prefix + "Z", z);

    for(size_t i = 0; i < data.size(); ++i)
    {
        writer.WriteElement(prefix + names[i], data[i]);
    }

    std::vector<double> mpi_rank_vec(Ncells, static_cast<double>(mpi_rank));
    writer.WriteElement(prefix + "MPI_rank", mpi_rank_vec);
}
} // namespace

void WritePoints(const std::vector<Vector3D> &points, const std::string &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names)
{
    int rank = 0;
    int ws = 0;
    std::string basePath;
#ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    if(rank == 0)
    {
        HDF5Writer w(filename, true);
        w.Close();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank > 0)
    {
        int dummy = 0;
        MPI_Recv(&dummy, 1, MPI_INT, rank - 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    basePath = "/rank" + std::to_string(rank);
#else
    basePath = "";
#endif

    {
        HDF5Writer writer(filename, rank == 0);
        WritePointsToWriter(writer, basePath, points, data, names, rank);
    }

#ifdef RICH_MPI
    if(rank < ws - 1)
    {
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, rank + 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

#ifdef RICH_MPI
void WritePointsParallel(const std::vector<Vector3D> &points, const std::string &filename, const std::vector<std::vector<double>> &data, const std::vector<std::string> &names)
{
    int rank = 0;
    int ws = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);

    fs::path path = fs::absolute(filename).parent_path();
    fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
    if(not fs::exists(ranks_files_path))
    {
        fs::create_directory(ranks_files_path);
    }
    std::string myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

    {
        HDF5Writer writer(myFilePath);
        WritePointsToWriter(writer, "", points, data, names, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0)
    {
        HDF5Writer globalWriter(filename);
        for(int _rank = 0; _rank < ws; _rank++)
        {
            std::string rankFile = (ranks_files_path / std::to_string(_rank)).string() + ".h5";
            std::string rankGroupName = "/rank" + std::to_string(_rank);
            globalWriter.AddExternalLink(rankFile, "/", rankGroupName);
        }
    }
}
#endif // RICH_MPI
