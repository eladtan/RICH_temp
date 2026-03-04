#include "read3D.hpp"
#include "cellData.hpp"

#ifdef RICH_MPI

Snapshot3D ReadSnapshot3DParallel_AOS(const string &fname, int fake_rank)
{
    Snapshot3D res;
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rank_to_read = (fake_rank >= 0) ? fake_rank : rank;

    HDF5Reader globalfile(fname);

    BoundingBox<Vector3D> box;
    globalfile.ReadElement("/Box", box);
    res.ll = box.getLL();
    res.ur = box.getUR();

    if(globalfile.Exists("/TracerNames"))
    {
        globalfile.ReadElement("/TracerNames", res.tracerstickernames.first);
    }
    if(globalfile.Exists("/StickerNames"))
    {
        globalfile.ReadElement("/StickerNames", res.tracerstickernames.second);
    }

    ComputationalCell3D::tracerNames = res.tracerstickernames.first;
    ComputationalCell3D::stickerNames = res.tracerstickernames.second;

    std::string dirname = std::filesystem::path(fname).replace_extension("").string();
    std::string rank_file = dirname + "/" + std::to_string(rank_to_read) + ".h5";

    HDF5Reader reader(rank_file);

    reader.ReadElement("/Points", res.mesh_points);
    reader.ReadElement("/Cells", res.cells);
    reader.ReadElement("/Volume", res.volumes);

    return res;
}

#endif // RICH_MPI
