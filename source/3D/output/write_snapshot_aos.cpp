#include "newtonian/three_dimensional/hdsim_3d.hpp"
#include "write3D.hpp"


void WriteData3DHelper_AOS(HDF5Writer &file, const std::string &prefix, const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells)
{
    size_t Ncells = tess.GetPointNo();

    std::vector<Vector3D> points = tess.getMeshPoints();
    points.resize(Ncells);
    file.WriteElement(prefix + "/Points", points);

    std::vector<ComputationalCell3D> cells_vec(cells.cbegin(), cells.cbegin() + Ncells);
    file.WriteElement(prefix + "/Cells", cells_vec);
    std::vector<Vector3D> cm = tess.GetAllCM();
    cm.resize(Ncells);
    file.WriteElement(prefix + "/CM", cm);
    std::vector<double> volume = tess.GetAllVolumes();
    volume.resize(Ncells);
    file.WriteElement(prefix + "/Volume", volume);
}

#ifdef RICH_MPI
void WriteSnapshot3DParallel_AOS(const Tessellation3D &tess, const std::vector<ComputationalCell3D> &cells, std::string const& filename)
{
    int rank = 0;
    int ws = 0; // MPI_COMM_WORLD size

    fs::path path = fs::absolute(filename).parent_path();
    fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
    if(not fs::exists(ranks_files_path))
    {
        fs::create_directory(ranks_files_path);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ws);
    std::string myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

    HDF5Writer filewriter(myFilePath);
    std::shared_ptr<HDF5Writer> globalFileWriter = (rank == 0) ? std::make_shared<HDF5Writer>(filename) : nullptr;
    
    if(rank == 0)
    {

        BoundingBox<Vector3D> box(tess.GetBoxCoordinates().first, tess.GetBoxCoordinates().second);
        globalFileWriter->WriteElement("/Box", box);
        globalFileWriter->WriteElement("/TracerNames", ComputationalCell3D::tracerNames);
        globalFileWriter->WriteElement("/StickerNames", ComputationalCell3D::stickerNames);
    }

    std::string prefix = "";
    WriteData3DHelper_AOS(filewriter, prefix, tess, cells);

    MPI_Barrier(MPI_COMM_WORLD);
    
    // only rank 0 makes the shared file
    if(rank == 0)
    {
        for(int _rank = 0; _rank < ws; _rank++)
        {
            // merge `_rank`'s file
            std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
            globalFileWriter->AddExternalLink(rankFile, "/", "/rank" + std::to_string(_rank));
        }
    }
}

void WriteSnapshot3DParallel_AOS(HDSim3D const &sim, std::string const& filename)
{
    WriteSnapshot3DParallel_AOS(sim.getTessellation(), sim.getCells(), filename);
}

#endif // RICH_MPI
