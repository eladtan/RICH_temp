#include "write3D.hpp"

struct VTU_Output
{
    std::vector<std::vector<double>> vtu_cell_variables;
    std::vector<std::string> vtu_cell_variable_names;
    std::vector<std::string> vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> vtu_cell_vectors;
};

void writeVTU(const std::string &filename, const HDSim3D &sim, const VTU_Output &data)
{
    std::filesystem::path vtu_name(filename);
    vtu_name.replace_extension("vtu");
    write_vtu3d::write_vtu_3d(vtu_name, data.vtu_cell_variable_names, data.vtu_cell_variables, data.vtu_cell_vectors_names, data.vtu_cell_vectors, sim.getTime(), sim.getCycle(), sim.getTesselation());
}

VTU_Output WriteSnapshot3DHelper(HDF5Writer &file, const std::string &prefix, const std::string &filename, const HDSim3D &sim, const std::vector<DiagnosticAppendix3D*> &appendices, bool write_vtu)
{
    Tessellation3D const &tess = sim.getTesselation();
    vector<ComputationalCell3D> const &cells = sim.getCells();
    
    VTU_Output vtu;
    std::vector<std::vector<double>> &vtu_cell_variables = vtu.vtu_cell_variables;
    std::vector<std::string> &vtu_cell_variable_names = vtu.vtu_cell_variable_names;
    std::vector<std::string> &vtu_cell_vectors_names = vtu.vtu_cell_vectors_names;
    std::vector<std::vector<Vector3D>> &vtu_cell_vectors = vtu.vtu_cell_vectors;

    size_t Ncells = tess.GetPointNo();

    vector<double> temp(Ncells);
    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetMeshPoint(i).x;
    }
    file.WriteElement(prefix +"/X", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetMeshPoint(i).y;
    }
    file.WriteElement(prefix + "/Y", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetMeshPoint(i).z;
    }
    file.WriteElement(prefix + "/Z", temp);
    
    if(write_vtu)
    {
        vtu_cell_vectors_names.push_back("Coordinates");
        std::vector<Vector3D> vel(Ncells);
        for(size_t i = 0; i < Ncells; ++i)
            vel[i] = tess.GetMeshPoint(i);
        vtu_cell_vectors.push_back(vel);
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetCellCM(i).x;
    }
    file.WriteElement(prefix + "/CMx", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetCellCM(i).y;
    }
    file.WriteElement(prefix + "/CMy", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetCellCM(i).z;
    }
    file.WriteElement(prefix + "/CMz", temp);

    Ncells = tess.GetPointNo();
    temp.resize(Ncells);
    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].density;
    }
    file.WriteElement(prefix + "/Density", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Density");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].pressure;
    }
    file.WriteElement(prefix + "/Pressure", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Pressure");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].internal_energy;
    }
    file.WriteElement(prefix + "/InternalEnergy", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("InternalEnergy");
    }

    std::vector<size_t> ids(Ncells);
    for(size_t i = 0; i < Ncells; ++i)
    {
        ids[i] = cells[i].ID;
    }
    file.WriteElement(prefix + "/ID", ids);

    if(write_vtu)
    {
        for(size_t i = 0; i < Ncells; ++i)
        {
            temp[i] = ids[i];
        }
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("ID");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].velocity.x;
    }
    file.WriteElement(prefix + "/Vx", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].velocity.y;
    }
    file.WriteElement(prefix + "/Vy", temp);

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].velocity.z;
    }
    file.WriteElement(prefix + "/Vz", temp);

    if(write_vtu)
    {
        vtu_cell_vectors_names.push_back("Velocity");
        std::vector<Vector3D> vel(Ncells);
        for(size_t i = 0; i < Ncells; ++i)
        {
            vel[i] = cells[i].velocity;
        }
        vtu_cell_vectors.push_back(vel);
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].temperature;
    }
    file.WriteElement(prefix + "/Temperature", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Temperature");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].Erad;
    }
    file.WriteElement(prefix + "/Erad", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Erad");
    }

    for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
        for(std::size_t i=0; i < Ncells; ++i){
            temp[i] = cells[i].Eg[g];
        }
        
        file.WriteElement(prefix + "/Eg_" + std::to_string(g), temp);
        
        if(write_vtu)
        {
            vtu_cell_variables.push_back(temp);
            vtu_cell_variable_names.push_back("Eg_" + std::to_string(g));
        }
    }

    for(size_t j = 0; j < ComputationalCell3D::tracerNames.size(); ++j)
    {
        for(size_t i = 0; i < Ncells; ++i)
        {
            temp[i] = cells[i].tracers[j];
        }
        file.WriteElement(prefix + "/tracers/" + ComputationalCell3D::tracerNames[j], temp);

        if(write_vtu)
        {
            vtu_cell_variables.push_back(temp);
            vtu_cell_variable_names.push_back(ComputationalCell3D::tracerNames[j]);
        }
    }

    for(size_t j = 0; j < ComputationalCell3D::stickerNames.size(); ++j)
    {
        for(size_t i = 0; i < Ncells; ++i)
        {
            temp[i] = cells[i].stickers[j];
        }
        file.WriteElement(prefix + "/stickers/" + ComputationalCell3D::stickerNames[j], temp);

        if(write_vtu)
        {
            vtu_cell_variables.push_back(temp);
            vtu_cell_variable_names.push_back(ComputationalCell3D::stickerNames[j]);
        }
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetVolume(i);
    }
    file.WriteElement(prefix + "/Volume", temp);

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Volume");
    }

    // Appendices
    for(size_t i = 0; i < appendices.size(); ++i)
    {
        temp = (*(appendices.at(i)))(sim);
        file.WriteElement(prefix + "/" + appendices.at(i)->getName(), temp);

        if(write_vtu)
        {
            vtu_cell_variables.push_back(temp);
            vtu_cell_variable_names.push_back(appendices.at(i)->getName());
        }
    }

    return vtu;
}

#ifdef RICH_MPI
    void WriteSnapshot3DParallel(HDSim3D const &sim, std::string const &filename, const vector<DiagnosticAppendix3D*> &appendices, bool write_vtu)
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
        Tessellation3D const &tess = sim.getTesselation();

        HDF5Writer filewriter(myFilePath);
        std::shared_ptr<HDF5Writer> globalFileWriter = (rank == 0) ? std::make_shared<HDF5Writer>(filename) : nullptr;
        
        if(rank == 0)
        {
            std::vector<double> box(6);
            box[0] = tess.GetBoxCoordinates().first.x;
            box[1] = tess.GetBoxCoordinates().first.y;
            box[2] = tess.GetBoxCoordinates().first.z;
            box[3] = tess.GetBoxCoordinates().second.x;
            box[4] = tess.GetBoxCoordinates().second.y;
            box[5] = tess.GetBoxCoordinates().second.z;
            globalFileWriter->WriteElement("/Box", box);
        }


        VTU_Output vtu = WriteSnapshot3DHelper(filewriter, "", filename, sim, appendices, write_vtu);

        if(rank == 0)
        {
            globalFileWriter->WriteElement("/Time", sim.getTime());
            globalFileWriter->WriteElement("/Cycle", sim.getCycle());
        }

        writeVTU(filename, sim, vtu);
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
#endif // RICH_MPI

void WriteSnapshot3D(HDSim3D const &sim, std::string const &filename, const vector<DiagnosticAppendix3D*> &appendices
#ifdef RICH_MPI
                                         , bool mpi_write
#endif // RICH_MPI
                                         , bool write_vtu)
{
    #ifdef RICH_MPI
        int rank = 0;
        int ws = 0; // MPI_COMM_WORLD size
        if(mpi_write)
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &ws);
        }
    #endif

    std::shared_ptr<HDF5Writer> filewriter = nullptr;

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
        filewriter = std::make_shared<HDF5Writer>(filename);
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    Tessellation3D const &tess = sim.getTesselation();

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
        std::vector<double> box(6);
        box[0] = tess.GetBoxCoordinates().first.x;
        box[1] = tess.GetBoxCoordinates().first.y;
        box[2] = tess.GetBoxCoordinates().first.z;
        box[3] = tess.GetBoxCoordinates().second.x;
        box[4] = tess.GetBoxCoordinates().second.y;
        box[5] = tess.GetBoxCoordinates().second.z;
        filewriter->WriteElement("/Box", box);
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    std::string prefix = "";

    #ifdef RICH_MPI
        if(mpi_write)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            int dummy = 0;
            if(rank > 0)
            {
                MPI_Recv(&dummy, 1, MPI_INT, rank - 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                filewriter = std::make_shared<HDF5Writer>(filename, false /* don't truncate */);
            }
            prefix = "/rank" + std::to_string(rank);
        }
        else
        {
            prefix = "";
        }
    #else
        prefix = "";
    #endif

    VTU_Output vtu = WriteSnapshot3DHelper(*filewriter, prefix, filename, sim, appendices, write_vtu);

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
        filewriter->WriteElement("/Time", sim.getTime());
        filewriter->WriteElement("/Cycle", sim.getCycle());
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI
    
    #ifdef RICH_MPI
        if(mpi_write)
        {
            if(rank < (ws - 1))
            {
                int dummy = 0;
                filewriter->Close();
                MPI_Send(&dummy, 1, MPI_INT, rank + 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else
        {
            filewriter->Close();
        }
    #else
        filewriter->Close();
    #endif

    if(write_vtu)
    {
        writeVTU(filename, sim, vtu);
    }
}