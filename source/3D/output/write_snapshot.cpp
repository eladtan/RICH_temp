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

VTU_Output WriteSnapshot3DHelper(H5File &file, Group &writegroup, Group &tracers, Group &stickers, const std::string &filename, const HDSim3D &sim, const std::vector<DiagnosticAppendix3D*> &appendices, bool write_vtu)
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
    write_std_vector_to_hdf5(writegroup, temp, "X");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetMeshPoint(i).y;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Y");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetMeshPoint(i).z;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Z");
    
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
    write_std_vector_to_hdf5(writegroup, temp, "CMx");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetCellCM(i).y;
    }
    write_std_vector_to_hdf5(writegroup, temp, "CMy");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = tess.GetCellCM(i).z;
    }
    write_std_vector_to_hdf5(writegroup, temp, "CMz");

    Ncells = tess.GetPointNo();
    temp.resize(Ncells);
    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].density;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Density");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Density");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].pressure;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Pressure");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Pressure");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].internal_energy;
    }
    write_std_vector_to_hdf5(writegroup, temp, "InternalEnergy");

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
    write_std_vector_to_hdf5(writegroup, ids, "ID");

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
    write_std_vector_to_hdf5(writegroup, temp, "Vx");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].velocity.y;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Vy");

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].velocity.z;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Vz");

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
    write_std_vector_to_hdf5(writegroup, temp, "Temperature");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Temperature");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].Erad;
    }
    write_std_vector_to_hdf5(writegroup, temp, "Erad");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Erad");
    }

    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].elastic_energy;
    }
    write_std_vector_to_hdf5(writegroup, temp, "ElasticEnergy");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("ElasticEnergy");
    }

    
    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].G;
    }
    write_std_vector_to_hdf5(writegroup, temp, "ShearModulus");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("ShearModulus");
    }


    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].Y0;
    }
    write_std_vector_to_hdf5(writegroup, temp, "PlasticYield");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("PlasticYield");
    }

    
    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].strain_plastic;
    }
    write_std_vector_to_hdf5(writegroup, temp, "EPS");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("EPS");
    }


    for(size_t i = 0; i < Ncells; ++i)
    {
        temp[i] = cells[i].strain_plastic_dt;
    }
    write_std_vector_to_hdf5(writegroup, temp, "EPSrate");

    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("EPSrate");
    }


    temp.resize(Ncells * 9);
    for(size_t i = 0; i < Ncells; ++i)
        for(size_t j = 0; j < 3; ++j)
            for(size_t k = 0; k < 3; ++k)
                temp[9*i + 3*j + k] = cells[i].stress.at(j, k);
    write_std_vector_to_hdf5(writegroup, temp, "Stress");
    temp.resize(Ncells);



    for(size_t j = 0; j < ComputationalCell3D::tracerNames.size(); ++j)
    {
        for(size_t i = 0; i < Ncells; ++i)
        {
            temp[i] = cells[i].tracers[j];
        }
        write_std_vector_to_hdf5(tracers, temp, ComputationalCell3D::tracerNames[j]);
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
        write_std_vector_to_hdf5(stickers, temp, ComputationalCell3D::stickerNames[j]);
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
    write_std_vector_to_hdf5(writegroup, temp, "Volume");
    if(write_vtu)
    {
        vtu_cell_variables.push_back(temp);
        vtu_cell_variable_names.push_back("Volume");
    }

    // Appendices
    for(size_t i = 0; i < appendices.size(); ++i)
    {
        temp = (*(appendices.at(i)))(sim);
        write_std_vector_to_hdf5(writegroup, temp, appendices.at(i)->getName());
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
        H5File file;

        fs::path path = fs::absolute(filename).parent_path();
        std::string myFilePath;
        fs::path ranks_files_path = path / fs::path(filename).filename().replace_extension();
        if(not fs::exists(ranks_files_path))
        {
            fs::create_directory(ranks_files_path);
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
        myFilePath = (ranks_files_path / std::to_string(rank)).string() + ".h5";

        // truncate my file and open it
        H5File file2(H5std_string(myFilePath), H5F_ACC_TRUNC);
        file2.close();
        file.openFile(H5std_string(myFilePath), H5F_ACC_RDWR);

        Group writegroup = file.openGroup("/");

        Tessellation3D const &tess = sim.getTesselation();

        if(rank == 0)
        {
            std::vector<double> box(6);
            box[0] = tess.GetBoxCoordinates().first.x;
            box[1] = tess.GetBoxCoordinates().first.y;
            box[2] = tess.GetBoxCoordinates().first.z;
            box[3] = tess.GetBoxCoordinates().second.x;
            box[4] = tess.GetBoxCoordinates().second.y;
            box[5] = tess.GetBoxCoordinates().second.z;
            write_std_vector_to_hdf5(file, box, "Box");
        }

        Group tracers, stickers;

        tracers = writegroup.createGroup("/tracers");
        stickers = writegroup.createGroup("/stickers");

        VTU_Output vtu = WriteSnapshot3DHelper(file, writegroup, tracers, stickers, filename, sim, appendices, write_vtu);

        if(rank == 0)
        {
            vector<double> time(1, sim.getTime());
            write_std_vector_to_hdf5(file, time, "Time");

            vector<int> cycle(1, static_cast<int>(sim.getCycle()));
            write_std_vector_to_hdf5(file, cycle, "Cycle");
        }

        stickers.close();
        tracers.close();
        writegroup.close();
        file.close();

        writeVTU(filename, sim, vtu);
        MPI_Barrier(MPI_COMM_WORLD);
        // only rank 0 makes the shared file
        if(rank == 0)
        {
            file2 = H5File(H5std_string(filename), H5F_ACC_TRUNC);
            file2.close();
            hid_t shared_file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

            for(int _rank = 0; _rank < ws; _rank++)
            {
                // merge `_rank`'s file
                std::string rankFile((ranks_files_path / std::to_string(_rank)).string() + ".h5");
                std::string rankGroupName("/rank" + std::to_string(_rank));
                H5Lcreate_external(rankFile.c_str(),
                                                    "/",
                                                    shared_file_id,
                                                    rankGroupName.c_str(),
                                                    H5P_DEFAULT,
                                                    H5P_DEFAULT);
            }
            H5Fclose(shared_file_id);
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
    H5File file;
    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
            H5File file2(H5std_string(filename), H5F_ACC_TRUNC);
            file2.close();
            file.openFile(H5std_string(filename), H5F_ACC_RDWR);
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
        write_std_vector_to_hdf5(file, box, "Box");
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI

    Group writegroup;
    #ifdef RICH_MPI
        if(mpi_write)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            int dummy = 0;
            if(rank > 0)
            {
                MPI_Recv(&dummy, 1, MPI_INT, rank - 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                file.openFile(H5std_string(filename), H5F_ACC_RDWR);
            }
            file.createGroup("/rank" + std::to_string(rank));
            writegroup = file.openGroup("/rank" + std::to_string(rank));
        }
        else
        {
            writegroup = file.openGroup("/");
        }
    #else
        writegroup = file.openGroup("/");
    #endif

    Group tracers, stickers;

    #ifdef RICH_MPI
        if(mpi_write)
        {
            tracers = writegroup.createGroup("/rank" + std::to_string(rank) + "/tracers");
            stickers = writegroup.createGroup("/rank" + std::to_string(rank) + "/stickers");
        }
        else
        {
            tracers = writegroup.createGroup("/tracers");
            stickers = writegroup.createGroup("/stickers");
        }
    #else // RICH_MPI
        tracers = writegroup.createGroup("/tracers");
        stickers = writegroup.createGroup("/stickers");
    #endif // RICH_MPI

    VTU_Output vtu = WriteSnapshot3DHelper(file, writegroup, tracers, stickers, filename, sim, appendices, write_vtu);

    #ifdef RICH_MPI
        if(rank == 0)
        {
    #endif // RICH_MPI
        vector<double> time(1, sim.getTime());
        write_std_vector_to_hdf5(file, time, "Time");

        vector<int> cycle(1, static_cast<int>(sim.getCycle()));
        write_std_vector_to_hdf5(file, cycle, "Cycle");
    #ifdef RICH_MPI
        }
    #endif // RICH_MPI
    
    #ifdef RICH_MPI
        if(mpi_write)
        {
            if(rank < (ws - 1))
            {
                int dummy = 0;
                tracers.close();
                stickers.close();
                writegroup.close();
                file.close();
                MPI_Send(&dummy, 1, MPI_INT, rank + 1, HDF5_WRITE_BLOCK_TAG, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
        else
        {
            file.close();
        }
    #else
        file.close();
    #endif

    writeVTU(filename, sim, vtu);
}