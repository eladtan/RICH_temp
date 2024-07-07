#include "read3D.hpp"

Snapshot3D ReadSnapshot3DHelper(H5File &file, H5File &globalfile, Group &read_location, bool const good_open)
{
    Snapshot3D res;
    const vector<double> box = read_double_vector_from_hdf5(globalfile, "Box");
    res.ll.Set(box[0], box[1], box[2]);
    res.ur.Set(box[3], box[4], box[5]);
    // Misc
    {
        const vector<double> time = read_double_vector_from_hdf5(globalfile, "Time");
        res.time = time.at(0);
        const vector<int> cycle = read_int_vector_from_hdf5(globalfile, "Cycle");
        res.cycle = cycle.at(0);
        Group g_tracers = read_location.openGroup("tracers");
        Group g_stickers = read_location.openGroup("stickers");
        vector<string> tracernames(g_tracers.getNumObjs());
        for(hsize_t n = 0; n < g_tracers.getNumObjs(); ++n)
        {
            const H5std_string name = g_tracers.getObjnameByIdx(n);
            tracernames[n] = name;
        }
        vector<string> stickernames(g_stickers.getNumObjs());
        for(hsize_t n = 0; n < g_stickers.getNumObjs(); ++n)
        {
            const H5std_string name = g_stickers.getObjnameByIdx(n);
            stickernames[n] = name;
        }
        res.tracerstickernames.first = tracernames;
        res.tracerstickernames.second = stickernames;
    }
    if(good_open)
    {
    // Mesh points
        {
            const vector<double> x = read_double_vector_from_hdf5(read_location, "X");
            const vector<double> y = read_double_vector_from_hdf5(read_location, "Y");
            const vector<double> z = read_double_vector_from_hdf5(read_location, "Z");
            res.mesh_points.resize(x.size());
            for(size_t i = 0; i < x.size(); ++i)
            {
                res.mesh_points.at(i) = Vector3D(x[i], y[i], z[i]);
            }
        }

        // Hydrodynamic
        {
            vector<double> Erad, temperature;
            const vector<double> density = read_double_vector_from_hdf5(read_location, "Density");
            if(H5Lexists(read_location.getId(), std::string("Erad").c_str(), H5P_DEFAULT ) > 0)
                Erad = read_double_vector_from_hdf5(read_location, "Erad");
            else
                Erad.resize(density.size(), 0);
            if(H5Lexists(read_location.getId(), std::string("Temperature").c_str(), H5P_DEFAULT ) > 0)
                temperature = read_double_vector_from_hdf5(read_location, "Temperature");
            else
                temperature.resize(density.size(), 0);
            const vector<double> pressure = read_double_vector_from_hdf5(read_location, "Pressure");
            const vector<double> energy = read_double_vector_from_hdf5(read_location, "InternalEnergy");
            vector<size_t> IDs(density.size(), 0);
            hsize_t objcount = read_location.getNumObjs();
            for(hsize_t i = 0; i < objcount; ++i)
            {
                std::string name = read_location.getObjnameByIdx(i);
                if(name.compare(std::string("ID")) == 0)
                {
                    IDs = read_sizet_vector_from_hdf5(read_location, "ID");
                }
            }
            const vector<double> x_velocity = read_double_vector_from_hdf5(read_location, "Vx");
            const vector<double> y_velocity = read_double_vector_from_hdf5(read_location, "Vy");
            const vector<double> z_velocity = read_double_vector_from_hdf5(read_location, "Vz");

            Group g_tracers = read_location.openGroup("tracers");
            Group g_stickers = read_location.openGroup("stickers");
            vector<vector<double>> tracers(g_tracers.getNumObjs());
            vector<string> tracernames(tracers.size());
            for(hsize_t n = 0; n < g_tracers.getNumObjs(); ++n)
            {
                const H5std_string name = g_tracers.getObjnameByIdx(n);
                tracernames[n] = name;
                tracers[n] = read_double_vector_from_hdf5(g_tracers, name);
            }

            vector<vector<int>> stickers(g_stickers.getNumObjs());
            vector<string> stickernames(stickers.size());
            for(hsize_t n = 0; n < g_stickers.getNumObjs(); ++n)
            {
                const H5std_string name = g_stickers.getObjnameByIdx(n);
                stickernames[n] = name;
                stickers[n] = read_int_vector_from_hdf5(g_stickers, name);
            }
            res.tracerstickernames.first = tracernames;
            res.tracerstickernames.second = stickernames;
            res.cells.resize(density.size());
            for(size_t i = 0; i < res.cells.size(); ++i)
            {
                res.cells.at(i).density = density.at(i);
                res.cells.at(i).Erad = Erad.at(i);
                res.cells.at(i).temperature = temperature.at(i);
                res.cells.at(i).pressure = pressure.at(i);
                res.cells.at(i).internal_energy = energy.at(i);
                res.cells.at(i).ID = IDs[i];
                res.cells.at(i).velocity.x = x_velocity.at(i);
                res.cells.at(i).velocity.y = y_velocity.at(i);
                res.cells.at(i).velocity.z = z_velocity.at(i);
                for(size_t j = 0; j < tracernames.size(); ++j)
                {
                    res.cells.at(i).tracers.at(j) = tracers.at(j).at(i);
                }
                for(size_t j = 0; j < stickernames.size(); ++j)
                {
                    res.cells.at(i).stickers.at(j) = (stickers.at(j).at(i) == 1);
                }
            }
        }

        // Volume
        {
            const vector<double> volume = read_double_vector_from_hdf5(read_location, "Volume");
            res.volumes = volume;
        }
    }

    return res;
}


Snapshot3D ReadSnapshot3D(const string &fname
#ifdef RICH_MPI
  , bool mpi_write, int fake_rank
#endif
)
{
    H5File file(fname, H5F_ACC_RDONLY);
    Group read_location = file.openGroup("/");
    bool good_open = true;
    #ifdef RICH_MPI
        if(mpi_write)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if(fake_rank >= 0)
            {
                rank = fake_rank;
            }
            try
            {
                read_location = file.openGroup("/rank" + std::to_string(rank));
            }
            catch (Exception& notFoundError)
            {
                good_open = false;
                read_location = file.openGroup("/rank" + std::to_string(0));
            }
        }
    #endif

    Snapshot3D res = ReadSnapshot3DHelper(file, file, read_location, good_open);

    read_location.close();
    file.close();

    return res;
}

#ifdef RICH_MPI
Snapshot3D ReadSnapshot3DParallel(const string &fname, int fake_rank)
{
    std::filesystem::path input_directory = std::filesystem::path(fname).replace_extension("");
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(fake_rank >= 0)
    {
        rank = fake_rank;
    }

    H5File file(input_directory / std::to_string(rank) / ".h5", H5F_ACC_RDONLY);
    Group read_location = file.openGroup("/");
    H5File globalfile(fname, H5F_ACC_RDONLY);

    Snapshot3D res = ReadSnapshot3DHelper(file, globalfile, read_location, true);

    globalfile.close();
    read_location.close();
    file.close();

    return res;
}
#endif // RICH_MPI