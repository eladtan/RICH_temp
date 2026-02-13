#include "read3D.hpp"

Snapshot3D ReadSnapshot3DHelper(const HDF5Reader &reader, const HDF5Reader &globalfile, bool const good_open)
{
    Snapshot3D res;
    std::vector<double> box;
    globalfile.ReadElement("/Box", box);

    res.ll.Set(box[0], box[1], box[2]);
    res.ur.Set(box[3], box[4], box[5]);
    // Misc
    {
        globalfile.ReadElement("/Time", res.time);
        globalfile.ReadElement("/Cycle", res.cycle);

        if(reader.Exists("/tracers"))
        {
            for(const std::string &tracerName : reader.ReadGroupNames("/tracers"))
            {
                res.tracerstickernames.first.push_back(tracerName);
            }   
        }
        if(reader.Exists("/stickers"))
        {
            for(const std::string &stickerName : reader.ReadGroupNames("/stickers"))
            {
                res.tracerstickernames.second.push_back(stickerName);
            }
        }
    }
    if(good_open)
    {
    // Mesh points
        {
            std::vector<double> x, y, z;
            reader.ReadElement("/X", x);
            reader.ReadElement("/Y", y);
            reader.ReadElement("/Z", z);
            res.mesh_points.resize(x.size());
            for(size_t i = 0; i < x.size(); ++i)
            {
                res.mesh_points.at(i) = Vector3D(x[i], y[i], z[i]);
            }
        }

        // Hydrodynamic
        {
            vector<double> density;
            reader.ReadElement("/Density", density);

            std::vector<double> Erad;
            if(reader.Exists("/Erad"))
            {
                reader.ReadElement("/Erad", Erad);
            }
            else
            {
                Erad.resize(density.size(), 0);
            }
            std::vector<double> temperature;
            if(reader.Exists("/Temperature"))
            {
                reader.ReadElement("/Temperature", temperature);
            }
            else
            {
                temperature.resize(density.size(), 0);
            }

            std::vector<double> pressure;
            reader.ReadElement("/Pressure", pressure);
            std::vector<double> energy;
            reader.ReadElement("/InternalEnergy", energy);

            vector<size_t> IDs(density.size(), 0);
            if(reader.Exists("/ID"))
            {
                reader.ReadElement("/ID", IDs);
            }

            std::vector<double> x_velocity;
            reader.ReadElement("/Vx", x_velocity);
            std::vector<double> y_velocity;
            reader.ReadElement("/Vy", y_velocity);
            std::vector<double> z_velocity;
            reader.ReadElement("/Vz", z_velocity);

            vector<vector<double>> tracers(res.tracerstickernames.first.size());

            for(size_t n = 0; n < res.tracerstickernames.first.size(); ++n)
            {
                std::string tracerName = res.tracerstickernames.first[n];
                reader.ReadElement("/tracers/" + tracerName, tracers[n]);
            }

            vector<vector<int>> stickers(res.tracerstickernames.second.size());
            for(size_t n = 0; n < res.tracerstickernames.second.size(); ++n)
            {
                std::string stickerName = res.tracerstickernames.second[n];
                reader.ReadElement("/stickers/" + stickerName, stickers[n]);
            }

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
                for(size_t j = 0; j < res.tracerstickernames.first.size(); ++j)
                {
                    res.cells.at(i).tracers.at(j) = tracers.at(j).at(i);
                }
                for(size_t j = 0; j < res.tracerstickernames.second.size(); ++j)
                {
                    res.cells.at(i).stickers.at(j) = (stickers.at(j).at(i) == 1);
                }
            }
            for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g)
            {
                if(reader.Exists("/Eg_" + std::to_string(g)))
                {
                    std::vector<double> Eg_temp;
                    reader.ReadElement("/Eg_" + std::to_string(g), Eg_temp);
                    for(size_t i = 0; i < res.cells.size(); ++i)
                        res.cells.at(i).Eg[g] = Eg_temp[i];
                }
                else
                {
                    if(ENERGY_GROUPS_NUM > 1)
                        throw std::runtime_error("Missing energy group Eg_" + std::to_string(g) + " in snapshot");
                }
            }
        }

        // Volume
        {
            std::vector<double> volume;
            reader.ReadElement("/Volume", volume);
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
    HDF5Reader globalfile(fname);
    
    #ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI
    
    std::shared_ptr<HDF5Reader> reader = nullptr;
    bool good_open = true;

    #ifdef RICH_MPI
        if(mpi_write)
        {
            int rank_to_read = (fake_rank >= 0)? fake_rank : rank;
            std::string dirname = std::filesystem::path(fname).replace_extension("").string();
            std::string rank_file = dirname + "/" + std::to_string(rank_to_read) + ".h5";
            if(not std::filesystem::exists(rank_file))
            {
                rank_file = dirname + "/0.h5";
                good_open = false;
            }
            reader = std::make_shared<HDF5Reader>(rank_file);
        }
        else
        {
            reader = std::make_shared<HDF5Reader>(fname);
        }
    #else // RICH_MPI
        reader = std::make_shared<HDF5Reader>(fname);
    #endif // RICH_MPI

    Snapshot3D res = ReadSnapshot3DHelper(*reader, globalfile, good_open);

    return res;
}

#ifdef RICH_MPI
Snapshot3D ReadSnapshot3DParallel(const string &fname, int fake_rank)
{
    return ReadSnapshot3D(fname, true, fake_rank);
}

rank_t GetNumberOfRanksInHDF(std::string const& fname)
{
    HDF5Reader file(fname);
    int counter = 0;
    std::vector<std::string> groupNames = file.ReadGroupNames("/");
    std::set<std::string> groupNamesSet(groupNames.cbegin(), groupNames.cend());

    while(true)
    {
        if(groupNamesSet.find("rank" + std::to_string(counter)) == groupNamesSet.end())
        {
            break;
        }
        ++counter;
    }
    return counter;
}

#endif // RICH_MPI