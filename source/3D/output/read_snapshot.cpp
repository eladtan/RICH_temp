#include "misc/universal_error.hpp"
#include "read3D.hpp"
#include <iostream>
#ifdef RICH_MPI
#include <mpi.h>
#endif

Snapshot3D ReadSnapshot3DHelper(const HDF5Reader &reader,
                                const HDF5Reader &globalfile,
                                bool const good_open,
                                const std::string& rank_prefix = "")
{
    Snapshot3D res;
    const auto rank_path = [&rank_prefix](const std::string& path)
    {
        return rank_prefix.empty() ? path : rank_prefix + path;
    };
    std::vector<double> box;
    globalfile.ReadElement("/Box", box);

    res.ll.Set(box[0], box[1], box[2]);
    res.ur.Set(box[3], box[4], box[5]);
    // Misc
    {
        try
        {
            globalfile.ReadElement("/Time", res.time);
            globalfile.ReadElement("/Cycle", res.cycle);
        }
        catch(...)
        {
            std::vector<double> time;
            std::vector<int> cycle;
            globalfile.ReadElement("/Time", time);
            globalfile.ReadElement("/Cycle", cycle);
            res.time = time[0];
            res.cycle = cycle[0];
        }

        if(reader.Exists(rank_path("/tracers")))
        {
            for(const std::string &tracerName : reader.ReadGroupNames(rank_path("/tracers")))
            {
                res.tracerstickernames.first.push_back(tracerName);
            }   
        }
        if(reader.Exists(rank_path("/stickers")))
        {
            for(const std::string &stickerName : reader.ReadGroupNames(rank_path("/stickers")))
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
            reader.ReadElement(rank_path("/X"), x);
            reader.ReadElement(rank_path("/Y"), y);
            reader.ReadElement(rank_path("/Z"), z);
            res.mesh_points.resize(x.size());
            for(size_t i = 0; i < x.size(); ++i)
            {
                res.mesh_points.at(i) = Vector3D(x[i], y[i], z[i]);
            }
        }

        // Hydrodynamic
        {
            vector<double> density;
            reader.ReadElement(rank_path("/Density"), density);

            std::vector<double> Erad;
            if(reader.Exists(rank_path("/Erad")))
            {
                reader.ReadElement(rank_path("/Erad"), Erad);
            }
            else
            {
                Erad.resize(density.size(), 0);
            }
            std::vector<double> temperature;
            if(reader.Exists(rank_path("/Temperature")))
            {
                reader.ReadElement(rank_path("/Temperature"), temperature);
            }
            else
            {
                temperature.resize(density.size(), 0);
            }

            std::vector<double> pressure;
            reader.ReadElement(rank_path("/Pressure"), pressure);
            std::vector<double> energy;
            reader.ReadElement(rank_path("/InternalEnergy"), energy);

            vector<size_t> IDs(density.size(), 0);
            if(reader.Exists(rank_path("/ID")))
            {
                reader.ReadElement(rank_path("/ID"), IDs);
            }

            std::vector<double> x_velocity;
            reader.ReadElement(rank_path("/Vx"), x_velocity);
            std::vector<double> y_velocity;
            reader.ReadElement(rank_path("/Vy"), y_velocity);
            std::vector<double> z_velocity;
            reader.ReadElement(rank_path("/Vz"), z_velocity);

            vector<vector<double>> tracers(res.tracerstickernames.first.size());

            for(size_t n = 0; n < res.tracerstickernames.first.size(); ++n)
            {
                std::string tracerName = res.tracerstickernames.first[n];
                reader.ReadElement(rank_path("/tracers/" + tracerName), tracers[n]);
            }

            vector<vector<int>> stickers(res.tracerstickernames.second.size());
            for(size_t n = 0; n < res.tracerstickernames.second.size(); ++n)
            {
                std::string stickerName = res.tracerstickernames.second[n];
                reader.ReadElement(rank_path("/stickers/" + stickerName), stickers[n]);
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
                if(reader.Exists(rank_path("/Eg_" + std::to_string(g))))
                {
                    std::vector<double> Eg_temp;
                    reader.ReadElement(rank_path("/Eg_" + std::to_string(g)), Eg_temp);
                    for(size_t i = 0; i < res.cells.size(); ++i)
                        res.cells.at(i).Eg[g] = Eg_temp[i];
                }
                else
                {
                    for(size_t i = 0; i < res.cells.size(); ++i)
                        res.cells.at(i).Eg[g] = 0;
                    static bool warned = false;
                    if(!warned)
                    {
                        int my_rank = 0;
#ifdef RICH_MPI
                        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
#endif
                        if(my_rank == 0)
                            std::cerr << "[Warning] Snapshot missing Eg_" << g
                                      << "; setting Eg to 0 for all cells" << std::endl;
                        warned = true;
                    }
                }
            }
        }

        // Volume
        {
            std::vector<double> volume;
            reader.ReadElement(rank_path("/Volume"), volume);
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
    #ifdef RICH_MPI
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    #endif // RICH_MPI

    HDF5Reader globalfile(fname);
    
    std::shared_ptr<HDF5Reader> reader = nullptr;
    bool good_open = true;
    std::string rank_prefix;

    #ifdef RICH_MPI
        if(mpi_write)
        {
            int rank_to_read = (fake_rank >= 0)? fake_rank : rank;
            std::string dirname = std::filesystem::path(fname).replace_extension("").string();
            std::string rank_file = dirname + "/" + std::to_string(rank_to_read) + ".h5";
            const std::string embedded_rank_group = "/rank" + std::to_string(rank_to_read);
            if(std::filesystem::exists(rank_file))
            {
                reader = std::make_shared<HDF5Reader>(rank_file);
            }
            else if(globalfile.Exists(embedded_rank_group))
            {
                reader = std::make_shared<HDF5Reader>(fname);
                rank_prefix = embedded_rank_group;
            }
            else
            {
                rank_file = dirname + "/0.h5";
                good_open = false;
                reader = std::make_shared<HDF5Reader>(rank_file);
            }
        }
        else
        {
            reader = std::make_shared<HDF5Reader>(fname);
        }
    #else // RICH_MPI
        reader = std::make_shared<HDF5Reader>(fname);
    #endif // RICH_MPI

    Snapshot3D res = ReadSnapshot3DHelper(*reader, globalfile, good_open, rank_prefix);

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