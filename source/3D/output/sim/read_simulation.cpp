#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include <filesystem>
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include "misc/universal_error.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "3D/tessellation/loadBalancing/io/HilbertLoadBalancerIOHandler.hpp"
    #include "3D/tessellation/loadBalancing/io/LoadBalancerIOHandlerFactory.hpp"
    #include "3D/tessellation/voronoi/Voronoi3D.hpp"
    #include "3D/tessellation/voronoi/pointsManager/io/HilbertPointsManagerIOHandler.hpp"
    #include "3D/tessellation/voronoi/pointsManager/io/PointsManagerIOHandlerFactory.hpp"
    #include "3D/hilbert/io/RectangularConvertorIOHandler.hpp"
    #include "3D/hilbert/io/ConvertorIOHandlerFactory.hpp"
    #include "utils/hdf5/HDF5ReaderParallel.hpp"
#endif

namespace fs = std::filesystem;

namespace
{
    void readGeneralInfo(const HDF5Reader &reader, Simulation &sim)
    {
        BoundingBox<Vector3D> box;
        reader.ReadElement("/Box", box);
        sim.getTessellation().SetBox(box.getLL(), box.getUR());

        double time = 0;
        reader.ReadElement("/Time", time);
        sim.SetTime(time);

        size_t cycle = 0;
        reader.ReadElement("/Cycle", cycle);
        sim.SetCycle(cycle);

        if(reader.Exists("/TimeStep"))
        {
            double dt = 0;
            reader.ReadElement("/TimeStep", dt);
            sim.SetTimeStep(dt);
        }

        if(reader.Exists("/WallclockTime"))
        {
            double wct = 0;
            reader.ReadElement("/WallclockTime", wct);
            sim.SetWallclockTime(wct);
        }
    }

    #ifdef RICH_MPI
    std::string readLoadBalancers(const HDF5Reader &reader, Simulation &sim)
    {
        std::string currentLBName;

        if(!reader.Exists("/load_balance"))
        {
            return currentLBName;
        }

        if(reader.Exists("/load_balance/current"))
        {
            reader.ReadElement("/load_balance/current", currentLBName);
        }

        auto lbEntries = reader.ReadGroupNames("/load_balance");
        for(const auto &name : lbEntries)
        {
            std::string group = "/load_balance/" + name;
            if(!reader.Exists(group + "/type"))
            {
                continue;
            }

            auto lb = LoadBalancerIO::readLoadBalancer(reader, group);
            sim.storeLoadBalance(name, lb);
        }

        return currentLBName;
    }
    #endif

    void readTessellation(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        Tessellation3D &tess = sim.getTessellation();


        if(reader.Exists(prefix + "/volumes"))
        {
            std::vector<double> vols;
            reader.ReadElement(prefix + "/volumes", vols);
        }
        if(reader.Exists(prefix + "/CM"))
        {
            std::vector<Vector3D> cm;
            reader.ReadElement(prefix + "/CM", cm);
        }

#ifdef RICH_MPI
        if(reader.Exists(prefix + "/points_manager/type"))
        {
            Voronoi3D *voronoi = dynamic_cast<Voronoi3D *>(&tess);
            if(voronoi)
            {
                auto coords = tess.GetBoxCoordinates();
                auto pm = PointsManagerIO::readPointsManager(reader, prefix + "/points_manager", coords.first, coords.second);
                voronoi->SetPointsManager(pm);
            }
        }
#endif

        int rank = 0;
        #ifdef RICH_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        #endif // RICH_MPI

        if(reader.Exists(prefix + "/mesh_points"))
        {
            std::vector<Vector3D> points;
            reader.ReadElement(prefix + "/mesh_points", points);
            #ifdef RICH_MPI
                std::cout << "Rank " << rank << " has " << points.size() << " points read." << std::endl;
                tess.BuildParallel(points, true, true);
            #else
                tess.Build(points);
            #endif
        }

        std::cout << "After first build, rank " << rank << " has " <<  tess.GetPointNo() << " points" << std::endl;
    }

    void readPhysicsGroups(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        for(auto &step : sim.getPhysicsSteps())
        {
            PhysicsStepIO::readStep(reader, prefix, *step);
        }
    }

    void readPrivateInfo(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        if(!reader.Exists(prefix + "/cells"))
        {
            return;
        }
        std::vector<ComputationalCell3D> cells;
        reader.ReadElement(prefix + "/cells", cells);
        sim.getCells() = std::move(cells);
    }
}

void ReadSimulation(const std::string &filename,
                    Simulation &sim
                    #ifdef RICH_MPI
                        , bool parallel
                        , int fake_rank
                    #endif
                    )
{
    #ifdef RICH_MPI
    std::string currentLBName;
    if(parallel)
    {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rank_to_read = (fake_rank >= 0) ? fake_rank : rank;

        // Backward compatibility: detect old per-rank file format
        std::string dir = fs::path(filename).replace_extension("").string();
        bool oldFormat = fs::exists(dir + "/0.h5");

        if(oldFormat)
        {
            std::string rankFile = dir + "/" + std::to_string(rank_to_read) + ".h5";
            if(!fs::exists(rankFile))
            {
                throw UniversalError("ReadSimulation: rank file not found: " + rankFile);
            }

            HDF5Reader globalReader(filename);
            readGeneralInfo(globalReader, sim);
            currentLBName = readLoadBalancers(globalReader, sim);

            HDF5Reader dataReader(rankFile);
            readTessellation(dataReader, "/tess", sim);
            readPhysicsGroups(dataReader, "", sim);
            readPrivateInfo(dataReader, "", sim);
        }
        else
        {
            HDF5ReaderParallel preader(filename, MPI_COMM_WORLD);
            HDF5Reader reader(preader.GetFileId());

            readGeneralInfo(reader, sim);
            currentLBName = readLoadBalancers(reader, sim);

            std::string rankPrefix = "/rank" + std::to_string(rank_to_read);
            readTessellation(reader, rankPrefix + "/tess", sim);
            readPhysicsGroups(reader, rankPrefix, sim);
            readPrivateInfo(reader, rankPrefix, sim);
        }

        if(!currentLBName.empty())
        {
            auto loads = sim.GetLoads();
            for(const auto &[name, lb] : loads)
            {
                if(name == currentLBName)
                {
                    sim.getTessellation().PresetLoadBalancer(lb);
                    break;
                }
            }
            sim.PresetLoadBalance(currentLBName);
        }
    }
    else
    #endif
    {
        HDF5Reader reader(filename);
        readGeneralInfo(reader, sim);
        readTessellation(reader, "/tess", sim);
        readPhysicsGroups(reader, "", sim);
        readPrivateInfo(reader, "", sim);
    }

    sim.recomputeMaxID();
}
