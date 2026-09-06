#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "newtonian/three_dimensional/conserved_3d.hpp"
#include "3D/output/cellData.hpp"
#include <filesystem>
#include <thread>
#include <chrono>
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include "misc/universal_error.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "3D/tessellation/io/load_balancing/HilbertLoadBalancerIOHandler.hpp"
    #include "3D/tessellation/io/load_balancing/OneDimensionalLoadBalancerIOHandler.hpp"
    #include "3D/tessellation/io/load_balancing/LoadBalancerIOHandlerFactory.hpp"
    #include "3D/tessellation/Voronoi3D.hpp"
    #include "3D/tessellation/io/points_manager/HilbertPointsManagerIOHandler.hpp"
    #include "3D/tessellation/io/points_manager/PointsManagerIOHandlerFactory.hpp"
    #include "3D/tessellation/io/hilbert/RectangularConvertorIOHandler.hpp"
    #include "3D/tessellation/io/hilbert/ConvertorIOHandlerFactory.hpp"
#endif

namespace fs = std::filesystem;

namespace
{
    void openReader(HDF5Reader &reader, const std::string &filename)
    {
        for(int attempt = 1; attempt <= 50; ++attempt)
        {
            try
            {
                reader.Load(filename);
                return;
            }
            catch(const H5::FileIException &)
            {
                if(attempt == 50)
                {
                    throw UniversalError("Failed to open HDF5 file after 50 attempts: " + filename);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }

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

    void readTessellation(const HDF5Reader &reader, const std::string &prefix, Simulation &sim
                          #ifdef RICH_MPI
                              , const std::shared_ptr<LoadBalancer<Vector3D>> &loadBalancer
                          #endif
                          )
    {
        Tessellation3D &tess = sim.getTessellation();

        if(reader.Exists(prefix + "/volumes"))
        {
            std::vector<double> vols;
            reader.ReadElement(prefix + "/volumes", vols);
            // tess.GetAllVolumes() = std::move(vols);
        }
        if(reader.Exists(prefix + "/CM"))
        {
            std::vector<Vector3D> cm;
            reader.ReadElement(prefix + "/CM", cm);
            // tess.GetAllCM() = std::move(cm);
        }

#ifdef RICH_MPI
        if(reader.Exists(prefix + "/points_manager/type"))
        {
            Voronoi3D *voronoi = dynamic_cast<Voronoi3D *>(&tess);
            if(voronoi)
            {
                std::pair<Vector3D, Vector3D> coords = tess.GetBoxCoordinates();
                std::shared_ptr<PointsManagerIO::PointsManager> pm =
                    PointsManagerIO::readPointsManager(reader, prefix + "/points_manager", coords.first, coords.second);
                voronoi->SetPointsManager(pm);
            }
        }
        if(loadBalancer != nullptr)
        {
            tess.PresetLoadBalancer(loadBalancer);
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

    void syncRestartState(Simulation &sim)
    {
        Tessellation3D &tess = sim.getTessellation();
        std::vector<ComputationalCell3D> &cells = sim.getCells();
        std::vector<Conserved3D> &extensives = sim.getExtensives();
        extensives.resize(cells.size());
        const std::size_t ownedCells = tess.GetPointNo();
        for(std::size_t cellIndex = 0; cellIndex < ownedCells; ++cellIndex)
        {
            PrimitiveToConserved(cells[cellIndex], tess.GetVolume(cellIndex), extensives[cellIndex]);
        }

        #ifdef RICH_MPI
            sim.buildDataTransfer();
            for(const std::shared_ptr<PhysicsStep> &step : sim.getPhysicsSteps())
            {
                step->afterLB();
            }
        #endif
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
    HDF5Reader globalReader;
    openReader(globalReader, filename);
    readGeneralInfo(globalReader, sim);

    std::shared_ptr<HDF5Reader> dataReader;

    #ifdef RICH_MPI
    std::string currentLBName;
    std::shared_ptr<LoadBalancer<Vector3D>> currentLoadBalancer;
    if(parallel)
    {
        currentLBName = readLoadBalancers(globalReader, sim);
        std::vector<std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>>> loads = sim.GetLoads();
        for(const std::pair<std::string, std::shared_ptr<LoadBalancer<Vector3D>>> &entry : loads)
        {
            if(entry.first == currentLBName)
            {
                currentLoadBalancer = entry.second;
                break;
            }
        }

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rank_to_read = (fake_rank >= 0) ? fake_rank : rank;

        std::string dir = fs::path(filename).replace_extension("").string();
        std::string rankFile = dir + "/" + std::to_string(rank_to_read) + ".h5";
        if(!fs::exists(rankFile))
        {
            throw UniversalError("ReadSimulation: rank file not found: " + rankFile);
        }

        dataReader = std::make_shared<HDF5Reader>();
        openReader(*dataReader, rankFile);
    }
    else
    #endif
    {
        dataReader = std::make_shared<HDF5Reader>();
        openReader(*dataReader, filename);
    }

    readTessellation(*dataReader, "/tess", sim
                     #ifdef RICH_MPI
                         , currentLoadBalancer
                     #endif
                     );

    #ifdef RICH_MPI
    if(parallel && !currentLBName.empty())
    {
        sim.PresetLoadBalance(currentLBName);
    }
    #endif

    readPhysicsGroups(*dataReader, "", sim);
    readPrivateInfo(*dataReader, "", sim);
    sim.recomputeMaxID();
    syncRestartState(sim);
}
