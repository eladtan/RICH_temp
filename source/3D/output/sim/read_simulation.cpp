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
    #include "3D/environment/kernels/io/KernelIOHandlerFactory.hpp"
    #include "3D/tessellation/voronoi/Voronoi3D.hpp"
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
    }

    #ifdef RICH_MPI
    void readLoadBalancers(const HDF5Reader &reader, Simulation &sim)
    {
        std::string currentLBName;
        reader.ReadElement("/load_balance/current", currentLBName);

        if(!reader.Exists("/load_balance"))
        {
            return;
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

        sim.setCurrentLoadBalance(currentLBName);
    }
    #endif

    void readTessellation(const HDF5Reader &reader, const std::string &prefix, Simulation &sim)
    {
        Tessellation3D &tess = sim.getTessellation();

        if(reader.Exists(prefix + "/mesh_points"))
        {
            std::vector<Vector3D> points;
            reader.ReadElement(prefix + "/mesh_points", points);
            #ifdef RICH_MPI
                tess.BuildParallel(points, true, true);
            #else
                tess.Build(points);
            #endif
        }
        if(reader.Exists(prefix + "/volumes"))
        {
            std::vector<double> vols;
            reader.ReadElement(prefix + "/volumes", vols);
            tess.GetAllVolumes() = std::move(vols);
        }
        if(reader.Exists(prefix + "/CM"))
        {
            std::vector<Vector3D> cm;
            reader.ReadElement(prefix + "/CM", cm);
            tess.GetAllCM() = std::move(cm);
        }

#ifdef RICH_MPI
        if(reader.Exists(prefix + "/kernel/type"))
        {
            auto kernel = KernelIO::readKernel(reader, prefix + "/kernel");
            Voronoi3D *voronoi = dynamic_cast<Voronoi3D *>(&tess);
            if(voronoi)
            {
                voronoi->SetKernel(kernel);
            }
        }
#endif
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
    HDF5Reader globalReader(filename);
    readGeneralInfo(globalReader, sim);

    std::shared_ptr<HDF5Reader> dataReader;

    #ifdef RICH_MPI
    if(parallel)
    {
        readLoadBalancers(globalReader, sim);

        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rank_to_read = (fake_rank >= 0) ? fake_rank : rank;

        std::string dir = fs::path(filename).replace_extension("").string();
        std::string rankFile = dir + "/" + std::to_string(rank_to_read) + ".h5";
        if(!fs::exists(rankFile))
        {
            throw UniversalError("ReadSimulation: rank file not found: " + rankFile);
        }

        dataReader = std::make_shared<HDF5Reader>(rankFile);
    }
    else
    #endif
    {
        dataReader = std::make_shared<HDF5Reader>(filename);
    }

    readTessellation(*dataReader, "/tess", sim);
    readPhysicsGroups(*dataReader, "", sim);
    readPrivateInfo(*dataReader, "", sim);
}
