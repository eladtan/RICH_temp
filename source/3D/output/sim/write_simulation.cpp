#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include <filesystem>
#include <thread>
#include <chrono>
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "3D/tessellation/io/load_balancing/HilbertLoadBalancerIOHandler.hpp"
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
    HDF5Writer openWriter(const std::string &filename)
    {
        for(int attempt = 1; attempt <= 50; ++attempt)
        {
            try
            {
                return HDF5Writer(filename);
            }
            catch(const H5::FileIException &)
            {
                if(attempt == 50)
                {
                    throw UniversalError("Failed to create HDF5 file after 50 attempts: " + filename);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
        throw UniversalError("Unreachable: openWriter");
    }

    void writeGeneralInfo(HDF5Writer &writer, const Simulation &sim)
    {
        const Tessellation3D &tess = sim.getTessellation();
        auto coords = tess.GetBoxCoordinates();
        BoundingBox<Vector3D> box(coords.first, coords.second);
        writer.WriteElement("/Box", box);
        writer.WriteElement("/Time", sim.GetTime());
        writer.WriteElement("/Cycle", sim.GetCycle());
        writer.WriteElement("/TimeStep", sim.GetTimeStep());
        writer.WriteElement("/WallclockTime", sim.GetWallclockTime());

        #ifdef RICH_MPI
        {
            std::vector<std::vector<std::string>> lb_table;
            for(const auto &step : sim.getPhysicsSteps())
            {
                lb_table.push_back({step->getName(), step->getRequiredLB()});
            }
            writer.WriteElement("/load_balance/names", lb_table);
            writer.WriteElement("/load_balance/current", sim.getCurrentLB());
        }
        #endif
    }

    #ifdef RICH_MPI
    void writeLoadBalancers(HDF5Writer &writer, const Simulation &sim)
    {
        auto loads = sim.GetLoads();
        for(const auto &[name, lb] : loads)
        {
            LoadBalancerIO::writeLoadBalancer(writer, "/load_balance/" + name, *lb);
        }
    }
    #endif

    void writeTessellation(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        const Tessellation3D &tess = sim.getTessellation();
        size_t N = tess.GetPointNo();

        writer.WriteSlice(prefix + "/mesh_points", tess.getMeshPoints(), N);
        writer.WriteSlice(prefix + "/volumes", tess.GetAllVolumes(), N);
        writer.WriteSlice(prefix + "/CM", tess.GetAllCM(), N);

#ifdef RICH_MPI
        const Voronoi3D *voronoi = dynamic_cast<const Voronoi3D*>(&tess);
        if(voronoi)
        {
            auto pm = voronoi->GetPointsManager();
            if(pm)
            {
                PointsManagerIO::writePointsManager(writer, prefix + "/points_manager", *pm);
            }
        }
#endif
    }

    void writePhysicsGroups(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        for(const auto &step : sim.getPhysicsSteps())
        {
            PhysicsStepIO::writeStep(writer, prefix, *step);
        }
    }

    void writePrivateInfo(HDF5Writer &writer, const std::string &prefix, const Simulation &sim)
    {
        const std::vector<ComputationalCell3D> &cells = sim.getCells();
        writer.WriteSlice(prefix + "/cells", cells, sim.getTessellation().GetPointNo());
    }
} // anonymous namespace

void WriteSimulation(const Simulation &sim, const std::string &filename
                     #ifdef RICH_MPI
                         , bool parallel
                     #endif
                     )
{
    MEMORY_DEBUG_PRINT("WriteSimulation: before");
    #ifdef RICH_MPI
        int rank = 0, ws = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &ws);
    #endif

    #ifdef RICH_MPI
    if(parallel)
    {
        fs::path path = fs::absolute(filename).parent_path();
        fs::path ranks_dir = path / fs::path(filename).filename().replace_extension();

        if(rank == 0)
        {
            if(fs::exists(ranks_dir))
            {
                fs::remove_all(ranks_dir);
            }
            fs::create_directory(ranks_dir);
        }
        MPI_Barrier(MPI_COMM_WORLD);

        std::string myFile = (ranks_dir / std::to_string(rank)).string() + ".h5";
        {
            HDF5Writer rankWriter = openWriter(myFile);
            writePrivateInfo(rankWriter, "", sim);
            writeTessellation(rankWriter, "/tess", sim);
            writePhysicsGroups(rankWriter, "", sim);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            HDF5Writer globalWriter = openWriter(filename);
            writeGeneralInfo(globalWriter, sim);
            writeLoadBalancers(globalWriter, sim);
            for(int r = 0; r < ws; ++r)
            {
                std::string relRankFile = ranks_dir.filename().string() + "/" + std::to_string(r) + ".h5";
                globalWriter.AddExternalLink(relRankFile, "/", "/rank" + std::to_string(r));
            }
        }
    }
    else
    #endif
    {
        HDF5Writer writer = openWriter(filename);
        writeGeneralInfo(writer, sim);
        writePrivateInfo(writer, "", sim);
        writeTessellation(writer, "/tess", sim);
        writePhysicsGroups(writer, "", sim);
    }
    MEMORY_DEBUG_PRINT("WriteSimulation: after");
}
