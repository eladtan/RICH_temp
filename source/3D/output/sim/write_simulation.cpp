#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include <filesystem>
#include "misc/universal_error.hpp"
#include "misc/memory_debug.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

#ifdef RICH_MPI
    #include <mpi.h>
    #include "3D/tessellation/loadBalancing/io/HilbertLoadBalancerIOHandler.hpp"
    #include "3D/tessellation/loadBalancing/io/LoadBalancerIOHandlerFactory.hpp"
    #include "3D/tessellation/voronoi/Voronoi3D.hpp"
    #include "3D/tessellation/voronoi/pointsManager/io/HilbertPointsManagerIOHandler.hpp"
    #include "3D/tessellation/voronoi/pointsManager/io/PointsManagerIOHandlerFactory.hpp"
    #include "3D/hilbert/io/RectangularConvertorIOHandler.hpp"
    #include "3D/hilbert/io/ConvertorIOHandlerFactory.hpp"
    #include "utils/hdf5/HDF5WriterParallel.hpp"
#endif

namespace fs = std::filesystem;

namespace
{
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
        HDF5WriterParallel pwriter(filename, MPI_COMM_WORLD);
        HDF5Writer writer(pwriter.GetFileId());

        std::string rankPrefix = pwriter.GetPrefix();

        writePrivateInfo(writer, rankPrefix, sim);
        writeTessellation(writer, rankPrefix + "/tess", sim);
        writePhysicsGroups(writer, rankPrefix, sim);

        if(rank == 0)
        {
            writeGeneralInfo(writer, sim);
            writeLoadBalancers(writer, sim);
        }
    }
    else
    #endif
    {
        HDF5Writer writer(filename);
        writeGeneralInfo(writer, sim);
        writePrivateInfo(writer, "", sim);
        writeTessellation(writer, "/tess", sim);
        writePhysicsGroups(writer, "", sim);
    }
    MEMORY_DEBUG_PRINT("WriteSimulation: after");
}
