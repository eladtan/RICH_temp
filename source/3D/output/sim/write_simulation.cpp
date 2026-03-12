#include "newtonian/three_dimensional/simulation/Simulation.hpp"
#include "3D/output/cellData.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/HydroStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/RadiationMCStepIOHandler.hpp"
#include "newtonian/three_dimensional/simulation/steps/io/PhysicsStepIOHandlerFactory.hpp"
#include <filesystem>
#include "misc/universal_error.hpp"
#include "newtonian/three_dimensional/computational_cell.hpp"

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
    void writeGeneralInfo(HDF5Writer &writer, const Simulation &sim)
    {
        const Tessellation3D &tess = sim.getTessellation();
        auto coords = tess.GetBoxCoordinates();
        BoundingBox<Vector3D> box(coords.first, coords.second);
        writer.WriteElement("/Box", box);
        writer.WriteElement("/Time", sim.GetTime());
        writer.WriteElement("/Cycle", sim.GetCycle());
        writer.WriteElement("/TimeStep", sim.GetTimeStep());

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
            auto kernel = voronoi->GetKernel();
            if(kernel)
            {
                KernelIO::writeKernel(writer, prefix + "/kernel", *kernel);
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
            HDF5Writer rankWriter(myFile);
            writePrivateInfo(rankWriter, "", sim);
            writeTessellation(rankWriter, "/tess", sim);
            writePhysicsGroups(rankWriter, "", sim);
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if(rank == 0)
        {
            HDF5Writer globalWriter(filename);
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
        HDF5Writer writer(filename);
        writeGeneralInfo(writer, sim);
        writePrivateInfo(writer, "", sim);
        writeTessellation(writer, "/tess", sim);
        writePhysicsGroups(writer, "", sim);
    }
}
