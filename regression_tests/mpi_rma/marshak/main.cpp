// Parallel timing driver outside STORM; reuses the existing Marshak problem 2.
#include <mpi.h>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include "examples/marshak_wave/MarshakCommon.hpp"
#include "manager/MonteCarloManagerFactory.hpp"

using Grid = STORM::examples::MarshakGrid;
using namespace STORM::examples;
using IMC = STORM::RadiationIMC<Vector3D, Grid, STORM::RadiationCell,
    STORM::SimpleExtensives, MarshakEOS, 1, MarshakOpacity<Vector3D, Grid>>;

class AuditedBoundary : public MarshakBoundary<Vector3D, Grid>
{
public:
    using MarshakBoundary<Vector3D, Grid>::MarshakBoundary;
    double injected = 0, escaped = 0;
    STORM::DeviceBoundaryFaceBehavior getDeviceBoundaryFaceBehavior(
        std::size_t face, std::size_t, std::size_t) const override
    {
        // Keep the emitting/escaping x-min face on the host for its ledger.
        // All other faces have the existing Marshak specular-reflection rule.
        const auto &box = this->grid.GetBoxCoordinates();
        if(std::abs(this->grid.FaceCM(face).x - box.first.x) <= 1e-12)
            return STORM::DeviceBoundaryFaceBehavior::HostOnly;
        return STORM::DeviceBoundaryFaceBehavior::ReflectingRigid;
    }
    std::vector<STORM::Particle<Vector3D>> generateNewBoundaryParticles(double dt) override
    {
        auto packets = MarshakBoundary::generateNewBoundaryParticles(dt);
        for(const auto &p : packets) injected += p.weight;
        return packets;
    }
    STORM::ParticleStatus apply(STORM::Particle<Vector3D> &p) override
    {
        auto status = MarshakBoundary::apply(p);
        if(status == STORM::ParticleStatus::REMOVE) escaped += p.weight;
        return status;
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    try
    {
        // nx, new/cell, boundary packets, dt_factor, backend, profile path.
        if(argc != 7) throw std::runtime_error("usage: marshak nx new boundary dt_factor {p2p|ofi|mpi} profile");
        const std::size_t nx = std::stoul(argv[1]), newPerCell = std::stoul(argv[2]);
        const std::size_t boundaryPackets = std::stoul(argv[3]);
        const double dtFactor = std::stod(argv[4]);
        const std::string backend = argv[5], output = argv[6];
        if(nx < static_cast<std::size_t>(ranks) || newPerCell == 0 || boundaryPackets == 0 ||
           !std::isfinite(dtFactor) || dtFactor <= 0 ||
           (backend != "p2p" && backend != "ofi" && backend != "mpi"))
            throw std::runtime_error("invalid benchmark arguments");
        {
            const auto params = GetProblemParams(2);
            const double dy = params.domainLength / nx;
            Grid grid(Vector3D(0, 0, 0), Vector3D(params.domainLength, dy, dy), nx, 1, 1);
            grid.BuildParallel(std::vector<double>(nx, 1.0));
            const std::size_t n = grid.GetPointNo();
            std::vector<STORM::RadiationCell> cells(n);
            std::vector<STORM::SimpleExtensives> extensives(n);
            std::vector<double> densities(n, 1.0);
            double initialMaterial = 0;
            for(std::size_t i = 0; i < n; ++i)
            {
                cells[i].temperature = 1e-2 * units::kev_kelvin;
                cells[i].internalEnergy = EOS_E_from_T(params, cells[i].temperature, 1.0) * grid.GetVolume(i);
                extensives[i].mass = grid.GetVolume(i);
                extensives[i].internal_energy = cells[i].internalEnergy;
                initialMaterial += cells[i].internalEnergy;
            }
            STORM::RadiationIMCParameters<1> settings;
            settings.newPhotonsPerCell = newPerCell;
            settings.withRandomWalk = true;
            // Exact transverse folding for the reflecting one-dimensional box.
            settings.withSlabTransport = true;
            settings.energyBoundaries = {0.0, 1e30};
            settings.energyBoundariesProvided = true;
            auto eos = std::make_shared<MarshakEOS>(params);
            auto opacity = std::make_shared<MarshakOpacity<Vector3D, Grid>>(
                params.kappaP0, params.kappaR0, params.alpha, params.betaRho, densities, cells);
            auto boundary = std::make_shared<AuditedBoundary>(grid, BathTemperature(params, params.initialDt), boundaryPackets);
            auto physics = std::make_shared<IMC>(grid, boundary, cells, extensives, eos, opacity, settings);
            std::shared_ptr<STORM::PopulationControl<Vector3D, Grid>> pop =
                std::make_shared<STORM::CombPopulationControl<Vector3D, Grid>>(grid, 4, 6.0);
            std::shared_ptr<STORM::BoundaryCondition<Vector3D, Grid>> bc = boundary;
            auto manager = STORM::CreateMonteCarloManager<Vector3D, Grid>(grid, physics, pop, bc,
                backend == "p2p" ? STORM::ManagerType::P2P : STORM::ManagerType::RDMA,
                backend == "mpi" ? STORM::RDMAEngine::MPI : STORM::RDMAEngine::OFI);
            manager.getParticles().clear();
            if(rank == 0)
                std::cout << "Marshak comparison: problem=2 nx=" << nx << " ranks=" << ranks
                          << " new=" << newPerCell << " boundary=" << boundaryPackets
                          << " dt_factor=" << dtFactor << " census_target=4 slab=1 backend=" << backend << std::endl;
            MPI_Barrier(MPI_COMM_WORLD);
            const auto start = std::chrono::steady_clock::now();
            double time = 0, dt = params.initialDt * dtFactor;
            std::size_t cycles = 0;
            while(time < params.tf)
            {
                boundary->SetTemperature(BathTemperature(params, std::max(time, params.initialDt)));
                dt = std::min(dt, params.tf - time);
                manager.step(dt);
                time += dt;
                ++cycles;
                dt = std::min(std::max(params.initialDt, time * 1e-3) * dtFactor, 5e-11 * dtFactor);
            }
            MPI_Barrier(MPI_COMM_WORLD);
            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            double maxElapsed = 0;
            MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            double ledger[5] = {initialMaterial, 0, 0, boundary->injected, boundary->escaped};
            int valid = 1;
            for(const auto &cell : cells)
            {
                ledger[1] += cell.internalEnergy;
                if(!std::isfinite(cell.temperature) || cell.temperature <= 0 ||
                   !std::isfinite(cell.internalEnergy) || cell.internalEnergy < 0) valid = 0;
            }
            for(const auto &packet : std::as_const(manager).getParticles()) ledger[2] += packet.weight;
            MPI_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            double totals[5] = {};
            MPI_Reduce(ledger, totals, 5, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            const auto &rad = physics->getEradTimeAvg();
            std::vector<double> local(3*n);
            for(std::size_t i = 0; i < n; ++i)
            {
                local[3*i] = grid.GetCellCM(i).x;
                local[3*i+1] = cells[i].temperature;
                local[3*i+2] = std::pow(std::max(rad[i], 0.0) / units::arad, 0.25);
            }
            int count = static_cast<int>(local.size());
            std::vector<int> counts(ranks), displacements(ranks);
            MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
            int total = 0;
            if(rank == 0) for(int i = 0; i < ranks; ++i) { displacements[i] = total; total += counts[i]; }
            std::vector<double> all(static_cast<std::size_t>(total));
            MPI_Gatherv(local.data(), count, MPI_DOUBLE, all.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            if(rank == 0)
            {
                if(total != static_cast<int>(3*nx)) throw std::runtime_error("global cell count mismatch");
                std::vector<std::array<double,3>> profile(nx);
                for(std::size_t i = 0; i < nx; ++i) profile[i] = {all[3*i], all[3*i+1], all[3*i+2]};
                std::sort(profile.begin(), profile.end());
                std::vector<double> x, temperature;
                std::ofstream stream(output);
                stream << std::scientific << std::setprecision(12);
                for(const auto &row : profile)
                {
                    stream << row[0] << ' ' << row[1] << ' ' << row[2] << '\n';
                    x.push_back(row[0]); temperature.push_back(row[1]);
                }
                if(!stream) throw std::runtime_error("cannot write profile");
                const auto reference = LoadReference(MARSHAK_REFERENCE);
                if(reference.empty()) throw std::runtime_error("missing Marshak diffusion reference");
                const double l1 = ComputeL1(x, temperature, reference, true);
                const double residual = (totals[0] + totals[3] - totals[4] - totals[1] - totals[2]) / totals[3];
                std::cout << std::setprecision(12)
                          << "MARSHAK_RESULT cycles=" << cycles << " time=" << time
                          << " seconds=" << maxElapsed << " energy_residual=" << residual
                          << " diffusion_l1=" << l1 << " valid=" << valid << std::endl;
                if(!valid || !std::isfinite(residual) || std::abs(residual) > 1e-6 || !std::isfinite(l1))
                    throw std::runtime_error("invalid temperatures or energy balance");
            }
        } // Destroy all MPI-owning objects before finalizing.
        MPI_Finalize();
        return 0;
    }
    catch(const std::exception &error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
