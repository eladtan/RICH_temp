// Subsonic ablative heat wave in gold — Case 1 (tau = 0), MMC radiation + pressure gradient
// Same physics/setup as runs/gold_heat_wave_tau0_imc/ but with MMC and diffusion pressure gradient.
// From simulation_setup.pdf:
//   gamma = 5/4, rho0 = 19.32 g/cm^3
//   e(T,rho) = 3.4e13 * (T/HeV)^1.6 * (rho)^{-0.14} erg/g
//   kappa_R = 7200 * (T/HeV)^{-1.5} * rho^{0.2} cm^2/g  (all absorption)
//   T_s(t) = 1 HeV,  T_bath(t) = [1 + 0.169141*(t/ns)^{-79/192}]^{1/4} * T_s
//   m_front(t) = 1.01889e-3 * (t/ns)^{33/64} g/cm^2
//   t_end = 2.0 ns

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/HydroStep.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationMCStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/Lagrangian3D.hpp"
#include "source/newtonian/three_dimensional/RoundCells3D.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CFL1D.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/SourceTerm3D.hpp"
#include "source/newtonian/three_dimensional/conserved_3d.hpp"
#include "source/3D/radiation/RadiationIMC.hpp"
#include "source/3D/radiation/PowerLawOpacity.hpp"
#include "source/monte/boundary/SideTemperature.hpp"
#include "source/monte/population/Comb.hpp"
#include <boost/math/special_functions/pow.hpp>
#include "source/Radiation/CMMC/src/units/units.hpp"
#include "source/3D/tessellation/utils/RandomOnFace.hpp"
#include "source/3D/output/write3D.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>
#include <fenv.h>
#include <libgen.h>
#include <string.h>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {


class XOnlyMotion3D : public PointMotion3D
{
public:
	explicit XOnlyMotion3D(const PointMotion3D& base) : base_(base) {}

	void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		double time, vector<Vector3D>& res) const override
	{
		base_(tess, cells, time, res);
		for (auto& v : res) { v.y = 0; v.z = 0; }
	}

	void ApplyFix(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
		double time, double dt, vector<Vector3D>& velocities) const override
	{
		base_.ApplyFix(tess, cells, time, dt, velocities);
		for (auto& v : velocities) { v.y = 0; v.z = 0; }
	}

private:
	const PointMotion3D& base_;
};

#ifdef RICH_MPI
class MCStepCostCalculator : public CostCalculator3D
{
public:
	explicit MCStepCostCalculator(const std::shared_ptr<MonteCarloManager3D> &manager) : manager_(manager) {}

	std::vector<double> CalculateCost(const Tessellation3D &tess, const vector<ComputationalCell3D> &) const override
	{
		size_t const N = tess.GetPointNo();
		const std::vector<size_t> &counters = manager_->GetCellsStepsCounters();
		std::vector<double> weights(N, 0.01);
		for (size_t j = 0; j < std::min(N, counters.size()); ++j)
			weights[j] = std::max(0.01, static_cast<double>(counters[j]));
		return weights;
	}

private:
	const std::shared_ptr<MonteCarloManager3D> manager_;
};
#endif

}  // namespace

int main(int argc, char* argv[])
{
#ifdef RICH_MPI
	MPI_Init(&argc, &argv);
#endif
	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

	// Energy groups (gray IMC — boundaries required, groups unused)
	size_t const G = ENERGY_GROUPS_NUM;
	double const Emin = units::kev * 1e-4;
	double const Emax = units::kev * 1e3;
	for (size_t g = 1; g <= G; ++g)
		ComputationalCell3D::energyBoundaries[g] = std::pow(Emax / Emin, static_cast<double>(g) / G) * Emin;

	// 100 eV in Kelvin
	double const HeV_K = 1.602176634e-10 / 1.380649e-16;

	// EOS: e = f * T^beta * rho^{-mu}
	IdealGas eos(5.0 / 4.0, 3.4e13 / std::pow(HeV_K, 1.6), 1.6, 0.14);

	// Opacity (all absorption): sigma_R = 7200 * rho^{1.2} * (T/HeV_K)^{-1.5} 1/cm
	double const sigmaA0 = 7200.0 * std::pow(HeV_K, 1.5);
	auto opacityPtr = std::make_shared<MCPowerLawOpacity>(sigmaA0, 0.0, 1.2, -1.5, 0.0, 0.0);

	// Domain: 1D slab along x (geometric grading near left boundary)
	size_t const Nx = 512;
	double const L = 1.5e-3;
	double const dy = L / 15;
	Vector3D ll(0, 0, 0), ur(L, dy, dy);
	Voronoi3D tess(ll, ur);

	std::vector<Vector3D> points;
	if (rank == 0)
	{
		double const dx0 = L / (Nx * 10.0);
		double const growth = 1.05;
		std::vector<double> widths(Nx);
		double geo_sum = 0;
		for (size_t i = 0; i < Nx; ++i)
		{
			double w_geo = dx0 * std::pow(growth, static_cast<double>(i));
			double remaining = L - geo_sum;
			size_t cells_left = Nx - i;
			double w_const = remaining / static_cast<double>(cells_left);
			if (w_const <= w_geo)
			{
				for (size_t j = i; j < Nx; ++j)
					widths[j] = w_const;
				break;
			}
			widths[i] = w_geo;
			geo_sum += w_geo;
		}
		points.resize(Nx);
		double x_acc = 0;
		for (size_t i = 0; i < Nx; ++i)
		{
			points[i] = Vector3D(x_acc + widths[i] * 0.5, dy * 0.5, dy * 0.5);
			x_acc += widths[i];
		}
	}
	try {
#ifdef RICH_MPI
		tess.BuildParallel(points);
#else
		tess.Build(points);
#endif
	} catch (UniversalError const& eo) {
		reportError(eo);
		throw;
	}

	// Initial conditions: cold uniform gold at rest
	double const T_init = 1e-3 * HeV_K;
	ComputationalCell3D init_cell;
	init_cell.density = 19.32;
	init_cell.temperature = T_init;
	init_cell.internal_energy = eos.dT2e(init_cell.density, T_init,
		init_cell.tracers, ComputationalCell3D::tracerNames);
	init_cell.pressure = eos.de2p(init_cell.density, init_cell.internal_energy,
		init_cell.tracers, ComputationalCell3D::tracerNames);
	init_cell.velocity = Vector3D(0, 0, 0);
	init_cell.Erad = CG::radiation_constant * std::pow(T_init, 4) / init_cell.density;

	std::vector<ComputationalCell3D> cells(tess.GetPointNo(), init_cell);

	// Hydro infrastructure
	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.5, 0.75);

	Lagrangian3D lagrangian;
	RoundCells3D round_cells(lagrangian, eos, 0.5);
	XOnlyMotion3D pm(round_cells);

	ZeroForce3D force;

	DefaultCellUpdater cu(false, 0, true, T_init, nullptr);

	RigidWallFlux3D rigidflux(rs);
	RegularFlux3D *regular_flux = new RegularFlux3D(rs);
	IsBoundaryFace3D *boundary_face = new IsBoundaryFace3D();
	IsBulkFace3D *bulk_face = new IsBulkFace3D();
	std::vector<std::pair<const ConditionActionFlux1::Condition3D *,
		const ConditionActionFlux1::Action3D *>> flux_vector;
	flux_vector.push_back({boundary_face, &rigidflux});
	flux_vector.push_back({bulk_face, regular_flux});
	ConditionActionFlux1 fc(flux_vector, interp);

	std::vector<std::pair<const ConditionExtensiveUpdater3D::Condition3D *,
		const ConditionExtensiveUpdater3D::Action3D *>> eu_sequence;
	ConditionExtensiveUpdater3D eu(eu_sequence);

	auto tsf = std::make_shared<CFL1D>(0.25, 1, force, std::vector<std::string>(), true);

	Simulation simulation(tess, cells, eos);
	simulation.SetTimeStepFunction(tsf);
	HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
		std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	ComputationalCell3D cold_state = init_cell;

	auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_LAGRANGIAN_1D,
		/*left_ext=*/nullptr, /*right_ext=*/&cold_state);

	// MMC radiation with diffusion pressure gradient
	double const t_start = 1e-14;
	double const T_bath_init = std::pow(1.0 + 0.169141 * std::pow(t_start / 1e-9, -79.0 / 192.0), 0.25) * HeV_K;

	constexpr size_t newPhotonsPerCell = 15;
	constexpr size_t maxPhotonsPerCell = 300;
	constexpr size_t boundaryPhotonsPerCell = 1000;
	constexpr size_t initialParticlesPerCell = 10;
	constexpr bool withHydro = true;

	auto eosPtr = std::make_shared<IdealGas>(eos);

	RadiationIMCParameters imc_params = {
		.newPhotonsPerCell = newPhotonsPerCell,
		.withHydro = withHydro,
		.diffusionPressureGradient = true,
		.MMC = true,
		.withMultigroupOpacity = false,
		.withRandomWalk = true,
		.withDDMC = true,
		.noHydroFeedback = false,
		.withEgTimeAvg = false,
		.withCompton = false,
	};

	auto& sim_cells = simulation.getCells();
	auto& extensives = simulation.getExtensives();

	auto boundaryCond = std::make_shared<SideTemperature<Vector3D, Tessellation3D>>(
		tess, sim_cells, T_bath_init, boundaryPhotonsPerCell);
	size_t const Nlocal = tess.GetPointNo();
	extensives.resize(Nlocal);
	for (size_t i = 0; i < Nlocal; ++i)
		PrimitiveToConserved(sim_cells[i], tess.GetVolume(i), extensives[i]);

	auto mc_physics = std::make_shared<RadiationIMC>(
		tess, boundaryCond, sim_cells, extensives, eosPtr, opacityPtr, imc_params);

	auto pop_control = std::make_shared<CombPopulationControl<Vector3D, Tessellation3D>>(tess, maxPhotonsPerCell);
	std::vector<Particle3D> initial_particles;
	auto mc_step = std::make_shared<RadiationMCStep>(
		tess, sim_cells, extensives, mc_physics, pop_control, boundaryCond,
		initial_particles, initialParticlesPerCell, withHydro
#ifdef RICH_MPI
		, RadiationMCStep::ManagerType::AUTO_RDMA
#endif
	);
#ifdef RICH_MPI
	mc_step->setCost(std::make_shared<MCStepCostCalculator>(mc_step->getManager()));
#endif

	simulation.addPhysics(hydroStep);
	simulation.addPhysics(mc_step);

	if (rank == 0)
		std::cout << "Gold heat wave tau=0 (MMC+pressure): Nx=" << Nx
			<< ", new/cell=" << newPhotonsPerCell
			<< ", max/cell=" << maxPhotonsPerCell
			<< ", bdy/cell=" << boundaryPhotonsPerCell << std::endl;

	char file_buf[4096];
	strncpy(file_buf, __FILE__, sizeof(file_buf) - 1);
	file_buf[sizeof(file_buf) - 1] = '\0';
	std::string dir_path = std::string(dirname(file_buf));
	std::string profile_path = dir_path + "/heat_wave_profile.txt";

	auto write_output = [&]()
	{
		size_t const N = tess.GetPointNo();
		auto const& out_cells = sim.getCells();
		auto const& Erad_tavg = mc_step->getEradTimeAvg();
		double const transverse_area = dy * dy;

		struct CellRow { double x, rho, T, Trad, vx, P, dm; };
		std::vector<CellRow> local_rows(N);
		for (size_t i = 0; i < N; ++i)
		{
			double erad = (i < Erad_tavg.size() && Erad_tavg[i] > 0)
				? Erad_tavg[i]
				: out_cells[i].Erad * out_cells[i].density;
			double Trad = std::pow(erad / CG::radiation_constant, 0.25);
			double vol = tess.GetVolume(i);
			local_rows[i] = {tess.GetMeshPoint(i).x, out_cells[i].density,
				out_cells[i].temperature, Trad, out_cells[i].velocity.x,
				out_cells[i].pressure, out_cells[i].density * vol / transverse_area};
		}

		std::vector<CellRow> all_rows;
#ifdef RICH_MPI
		int local_n = static_cast<int>(N);
		int world_size = 1;
		MPI_Comm_size(MPI_COMM_WORLD, &world_size);
		std::vector<int> counts(world_size), displs(world_size);
		MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
		if (rank == 0)
		{
			displs[0] = 0;
			for (int r = 1; r < world_size; ++r)
				displs[r] = displs[r - 1] + counts[r - 1];
			int total = displs[world_size - 1] + counts[world_size - 1];
			all_rows.resize(static_cast<size_t>(total));
		}
		int send_bytes = local_n * static_cast<int>(sizeof(CellRow));
		std::vector<int> byte_counts(world_size), byte_displs(world_size);
		if (rank == 0)
		{
			for (int r = 0; r < world_size; ++r)
			{
				byte_counts[r] = counts[r] * static_cast<int>(sizeof(CellRow));
				byte_displs[r] = displs[r] * static_cast<int>(sizeof(CellRow));
			}
		}
		MPI_Gatherv(local_rows.data(), send_bytes, MPI_BYTE,
			all_rows.data(), byte_counts.data(), byte_displs.data(), MPI_BYTE,
			0, MPI_COMM_WORLD);
#else
		all_rows = std::move(local_rows);
#endif

		if (rank == 0)
		{
			std::sort(all_rows.begin(), all_rows.end(),
				[](const CellRow& a, const CellRow& b) { return a.x < b.x; });

			std::ofstream out(profile_path);
			out << std::scientific << std::setprecision(12);
			out << "# time = " << simulation.GetTime() << "\n";
			out << "# x  density  temperature  T_rad  velocity_x  pressure  mass_coord\n";

			double mass_cumulative = 0;
			for (auto const& r : all_rows)
			{
				mass_cumulative += r.dm;
				out << r.x << " " << r.rho << " " << r.T << " "
					<< r.Trad << " " << r.vx << " " << r.P << " "
					<< mass_cumulative << "\n";
			}
		}
	};

	double const tf = 2e-9;
	double old_dt = 1e-15;
	simulation.SetTime(t_start);
	while (simulation.GetTime() < tf)
	{
		double const t_now = simulation.GetTime();
		double const t_ns = t_now / 1e-9;
		double const T_bath = std::pow(1.0 + 0.169141 * std::pow(t_ns, -79.0 / 192.0), 0.25) * HeV_K;
		boundaryCond->SetTemperature(T_bath);

		try
		{
			simulation.SetTimeStep(old_dt);
			simulation.step();
			double new_dt = mc_step->suggestTimeStep();
			new_dt = std::min(new_dt, tsf->GetTimeStep());
			new_dt = std::max(1e-15, new_dt);
			old_dt = new_dt;
		}
		catch (UniversalError const& eo)
		{
			reportError(eo);
			throw;
		}

		if (rank == 0)
			std::cout << "Cycle " << simulation.GetCycle() << " Time " << simulation.GetTime()
				<< " dt " << old_dt << std::endl;

		if (simulation.GetCycle() % 200 == 0)
			write_output();
	}

	write_output();
	WriteSnapshot3D(sim, dir_path + "/final.h5");

	if (rank == 0)
	{
		double const t_ns_end = tf / 1e-9;
		double const m_front = 1.01889e-3 * std::pow(t_ns_end, 33.0 / 64.0);
		std::cout << "Analytic m_front at t=" << tf << " s: " << m_front << " g/cm^2" << std::endl;
		std::cout << "Done" << std::endl;
	}
#ifdef RICH_MPI
	MPI_Finalize();
#endif
	return 0;
}
