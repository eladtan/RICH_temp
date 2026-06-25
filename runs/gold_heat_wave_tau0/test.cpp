// Subsonic ablative heat wave in gold — Case 1 (tau = 0)
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
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include "source/misc/mesh_generator3D.hpp"
// #include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/PCM3D.hpp"
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
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/3D/output/write3D.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>
#include <fenv.h>
#include <libgen.h>
#include <string.h>
#ifdef RICH_MPI
#include <mpi.h>
#endif

size_t find_leftmost_cell(Tessellation3D const& tess)
{
	size_t left_i = 0;
	double xmin = std::numeric_limits<double>::max();
	for (size_t i = 0; i < tess.GetPointNo(); ++i)
	{
		double const x = tess.GetMeshPoint(i).x;
		if (x < xmin) { xmin = x; left_i = i; }
	}
	return left_i;
}

size_t find_rightmost_cell(Tessellation3D const& tess)
{
	size_t right_i = 0;
	double xmax = -std::numeric_limits<double>::max();
	for (size_t i = 0; i < tess.GetPointNo(); ++i)
	{
		double const x = tess.GetMeshPoint(i).x;
		if (x > xmax) { xmax = x; right_i = i; }
	}
	return right_i;
}

double shock_speed_from_pressure(double const rho0, double const rho2, double const P2)
{
	return std::sqrt(P2 / std::max(rho0 * (1.0 - rho0 / rho2), 1e-300));
}

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

	// 100 eV in Kelvin
	double const HeV_K = 1.602176634e-10 / 1.380649e-16;

	// EOS: e = f * T^beta * rho^{-mu}
	// e = 3.4e13 * (T/HeV_K)^1.6 * rho^{-0.14}
	double const f_eos = 3.4e13 / std::pow(HeV_K, 1.6);
	IdealGas eos(5.0 / 4.0, f_eos, 1.6, 0.14);

	// Opacity (all absorption, zero scattering):
	// kappa_R = 7200 * (T/HeV)^{-1.5} * rho^{0.2} cm^2/g
	// sigma_R = rho * kappa_R = 7200 * rho^{1.2} * (T/HeV_K)^{-1.5}
	// D = c / (3*sigma_R) = [c / (3*7200)] * HeV_K^{1.5} * T^{1.5} * rho^{-1.2}
	double const D0 = CG::speed_of_light / (3.0 * 7200.0 * std::pow(HeV_K, 1.5));
	// sigma_P = sigma_R = 7200 * HeV_K^{1.5} * rho^{1.2} * T^{-1.5}
	double const planck0 = 100*7200.0 * std::pow(HeV_K, 1.5);
	PowerLawOpacity opacity(D0, -1.2, 1.5, planck0, 1.2, -1.5);

	// Domain: 1D slab along x
	size_t const Nx = 512;
	double const L = 1.5e-3;
	double const dy = L / 50;
	Vector3D ll(0, 0, 0), ur(L, dy, dy);
	Voronoi3D tess(ll, ur);

	std::vector<Vector3D> points;
	if (rank == 0)
	{
		double const dx0 = L / (Nx * 4.0);
		double const growth = 1.025;
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
		std::cout << "Initial mesh points (" << points.size() << "):" << std::endl;
		for (size_t i = 0; i < points.size(); ++i)
		{
			std::cout << "  [" << i << "] "
				<< points[i].x << " " << points[i].y << " " << points[i].z
				<< std::endl;
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
	double const T_init = 1e-4 * HeV_K;
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
	// PCM3D interp(ghost);

	Lagrangian3D lagrangian;
	RoundCells3D round_cells(lagrangian, eos, 0.1);
	XOnlyMotion3D pm(round_cells);

	double const t_start = 1e-14 * 512 / Nx;

	// --- Boundary choice ---
	// Test A (Dirichlet diagnostic): DiffusionDirichletBoundary D_boundary(HeV_K, opacity);
	// Test B (old Marshak):          DiffusionSideBoundary D_boundary(T_bath_init);
	// Test C (moving Marshak):
	double const T_bath_init = std::pow(1.0 + 0.169141 * std::pow(t_start / 1e-9, -79.0 / 192.0), 0.25) * HeV_K;
	DiffusionMovingMarshakBoundary D_boundary(T_bath_init);
	D_boundary.SetDebug(false);

	// Diffusion: no flux limiter, hydro on, no compton
	Diffusion diffusion(opacity, D_boundary, eos, std::vector<std::string>(),
		false, true, false, false, 1e50);
	DiffusionForce force(diffusion, eos);

	DefaultCellUpdater cu(false, 0, true, 0, &diffusion);

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

	auto tsf = std::make_shared<CFL1D>(0.25, 1, force, std::vector<std::string> (), true);

	Simulation simulation(tess, cells, eos);
	simulation.SetTimeStepFunction(tsf);
	HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
		std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	// Right boundary: matched cold state so right side stays stationary
	ComputationalCell3D cold_state = init_cell;

	auto hydroStep = std::make_shared<HydroStep>(sim, HydroStep::TIMEADVANCE_LAGRANGIAN_1D,
		/*left_ext=*/nullptr, /*right_ext=*/&cold_state);

	auto radStep = std::make_shared<RadiationStep>(tess, simulation.getCells(), simulation.getExtensives(),
		simulation.getTracker(),
#ifdef RICH_MPI
		sim.cost_calc_,
#endif
		diffusion, false);

	simulation.addPhysics(hydroStep);
	simulation.addPhysics(radStep);

	char file_buf[4096];
	strncpy(file_buf, __FILE__, sizeof(file_buf) - 1);
	file_buf[sizeof(file_buf) - 1] = '\0';
	std::string dir_path = std::string(dirname(file_buf));
	std::string profile_path = dir_path + "/heat_wave_profile.txt";

	auto write_output = [&]()
	{
		size_t const N = tess.GetPointNo();
		auto const& out_cells = sim.getCells();
		double const transverse_area = dy * dy;

		struct CellRow { double x, rho, T, Trad, vx, P, dm; };
		std::vector<CellRow> local_rows(N);
		for (size_t i = 0; i < N; ++i)
		{
			double Trad = std::pow(out_cells[i].Erad * out_cells[i].density
				/ CG::radiation_constant, 0.25);
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
	double old_dt = 1e-16;
	simulation.SetTime(t_start);

	std::string diag_path = dir_path + "/boundary_diagnostics.txt";
	std::ofstream diag_out;
	if (rank == 0)
	{
		diag_out.open(diag_path);
		diag_out << "# time T_bath_HeV T_surface_mat_HeV T_surface_rad_HeV rho_surface P_surface v_surface v_face_left x_left_box x_left_cell\n";
	}

	while (simulation.GetTime() < tf)
	{
		double const dt = std::min(old_dt, tf - simulation.GetTime());

		// Update bath temperature at midpoint of timestep
		double const t_mid_ns = (simulation.GetTime() + 0.5 * dt) / 1e-9;
		double const T_bath = std::pow(1.0 + 0.169141 * std::pow(t_mid_ns, -79.0 / 192.0), 0.25) * HeV_K;
		D_boundary.SetTemperature(T_bath);

		// Set face velocities from globally leftmost/rightmost cells
		{
			auto const& cur_tess = sim.getTessellation();
			auto const& cur_cells = sim.getCells();
			size_t const left_i = find_leftmost_cell(cur_tess);
			size_t const right_i = find_rightmost_cell(cur_tess);

			double left_x = cur_tess.GetMeshPoint(left_i).x;
			double right_x = cur_tess.GetMeshPoint(right_i).x;
			Vector3D left_vel = cur_cells[left_i].velocity;
			Vector3D right_vel = cur_cells[right_i].velocity;

#ifdef RICH_MPI
			struct { double x; int rank; } left_data, right_data;
			left_data.x = left_x;
			left_data.rank = rank;
			right_data.x = right_x;
			right_data.rank = rank;

			MPI_Allreduce(MPI_IN_PLACE, &left_data, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &right_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

			MPI_Bcast(&left_vel, 3, MPI_DOUBLE, left_data.rank, MPI_COMM_WORLD);
			MPI_Bcast(&right_vel, 3, MPI_DOUBLE, right_data.rank, MPI_COMM_WORLD);
#endif
			D_boundary.SetLeftFaceVelocity(left_vel);
			D_boundary.SetRightFaceVelocity(right_vel);
		}

		try
		{
			simulation.SetTimeStep(dt);
			simulation.step();
			double new_dt = radStep->suggestTimeStep();
			new_dt = std::min(new_dt, tsf->GetTimeStep());
			new_dt = std::max(1e-16 * 512 / Nx, new_dt);
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

		// Write surface diagnostics (gather global leftmost cell info)
		if (simulation.GetCycle() % 50 == 0)
		{
			auto const& cur_tess = sim.getTessellation();
			auto const& cur_cells = sim.getCells();
			size_t const left_i = find_leftmost_cell(cur_tess);

			struct { double T_mat, T_rad, rho, P, vx, x_cell; } surf;
			surf.x_cell = cur_tess.GetMeshPoint(left_i).x;
			ComputationalCell3D const& c = cur_cells[left_i];
			surf.T_mat = c.temperature;
			double const Er_vol = c.Erad * c.density;
			surf.T_rad = std::pow(std::max(Er_vol / CG::radiation_constant, 1e-300), 0.25);
			surf.rho = c.density;
			surf.P = c.pressure;
			surf.vx = c.velocity.x;

#ifdef RICH_MPI
			struct { double x; int rank; } loc_data;
			loc_data.x = surf.x_cell;
			loc_data.rank = rank;
			MPI_Allreduce(MPI_IN_PLACE, &loc_data, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
			MPI_Bcast(&surf, 6, MPI_DOUBLE, loc_data.rank, MPI_COMM_WORLD);
#endif

			if (rank == 0)
			{
				diag_out << std::scientific << std::setprecision(12)
					<< simulation.GetTime() << " "
					<< T_bath / HeV_K << " "
					<< surf.T_mat / HeV_K << " "
					<< surf.T_rad / HeV_K << " "
					<< surf.rho << " "
					<< surf.P << " "
					<< surf.vx << " "
					<< surf.vx << " "
					<< cur_tess.GetBoxCoordinates().first.x << " "
					<< surf.x_cell << "\n";
			}
		}

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
