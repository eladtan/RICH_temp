#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
#include "source/newtonian/three_dimensional/simulation/Simulation.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Hllc3D.hpp"
#include "source/misc/simple_io.hpp"
#include "source/newtonian/three_dimensional/eulerian_3d.hpp"
#include "source/newtonian/three_dimensional/default_cell_updater.hpp"
#include "source/newtonian/three_dimensional/ConditionActionFlux1.hpp"
#include "source/newtonian/three_dimensional/ConditionExtensiveUpdater3D.hpp"
#include "source/newtonian/three_dimensional/CourantFriedrichsLewy.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/Radiation/Diffusion.hpp"
#include "source/Radiation/DiffusionForce.hpp"
#include "source/newtonian/three_dimensional/simulation/steps/RadiationStep.hpp"
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fenv.h>
#include <libgen.h>
#include <string.h>

// Problem 3: Derei et al. (2024) Test 1
// kappa_R = 40*(T/keV)^{-1.5} * rho^{1.2}
// kappa_P = 0.0025*kappa_R
// u(T,rho) = 1e14*(T/keV)^{3.4} * rho^{0.86}
// rho(x) = x^{20/19}, uniform grid [0,1]
// T_bath(t) = 1.0470478*(t/ns)^{86/57} keV

int main(void)
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

	double const keV_K = 1.602176634e-9 / 1.380649e-16;

	// EOS: e = f*T^beta*rho^{-mu}
	// u = rho*e = f*T^beta*rho^{1-mu}
	// u(T,rho) = 1e14*(T/keV_K)^{3.4}*rho^{0.86}
	// => 1-mu = 0.86 => mu = 0.14
	double const f_eos = 1e14 / std::pow(keV_K, 3.4);
	IdealGas eos(1.4, f_eos, 3.4, 0.14);

	// kappa_R = 40*(keV_K/T)^{1.5}*rho^{1.2}
	// D = c/(3*kappa_R) = c*T^{1.5}/(120*keV_K^{1.5}*rho^{1.2})
	// => D0 = c/(120*keV_K^{1.5}), alpha = -1.2, beta = 1.5
	// kappa_P = 0.0025*40*(keV_K/T)^{1.5}*rho^{1.2} = 0.1*keV_K^{1.5}*T^{-1.5}*rho^{1.2}
	double const D0 = CG::speed_of_light / (120.0 * std::pow(keV_K, 1.5));
	double const planck0 = 0.1 * std::pow(keV_K, 1.5);
	PowerLawOpacity opacity(D0, -1.2, 1.5, planck0, 1.2, -1.5);

	size_t const Nx = 512;
	double const width = 1.0;
	double const dy = width / Nx;
	Vector3D ll(0, 0, 0), ur(width, dy, dy);
	Voronoi3D tess(ll, ur);

	std::vector<Vector3D> points = CartesianMesh(Nx, 1, 1, ll, ur);
	try {
		tess.Build(points);
	} catch (UniversalError const& eo) {
		reportError(eo);
		throw;
	}

	double const T_init = 1e-3 * keV_K;
	std::vector<ComputationalCell3D> cells(tess.GetPointNo());
	for (size_t i = 0; i < tess.GetPointNo(); ++i)
	{
		double const x = tess.GetCellCM(i).x;
		cells[i].density = std::pow(x, 20.0 / 19.0);
		cells[i].temperature = T_init;
		cells[i].internal_energy = eos.dT2e(cells[i].density, cells[i].temperature,
			cells[i].tracers, ComputationalCell3D::tracerNames);
		cells[i].pressure = eos.de2p(cells[i].density, cells[i].internal_energy,
			cells[i].tracers, ComputationalCell3D::tracerNames);
		cells[i].Erad = CG::radiation_constant * std::pow(T_init, 4) / cells[i].density;
	}

	Hllc3D rs;
	RigidWallGenerator3D ghost;
	LinearGauss3D interp(eos, ghost, true, 0.2, 0.25, 0.75);
	Eulerian3D pm;

	double const tau = 86.0 / 57.0;
	double const T_bath_init = 1.0470478 * std::pow(1e-15 / 1e-9, tau) * keV_K;
	DiffusionSideBoundary D_boundary(T_bath_init);
	Diffusion diffusion(opacity, D_boundary, eos, std::vector<std::string>(),
		false, false, false);
	ZeroForce3D force;

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

	auto tsf = std::make_shared<CourantFriedrichsLewy>(0.25, 1, force);

	Simulation simulation(tess, cells, eos);
	simulation.SetTimeStepFunction(tsf);
	HDSim3D sim(tess, simulation.getCells(), simulation.getExtensives(), eos, simulation.getTracker(), pm, *tsf, fc, cu, eu, force,
		std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	double const tf = 1e-9;
	double old_dt = 1e-15;

	auto radStep = std::make_shared<RadiationStep>(tess, simulation.getCells(), simulation.getExtensives(),
		simulation.getTracker(),
#ifdef RICH_MPI
		nullptr,
#endif
		diffusion, true);
	simulation.addPhysics(radStep);

	while (simulation.GetTime() < tf)
	{
		double const t_now = std::max(simulation.GetTime(), 1e-15);
		double const T_bath = 1.0470478 * std::pow(t_now / 1e-9, tau) * keV_K;
		D_boundary.SetTemperature(T_bath);

		try
		{
			simulation.SetTimeStep(old_dt);
			simulation.step();
			double new_dt = radStep->suggestTimeStep();
			new_dt = std::min(std::max(1e-15, simulation.GetTime() * 1e-3), new_dt);
			old_dt = new_dt;
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}

		std::cout << "\nCycle " << simulation.GetCycle() << " Time " << simulation.GetTime()
			<< " dt " << old_dt << "\n" << std::endl;
	}

	char file_buf[4096];
	strncpy(file_buf, __FILE__, sizeof(file_buf) - 1);
	file_buf[sizeof(file_buf) - 1] = '\0';
	std::string dir_path = std::string(dirname(file_buf));
	std::string profile_path = dir_path + "/marshak_profile.txt";

	{
		size_t const N = tess.GetPointNo();
		std::vector<size_t> idx(N);
		std::iota(idx.begin(), idx.end(), 0);
		std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
			return tess.GetMeshPoint(a).x < tess.GetMeshPoint(b).x;
		});

		auto const& final_cells = sim.getCells();
		std::ofstream out(profile_path);
		out << std::scientific << std::setprecision(12);
		for (size_t i = 0; i < N; ++i)
		{
			size_t const k = idx[i];
			double const Trad = std::pow(final_cells[k].Erad * final_cells[k].density
				/ CG::radiation_constant, 0.25);
			out << tess.GetMeshPoint(k).x << " "
				<< final_cells[k].temperature << " "
				<< Trad << "\n";
		}
	}

	{
		std::ofstream pf(dir_path + "/problem_number.txt");
		pf << "3\n";
	}

	std::cout << "Done" << std::endl;
	return 0;
}
