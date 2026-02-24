#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/newtonian/three_dimensional/hdsim_3d.hpp"
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
#include <fstream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fenv.h>
#include <libgen.h>
#include <string.h>

// Problem 4: Derei et al. (2024) Test 3
// kappa_R = 2*(T/keV)^{-4.5} * rho^{1.9}
// kappa_P = 5e-4*kappa_R
// u(T,rho) = 1e14*(T/keV)^6 * rho^{0.7}
// rho(x) = x^{-40/139}, stretched grid (Eq 5.17 from paper)
// T_bath(t) = 1.01008116*(t/ns)^{14/139} keV

static std::vector<double> generate_stretched_grid(double dx, double L, size_t N)
{
	std::vector<double> centers(N);
	if (std::abs(L - double(N) * dx) < 1e-12 * L)
	{
		for (size_t i = 0; i < N; ++i)
			centers[i] = (double(i) + 0.5) * dx;
		return centers;
	}
	// Solve dx * (r^N - 1) / (r - 1) = L for the geometric ratio r via bisection
	double r_lo = 1.0 + 1e-15;
	double r_hi = std::pow(L / dx, 1.0 / double(N - 1));
	for (int iter = 0; iter < 200; ++iter)
	{
		double const r_mid = 0.5 * (r_lo + r_hi);
		double const S = dx * (std::pow(r_mid, double(N)) - 1.0) / (r_mid - 1.0);
		if (S < L)
			r_lo = r_mid;
		else
			r_hi = r_mid;
		if (r_hi - r_lo < 1e-15 * r_lo)
			break;
	}
	double const r = 0.5 * (r_lo + r_hi);
	double edge = 0.0;
	double width = dx;
	for (size_t i = 0; i < N; ++i)
	{
		centers[i] = edge + 0.5 * width;
		edge += width;
		width *= r;
	}
	return centers;
}

int main(void)
{
	feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);

	double const keV_K = 1.602176634e-9 / 1.380649e-16;

	// EOS: e = f*T^beta*rho^{-mu}
	// u = rho*e = f*T^beta*rho^{1-mu}
	// u(T,rho) = 1e14*(T/keV_K)^6*rho^{0.7}
	// => 1-mu = 0.7 => mu = 0.3
	double const f_eos = 1e14 / std::pow(keV_K, 6.0);
	IdealGas eos(1.4, f_eos, 6.0, 0.3);

	// kappa_R = 2*(keV_K/T)^{4.5}*rho^{1.9}
	// D = c/(3*2*keV_K^{4.5}*T^{-4.5}*rho^{1.9}) = c*T^{4.5}/(6*keV_K^{4.5}*rho^{1.9})
	// => D0 = c/(6*keV_K^{4.5}), alpha = -1.9, beta = 4.5
	// kappa_P = 5e-4*kappa_R = 0.001*keV_K^{4.5}*T^{-4.5}*rho^{1.9}
	double const D0 = CG::speed_of_light / (6.0 * std::pow(keV_K, 4.5));
	double const planck0 = 0.001 * std::pow(keV_K, 4.5);
	PowerLawOpacity opacity(D0, -1.9, 4.5, planck0, 1.9, -4.5);

	size_t const Nx = 512;
	// size_t const Nx = 1024;
	double const x_offset = 1e-5;
	// double const dx0 = 2.24e-2 * (1.0075 - 1.0);
	double const dx0 = 7e-5;
	// double const L_grid = 2.24e-2 * (std::pow(1.0075, double(Nx)) - 1.0);
	double const L_grid = 1.0;
	std::vector<double> cx = generate_stretched_grid(dx0, L_grid, Nx);

	double const x_max = x_offset + L_grid;
	double const dy = x_max / Nx;
	Vector3D ll(0, 0, 0), ur(x_max, dy, dy);
	Voronoi3D tess(ll, ur);

	std::vector<Vector3D> points;
	for (size_t i = 0; i < Nx; ++i)
		points.push_back(Vector3D(x_offset + cx[i], 0.5 * dy, 0.5 * dy));
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
		cells[i].density = std::pow(x, -40.0 / 139.0);
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

	double const tau = 14.0 / 139.0;
	double const T_bath_init = 1.01008116 * std::pow(1e-15 / 1e-9, tau) * keV_K;
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

	CourantFriedrichsLewy tsf(0.25, 1, force);

	HDSim3D sim(tess, cells, eos, pm, tsf, fc, cu, eu, force,
		std::make_pair(ComputationalCell3D::tracerNames, ComputationalCell3D::stickerNames));

	double const tf = 1e-9;
	double old_dt = 1e-17;
	tsf.SetTimeStep(old_dt);

	while (sim.getTime() < tf)
	{
		double const t_now = std::max(sim.getTime(), 1e-20);
		double const T_bath = 1.01008116 * std::pow(t_now * 1e9, tau) * keV_K;
		D_boundary.SetTemperature(T_bath);

		try
		{
			double new_dt = sim.RadiationTimeStep(old_dt, diffusion, true);
			new_dt = std::min(std::max(1e-20, sim.getTime() * 1e-3), new_dt);
			tsf.SetTimeStep(new_dt);
			old_dt = new_dt;
		}
		catch (UniversalError const &eo)
		{
			reportError(eo);
			throw;
		}

		std::cout << "\nCycle " << sim.getCycle() << " Time " << sim.getTime()
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
		pf << "4\n";
	}

	std::cout << "Done" << std::endl;
	return 0;
}
