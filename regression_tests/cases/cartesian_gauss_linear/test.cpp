#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <random>

#include "source/3D/tessellation/voronoi/Voronoi3D.hpp"
#include "source/misc/mesh_generator3D.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/newtonian/three_dimensional/Ghost3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/SphericalLinearGauss3D.hpp"
#include "source/misc/universal_error.hpp"

namespace {

struct CartLinCoeffs {
	double a0, ax, ay, az;
	double eval(Vector3D const& p) const { return a0 + ax*p.x + ay*p.y + az*p.z; }
};

CartLinCoeffs const rho_c  = { 5.0,  0.3, -0.2,  0.1 };
CartLinCoeffs const pres_c = { 3.0, -0.1,  0.4,  0.05 };
CartLinCoeffs const ie_c   = { 2.0,  0.2,  0.1, -0.15 };
CartLinCoeffs const vx_c   = { 1.0,  0.1, -0.05, 0.02 };
CartLinCoeffs const vy_c   = {-0.5,  0.2,  0.0,  0.1 };
CartLinCoeffs const vz_c   = { 0.3, -0.1,  0.15, 0.0 };

void fill_cell(ComputationalCell3D &cell, Vector3D const& pos)
{
	cell.density = rho_c.eval(pos);
	cell.pressure = pres_c.eval(pos);
	cell.internal_energy = ie_c.eval(pos);
	cell.velocity = Vector3D(vx_c.eval(pos), vy_c.eval(pos), vz_c.eval(pos));
}

struct FieldErrors {
	double max_abs;
	double max_rel;
	double worst_interp;
	double worst_exact;
	size_t worst_cell_index;
	size_t worst_cell_id;
	Vector3D worst_pos;
	size_t count;
	FieldErrors() : max_abs(0), max_rel(0), worst_interp(0), worst_exact(0),
		worst_cell_index(0), worst_cell_id(0), worst_pos(), count(0) {}
	void update(double interp_v, double exact_v, Vector3D const& pos, size_t cell_index, size_t cell_id) {
		double err = std::abs(interp_v - exact_v);
		double scale = std::max(std::abs(exact_v), 1e-10);
		double rel = err / scale;
		if (err > max_abs) max_abs = err;
		if (rel > max_rel) {
			max_rel = rel;
			worst_pos = pos;
			worst_interp = interp_v;
			worst_exact = exact_v;
			worst_cell_index = cell_index;
			worst_cell_id = cell_id;
		}
		++count;
	}
};

} // namespace

int main(void)
{
	double const box_half = 4.0;
	Vector3D ll(-box_half, -box_half, -box_half);
	Vector3D ur(box_half, box_half, box_half);

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(0.0, 1.0);
	std::vector<Vector3D> points;
	size_t const N_target = 8000;
	points.reserve(N_target);
	while (points.size() < N_target) {
		double r = 1.0 + dist(gen);
		double phi = 2.0 * M_PI * dist(gen);
		double costheta = 2.0 * dist(gen) - 1.0;
		double sintheta = std::sqrt(1.0 - costheta * costheta);
		points.push_back(Vector3D(r*sintheta*std::cos(phi),
		                          r*sintheta*std::sin(phi),
		                          r*costheta));
	}

	auto outer_pts = RandSphereR2(30000, ll, ur, 2.0, box_half * std::sqrt(3.0));
	points.insert(points.end(), outer_pts.begin(), outer_pts.end());
	auto inner_pts = RandSphereR(3000, ll, ur, 0.0, 1.0);
	points.insert(points.end(), inner_pts.begin(), inner_pts.end());

	Voronoi3D tess(ll, ur);
	try {
		tess.Build(points);
	} catch (UniversalError const& eo) {
		reportError(eo);
		return 1;
	}
	std::cout << "Tessellation built with " << tess.GetPointNo() << " cells" << std::endl;

	size_t const Nlocal = tess.GetPointNo();
	std::vector<ComputationalCell3D> cells(Nlocal);
	for (size_t i = 0; i < Nlocal; ++i) {
		Vector3D pos = tess.GetCellCM(i);
		fill_cell(cells[i], pos);
		cells[i].ID = i;
	}

	IdealGas eos(5.0 / 3.0);
	RigidWallGenerator3D ghost;

	LinearGauss3D cart_interp(eos, ghost,
	    /*slf=*/false, /*delta_v=*/0.2, /*theta=*/0.5,
	    /*delta_P=*/0.7, /*SR=*/false, /*calc_tracers=*/{},
	    /*skip_key=*/"", /*pressure_calc=*/false);

	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> cart_face_values;
	try {
		cart_interp(tess, cells, 0.0, cart_face_values);
	} catch (UniversalError const& eo) {
		reportError(eo);
		return 1;
	}
	std::cout << "Cartesian interpolation computed for " << cart_face_values.size() << " faces" << std::endl;

	Vector3D origin(0, 0, 0);
	SphericalLinearGauss3D sph_interp(eos, ghost, origin,
	    /*slf=*/true, /*delta_v=*/0.2, /*theta=*/0.5,
	    /*delta_P=*/0.7, /*SR=*/false, /*calc_tracers=*/{},
	    /*skip_key=*/"", /*pressure_calc=*/false);

	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> sph_face_values;
	try {
		sph_interp(tess, cells, 0.0, sph_face_values);
	} catch (UniversalError const& eo) {
		reportError(eo);
		return 1;
	}
	std::cout << "Spherical interpolation computed for " << sph_face_values.size() << " faces" << std::endl;

	FieldErrors cart_rho_err, cart_ie_err, cart_vx_err, cart_vy_err, cart_vz_err;
	FieldErrors sph_rho_err, sph_ie_err, sph_vel_err;
	Vector3D sph_vel_worst_interp, sph_vel_worst_exact;
size_t sph_vel_worst_cell_index = 0;
size_t sph_vel_worst_cell_id = 0;

	for (size_t f = 0; f < tess.GetTotalFacesNumber(); ++f) {
		auto const& neigh = tess.GetFaceNeighbors(f);
		if (neigh.first >= Nlocal || neigh.second >= Nlocal)
			continue;

		Vector3D cm1 = tess.GetCellCM(neigh.first);
		Vector3D cm2 = tess.GetCellCM(neigh.second);
		double r1 = abs(cm1), r2 = abs(cm2);
		if (r1 < 1.1 || r1 > 1.9 || r2 < 1.1 || r2 > 1.9)
			continue;

		Vector3D face_cm = tess.FaceCM(f);

		double exact_rho = rho_c.eval(face_cm);
		double exact_ie = ie_c.eval(face_cm);
		Vector3D exact_vel(vx_c.eval(face_cm), vy_c.eval(face_cm), vz_c.eval(face_cm));
		double vel_mag = abs(exact_vel);

		for (int side = 0; side < 2; ++side) {
			size_t src_index = static_cast<size_t>((side == 0) ? neigh.first : neigh.second);
			size_t src_id = static_cast<size_t>(cells[src_index].ID);
			ComputationalCell3D const& cv =
			    (side == 0) ? cart_face_values[f].first : cart_face_values[f].second;
			cart_rho_err.update(cv.density, exact_rho, face_cm, src_index, src_id);
			cart_ie_err.update(cv.internal_energy, exact_ie, face_cm, src_index, src_id);
			cart_vx_err.update(cv.velocity.x, exact_vel.x, face_cm, src_index, src_id);
			cart_vy_err.update(cv.velocity.y, exact_vel.y, face_cm, src_index, src_id);
			cart_vz_err.update(cv.velocity.z, exact_vel.z, face_cm, src_index, src_id);

			ComputationalCell3D const& sv =
			    (side == 0) ? sph_face_values[f].first : sph_face_values[f].second;
			sph_rho_err.update(sv.density, exact_rho, face_cm, src_index, src_id);
			sph_ie_err.update(sv.internal_energy, exact_ie, face_cm, src_index, src_id);
			double dv = abs(sv.velocity - exact_vel);
			double dv_rel = dv / std::max(vel_mag, 1e-10);
			if (dv > sph_vel_err.max_abs) sph_vel_err.max_abs = dv;
			if (dv_rel > sph_vel_err.max_rel) {
				sph_vel_err.max_rel = dv_rel;
				sph_vel_err.worst_pos = face_cm;
				sph_vel_worst_interp = sv.velocity;
				sph_vel_worst_exact = exact_vel;
				sph_vel_worst_cell_index = src_index;
				sph_vel_worst_cell_id = src_id;
			}
			++sph_vel_err.count;
		}
	}

	std::cout << std::scientific << std::setprecision(6);

	auto print_worst = [](std::string const& label, FieldErrors const& fe) {
		double r = abs(fe.worst_pos);
		double theta = (r > 1e-14) ? std::acos(std::max(-1.0, std::min(1.0, fe.worst_pos.z / r))) : 0.0;
		double phi = std::atan2(fe.worst_pos.y, fe.worst_pos.x);
		std::cout << "  worst " << label << " at cart=("
		          << fe.worst_pos.x << ", " << fe.worst_pos.y << ", " << fe.worst_pos.z
		          << ")  sph=(r=" << r << ", theta=" << theta << ", phi=" << phi << ")\n";
	};

	std::cout << "\n--- Cartesian per-field max absolute / relative errors ---\n";
	std::cout << "density:   abs=" << cart_rho_err.max_abs << "  rel=" << cart_rho_err.max_rel
	          << "  (N=" << cart_rho_err.count << ")\n";
	print_worst("density", cart_rho_err);
	std::cout << "int_ener:  abs=" << cart_ie_err.max_abs << "  rel=" << cart_ie_err.max_rel
	          << "  (N=" << cart_ie_err.count << ")\n";
	print_worst("int_ener", cart_ie_err);
	std::cout << "vx:        abs=" << cart_vx_err.max_abs << "  rel=" << cart_vx_err.max_rel
	          << "  (N=" << cart_vx_err.count << ")\n";
	print_worst("vx", cart_vx_err);
	std::cout << "  interp=" << cart_vx_err.worst_interp
	          << "  exact=" << cart_vx_err.worst_exact
	          << "  cell_index=" << cart_vx_err.worst_cell_index
	          << "  cell_id=" << cart_vx_err.worst_cell_id << "\n";
	std::cout << "vy:        abs=" << cart_vy_err.max_abs << "  rel=" << cart_vy_err.max_rel
	          << "  (N=" << cart_vy_err.count << ")\n";
	print_worst("vy", cart_vy_err);
	std::cout << "  interp=" << cart_vy_err.worst_interp
	          << "  exact=" << cart_vy_err.worst_exact
	          << "  cell_index=" << cart_vy_err.worst_cell_index
	          << "  cell_id=" << cart_vy_err.worst_cell_id << "\n";
	std::cout << "vz:        abs=" << cart_vz_err.max_abs << "  rel=" << cart_vz_err.max_rel
	          << "  (N=" << cart_vz_err.count << ")\n";
	print_worst("vz", cart_vz_err);
	std::cout << "  interp=" << cart_vz_err.worst_interp
	          << "  exact=" << cart_vz_err.worst_exact
	          << "  cell_index=" << cart_vz_err.worst_cell_index
	          << "  cell_id=" << cart_vz_err.worst_cell_id << "\n";

	double cart_scalar_max_rel = std::max(cart_rho_err.max_rel, cart_ie_err.max_rel);
	double cart_vel_max_rel = std::max({cart_vx_err.max_rel, cart_vy_err.max_rel, cart_vz_err.max_rel});
	std::cout << "\nCartesian scalar max relative error: " << cart_scalar_max_rel << std::endl;
	std::cout << "Cartesian velocity max relative error: " << cart_vel_max_rel << std::endl;

	std::cout << "\n--- Spherical per-field max absolute / relative errors ---\n";
	std::cout << "density:   abs=" << sph_rho_err.max_abs << "  rel=" << sph_rho_err.max_rel
	          << "  (N=" << sph_rho_err.count << ")\n";
	print_worst("density", sph_rho_err);
	std::cout << "int_ener:  abs=" << sph_ie_err.max_abs << "  rel=" << sph_ie_err.max_rel
	          << "  (N=" << sph_ie_err.count << ")\n";
	print_worst("int_ener", sph_ie_err);
	std::cout << "velocity:  abs=" << sph_vel_err.max_abs << "  rel=" << sph_vel_err.max_rel
	          << "  (|dv|/|v|, N=" << sph_vel_err.count << ")\n";
	print_worst("velocity", sph_vel_err);
	std::cout << "  interp=(" << sph_vel_worst_interp.x << ", " << sph_vel_worst_interp.y
	          << ", " << sph_vel_worst_interp.z << ")  |interp|=" << abs(sph_vel_worst_interp) << "\n";
	std::cout << "  exact =(" << sph_vel_worst_exact.x << ", " << sph_vel_worst_exact.y
	          << ", " << sph_vel_worst_exact.z << ")  |exact|=" << abs(sph_vel_worst_exact) << "\n";
	std::cout << "  dcomp =(" << (sph_vel_worst_interp.x - sph_vel_worst_exact.x)
	          << ", " << (sph_vel_worst_interp.y - sph_vel_worst_exact.y)
	          << ", " << (sph_vel_worst_interp.z - sph_vel_worst_exact.z) << ")"
	          << "  cell_index=" << sph_vel_worst_cell_index
	          << "  cell_id=" << sph_vel_worst_cell_id << "\n";

	double sph_scalar_max_rel = std::max(sph_rho_err.max_rel, sph_ie_err.max_rel);
	double sph_vel_max_rel = sph_vel_err.max_rel;
	std::cout << "\nSpherical scalar max relative error: " << sph_scalar_max_rel << std::endl;
	std::cout << "Spherical velocity max relative error: " << sph_vel_max_rel << std::endl;

	{
		std::ofstream mf("cart_gauss_linear_metrics.txt");
		mf << std::scientific << std::setprecision(12);
		mf << "cart_scalar_max_rel_error " << cart_scalar_max_rel << "\n";
		mf << "cart_velocity_max_rel_error " << cart_vel_max_rel << "\n";
		mf << "cart_density_max_rel " << cart_rho_err.max_rel << "\n";
		mf << "cart_ie_max_rel " << cart_ie_err.max_rel << "\n";
		mf << "cart_vx_max_abs " << cart_vx_err.max_abs << "\n";
		mf << "cart_vy_max_abs " << cart_vy_err.max_abs << "\n";
		mf << "cart_vz_max_abs " << cart_vz_err.max_abs << "\n";
		mf << "faces_checked " << cart_rho_err.count << "\n";
		mf << "sph_scalar_max_rel_error " << sph_scalar_max_rel << "\n";
		mf << "sph_velocity_max_rel_error " << sph_vel_max_rel << "\n";
		mf << "sph_density_max_rel " << sph_rho_err.max_rel << "\n";
		mf << "sph_ie_max_rel " << sph_ie_err.max_rel << "\n";
		mf << "sph_vel_max_abs " << sph_vel_err.max_abs << "\n";
	}

	double scalar_tol = 1e-6;
	double vel_tol = 0.1;
	double sph_vel_tol = 0.5;

	bool pass = (cart_scalar_max_rel < scalar_tol) && (cart_vel_max_rel < vel_tol) &&
	            (cart_rho_err.count > 0) &&
	            (cart_scalar_max_rel < sph_scalar_max_rel) &&
	            (sph_vel_max_rel < sph_vel_tol);
	std::cout << (pass ? "PASS" : "FAIL") << std::endl;
	if (cart_vel_max_rel >= vel_tol)
		std::cout << "Cartesian velocity error " << cart_vel_max_rel
		          << " exceeds tolerance " << vel_tol << std::endl;
	if (sph_vel_max_rel >= sph_vel_tol)
		std::cout << "Spherical velocity error " << sph_vel_max_rel
		          << " exceeds tolerance " << sph_vel_tol << std::endl;
	return pass ? 0 : 1;
}
