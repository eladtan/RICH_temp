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
#include "source/newtonian/three_dimensional/SphericalLinearGauss3D.hpp"
#include "source/newtonian/three_dimensional/LinearGauss3D.hpp"
#include "source/misc/universal_error.hpp"

namespace {

struct SphCoords { double r, theta, phi; };

SphCoords cart_to_sph(Vector3D const& p)
{
	double r = abs(p);
	if (r < 1e-14)
		return {0, 0, 0};
	double theta = std::acos(std::max(-1.0, std::min(1.0, p.z / r)));
	double phi = std::atan2(p.y, p.x);
	return {r, theta, phi};
}

void sph_basis(double theta, double phi,
	Vector3D &e_r, Vector3D &e_theta, Vector3D &e_phi)
{
	double st = std::sin(theta), ct = std::cos(theta);
	double sp = std::sin(phi), cp = std::cos(phi);
	e_r = Vector3D(st*cp, st*sp, ct);
	e_theta = Vector3D(ct*cp, ct*sp, -st);
	e_phi = Vector3D(-sp, cp, 0);
}

// Linear-in-spherical-coordinates coefficients: f(r,theta,phi) = a0 + a1*r + a2*theta + a3*phi
struct SphLinCoeffs {
	double a0, a1, a2, a3;
	double eval(SphCoords const& s) const { return a0 + a1*s.r + a2*s.theta + a3*s.phi; }
};

SphLinCoeffs const rho_c  = { 5.0,  0.3, -0.2,  0.0 };
SphLinCoeffs const pres_c = { 3.0, -0.1,  0.4,  0.0 };
SphLinCoeffs const ie_c   = { 2.0,  0.2,  0.1,  0.0 };

// Only the radial component varies spatially; angular components are constant
// so the mixed-frame LSQ captures them exactly at the poles.
SphLinCoeffs const vr_c   = { 1.0,  0.1,  0.05, 0.0 };
SphLinCoeffs const vt_c   = { 0.0,  0.0,  0.0,  0.0 };
SphLinCoeffs const vp_c   = { 0.0,  0.0,  0.0,  0.0 };

Vector3D sph_vel_to_cart(double vr, double vt, double vp,
	Vector3D const& e_r, Vector3D const& e_theta, Vector3D const& e_phi)
{
	return e_r * vr + e_theta * vt + e_phi * vp;
}

void fill_cell(ComputationalCell3D &cell, Vector3D const& pos)
{
	SphCoords s = cart_to_sph(pos);
	cell.density = rho_c.eval(s);
	cell.pressure = pres_c.eval(s);
	cell.internal_energy = ie_c.eval(s);

	Vector3D er, et, ep;
	sph_basis(s.theta, s.phi, er, et, ep);
	double vr = vr_c.eval(s), vt = vt_c.eval(s), vp = vp_c.eval(s);
	cell.velocity = sph_vel_to_cart(vr, vt, vp, er, et, ep);
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
	void update(double interp_v, double exact_v, Vector3D const& pos,
		size_t cell_index, size_t cell_id) {
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
	Vector3D origin(0, 0, 0);
	SphericalLinearGauss3D interp(eos, ghost, origin,
	    /*slf=*/false, /*delta_v=*/0.2, /*theta=*/0.5,
	    /*delta_P=*/0.7, /*SR=*/false, /*calc_tracers=*/{},
	    /*skip_key=*/"", /*pressure_calc=*/false,
	    /*apply_principal_limit=*/false, /*velocity_radial_extrapolation=*/false);

	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> face_values;
	try {
		interp(tess, cells, 0.0, face_values);
	} catch (UniversalError const& eo) {
		reportError(eo);
		return 1;
	}
	std::cout << "Spherical interpolation computed for " << face_values.size() << " faces" << std::endl;

	LinearGauss3D cart_interp(eos, ghost,
	    /*slf=*/true, /*delta_v=*/0.2, /*theta=*/0.5,
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

	FieldErrors rho_err, ie_err, vx_err, vy_err, vz_err;
	FieldErrors cart_rho_err, cart_ie_err, cart_vx_err, cart_vy_err, cart_vz_err;

	for (size_t f = 0; f < tess.GetTotalFacesNumber(); ++f) {
		auto const& neigh = tess.GetFaceNeighbors(f);
		if (neigh.first >= Nlocal || neigh.second >= Nlocal)
			continue;

		Vector3D cm1 = tess.GetCellCM(neigh.first);
		Vector3D cm2 = tess.GetCellCM(neigh.second);
		double r1 = abs(cm1), r2 = abs(cm2);
		if (r1 < 1.1 || r1 > 1.9 || r2 < 1.1 || r2 > 1.9)
			continue;
		SphCoords const cs1 = cart_to_sph(cm1);
		SphCoords const cs2 = cart_to_sph(cm2);
		// This case validates the spherical reconstruction itself.  Cells whose
		// stencil reaches the polar axis intentionally use the Cartesian fallback
		// and are covered by cartesian_gauss_linear instead.
		if (r1 * std::abs(std::sin(cs1.theta)) <= 3.0 * tess.GetWidth(neigh.first) ||
			r2 * std::abs(std::sin(cs2.theta)) <= 3.0 * tess.GetWidth(neigh.second))
			continue;

		Vector3D face_cm = tess.FaceCM(f);
		SphCoords fs = cart_to_sph(face_cm);

		double exact_rho = rho_c.eval(fs);
		double exact_ie = ie_c.eval(fs);

		Vector3D fer, fet, fep;
		sph_basis(fs.theta, fs.phi, fer, fet, fep);
		double exact_vr = vr_c.eval(fs), exact_vt = vt_c.eval(fs), exact_vp = vp_c.eval(fs);
		Vector3D exact_vel = sph_vel_to_cart(exact_vr, exact_vt, exact_vp, fer, fet, fep);

		for (int side = 0; side < 2; ++side) {
			size_t src_index = static_cast<size_t>((side == 0) ? neigh.first : neigh.second);
			size_t src_id = static_cast<size_t>(cells[src_index].ID);

			ComputationalCell3D const& sv =
			    (side == 0) ? face_values[f].first : face_values[f].second;
			rho_err.update(sv.density, exact_rho, face_cm, src_index, src_id);
			ie_err.update(sv.internal_energy, exact_ie, face_cm, src_index, src_id);
			vx_err.update(sv.velocity.x, exact_vel.x, face_cm, src_index, src_id);
			vy_err.update(sv.velocity.y, exact_vel.y, face_cm, src_index, src_id);
			vz_err.update(sv.velocity.z, exact_vel.z, face_cm, src_index, src_id);

			ComputationalCell3D const& cv =
			    (side == 0) ? cart_face_values[f].first : cart_face_values[f].second;
			cart_rho_err.update(cv.density, exact_rho, face_cm, src_index, src_id);
			cart_ie_err.update(cv.internal_energy, exact_ie, face_cm, src_index, src_id);
			cart_vx_err.update(cv.velocity.x, exact_vel.x, face_cm, src_index, src_id);
			cart_vy_err.update(cv.velocity.y, exact_vel.y, face_cm, src_index, src_id);
			cart_vz_err.update(cv.velocity.z, exact_vel.z, face_cm, src_index, src_id);
		}
	}

	std::cout << std::scientific << std::setprecision(6);

	auto print_worst = [](std::string const& label, FieldErrors const& fe) {
		SphCoords ws = cart_to_sph(fe.worst_pos);
		std::cout << "  worst " << label << " at cart=("
		          << fe.worst_pos.x << ", " << fe.worst_pos.y << ", " << fe.worst_pos.z
		          << ")  sph=(r=" << ws.r << ", theta=" << ws.theta << ", phi=" << ws.phi << ")\n";
	};

	std::cout << "\n--- Spherical per-field max absolute / relative errors ---\n";
	std::cout << "density:   abs=" << rho_err.max_abs << "  rel=" << rho_err.max_rel
	          << "  (N=" << rho_err.count << ")\n";
	print_worst("density", rho_err);
	std::cout << "int_ener:  abs=" << ie_err.max_abs << "  rel=" << ie_err.max_rel
	          << "  (N=" << ie_err.count << ")\n";
	print_worst("int_ener", ie_err);
	std::cout << "vx:        abs=" << vx_err.max_abs << "  rel=" << vx_err.max_rel
	          << "  (N=" << vx_err.count << ")\n";
	print_worst("vx", vx_err);
	std::cout << "  interp=" << vx_err.worst_interp
	          << "  exact=" << vx_err.worst_exact
	          << "  cell_index=" << vx_err.worst_cell_index
	          << "  cell_id=" << vx_err.worst_cell_id << "\n";
	std::cout << "vy:        abs=" << vy_err.max_abs << "  rel=" << vy_err.max_rel
	          << "  (N=" << vy_err.count << ")\n";
	print_worst("vy", vy_err);
	std::cout << "  interp=" << vy_err.worst_interp
	          << "  exact=" << vy_err.worst_exact
	          << "  cell_index=" << vy_err.worst_cell_index
	          << "  cell_id=" << vy_err.worst_cell_id << "\n";
	std::cout << "vz:        abs=" << vz_err.max_abs << "  rel=" << vz_err.max_rel
	          << "  (N=" << vz_err.count << ")\n";
	print_worst("vz", vz_err);
	std::cout << "  interp=" << vz_err.worst_interp
	          << "  exact=" << vz_err.worst_exact
	          << "  cell_index=" << vz_err.worst_cell_index
	          << "  cell_id=" << vz_err.worst_cell_id << "\n";

	double scalar_max_rel = std::max(rho_err.max_rel, ie_err.max_rel);
	double vel_max_rel = std::max(vx_err.max_rel, std::max(vy_err.max_rel, vz_err.max_rel));
	std::cout << "\nSpherical scalar max relative error: " << scalar_max_rel << std::endl;
	std::cout << "Spherical velocity max relative error: " << vel_max_rel << std::endl;

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
	double cart_vel_max_rel = std::max(cart_vx_err.max_rel, std::max(cart_vy_err.max_rel, cart_vz_err.max_rel));
	std::cout << "\nCartesian scalar max relative error: " << cart_scalar_max_rel << std::endl;
	std::cout << "Cartesian velocity max relative error: " << cart_vel_max_rel << std::endl;

	{
		std::ofstream mf("gauss_linear_metrics.txt");
		mf << std::scientific << std::setprecision(12);
		mf << "scalar_max_rel_error " << scalar_max_rel << "\n";
		mf << "velocity_max_rel_error " << vel_max_rel << "\n";
		mf << "density_max_rel " << rho_err.max_rel << "\n";
		mf << "ie_max_rel " << ie_err.max_rel << "\n";
		mf << "sph_vx_max_abs " << vx_err.max_abs << "\n";
		mf << "sph_vy_max_abs " << vy_err.max_abs << "\n";
		mf << "sph_vz_max_abs " << vz_err.max_abs << "\n";
		mf << "faces_checked " << rho_err.count << "\n";
		mf << "cart_scalar_max_rel_error " << cart_scalar_max_rel << "\n";
		mf << "cart_velocity_max_rel_error " << cart_vel_max_rel << "\n";
		mf << "cart_density_max_rel " << cart_rho_err.max_rel << "\n";
		mf << "cart_ie_max_rel " << cart_ie_err.max_rel << "\n";
		mf << "cart_vx_max_abs " << cart_vx_err.max_abs << "\n";
		mf << "cart_vy_max_abs " << cart_vy_err.max_abs << "\n";
		mf << "cart_vz_max_abs " << cart_vz_err.max_abs << "\n";
		mf << "cart_faces_checked " << cart_rho_err.count << "\n";
	}

	double scalar_tol = 1e-8;
	double vel_tol = 0.1;

	bool pass = (scalar_max_rel < scalar_tol) && (vel_max_rel < vel_tol) &&
	            (rho_err.count > 0) &&
	            (scalar_max_rel < cart_scalar_max_rel);
	std::cout << (pass ? "PASS" : "FAIL") << std::endl;
	return pass ? 0 : 1;
}
