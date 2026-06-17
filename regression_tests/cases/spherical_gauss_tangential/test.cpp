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

struct SphLinCoeffs {
	double a0, a1, a2, a3;
	double eval(SphCoords const& s) const { return a0 + a1*s.r + a2*s.theta + a3*s.phi; }
};

SphLinCoeffs const rho_c  = { 5.0,  0.3, -0.2,  0.0 };
SphLinCoeffs const pres_c = { 3.0, -0.1,  0.4,  0.0 };
SphLinCoeffs const ie_c   = { 2.0,  0.2,  0.1,  0.0 };

// Constant-vphi face-basis smoke test.
// v_phi = 0.3 everywhere, v_r = v_theta = 0. The spherical component
// is constant so LSQ reconstructs it exactly; only the final
// sph-to-cart conversion matters. The old mixed-basis code fails this
// because it uses e_phi(cell) instead of e_phi(face). A separate
// angularly varying test (e.g. v_phi = v*sin(phi)) would additionally
// exercise slope accuracy but is not the purpose of this case.
SphLinCoeffs const vr_c   = { 0.0,  0.0,  0.0,  0.0 };
SphLinCoeffs const vt_c   = { 0.0,  0.0,  0.0,  0.0 };
SphLinCoeffs const vp_c   = { 0.3,  0.0,  0.0,  0.0 };

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
	size_t count;
	FieldErrors() : max_abs(0), max_rel(0), count(0) {}
	void update(double interp_v, double exact_v) {
		double err = std::abs(interp_v - exact_v);
		double scale = std::max(std::abs(exact_v), 1e-10);
		double rel = err / scale;
		if (err > max_abs) max_abs = err;
		if (rel > max_rel) max_rel = rel;
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

	FieldErrors vx_err, vy_err, vz_err;
	size_t faces_checked = 0;

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
		SphCoords fs = cart_to_sph(face_cm);

		Vector3D fer, fet, fep;
		sph_basis(fs.theta, fs.phi, fer, fet, fep);
		Vector3D exact_vel = sph_vel_to_cart(0.0, 0.0, 0.3, fer, fet, fep);

		for (int side = 0; side < 2; ++side) {
			ComputationalCell3D const& sv =
			    (side == 0) ? face_values[f].first : face_values[f].second;
			vx_err.update(sv.velocity.x, exact_vel.x);
			vy_err.update(sv.velocity.y, exact_vel.y);
			vz_err.update(sv.velocity.z, exact_vel.z);
		}
		++faces_checked;
	}

	double vel_max_abs = std::max(vx_err.max_abs, std::max(vy_err.max_abs, vz_err.max_abs));
	double vel_max_rel = std::max(vx_err.max_rel, std::max(vy_err.max_rel, vz_err.max_rel));

	std::cout << std::scientific << std::setprecision(6);
	std::cout << "\n--- Tangential face-basis velocity errors ---\n";
	std::cout << "vx:  abs=" << vx_err.max_abs << "  rel=" << vx_err.max_rel << "\n";
	std::cout << "vy:  abs=" << vy_err.max_abs << "  rel=" << vy_err.max_rel << "\n";
	std::cout << "vz:  abs=" << vz_err.max_abs << "  rel=" << vz_err.max_rel << "\n";
	std::cout << "max: abs=" << vel_max_abs << "  rel=" << vel_max_rel << "\n";
	std::cout << "faces_checked: " << faces_checked << "\n";

	double vel_abs_tol = 5e-10;
	double vel_rel_tol = 5e-10;
	bool pass = (vel_max_abs < vel_abs_tol) && (vel_max_rel < vel_rel_tol) && (faces_checked > 0);

	{
		std::ofstream mf("spherical_gauss_tangential_metrics.txt");
		mf << std::scientific << std::setprecision(12);
		mf << "sph_tangential_vx_max_abs " << vx_err.max_abs << "\n";
		mf << "sph_tangential_vy_max_abs " << vy_err.max_abs << "\n";
		mf << "sph_tangential_vz_max_abs " << vz_err.max_abs << "\n";
		mf << "sph_tangential_velocity_max_abs " << vel_max_abs << "\n";
		mf << "sph_tangential_velocity_max_rel " << vel_max_rel << "\n";
		mf << "faces_checked " << faces_checked << "\n";
		mf << "pass " << (pass ? 1 : 0) << "\n";
	}

	std::cout << (pass ? "PASS" : "FAIL") << std::endl;
	return pass ? 0 : 1;
}
