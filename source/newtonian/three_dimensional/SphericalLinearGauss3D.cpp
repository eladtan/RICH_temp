#include "SphericalLinearGauss3D.hpp"
#include "../../misc/utils.hpp"
#include <array>
#include <iostream>
#include <cstring>
#include <fstream>
#include <cmath>
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
	// ========== Common helpers ==========

	void CheckCell(ComputationalCell3D const& cell)
	{
		if ((!(cell.density > 0)) || (!(cell.internal_energy > 0)) || (!std::isfinite(cell.velocity.x)) || (!std::isfinite(cell.velocity.y)) || (!std::isfinite(cell.velocity.z)))
			throw UniversalError("Bad cell after interpolation in SphericalLinearGauss3D");
	}

	inline Vector3D cart_to_sph_coords(Vector3D const& pos, Vector3D const& origin)
	{
		Vector3D d = pos - origin;
		double r = abs(d);
		if (r < 1e-14)
			return Vector3D(0, 0, 0);
		double theta = std::acos(std::max(-1.0, std::min(1.0, d.z / r)));
		double phi = std::atan2(d.y, d.x);
		return Vector3D(r, theta, phi);
	}

	inline void sph_basis_at(double theta, double phi,
		Vector3D &e_r, Vector3D &e_theta, Vector3D &e_phi)
	{
		double st = std::sin(theta), ct = std::cos(theta);
		double sp = std::sin(phi), cp = std::cos(phi);
		e_r = Vector3D(st * cp, st * sp, ct);
		e_theta = Vector3D(ct * cp, ct * sp, -st);
		e_phi = Vector3D(-sp, cp, 0);
	}

	inline double wrap_dphi(double dp)
	{
		return std::atan2(std::sin(dp), std::cos(dp));
	}

	inline Vector3D cart_to_sph_vec(Vector3D const& v,
		Vector3D const& e_r, Vector3D const& e_theta, Vector3D const& e_phi)
	{
		return Vector3D(ScalarProd(e_r, v), ScalarProd(e_theta, v), ScalarProd(e_phi, v));
	}

	inline Vector3D sph_to_cart_vec(Vector3D const& v_sph,
		Vector3D const& e_r, Vector3D const& e_theta, Vector3D const& e_phi)
	{
		return e_r * v_sph.x + e_theta * v_sph.y + e_phi * v_sph.z;
	}

	inline Vector3D sph_velocity_theta_derivative(Vector3D const& v_sph,
		Vector3D const& dv_dtheta)
	{
		return Vector3D(dv_dtheta.x - v_sph.y, dv_dtheta.y + v_sph.x,
			dv_dtheta.z);
	}

	inline Vector3D sph_velocity_phi_derivative(Vector3D const& v_sph,
		Vector3D const& dv_dphi, double theta)
	{
		double st = std::sin(theta), ct = std::cos(theta);
		return Vector3D(dv_dphi.x - st * v_sph.z,
			dv_dphi.y - ct * v_sph.z,
			dv_dphi.z + st * v_sph.x + ct * v_sph.y);
	}

	void GetNeighborCM(Tessellation3D const& tess, size_t cell_index,
		vector<Vector3D> &res, face_vec const& faces)
	{
		res.resize(faces.size());
		const size_t nloop = faces.size();
		for (size_t i = 0; i < nloop; ++i)
		{
			if (tess.GetFaceNeighbors(faces[i]).first == cell_index)
				res[i] = tess.GetCellCM(tess.GetFaceNeighbors(faces[i]).second);
			else
				res[i] = tess.GetCellCM(tess.GetFaceNeighbors(faces[i]).first);
		}
	}

	void GetNeighborCells(Tessellation3D const& tess, size_t cell_index,
		vector<ComputationalCell3D> const& cells, face_vec const& faces, vector<ComputationalCell3D> &res)
	{
		const size_t nloop = faces.size();
		res.resize(nloop);
		for (size_t i = 0; i < nloop; ++i)
		{
			size_t other_cell = (tess.GetFaceNeighbors(faces[i]).first == cell_index) ?
				tess.GetFaceNeighbors(faces[i]).second : tess.GetFaceNeighbors(faces[i]).first;
			ReplaceComputationalCell(res[i], cells[other_cell]);
		}
	}

	inline bool is_near_pole(double r, double theta, double cell_width)
	{
		if (!(r > 0.0) || !(cell_width > 0.0) ||
			!std::isfinite(r) || !std::isfinite(theta) ||
			!std::isfinite(cell_width))
			return true;

		// Spherical azimuth is ill-conditioned when a cell stencil reaches the
		// polar axis.  Use the existing Cartesian reconstruction for those cells;
		// scaling by the local cell width keeps the criterion mesh independent.
		double const distance_to_axis = r * std::abs(std::sin(theta));
		return distance_to_axis <= 3.0 * cell_width;
	}

	inline bool is_ghost_index(size_t index, size_t local_count)
	{
		return index >= local_count;
	}

	inline bool is_valid_radius(double r)
	{
		return std::isfinite(r) && r > 0;
	}

	inline bool same_shell_radius(double r1, double r2,
		double abs_tol, double rel_tol)
	{
		double scale = std::max(std::max(r1, r2), 1.0);
		return std::abs(r1 - r2) <= abs_tol + rel_tol * scale;
	}

	bool same_shell_face(Tessellation3D const& tess,
		size_t cell_index, size_t face, Vector3D const& origin,
		SphericalLinearGauss3D::FaceRadiusPolicy policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol)
	{
		if (policy == SphericalLinearGauss3D::FaceRadiusPolicy::PhysicalFaceCM)
			return false;
		if (tess.BoundaryFace(face))
			return false;

		auto neigh = tess.GetFaceNeighbors(face);
		if (neigh.first != cell_index && neigh.second != cell_index)
			return false;
		size_t const other = (neigh.first == cell_index) ? neigh.second : neigh.first;

		// Non-boundary ghost points are intentionally handled like local points:
		// if the tessellation has a mesh point for both sides, it can define the shell radius.
		double const r1 = abs(tess.GetMeshPoint(cell_index) - origin);
		double const r2 = abs(tess.GetMeshPoint(other) - origin);
		if (!is_valid_radius(r1) || !is_valid_radius(r2))
			return false;

		return same_shell_radius(r1, r2, shell_radius_abs_tol, shell_radius_rel_tol);
	}

	Vector3D effective_face_sph(Tessellation3D const& tess,
		size_t cell_index, size_t face, Vector3D const& origin,
		Vector3D const& cell_coords, Vector3D const& physical_face_sph,
		SphericalLinearGauss3D::FaceRadiusPolicy policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol)
	{
		Vector3D res = physical_face_sph;
		if (same_shell_face(tess, cell_index, face, origin, policy,
			shell_radius_abs_tol, shell_radius_rel_tol))
			res.x = cell_coords.x;
		return res;
	}

	void build_effective_face_sph_cache(Tessellation3D const& tess,
		size_t cell_index, face_vec const& faces, Vector3D const& origin,
		Vector3D const& cell_coords, vector<Vector3D> const& physical_face_sph_cache,
		SphericalLinearGauss3D::FaceRadiusPolicy policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol,
		vector<Vector3D>& effective_face_sph_cache)
	{
		effective_face_sph_cache = physical_face_sph_cache;
		if (policy == SphericalLinearGauss3D::FaceRadiusPolicy::PhysicalFaceCM)
			return;

		for (size_t i = 0; i < faces.size(); ++i)
			effective_face_sph_cache[i] = effective_face_sph(tess, cell_index, faces[i],
				origin, cell_coords, physical_face_sph_cache[i], policy,
				shell_radius_abs_tol, shell_radius_rel_tol);
	}

	// ========== Cartesian (LinearGauss3D) utilities for pole cells ==========

	void cart_GetNeighborMesh(Tessellation3D const& tess, size_t cell_index,
		vector<Vector3D> &res, face_vec const& faces)
	{
		res.resize(faces.size());
		const size_t nloop = faces.size();
		for (size_t i = 0; i < nloop; ++i)
		{
			auto neigh = tess.GetFaceNeighbors(faces[i]);
			if (neigh.first == cell_index)
				res[i] = tess.GetMeshPoint(neigh.second);
			else
				res[i] = tess.GetMeshPoint(neigh.first);
		}
	}

	void cart_calc_naive_slope(ComputationalCell3D const& cell,
		Vector3D const& center, Vector3D const& cell_cm, double cell_volume,
		vector<ComputationalCell3D> const& neighbors,
		vector<Vector3D> const& neighbor_centers,
		vector<Vector3D> const& neigh_cm, Tessellation3D const& tess,
		Slope3D &res, Slope3D &temp, size_t /*index*/, face_vec const& faces,
		std::vector<Vector3D> &c_ij,
		vector<Vector3D>& face_cms_cache, vector<double>& face_areas_cache)
	{
		size_t n = neighbor_centers.size();
		std::array<double, 9> m;
		std::fill_n(m.begin(), 9, 0.0);
		c_ij.resize(n);

		face_cms_cache.resize(n);
		face_areas_cache.resize(n);
		for (size_t i = 0; i < n; ++i)
		{
			face_cms_cache[i] = tess.FaceCM(faces[i]);
			face_areas_cache[i] = tess.GetArea(faces[i]);
		}

		for (size_t i = 0; i < n; i++)
		{
			c_ij[i] = neigh_cm[i];
			c_ij[i] += cell_cm;
			c_ij[i] *= -0.5;
			c_ij[i] += face_cms_cache[i];
			const Vector3D r_ij = normalize(neighbor_centers[i] - center);
			const double A = face_areas_cache[i];
			m[0] -= c_ij[i].x*r_ij.x*A;
			m[1] -= c_ij[i].y*r_ij.x*A;
			m[2] -= c_ij[i].z*r_ij.x*A;
			m[3] -= c_ij[i].x*r_ij.y*A;
			m[4] -= c_ij[i].y*r_ij.y*A;
			m[5] -= c_ij[i].z*r_ij.y*A;
			m[6] -= c_ij[i].x*r_ij.z*A;
			m[7] -= c_ij[i].y*r_ij.z*A;
			m[8] -= c_ij[i].z*r_ij.z*A;
			if (i == 0)
			{
				ReplaceComputationalCell(temp.xderivative, neighbors[i]);
				temp.xderivative *= r_ij.x*A;
				ReplaceComputationalCell(temp.yderivative, neighbors[i]);
				temp.yderivative *= r_ij.y*A;
				ReplaceComputationalCell(temp.zderivative, neighbors[i]);
				temp.zderivative *= r_ij.z*A;
			}
			else
			{
				ComputationalCellAddMult(temp.xderivative, neighbors[i], r_ij.x*A);
				ComputationalCellAddMult(temp.yderivative, neighbors[i], r_ij.y*A);
				ComputationalCellAddMult(temp.zderivative, neighbors[i], r_ij.z*A);
			}
			ComputationalCellAddMult(temp.xderivative, cell, r_ij.x*A);
			ComputationalCellAddMult(temp.yderivative, cell, r_ij.y*A);
			ComputationalCellAddMult(temp.zderivative, cell, r_ij.z*A);
		}
		double v_inv = 1.0 / cell_volume;
		for (size_t i = 0; i < 9; ++i)
			m[i] *= v_inv;
		m[0] += 1;
		m[4] += 1;
		m[8] += 1;
		const double det = -m[2]*m[4]*m[6] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7]
			- m[0]*m[5]*m[7] - m[1]*m[3]*m[8] + m[0]*m[4]*m[8];
		if (std::abs(det) < 1e-10)
		{
			UniversalError eo("Singular matrix in cart_calc_naive_slope (pole cell)");
			eo.addEntry("Cell x cor", center.x);
			eo.addEntry("Cell y cor", center.y);
			eo.addEntry("Cell z cor", center.z);
			eo.addEntry("Cell volume", cell_volume);
			eo.addEntry("Det was", det);
			throw eo;
		}
		std::array<double, 9> m_inv;
		m_inv[0] = m[4]*m[8] - m[5]*m[7];
		m_inv[1] = m[2]*m[7] - m[1]*m[8];
		m_inv[2] = m[1]*m[5] - m[2]*m[4];
		m_inv[3] = m[5]*m[6] - m[3]*m[8];
		m_inv[4] = m[0]*m[8] - m[2]*m[6];
		m_inv[5] = m[2]*m[3] - m[5]*m[0];
		m_inv[6] = m[3]*m[7] - m[6]*m[4];
		m_inv[7] = m[6]*m[1] - m[0]*m[7];
		m_inv[8] = m[4]*m[0] - m[1]*m[3];
		for (size_t i = 0; i < 9; ++i)
			m_inv[i] /= (2 * cell_volume * det);

		ReplaceComputationalCell(res.xderivative, temp.xderivative);
		res.xderivative *= m_inv[0];
		ComputationalCellAddMult(res.xderivative, temp.yderivative, m_inv[1]);
		ComputationalCellAddMult(res.xderivative, temp.zderivative, m_inv[2]);

		ReplaceComputationalCell(res.yderivative, temp.xderivative);
		res.yderivative *= m_inv[3];
		ComputationalCellAddMult(res.yderivative, temp.yderivative, m_inv[4]);
		ComputationalCellAddMult(res.yderivative, temp.zderivative, m_inv[5]);

		ReplaceComputationalCell(res.zderivative, temp.xderivative);
		res.zderivative *= m_inv[6];
		ComputationalCellAddMult(res.zderivative, temp.yderivative, m_inv[7]);
		ComputationalCellAddMult(res.zderivative, temp.zderivative, m_inv[8]);
	}

	double cart_shock_weight(Slope3D const& naive_slope, double cell_width, double shock_ratio,
		ComputationalCell3D const& cell, vector<ComputationalCell3D> const& neighbor_list,
		double pressure_ratio, double cs)
	{
		double div_v = (naive_slope.xderivative.velocity.x + naive_slope.yderivative.velocity.y +
			naive_slope.zderivative.velocity.z) * cell_width;
		double t_div = std::max(0.0, std::min(1.0, -div_v / (shock_ratio * cs)));
		double p = cell.pressure;
		double p_ratio = 1.0;
		for (size_t i = 0; i < neighbor_list.size(); i++)
		{
			if (p > neighbor_list[i].pressure)
				p_ratio = std::min(p_ratio, neighbor_list[i].pressure / p);
			else
				p_ratio = std::min(p_ratio, p / neighbor_list[i].pressure);
		}
		double t_pres = std::max(0.0, std::min(1.0, (pressure_ratio - p_ratio) / (0.2 * pressure_ratio)));
		return std::max(t_div, t_pres);
	}

	bool cart_build_principal_frame(Vector3D const& velocity, Vector3D &e1, Vector3D &e2, Vector3D &e3)
	{
		double vmag = abs(velocity);
		if (vmag < 1e-14)
			return false;
		e1 = velocity / vmag;
		double sign_z = std::copysign(1.0, e1.z);
		double a = -1.0 / (sign_z + e1.z);
		double b = e1.x * e1.y * a;
		e2 = Vector3D(1.0 + sign_z * e1.x * e1.x * a, sign_z * b, -sign_z * e1.x);
		e3 = Vector3D(b, sign_z + e1.y * e1.y * a, -e1.y);
		return true;
	}

	void cart_apply_principal_limit(Vector3D &sv, Vector3D const& e1, Vector3D const& e2,
		Vector3D const& e3, double psi1, double psi2, double psi3)
	{
		double c1 = ScalarProd(sv, e1);
		double c2 = ScalarProd(sv, e2);
		double c3 = ScalarProd(sv, e3);
		sv.x = e1.x*c1*psi1 + e2.x*c2*psi2 + e3.x*c3*psi3;
		sv.y = e1.y*c1*psi1 + e2.y*c2*psi2 + e3.y*c3*psi3;
		sv.z = e1.z*c1*psi1 + e2.z*c2*psi2 + e3.z*c3*psi3;
	}

	ComputationalCell3D cart_interp(ComputationalCell3D const& cell, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm, EquationOfState const& eos,
		bool pressure_calc)
	{
		ComputationalCell3D res(cell);
		ComputationalCellAddMult(res, slope.xderivative, target.x - cm.x);
		ComputationalCellAddMult(res, slope.yderivative, target.y - cm.y);
		ComputationalCellAddMult(res, slope.zderivative, target.z - cm.z);
		if (pressure_calc)
			try
		{
			res.internal_energy = eos.dp2e(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("density", res.density);
			eo.addEntry("internal energy", res.internal_energy);
			throw eo;
		}
		return res;
	}

	void cart_interp23Dsimple(ComputationalCell3D &res, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm)
	{
		ComputationalCellAddMult3(res, slope.xderivative, slope.yderivative, slope.zderivative,
			target.x - cm.x, target.y - cm.y, target.z - cm.z);
	}

	void cart_interp23D(ComputationalCell3D &res, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm, EquationOfState const& eos,
		bool pressure_calc)
	{
		ComputationalCellAddMult3(res, slope.xderivative, slope.yderivative, slope.zderivative,
			target.x - cm.x, target.y - cm.y, target.z - cm.z);
		try
		{
			if (!pressure_calc)
				res.pressure = eos.de2p(res.density, res.internal_energy, res.tracers, ComputationalCell3D::tracerNames);
			else
				res.internal_energy = eos.dp2e(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("density", res.density);
			eo.addEntry("internal energy", res.internal_energy);
			throw eo;
		}
	}

	void cart_blended_slope_limit(ComputationalCell3D const& cell, Vector3D const& cm,
		vector<ComputationalCell3D> const& neighbors, Slope3D &slope, ComputationalCell3D &cmax,
		ComputationalCell3D &cmin, ComputationalCell3D &maxdiff, ComputationalCell3D &mindiff,
		double diffusecoeff, double shock_w,
		string const& skip_key, Tessellation3D const& tess,
		size_t cell_index, face_vec const& faces, EquationOfState const& eos,
		vector<Vector3D> const& face_cms_cache, bool apply_principal_limit_flag)
	{
		ReplaceComputationalCell(cmax, cell);
		ReplaceComputationalCell(cmin, cell);
		size_t nloop = neighbors.size();
		size_t ntracer = ComputationalCell3D::tracerNames.size();

		Vector3D e1, e2, e3;
		bool has_frame = apply_principal_limit_flag &&
			cart_build_principal_frame(cell.velocity, e1, e2, e3);
		double cell_v1 = has_frame ? ScalarProd(cell.velocity, e1) : 0.0;
		double cell_v2 = has_frame ? ScalarProd(cell.velocity, e2) : 0.0;
		double cell_v3 = has_frame ? ScalarProd(cell.velocity, e3) : 0.0;
		double v1_max = cell_v1, v1_min = cell_v1;
		double v2_max = cell_v2, v2_min = cell_v2;
		double v3_max = cell_v3, v3_min = cell_v3;

		for (size_t i = 0; i < nloop; ++i)
		{
			ComputationalCell3D const& cell_temp = neighbors[i];
			if (!skip_key.empty() && *safe_retrieve(cell_temp.stickers.begin(), ComputationalCell3D::stickerNames.begin(),
								ComputationalCell3D::stickerNames.end(), skip_key))
				continue;
			cmax.density = std::max(cmax.density, cell_temp.density);
			cmax.pressure = std::max(cmax.pressure, cell_temp.pressure);
			cmax.internal_energy = std::max(cmax.internal_energy, cell_temp.internal_energy);
			cmin.density = std::min(cmin.density, cell_temp.density);
			cmin.pressure = std::min(cmin.pressure, cell_temp.pressure);
			cmin.internal_energy = std::min(cmin.internal_energy, cell_temp.internal_energy);
			if (has_frame)
			{
				double nv1 = ScalarProd(cell_temp.velocity, e1);
				double nv2 = ScalarProd(cell_temp.velocity, e2);
				double nv3 = ScalarProd(cell_temp.velocity, e3);
				v1_max = std::max(v1_max, nv1); v1_min = std::min(v1_min, nv1);
				v2_max = std::max(v2_max, nv2); v2_min = std::min(v2_min, nv2);
				v3_max = std::max(v3_max, nv3); v3_min = std::min(v3_min, nv3);
			}
			else
			{
				cmax.velocity.x = std::max(cmax.velocity.x, cell_temp.velocity.x);
				cmax.velocity.y = std::max(cmax.velocity.y, cell_temp.velocity.y);
				cmax.velocity.z = std::max(cmax.velocity.z, cell_temp.velocity.z);
				cmin.velocity.x = std::min(cmin.velocity.x, cell_temp.velocity.x);
				cmin.velocity.y = std::min(cmin.velocity.y, cell_temp.velocity.y);
				cmin.velocity.z = std::min(cmin.velocity.z, cell_temp.velocity.z);
			}
			for (size_t j = 0; j < ntracer; ++j)
			{
				cmax.tracers[j] = std::max(cmax.tracers[j], cell_temp.tracers[j]);
				cmin.tracers[j] = std::min(cmin.tracers[j], cell_temp.tracers[j]);
			}
		}
		ReplaceComputationalCell(maxdiff, cmax);
		maxdiff -= cell;
		ReplaceComputationalCell(mindiff, cmin);
		mindiff -= cell;

		double v1_maxdiff = v1_max - cell_v1, v1_mindiff = v1_min - cell_v1;
		double v2_maxdiff = v2_max - cell_v2, v2_mindiff = v2_min - cell_v2;
		double v3_maxdiff = v3_max - cell_v3, v3_mindiff = v3_min - cell_v3;

		ComputationalCell3D centroid_val = cart_interp(cell, slope, face_cms_cache[0], cm, eos, false);
		ComputationalCell3D dphi = centroid_val - cell;
		vector<double> psi(6 + cell.tracers.size(), 1);
		double psi_v1 = 1.0, psi_v2 = 1.0, psi_v3 = 1.0;
		const size_t nedges = faces.size();
		const double skipfactor = 1e-3;
		const double sf = 1e-9;
		const double w = shock_w;
		const double w1 = 1.0 - w;
		auto blend_psi = [w, w1](double psi_bj, double psi_sh) { return w1 * psi_bj + w * psi_sh; };

		for (size_t i = 0; i < nedges; i++)
		{
			if (i > 0)
			{
				ReplaceComputationalCell(centroid_val, cell);
				cart_interp23Dsimple(centroid_val, slope, face_cms_cache[i], cm);
				ReplaceComputationalCell(dphi, centroid_val);
				dphi -= cell;
			}
			// density
			{
				double psi_bj = 1.0, psi_sh = 1.0;
				if (std::abs(dphi.density) > skipfactor*std::max(std::abs(maxdiff.density), std::abs(mindiff.density)) || centroid_val.density*cell.density < 0)
				{
					if (dphi.density > 1e-9*cell.density)
						psi_bj = std::max(maxdiff.density / dphi.density, 0.0);
					else if (dphi.density < -1e-9*cell.density)
						psi_bj = std::max(mindiff.density / dphi.density, 0.0);
				}
				if (std::abs(dphi.density) > sf*std::max(std::abs(cmax.density), std::abs(cmin.density)) || centroid_val.density*cell.density < 0)
				{
					if (std::abs(dphi.density) > 1e-9*cell.density)
						psi_sh = std::max(diffusecoeff*(neighbors[i].density - cell.density) / dphi.density, 0.0);
				}
				psi[0] = std::min(psi[0], w1*psi_bj + w*psi_sh);
			}
			// pressure
			{
				double psi_bj = 1.0, psi_sh = 1.0;
				if (std::abs(dphi.pressure) > skipfactor*std::max(std::abs(maxdiff.pressure), std::abs(mindiff.pressure)) || centroid_val.pressure*cell.pressure < 0)
				{
					if (dphi.pressure > 1e-9*cell.pressure)
						psi_bj = std::max(maxdiff.pressure / dphi.pressure, 0.0);
					else if (dphi.pressure < -1e-9*cell.pressure)
						psi_bj = std::max(mindiff.pressure / dphi.pressure, 0.0);
				}
				if (std::abs(dphi.pressure) > sf*std::max(std::abs(cmax.pressure), std::abs(cmin.pressure)) || centroid_val.pressure*cell.pressure < 0)
				{
					if (std::abs(dphi.pressure) > 1e-9*cell.pressure)
						psi_sh = std::max(diffusecoeff*(neighbors[i].pressure - cell.pressure) / dphi.pressure, 0.0);
				}
				psi[1] = std::min(psi[1], blend_psi(psi_bj, psi_sh));
			}
			// internal_energy
			{
				double psi_bj = 1.0, psi_sh = 1.0;
				if (std::abs(dphi.internal_energy) > skipfactor*std::max(std::abs(maxdiff.internal_energy), std::abs(mindiff.internal_energy)) || centroid_val.internal_energy*cell.internal_energy < 0)
				{
					if (dphi.internal_energy > 1e-9*cell.internal_energy)
						psi_bj = std::max(maxdiff.internal_energy / dphi.internal_energy, 0.0);
					else if (dphi.internal_energy < -1e-9*cell.internal_energy)
						psi_bj = std::max(mindiff.internal_energy / dphi.internal_energy, 0.0);
				}
				if (std::abs(dphi.internal_energy) > sf*std::max(std::abs(cmax.internal_energy), std::abs(cmin.internal_energy)) || centroid_val.internal_energy*cell.internal_energy < 0)
				{
					if (std::abs(dphi.internal_energy) > 1e-9*cell.internal_energy)
						psi_sh = std::max(diffusecoeff*(neighbors[i].internal_energy - cell.internal_energy) / dphi.internal_energy, 0.0);
				}
				psi[5] = std::min(psi[5], blend_psi(psi_bj, psi_sh));
			}
			// velocity
			if (has_frame)
			{
				double iv1 = ScalarProd(centroid_val.velocity, e1);
				double iv2 = ScalarProd(centroid_val.velocity, e2);
				double iv3 = ScalarProd(centroid_val.velocity, e3);
				double dv1 = iv1 - cell_v1, dv2 = iv2 - cell_v2, dv3 = iv3 - cell_v3;
				double nv1 = ScalarProd(neighbors[i].velocity, e1);
				double nv2 = ScalarProd(neighbors[i].velocity, e2);
				double nv3 = ScalarProd(neighbors[i].velocity, e3);
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dv1) > skipfactor*std::max(std::abs(v1_maxdiff), std::abs(v1_mindiff)) || iv1*cell_v1 < 0)
					{
						if (dv1 > std::abs(1e-9*cell_v1))
							psi_bj = std::max(v1_maxdiff / dv1, 0.0);
						else if (dv1 < -std::abs(1e-9*cell_v1))
							psi_bj = std::max(v1_mindiff / dv1, 0.0);
					}
					if (std::abs(dv1) > sf*std::max(std::abs(v1_max), std::abs(v1_min)) || iv1*cell_v1 < 0)
					{
						if (std::abs(dv1) > 1e-9*std::abs(cell_v1))
							psi_sh = std::max(diffusecoeff*(nv1 - cell_v1) / dv1, 0.0);
					}
					psi_v1 = std::min(psi_v1, blend_psi(psi_bj, psi_sh));
				}
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dv2) > skipfactor*std::max(std::abs(v2_maxdiff), std::abs(v2_mindiff)) || iv2*cell_v2 < 0)
					{
						if (dv2 > std::abs(1e-9*cell_v2))
							psi_bj = std::max(v2_maxdiff / dv2, 0.0);
						else if (dv2 < -std::abs(1e-9*cell_v2))
							psi_bj = std::max(v2_mindiff / dv2, 0.0);
					}
					if (std::abs(dv2) > sf*std::max(std::abs(v2_max), std::abs(v2_min)) || iv2*cell_v2 < 0)
					{
						if (std::abs(dv2) > 1e-9*std::abs(cell_v2))
							psi_sh = std::max(diffusecoeff*(nv2 - cell_v2) / dv2, 0.0);
					}
					psi_v2 = std::min(psi_v2, blend_psi(psi_bj, psi_sh));
				}
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dv3) > skipfactor*std::max(std::abs(v3_maxdiff), std::abs(v3_mindiff)) || iv3*cell_v3 < 0)
					{
						if (dv3 > std::abs(1e-9*cell_v3))
							psi_bj = std::max(v3_maxdiff / dv3, 0.0);
						else if (dv3 < -std::abs(1e-9*cell_v3))
							psi_bj = std::max(v3_mindiff / dv3, 0.0);
					}
					if (std::abs(dv3) > sf*std::max(std::abs(v3_max), std::abs(v3_min)) || iv3*cell_v3 < 0)
					{
						if (std::abs(dv3) > 1e-9*std::abs(cell_v3))
							psi_sh = std::max(diffusecoeff*(nv3 - cell_v3) / dv3, 0.0);
					}
					psi_v3 = std::min(psi_v3, blend_psi(psi_bj, psi_sh));
				}
			}
			else
			{
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dphi.velocity.x) > skipfactor*std::max(std::abs(maxdiff.velocity.x), std::abs(mindiff.velocity.x)) || centroid_val.velocity.x*cell.velocity.x < 0)
					{
						if (dphi.velocity.x > std::abs(1e-9*cell.velocity.x))
							psi_bj = std::max(maxdiff.velocity.x / dphi.velocity.x, 0.0);
						else if (dphi.velocity.x < -std::abs(1e-9*cell.velocity.x))
							psi_bj = std::max(mindiff.velocity.x / dphi.velocity.x, 0.0);
					}
					if (std::abs(dphi.velocity.x) > sf*std::max(std::abs(cmax.velocity.x), std::abs(cmin.velocity.x)) || centroid_val.velocity.x*cell.velocity.x < 0)
					{
						if (std::abs(dphi.velocity.x) > 1e-9*std::abs(cell.velocity.x))
							psi_sh = std::max(diffusecoeff*(neighbors[i].velocity.x - cell.velocity.x) / dphi.velocity.x, 0.0);
					}
					psi[2] = std::min(psi[2], blend_psi(psi_bj, psi_sh));
				}
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dphi.velocity.y) > skipfactor*std::max(std::abs(maxdiff.velocity.y), std::abs(mindiff.velocity.y)) || centroid_val.velocity.y*cell.velocity.y < 0)
					{
						if (dphi.velocity.y > std::abs(1e-9*cell.velocity.y))
							psi_bj = std::max(maxdiff.velocity.y / dphi.velocity.y, 0.0);
						else if (dphi.velocity.y < -std::abs(1e-9*cell.velocity.y))
							psi_bj = std::max(mindiff.velocity.y / dphi.velocity.y, 0.0);
					}
					if (std::abs(dphi.velocity.y) > sf*std::max(std::abs(cmax.velocity.y), std::abs(cmin.velocity.y)) || centroid_val.velocity.y*cell.velocity.y < 0)
					{
						if (std::abs(dphi.velocity.y) > 1e-9*std::abs(cell.velocity.y))
							psi_sh = std::max(diffusecoeff*(neighbors[i].velocity.y - cell.velocity.y) / dphi.velocity.y, 0.0);
					}
					psi[3] = std::min(psi[3], blend_psi(psi_bj, psi_sh));
				}
				{
					double psi_bj = 1.0, psi_sh = 1.0;
					if (std::abs(dphi.velocity.z) > skipfactor*std::max(std::abs(maxdiff.velocity.z), std::abs(mindiff.velocity.z)) || centroid_val.velocity.z*cell.velocity.z < 0)
					{
						if (dphi.velocity.z > std::abs(1e-9*cell.velocity.z))
							psi_bj = std::max(maxdiff.velocity.z / dphi.velocity.z, 0.0);
						else if (dphi.velocity.z < -std::abs(1e-9*cell.velocity.z))
							psi_bj = std::max(mindiff.velocity.z / dphi.velocity.z, 0.0);
					}
					if (std::abs(dphi.velocity.z) > sf*std::max(std::abs(cmax.velocity.z), std::abs(cmin.velocity.z)) || centroid_val.velocity.z*cell.velocity.z < 0)
					{
						if (std::abs(dphi.velocity.z) > 1e-9*std::abs(cell.velocity.z))
							psi_sh = std::max(diffusecoeff*(neighbors[i].velocity.z - cell.velocity.z) / dphi.velocity.z, 0.0);
					}
					psi[4] = std::min(psi[4], blend_psi(psi_bj, psi_sh));
				}
				double psi_v_unified = std::min({psi[2], psi[3], psi[4]});
				psi[2] = psi[3] = psi[4] = psi_v_unified;
			}
			// tracers
			for (size_t j = 0; j < ntracer; ++j)
			{
				double cell_tracer = cell.tracers[j];
				double psi_bj = 1.0, psi_sh = 1.0;
				if (std::abs(dphi.tracers[j]) > skipfactor*std::max(std::abs(maxdiff.tracers[j]), std::abs(mindiff.tracers[j])) ||
					centroid_val.tracers[j] * cell_tracer < 0)
				{
					if (dphi.tracers[j] > std::abs(1e-9*cell_tracer))
						psi_bj = std::max(maxdiff.tracers[j] / dphi.tracers[j], 0.0);
					else if (dphi.tracers[j] < -std::abs(1e-9 * cell_tracer))
						psi_bj = std::max(mindiff.tracers[j] / dphi.tracers[j], 0.0);
				}
				if (std::abs(dphi.tracers[j]) > 0.001*std::max(std::abs(cmax.tracers[j]), std::abs(cmin.tracers[j])) ||
					centroid_val.tracers[j] * cell_tracer < 0)
				{
					if (std::abs(dphi.tracers[j]) > std::abs(1e-9*cell_tracer))
						psi_sh = std::max(diffusecoeff*(neighbors[i].tracers[j] - cell_tracer) / dphi.tracers[j], 0.0);
				}
				psi[6 + j] = std::min(psi[6 + j], blend_psi(psi_bj, psi_sh));
			}
		}
		psi[1] = std::min(psi[1], psi[5]);
		psi[5] = psi[1];
		if (shock_w > 0.85)
		{
			double psi_scalar_min = std::min(psi[1], psi[5]);
			psi[0] = std::min(psi[0], psi_scalar_min);
			psi[2] = std::min(psi[2], psi_scalar_min);
			psi[3] = std::min(psi[3], psi_scalar_min);
			psi[4] = std::min(psi[4], psi_scalar_min);
			if (has_frame)
			{
				psi_v1 = std::min(psi_v1, psi_scalar_min);
				psi_v2 = std::min(psi_v2, psi_scalar_min);
				psi_v3 = std::min(psi_v3, psi_scalar_min);
			}
		}
		slope.xderivative.density *= psi[0];
		slope.yderivative.density *= psi[0];
		slope.zderivative.density *= psi[0];
		slope.xderivative.pressure *= psi[1];
		slope.yderivative.pressure *= psi[1];
		slope.zderivative.pressure *= psi[1];
		if (has_frame && apply_principal_limit_flag)
		{
			double psi_vt = std::min(psi_v2, psi_v3);
			cart_apply_principal_limit(slope.xderivative.velocity, e1, e2, e3, psi_v1, psi_vt, psi_vt);
			cart_apply_principal_limit(slope.yderivative.velocity, e1, e2, e3, psi_v1, psi_vt, psi_vt);
			cart_apply_principal_limit(slope.zderivative.velocity, e1, e2, e3, psi_v1, psi_vt, psi_vt);
		}
		else
		{
			if (has_frame)
			{
				double psi_v_unified = std::min({psi_v1, psi_v2, psi_v3});
				psi[2] = psi[3] = psi[4] = psi_v_unified;
			}
			slope.xderivative.velocity.x *= psi[2];
			slope.yderivative.velocity.x *= psi[2];
			slope.zderivative.velocity.x *= psi[2];
			slope.xderivative.velocity.y *= psi[3];
			slope.yderivative.velocity.y *= psi[3];
			slope.zderivative.velocity.y *= psi[3];
			slope.xderivative.velocity.z *= psi[4];
			slope.yderivative.velocity.z *= psi[4];
			slope.zderivative.velocity.z *= psi[4];
		}
		slope.xderivative.internal_energy *= psi[5];
		slope.yderivative.internal_energy *= psi[5];
		slope.zderivative.internal_energy *= psi[5];
		size_t counter = 6;
		size_t N = slope.xderivative.tracers.size();
		for (size_t k = 0; k < N; ++k)
		{
			slope.xderivative.tracers[k] *= psi[counter];
			slope.yderivative.tracers[k] *= psi[counter];
			slope.zderivative.tracers[k] *= psi[counter];
			++counter;
		}
		if (shock_w > 0)
		{
			double maxDv = ScalarProd(slope.xderivative.velocity, slope.xderivative.velocity)
				+ ScalarProd(slope.yderivative.velocity, slope.yderivative.velocity)
				+ ScalarProd(slope.zderivative.velocity, slope.zderivative.velocity);
			maxDv *= tess.GetWidth(cell_index) * tess.GetWidth(cell_index);
			if (maxDv > 100 * ScalarProd(cell.velocity, cell.velocity))
			{
				double sfactor = fastsqrt(100 * ScalarProd(cell.velocity, cell.velocity) / maxDv);
				slope.xderivative.velocity *= sfactor;
				slope.yderivative.velocity *= sfactor;
				slope.zderivative.velocity *= sfactor;
			}
		}
	}

	// ========== Spherical slope utilities ==========

	void calc_lsq_slope(ComputationalCell3D const& cell_sph,
		Vector3D const& cell_cm, Vector3D const& cell_coords, Vector3D const& origin,
		vector<ComputationalCell3D> const& neighbors, vector<Vector3D> const& neigh_cm,
		Tessellation3D const& tess, size_t cell_index, face_vec const& faces,
		Slope3D &res, Slope3D &temp,
		ComputationalCell3D &neigh_sph_buf,
		vector<Vector3D>& face_sph_cache, vector<double>& face_areas_cache,
		SphericalLinearGauss3D::FaceRadiusPolicy face_radius_policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol)
	{
		size_t n = neigh_cm.size();
		double r_i = cell_coords.x, theta_i = cell_coords.y, phi_i = cell_coords.z;

		face_sph_cache.resize(n);
		face_areas_cache.resize(n);
		for (size_t i = 0; i < n; ++i)
		{
			face_sph_cache[i] = cart_to_sph_coords(tess.FaceCM(faces[i]), origin);
			face_areas_cache[i] = tess.GetArea(faces[i]);
		}

		std::array<double, 9> m;
		std::fill_n(m.begin(), 9, 0.0);

		for (size_t i = 0; i < n; ++i)
		{
			Vector3D nc = cart_to_sph_coords(neigh_cm[i], origin);
			bool const same_shell = same_shell_face(tess, cell_index, faces[i],
				origin, face_radius_policy, shell_radius_abs_tol, shell_radius_rel_tol);
			double dr = same_shell ? 0.0 : nc.x - r_i;
			double dtheta = nc.y - theta_i;
			double dphi = wrap_dphi(nc.z - phi_i);

			Vector3D diff = neigh_cm[i] - cell_cm;
			double dist_sq = ScalarProd(diff, diff);
			if (dist_sq < 1e-28)
				dist_sq = 1e-28;
			(void)dist_sq;
			// Keep equal LSQ weights in spherical coordinates. Area/distance weights
			// bias non-uniform shell meshes and noticeably break spherical symmetry.
			double w = 1.0;

			double w_dr = w * dr, w_dt = w * dtheta, w_dp = w * dphi;

			m[0] += w * dr * dr;
			m[1] += w * dr * dtheta;
			m[2] += w * dr * dphi;
			m[4] += w * dtheta * dtheta;
			m[5] += w * dtheta * dphi;
			m[8] += w * dphi * dphi;

			ReplaceComputationalCell(neigh_sph_buf, neighbors[i]);

			Vector3D neigh_er, neigh_et, neigh_ep;
			sph_basis_at(nc.y, nc.z, neigh_er, neigh_et, neigh_ep);
			neigh_sph_buf.velocity = cart_to_sph_vec(neighbors[i].velocity, neigh_er, neigh_et, neigh_ep);

			if (i == 0)
			{
				ReplaceComputationalCell(temp.xderivative, neigh_sph_buf);
				ComputationalCellAddMult(temp.xderivative, cell_sph, -1.0);
				temp.xderivative *= w_dr;

				ReplaceComputationalCell(temp.yderivative, neigh_sph_buf);
				ComputationalCellAddMult(temp.yderivative, cell_sph, -1.0);
				temp.yderivative *= w_dt;

				ReplaceComputationalCell(temp.zderivative, neigh_sph_buf);
				ComputationalCellAddMult(temp.zderivative, cell_sph, -1.0);
				temp.zderivative *= w_dp;
			}
			else
			{
				ComputationalCellAddMult(temp.xderivative, neigh_sph_buf, w_dr);
				ComputationalCellAddMult(temp.xderivative, cell_sph, -w_dr);

				ComputationalCellAddMult(temp.yderivative, neigh_sph_buf, w_dt);
				ComputationalCellAddMult(temp.yderivative, cell_sph, -w_dt);

				ComputationalCellAddMult(temp.zderivative, neigh_sph_buf, w_dp);
				ComputationalCellAddMult(temp.zderivative, cell_sph, -w_dp);
			}
		}

		m[3] = m[1];
		m[6] = m[2];
		m[7] = m[5];

		const double det = -m[2]*m[4]*m[6] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7]
			- m[0]*m[5]*m[7] - m[1]*m[3]*m[8] + m[0]*m[4]*m[8];
		if (std::abs(det) < 1e-30)
		{
			UniversalError eo("Singular LSQ matrix in SphericalLinearGauss3D");
			eo.addEntry("Cell x", cell_cm.x);
			eo.addEntry("Cell y", cell_cm.y);
			eo.addEntry("Cell z", cell_cm.z);
			eo.addEntry("r", r_i);
			eo.addEntry("Det was", det);
			throw eo;
		}

		std::array<double, 9> mi;
		mi[0] = (m[4]*m[8] - m[5]*m[7]) / det;
		mi[1] = (m[2]*m[7] - m[1]*m[8]) / det;
		mi[2] = (m[1]*m[5] - m[2]*m[4]) / det;
		mi[3] = (m[5]*m[6] - m[3]*m[8]) / det;
		mi[4] = (m[0]*m[8] - m[2]*m[6]) / det;
		mi[5] = (m[2]*m[3] - m[0]*m[5]) / det;
		mi[6] = (m[3]*m[7] - m[4]*m[6]) / det;
		mi[7] = (m[1]*m[6] - m[0]*m[7]) / det;
		mi[8] = (m[0]*m[4] - m[1]*m[3]) / det;

		ReplaceComputationalCell(res.xderivative, temp.xderivative);
		res.xderivative *= mi[0];
		ComputationalCellAddMult(res.xderivative, temp.yderivative, mi[1]);
		ComputationalCellAddMult(res.xderivative, temp.zderivative, mi[2]);

		ReplaceComputationalCell(res.yderivative, temp.xderivative);
		res.yderivative *= mi[3];
		ComputationalCellAddMult(res.yderivative, temp.yderivative, mi[4]);
		ComputationalCellAddMult(res.yderivative, temp.zderivative, mi[5]);

		ReplaceComputationalCell(res.zderivative, temp.xderivative);
		res.zderivative *= mi[6];
		ComputationalCellAddMult(res.zderivative, temp.yderivative, mi[7]);
		ComputationalCellAddMult(res.zderivative, temp.zderivative, mi[8]);
	}

	double PressureRatio(ComputationalCell3D const& cell, vector<ComputationalCell3D> const& neigh)
	{
		double res = 1.0;
		double p = cell.pressure;
		size_t N = neigh.size();
		for (size_t i = 0; i < N; i++)
		{
			if (p > neigh[i].pressure)
				res = std::min(res, neigh[i].pressure / p);
			else
				res = std::min(res, p / neigh[i].pressure);
		}
		return res;
	}

	double shock_weight(Slope3D const& naive_slope, double cell_width, double shock_ratio,
		ComputationalCell3D const& cell, vector<ComputationalCell3D> const& neighbor_list,
		double pressure_ratio, double cs, double r, double theta)
	{
		(void)naive_slope;
		(void)cell_width;
		(void)shock_ratio;
		(void)cell;
		(void)neighbor_list;
		(void)pressure_ratio;
		(void)cs;
		(void)r;
		(void)theta;
		// Keep the blended limiter fully in its shock branch for spherical shells.
		// The shock sensor varies with non-uniform mesh spacing and harms symmetry.
		return 1.0;
	}

	void interp_coord_simple(ComputationalCell3D &res, Slope3D const& slope,
		double dr, double dtheta, double dphi)
	{
		ComputationalCellAddMult3(res, slope.xderivative, slope.yderivative, slope.zderivative,
			dr, dtheta, dphi);
	}

	void interp_coord_eos(ComputationalCell3D &res, Slope3D const& slope,
		double dr, double dtheta, double dphi,
		EquationOfState const& eos, bool pressure_calc)
	{
		interp_coord_simple(res, slope, dr, dtheta, dphi);
		try
		{
			if (!pressure_calc)
				res.pressure = eos.de2p(res.density, res.internal_energy, res.tracers, ComputationalCell3D::tracerNames);
			else
				res.internal_energy = eos.dp2e(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("density", res.density);
			eo.addEntry("internal energy", res.internal_energy);
			throw eo;
		}
	}

	void spherical_blended_slope_limit(ComputationalCell3D const& cell,
		Vector3D const& cell_coords,
		vector<ComputationalCell3D> const& neighbors, Slope3D &slope,
		ComputationalCell3D &cmax, ComputationalCell3D &cmin,
		ComputationalCell3D &maxdiff, ComputationalCell3D &mindiff,
		double diffusecoeff, double shock_w,
		string const& skip_key, Tessellation3D const& tess,
		size_t cell_index, face_vec const& faces,
		vector<Vector3D> const& effective_face_sph_cache,
		vector<double> const& face_vel_dr_correction)
	{
		ReplaceComputationalCell(cmax, cell);
		ReplaceComputationalCell(cmin, cell);
		size_t nloop = neighbors.size();
		size_t ntracer = ComputationalCell3D::tracerNames.size();

		double cell_vr = cell.velocity.x, cell_vt = cell.velocity.y, cell_vp = cell.velocity.z;
		double vr_max = cell_vr, vr_min = cell_vr;
		double vt_max = cell_vt, vt_min = cell_vt;
		double vp_max = cell_vp, vp_min = cell_vp;
		auto skip_neighbor = [&skip_key](ComputationalCell3D const& ct)
		{
			return !skip_key.empty() && *safe_retrieve(ct.stickers.begin(),
				ComputationalCell3D::stickerNames.begin(),
				ComputationalCell3D::stickerNames.end(), skip_key);
		};

		for (size_t i = 0; i < nloop; ++i)
		{
			ComputationalCell3D const& ct = neighbors[i];
			if (skip_neighbor(ct))
				continue;
			cmax.density = std::max(cmax.density, ct.density);
			cmax.pressure = std::max(cmax.pressure, ct.pressure);
			cmax.internal_energy = std::max(cmax.internal_energy, ct.internal_energy);
			cmin.density = std::min(cmin.density, ct.density);
			cmin.pressure = std::min(cmin.pressure, ct.pressure);
			cmin.internal_energy = std::min(cmin.internal_energy, ct.internal_energy);
			vr_max = std::max(vr_max, ct.velocity.x);
			vr_min = std::min(vr_min, ct.velocity.x);
			vt_max = std::max(vt_max, ct.velocity.y);
			vt_min = std::min(vt_min, ct.velocity.y);
			vp_max = std::max(vp_max, ct.velocity.z);
			vp_min = std::min(vp_min, ct.velocity.z);
			for (size_t j = 0; j < ntracer; ++j)
			{
				cmax.tracers[j] = std::max(cmax.tracers[j], ct.tracers[j]);
				cmin.tracers[j] = std::min(cmin.tracers[j], ct.tracers[j]);
			}
		}

		ReplaceComputationalCell(maxdiff, cmax);
		maxdiff -= cell;
		ReplaceComputationalCell(mindiff, cmin);
		mindiff -= cell;
		double vr_maxdiff = vr_max - cell_vr, vr_mindiff = vr_min - cell_vr;
		double vt_maxdiff = vt_max - cell_vt, vt_mindiff = vt_min - cell_vt;
		double vp_maxdiff = vp_max - cell_vp, vp_mindiff = vp_min - cell_vp;

		vector<double> psi(6 + cell.tracers.size(), 1);
		double psi_vr = 1.0, psi_vt = 1.0, psi_vp = 1.0;
		const size_t nedges = faces.size();
		const double skipfactor = 1e-3;
		const double sf = 1e-9;
		const double w = shock_w;
		const double w1 = 1.0 - w;
		auto blend_psi = [w, w1](double pb, double ps) { return w1 * pb + w * ps; };

		ComputationalCell3D centroid_val(cell), dphi_cell(cell);

		for (size_t i = 0; i < nedges; ++i)
		{
			if (skip_neighbor(neighbors[i]))
				continue;
			double dri = effective_face_sph_cache[i].x - cell_coords.x;
			double dti = effective_face_sph_cache[i].y - cell_coords.y;
			double dpi = wrap_dphi(effective_face_sph_cache[i].z - cell_coords.z);

			ReplaceComputationalCell(centroid_val, cell);
			interp_coord_simple(centroid_val, slope, dri, dti, dpi);
			if (i < face_vel_dr_correction.size())
			{
				double vel_dr_extra = face_vel_dr_correction[i];
				centroid_val.velocity.x += slope.xderivative.velocity.x * vel_dr_extra;
				centroid_val.velocity.y += slope.xderivative.velocity.y * vel_dr_extra;
				centroid_val.velocity.z += slope.xderivative.velocity.z * vel_dr_extra;
			}
			ReplaceComputationalCell(dphi_cell, centroid_val);
			dphi_cell -= cell;

			// density
			{
				double pb = 1.0, ps = 1.0;
				if (std::abs(dphi_cell.density) > skipfactor*std::max(std::abs(maxdiff.density), std::abs(mindiff.density)) || centroid_val.density*cell.density < 0)
				{
					if (dphi_cell.density > 1e-9*cell.density)
						pb = std::max(maxdiff.density / dphi_cell.density, 0.0);
					else if (dphi_cell.density < -1e-9*cell.density)
						pb = std::max(mindiff.density / dphi_cell.density, 0.0);
				}
				if (std::abs(dphi_cell.density) > sf*std::max(std::abs(cmax.density), std::abs(cmin.density)) || centroid_val.density*cell.density < 0)
				{
					if (std::abs(dphi_cell.density) > 1e-9*cell.density)
						ps = std::max(diffusecoeff*(neighbors[i].density - cell.density) / dphi_cell.density, 0.0);
				}
				psi[0] = std::min(psi[0], blend_psi(pb, ps));
			}
			// pressure
			{
				double pb = 1.0, ps = 1.0;
				if (std::abs(dphi_cell.pressure) > skipfactor*std::max(std::abs(maxdiff.pressure), std::abs(mindiff.pressure)) || centroid_val.pressure*cell.pressure < 0)
				{
					if (dphi_cell.pressure > 1e-9*cell.pressure)
						pb = std::max(maxdiff.pressure / dphi_cell.pressure, 0.0);
					else if (dphi_cell.pressure < -1e-9*cell.pressure)
						pb = std::max(mindiff.pressure / dphi_cell.pressure, 0.0);
				}
				if (std::abs(dphi_cell.pressure) > sf*std::max(std::abs(cmax.pressure), std::abs(cmin.pressure)) || centroid_val.pressure*cell.pressure < 0)
				{
					if (std::abs(dphi_cell.pressure) > 1e-9*cell.pressure)
						ps = std::max(diffusecoeff*(neighbors[i].pressure - cell.pressure) / dphi_cell.pressure, 0.0);
				}
				psi[1] = std::min(psi[1], blend_psi(pb, ps));
			}
			// internal_energy
			{
				double pb = 1.0, ps = 1.0;
				if (std::abs(dphi_cell.internal_energy) > skipfactor*std::max(std::abs(maxdiff.internal_energy), std::abs(mindiff.internal_energy)) || centroid_val.internal_energy*cell.internal_energy < 0)
				{
					if (dphi_cell.internal_energy > 1e-9*cell.internal_energy)
						pb = std::max(maxdiff.internal_energy / dphi_cell.internal_energy, 0.0);
					else if (dphi_cell.internal_energy < -1e-9*cell.internal_energy)
						pb = std::max(mindiff.internal_energy / dphi_cell.internal_energy, 0.0);
				}
				if (std::abs(dphi_cell.internal_energy) > sf*std::max(std::abs(cmax.internal_energy), std::abs(cmin.internal_energy)) || centroid_val.internal_energy*cell.internal_energy < 0)
				{
					if (std::abs(dphi_cell.internal_energy) > 1e-9*cell.internal_energy)
						ps = std::max(diffusecoeff*(neighbors[i].internal_energy - cell.internal_energy) / dphi_cell.internal_energy, 0.0);
				}
				psi[5] = std::min(psi[5], blend_psi(pb, ps));
			}
			// velocity
			{
				double dv_r = centroid_val.velocity.x - cell_vr;
				double dv_t = centroid_val.velocity.y - cell_vt;
				double dv_p = centroid_val.velocity.z - cell_vp;
				double nv_r = neighbors[i].velocity.x;
				double nv_t = neighbors[i].velocity.y;
				double nv_p = neighbors[i].velocity.z;
				{
					double pb = 1.0, ps = 1.0;
					if (std::abs(dv_r) > skipfactor*std::max(std::abs(vr_maxdiff), std::abs(vr_mindiff)) || centroid_val.velocity.x*cell_vr < 0)
					{
						if (dv_r > std::abs(1e-9*cell_vr))
							pb = std::max(vr_maxdiff / dv_r, 0.0);
						else if (dv_r < -std::abs(1e-9*cell_vr))
							pb = std::max(vr_mindiff / dv_r, 0.0);
					}
					if (std::abs(dv_r) > sf*std::max(std::abs(vr_max), std::abs(vr_min)) || centroid_val.velocity.x*cell_vr < 0)
					{
						if (std::abs(dv_r) > 1e-9*std::abs(cell_vr))
							ps = std::max(diffusecoeff*(nv_r - cell_vr) / dv_r, 0.0);
					}
					psi_vr = std::min(psi_vr, blend_psi(pb, ps));
				}
				{
					double pb = 1.0, ps = 1.0;
					if (std::abs(dv_t) > skipfactor*std::max(std::abs(vt_maxdiff), std::abs(vt_mindiff)) || centroid_val.velocity.y*cell_vt < 0)
					{
						if (dv_t > std::abs(1e-9*cell_vt))
							pb = std::max(vt_maxdiff / dv_t, 0.0);
						else if (dv_t < -std::abs(1e-9*cell_vt))
							pb = std::max(vt_mindiff / dv_t, 0.0);
					}
					if (std::abs(dv_t) > sf*std::max(std::abs(vt_max), std::abs(vt_min)) || centroid_val.velocity.y*cell_vt < 0)
					{
						if (std::abs(dv_t) > 1e-9*std::abs(cell_vt))
							ps = std::max(diffusecoeff*(nv_t - cell_vt) / dv_t, 0.0);
					}
					psi_vt = std::min(psi_vt, blend_psi(pb, ps));
				}
				{
					double pb = 1.0, ps = 1.0;
					if (std::abs(dv_p) > skipfactor*std::max(std::abs(vp_maxdiff), std::abs(vp_mindiff)) || centroid_val.velocity.z*cell_vp < 0)
					{
						if (dv_p > std::abs(1e-9*cell_vp))
							pb = std::max(vp_maxdiff / dv_p, 0.0);
						else if (dv_p < -std::abs(1e-9*cell_vp))
							pb = std::max(vp_mindiff / dv_p, 0.0);
					}
					if (std::abs(dv_p) > sf*std::max(std::abs(vp_max), std::abs(vp_min)) || centroid_val.velocity.z*cell_vp < 0)
					{
						if (std::abs(dv_p) > 1e-9*std::abs(cell_vp))
							ps = std::max(diffusecoeff*(nv_p - cell_vp) / dv_p, 0.0);
					}
					psi_vp = std::min(psi_vp, blend_psi(pb, ps));
				}
			}
			// tracers
			for (size_t j = 0; j < ntracer; ++j)
			{
				double ct2 = cell.tracers[j];
				double pb = 1.0, ps = 1.0;
				if (std::abs(dphi_cell.tracers[j]) > skipfactor*std::max(std::abs(maxdiff.tracers[j]), std::abs(mindiff.tracers[j])) ||
					centroid_val.tracers[j] * ct2 < 0)
				{
					if (dphi_cell.tracers[j] > std::abs(1e-9*ct2))
						pb = std::max(maxdiff.tracers[j] / dphi_cell.tracers[j], 0.0);
					else if (dphi_cell.tracers[j] < -std::abs(1e-9*ct2))
						pb = std::max(mindiff.tracers[j] / dphi_cell.tracers[j], 0.0);
				}
				if (std::abs(dphi_cell.tracers[j]) > 0.001*std::max(std::abs(cmax.tracers[j]), std::abs(cmin.tracers[j])) ||
					centroid_val.tracers[j] * ct2 < 0)
				{
					if (std::abs(dphi_cell.tracers[j]) > std::abs(1e-9*ct2))
						ps = std::max(diffusecoeff*(neighbors[i].tracers[j] - ct2) / dphi_cell.tracers[j], 0.0);
				}
				psi[6 + j] = std::min(psi[6 + j], blend_psi(pb, ps));
			}
		}

		psi[1] = std::min(psi[1], psi[5]);
		psi[5] = psi[1];
		if (shock_w > 0.85)
		{
			double psi_scalar_min = std::min(psi[1], psi[5]);
			psi[0] = std::min(psi[0], psi_scalar_min);
			psi_vr = std::min(psi_vr, psi_scalar_min);
			psi_vt = std::min(psi_vt, psi_scalar_min);
			psi_vp = std::min(psi_vp, psi_scalar_min);
		}

		const double psi_floor = 0.01;
		for (auto& p : psi)
			if (p < psi_floor) p = 0.0;
		if (psi_vr < psi_floor) psi_vr = 0.0;
		if (psi_vt < psi_floor) psi_vt = 0.0;
		if (psi_vp < psi_floor) psi_vp = 0.0;

		slope.xderivative.density *= psi[0];
		slope.yderivative.density *= psi[0];
		slope.zderivative.density *= psi[0];
		slope.xderivative.pressure *= psi[1];
		slope.yderivative.pressure *= psi[1];
		slope.zderivative.pressure *= psi[1];

		double psi_tang = std::min(psi_vt, psi_vp);
		slope.xderivative.velocity.x *= psi_vr;
		slope.yderivative.velocity.x *= psi_vr;
		slope.zderivative.velocity.x *= psi_vr;
		slope.xderivative.velocity.y *= psi_tang;
		slope.yderivative.velocity.y *= psi_tang;
		slope.zderivative.velocity.y *= psi_tang;
		slope.xderivative.velocity.z *= psi_tang;
		slope.yderivative.velocity.z *= psi_tang;
		slope.zderivative.velocity.z *= psi_tang;

		slope.xderivative.internal_energy *= psi[5];
		slope.yderivative.internal_energy *= psi[5];
		slope.zderivative.internal_energy *= psi[5];

		size_t counter = 6;
		size_t N = slope.xderivative.tracers.size();
		for (size_t k = 0; k < N; ++k)
		{
			slope.xderivative.tracers[k] *= psi[counter];
			slope.yderivative.tracers[k] *= psi[counter];
			slope.zderivative.tracers[k] *= psi[counter];
			++counter;
		}

		if (shock_w > 0)
		{
			double safe_r = std::max(cell_coords.x, 1e-14);
			double safe_rsint = std::max(safe_r * std::abs(std::sin(cell_coords.y)), 1e-14);
			double inv_r = 1.0 / safe_r;
			double inv_rsint = 1.0 / safe_rsint;
			Vector3D dv_dr = slope.xderivative.velocity;
			Vector3D dv_dtheta = sph_velocity_theta_derivative(cell.velocity,
				slope.yderivative.velocity);
			Vector3D dv_dphi = sph_velocity_phi_derivative(cell.velocity,
				slope.zderivative.velocity, cell_coords.y);
			double maxDv = ScalarProd(dv_dr, dv_dr)
				+ ScalarProd(dv_dtheta, dv_dtheta) * inv_r * inv_r
				+ ScalarProd(dv_dphi, dv_dphi) * inv_rsint * inv_rsint;
			maxDv *= tess.GetWidth(cell_index) * tess.GetWidth(cell_index);
			if (maxDv > 100 * ScalarProd(cell.velocity, cell.velocity))
			{
				double sfactor = fastsqrt(100 * ScalarProd(cell.velocity, cell.velocity) / maxDv);
				slope.xderivative.velocity *= sfactor;
				slope.yderivative.velocity *= sfactor;
				slope.zderivative.velocity *= sfactor;
			}
		}
	}

	// ========== Slope orchestrators ==========

	void zero_radiation_fields(Slope3D &res, const vector<string>& calc_tracers)
	{
		for (size_t i = 0; i < ComputationalCell3D::tracerNames.size(); ++i)
		{
			if (std::find(calc_tracers.begin(), calc_tracers.end(), ComputationalCell3D::tracerNames[i]) == calc_tracers.end())
			{
				res.xderivative.tracers[i] = 0;
				res.yderivative.tracers[i] = 0;
				res.zderivative.tracers[i] = 0;
			}
		}
		res.xderivative.Erad = 0;  res.yderivative.Erad = 0;  res.zderivative.Erad = 0;
		res.xderivative.Erad_dt = 0;  res.yderivative.Erad_dt = 0;  res.zderivative.Erad_dt = 0;
		res.xderivative.Erad_dt_dt = 0;  res.yderivative.Erad_dt_dt = 0;  res.zderivative.Erad_dt_dt = 0;
		for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.xderivative.Eg[j] = 0;
		for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.yderivative.Eg[j] = 0;
		for (size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.zderivative.Eg[j] = 0;
		res.xderivative.temperature = 0;
		res.yderivative.temperature = 0;
		res.zderivative.temperature = 0;
	}

	// Cartesian slope computation for cells near the poles
	void calc_slope_cartesian(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
		size_t cell_index, bool slf, double shockratio, double diffusecoeff, double pressure_ratio,
		EquationOfState const& eos, const vector<string>& calc_tracers,
		Vector3D const& origin,
		Vector3D &e_r, Vector3D &e_theta, Vector3D &e_phi,
		Vector3D &cell_coords,
		Slope3D &naive_slope_, Slope3D &res, Slope3D &temp1,
		ComputationalCell3D &temp2, ComputationalCell3D &temp3,
		ComputationalCell3D &temp4, ComputationalCell3D &temp5,
		vector<Vector3D> &neighbor_mesh_list,
		vector<Vector3D> &neighbor_cm_list,
		string const& skip_key,
		vector<ComputationalCell3D> &neighbor_list,
		std::vector<Vector3D> &c_ij,
		vector<Vector3D>& face_cart_cache, vector<double>& face_areas_cache,
		bool apply_principal_limit_flag)
	{
		face_vec const& faces = tess.GetCellFaces(cell_index);
		cart_GetNeighborMesh(tess, cell_index, neighbor_mesh_list, faces);
		GetNeighborCM(tess, cell_index, neighbor_cm_list, faces);
		GetNeighborCells(tess, cell_index, cells, faces, neighbor_list);

		Vector3D cell_cm = tess.GetCellCM(cell_index);
		cell_coords = cart_to_sph_coords(cell_cm, origin);

		if (cell_coords.x >= 1e-14)
			sph_basis_at(cell_coords.y, cell_coords.z, e_r, e_theta, e_phi);
		else
		{
			e_r = Vector3D(0, 0, 1);
			e_theta = Vector3D(1, 0, 0);
			e_phi = Vector3D(0, 1, 0);
		}

		ComputationalCell3D const& cell = cells[cell_index];

		cart_calc_naive_slope(cell, tess.GetMeshPoint(cell_index), cell_cm,
			tess.GetVolume(cell_index), neighbor_list, neighbor_mesh_list, neighbor_cm_list,
			tess, res, temp1, cell_index, faces, c_ij, face_cart_cache, face_areas_cache);

		naive_slope_ = res;
		zero_radiation_fields(res, calc_tracers);

		if (slf)
		{
			double sw = cart_shock_weight(res, tess.GetWidth(cell_index), shockratio, cell,
				neighbor_list, pressure_ratio,
				eos.de2c(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames));
			cart_blended_slope_limit(cell, cell_cm, neighbor_list, res,
				temp2, temp3, temp4, temp5,
				diffusecoeff, sw, skip_key, tess, cell_index, faces, eos, face_cart_cache,
				apply_principal_limit_flag);
		}
	}

	// Spherical slope computation for cells away from poles
	void calc_slope_spherical(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
		size_t cell_index, bool slf, double shockratio, double diffusecoeff, double pressure_ratio,
		EquationOfState const& eos, const vector<string>& calc_tracers,
		Vector3D const& origin,
		Vector3D &e_r, Vector3D &e_theta, Vector3D &e_phi,
		Vector3D &cell_coords,
		Slope3D &naive_slope_, Slope3D &res, Slope3D &temp1,
		ComputationalCell3D &temp2, ComputationalCell3D &temp3,
		ComputationalCell3D &temp4, ComputationalCell3D &temp5,
		ComputationalCell3D &neigh_sph_buf,
		vector<Vector3D> &neighbor_cm_list,
		string const& skip_key,
		vector<ComputationalCell3D> &neighbor_list,
		vector<Vector3D>& face_sph_cache, vector<double>& face_areas_cache,
		bool velocity_radial_extrapolation,
		SphericalLinearGauss3D::FaceRadiusPolicy face_radius_policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol,
		vector<double>& face_vel_dr_correction)
	{
		face_vec const& faces = tess.GetCellFaces(cell_index);
		GetNeighborCM(tess, cell_index, neighbor_cm_list, faces);
		GetNeighborCells(tess, cell_index, cells, faces, neighbor_list);

		Vector3D cell_cm = tess.GetCellCM(cell_index);
		cell_coords = cart_to_sph_coords(cell_cm, origin);

		if (cell_coords.x < 1e-14)
		{
			e_r = Vector3D(0, 0, 1);
			e_theta = Vector3D(1, 0, 0);
			e_phi = Vector3D(0, 1, 0);
			res.xderivative *= 0;
			res.yderivative *= 0;
			res.zderivative *= 0;
			naive_slope_ = res;
			face_sph_cache.resize(faces.size());
			face_areas_cache.resize(faces.size());
			for (size_t i = 0; i < faces.size(); ++i)
			{
				face_sph_cache[i] = cart_to_sph_coords(tess.FaceCM(faces[i]), origin);
				face_areas_cache[i] = tess.GetArea(faces[i]);
			}
			face_vel_dr_correction.assign(faces.size(), 0.0);
			return;
		}

		sph_basis_at(cell_coords.y, cell_coords.z, e_r, e_theta, e_phi);

		ComputationalCell3D cell_sph;
		ReplaceComputationalCell(cell_sph, cells[cell_index]);
		cell_sph.velocity = cart_to_sph_vec(cells[cell_index].velocity, e_r, e_theta, e_phi);

		calc_lsq_slope(cell_sph, cell_cm, cell_coords, origin,
			neighbor_list, neighbor_cm_list, tess, cell_index, faces,
			res, temp1, neigh_sph_buf, face_sph_cache, face_areas_cache,
			face_radius_policy, shell_radius_abs_tol, shell_radius_rel_tol);
		vector<Vector3D> physical_face_sph_cache = face_sph_cache;
		build_effective_face_sph_cache(tess, cell_index, faces, origin, cell_coords,
			face_sph_cache, face_radius_policy, shell_radius_abs_tol,
			shell_radius_rel_tol, face_sph_cache);

		naive_slope_ = res;
		zero_radiation_fields(res, calc_tracers);

		// Compute velocity radial correction for each face
		face_vel_dr_correction.assign(faces.size(), 0.0);
		if (velocity_radial_extrapolation &&
			face_radius_policy == SphericalLinearGauss3D::FaceRadiusPolicy::PhysicalFaceCM)
		{
			double r_gen_cell = abs(tess.GetMeshPoint(cell_index) - origin);
			for (size_t fi = 0; fi < faces.size(); ++fi)
			{
				auto neigh = tess.GetFaceNeighbors(faces[fi]);
				size_t other = (neigh.first == cell_index) ? neigh.second : neigh.first;
				double r_gen_other = abs(tess.GetMeshPoint(other) - origin);
				double r_avg = 0.5 * (r_gen_cell + r_gen_other);
				face_vel_dr_correction[fi] = r_avg - physical_face_sph_cache[fi].x;
			}
		}

		if (slf)
		{
		for (size_t i = 0; i < neighbor_list.size(); ++i)
		{
			Vector3D nc = cart_to_sph_coords(neighbor_cm_list[i], origin);
			Vector3D neigh_er, neigh_et, neigh_ep;
			sph_basis_at(nc.y, nc.z, neigh_er, neigh_et, neigh_ep);
			neighbor_list[i].velocity = cart_to_sph_vec(neighbor_list[i].velocity, neigh_er, neigh_et, neigh_ep);
		}
			double sw = shock_weight(res, tess.GetWidth(cell_index), shockratio, cell_sph,
				neighbor_list, pressure_ratio,
				eos.de2c(cell_sph.density, cell_sph.internal_energy, cell_sph.tracers, ComputationalCell3D::tracerNames),
				cell_coords.x, cell_coords.y);
			spherical_blended_slope_limit(cell_sph, cell_coords, neighbor_list, res,
				temp2, temp3, temp4, temp5,
				diffusecoeff, sw, skip_key, tess, cell_index, faces, face_sph_cache,
				face_vel_dr_correction);
		}
	}

	// ========== Slope conversion: spherical -> Cartesian ==========

	// Converts a spherical-coordinate slope (d/dr, d/dtheta, d/dphi with spherical
	// velocity components) to Cartesian (d/dx, d/dy, d/dz with Cartesian velocity).
	// Modifies slope in-place using temp as scratch space.
	void coord_sph_slope_to_cart(Slope3D &slope,
		double r, double theta,
		Vector3D const& v_sph,
		Vector3D const& e_r, Vector3D const& e_theta, Vector3D const& e_phi,
		Slope3D &temp)
	{
		double safe_r = std::max(r, 1e-14);
		double st = std::sin(theta);
		double safe_rsint = std::max(safe_r * st, 1e-14);
		double inv_r = 1.0 / safe_r;
		double inv_rsint = 1.0 / safe_rsint;

		// Save derivatives before applying inverse Jacobian.
		ReplaceComputationalCell(temp.xderivative, slope.xderivative);
		ReplaceComputationalCell(temp.yderivative, slope.yderivative);
		ReplaceComputationalCell(temp.zderivative, slope.zderivative);

		// d/dx = e_r.x * d/dr + (e_theta.x / r) * d/dtheta + (e_phi.x / (r*sin_theta)) * d/dphi
		ReplaceComputationalCell(slope.xderivative, temp.xderivative);
		slope.xderivative *= e_r.x;
		ComputationalCellAddMult(slope.xderivative, temp.yderivative, e_theta.x * inv_r);
		ComputationalCellAddMult(slope.xderivative, temp.zderivative, e_phi.x * inv_rsint);

		ReplaceComputationalCell(slope.yderivative, temp.xderivative);
		slope.yderivative *= e_r.y;
		ComputationalCellAddMult(slope.yderivative, temp.yderivative, e_theta.y * inv_r);
		ComputationalCellAddMult(slope.yderivative, temp.zderivative, e_phi.y * inv_rsint);

		ReplaceComputationalCell(slope.zderivative, temp.xderivative);
		slope.zderivative *= e_r.z;
		ComputationalCellAddMult(slope.zderivative, temp.yderivative, e_theta.z * inv_r);
		ComputationalCellAddMult(slope.zderivative, temp.zderivative, e_phi.z * inv_rsint);

		Vector3D dv_dr_cart = sph_to_cart_vec(temp.xderivative.velocity,
			e_r, e_theta, e_phi);
		Vector3D dv_dtheta_cart = sph_to_cart_vec(
			sph_velocity_theta_derivative(v_sph, temp.yderivative.velocity),
			e_r, e_theta, e_phi);
		Vector3D dv_dphi_cart = sph_to_cart_vec(
			sph_velocity_phi_derivative(v_sph, temp.zderivative.velocity, theta),
			e_r, e_theta, e_phi);

		slope.xderivative.velocity = dv_dr_cart * e_r.x
			+ dv_dtheta_cart * (e_theta.x * inv_r)
			+ dv_dphi_cart * (e_phi.x * inv_rsint);
		slope.yderivative.velocity = dv_dr_cart * e_r.y
			+ dv_dtheta_cart * (e_theta.y * inv_r)
			+ dv_dphi_cart * (e_phi.y * inv_rsint);
		slope.zderivative.velocity = dv_dr_cart * e_r.z
			+ dv_dtheta_cart * (e_theta.z * inv_r)
			+ dv_dphi_cart * (e_phi.z * inv_rsint);
	}

	// Converts a Cartesian-coordinate slope (d/dx, d/dy, d/dz with Cartesian
	// velocity components) to spherical (d/dr, d/dtheta, d/dphi with spherical
	// velocity). Modifies slope in-place using temp as scratch space.
	void coord_cart_slope_to_sph(Slope3D &slope,
		double r, double theta,
		Vector3D const& e_r, Vector3D const& e_theta, Vector3D const& e_phi,
		Slope3D &temp)
	{
		double st = std::sin(theta);

		// Step 1: apply the forward Jacobian to transform derivative directions
		// from (d/dx, d/dy, d/dz) to (d/dr, d/dtheta, d/dphi).
		ReplaceComputationalCell(temp.xderivative, slope.xderivative);
		ReplaceComputationalCell(temp.yderivative, slope.yderivative);
		ReplaceComputationalCell(temp.zderivative, slope.zderivative);

		// d/dr = e_r.x * d/dx + e_r.y * d/dy + e_r.z * d/dz
		ReplaceComputationalCell(slope.xderivative, temp.xderivative);
		slope.xderivative *= e_r.x;
		ComputationalCellAddMult(slope.xderivative, temp.yderivative, e_r.y);
		ComputationalCellAddMult(slope.xderivative, temp.zderivative, e_r.z);

		// d/dtheta = r * (e_theta.x * d/dx + e_theta.y * d/dy + e_theta.z * d/dz)
		ReplaceComputationalCell(slope.yderivative, temp.xderivative);
		slope.yderivative *= r * e_theta.x;
		ComputationalCellAddMult(slope.yderivative, temp.yderivative, r * e_theta.y);
		ComputationalCellAddMult(slope.yderivative, temp.zderivative, r * e_theta.z);

		// d/dphi = r*sin(theta) * (e_phi.x * d/dx + e_phi.y * d/dy + e_phi.z * d/dz)
		double rsint = r * st;
		ReplaceComputationalCell(slope.zderivative, temp.xderivative);
		slope.zderivative *= rsint * e_phi.x;
		ComputationalCellAddMult(slope.zderivative, temp.yderivative, rsint * e_phi.y);
		ComputationalCellAddMult(slope.zderivative, temp.zderivative, rsint * e_phi.z);

		// Step 2: rotate velocity from Cartesian to spherical basis
		slope.xderivative.velocity = cart_to_sph_vec(slope.xderivative.velocity, e_r, e_theta, e_phi);
		slope.yderivative.velocity = cart_to_sph_vec(slope.yderivative.velocity, e_r, e_theta, e_phi);
		slope.zderivative.velocity = cart_to_sph_vec(slope.zderivative.velocity, e_r, e_theta, e_phi);
	}

#ifdef RICH_MPI
	void exchange_ghost_slopes(Tessellation3D const& tess, vector<Slope3D> & slopes)
	{
		Slope3D sdummy;
		MPI_exchange_data(tess, slopes, true);
	}
#endif
}

// ========== Public methods ==========

	SphericalLinearGauss3D::SphericalLinearGauss3D(EquationOfState const& eos, Ghost3D const& ghost,
		Vector3D const& origin, bool slf, double delta_v, double theta,
		double delta_P, bool SR, const vector<string>& calc_tracers,
		const string& skip_key, bool pressure_calc, bool apply_principal_limit,
		bool velocity_radial_extrapolation,
		SphericalLinearGauss3D::FaceRadiusPolicy face_radius_policy,
		double shell_radius_abs_tol, double shell_radius_rel_tol)
		: eos_(eos), ghost_(ghost), origin_(origin), rslopes_(), naive_rslopes_(),
		er_(), etheta_(), ephi_(),
		slf_(slf), shockratio_(delta_v), diffusecoeff_(theta), pressure_ratio_(delta_P), SR_(SR),
		calc_tracers_(calc_tracers), skip_key_(skip_key), pressure_calc_(pressure_calc),
		apply_principal_limit_(apply_principal_limit),
		velocity_radial_extrapolation_(velocity_radial_extrapolation),
		face_radius_policy_(face_radius_policy),
		shell_radius_abs_tol_(shell_radius_abs_tol),
		shell_radius_rel_tol_(shell_radius_rel_tol) {}

void SphericalLinearGauss3D::BuildSlopes(Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells, double time)
{
	prev_rslopes_ = rslopes_;
	prev_naive_rslopes_ = naive_rslopes_;

	const size_t CellNumber = tess.GetPointNo();
	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);

	vector<ComputationalCell3D> new_cells(cells);
	new_cells.resize(tess.GetTotalPointNumber());
	for (auto it = ghost_cells.begin(); it != ghost_cells.end(); ++it)
		new_cells[it->first] = it->second;
	if (SR_)
	{
		for (size_t j = 0; j < new_cells.size(); ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells[j].velocity, new_cells[j].velocity));
			new_cells[j].velocity *= gamma;
		}
	}

	rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	naive_rslopes_.resize(CellNumber);
	er_.resize(CellNumber);
	etheta_.resize(CellNumber);
	ephi_.resize(CellNumber);
	is_pole_cell_.resize(CellNumber, false);

	Slope3D temp1(cells[0], cells[0], cells[0]);
	ComputationalCell3D temp2(cells[0]), temp3(cells[0]), temp4(cells[0]), temp5(cells[0]);
	ComputationalCell3D neigh_sph_buf(cells[0]);
	vector<ComputationalCell3D> neighbor_list;
	vector<Vector3D> neighbor_cm_list;
	vector<Vector3D> neighbor_mesh_list;
	vector<Vector3D> face_sph_cache;
	vector<Vector3D> face_cart_cache;
	vector<double> face_areas_cache;
	vector<double> face_vel_dr_correction;
	std::vector<Vector3D> c_ij;
	Vector3D cell_coords;

	for (size_t i = 0; i < CellNumber; ++i)
	{
		Vector3D cm_i = tess.GetCellCM(i);
		Vector3D sc_i = cart_to_sph_coords(cm_i, origin_);
		bool pole_cell = is_near_pole(sc_i.x, sc_i.y, tess.GetWidth(i));
		is_pole_cell_[i] = pole_cell;

		if (pole_cell)
		{
			calc_slope_cartesian(tess, new_cells, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_,
				eos_, calc_tracers_, origin_, er_[i], etheta_[i], ephi_[i], cell_coords,
				naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
				neighbor_mesh_list, neighbor_cm_list, skip_key_, neighbor_list,
				c_ij, face_cart_cache, face_areas_cache, apply_principal_limit_);
		}
		else
		{
				calc_slope_spherical(tess, new_cells, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_,
					eos_, calc_tracers_, origin_, er_[i], etheta_[i], ephi_[i], cell_coords,
					naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
					neigh_sph_buf, neighbor_cm_list, skip_key_, neighbor_list,
					face_sph_cache, face_areas_cache,
					velocity_radial_extrapolation_, face_radius_policy_,
					shell_radius_abs_tol_, shell_radius_rel_tol_, face_vel_dr_correction);
		}
	}

	// Exchange ghost slopes in native form (before converting to Cartesian)
#ifdef RICH_MPI
	exchange_ghost_slopes(tess, rslopes_);
#endif

	// Convert all slopes to Cartesian - local cells
	for (size_t i = 0; i < CellNumber; ++i)
	{
		if (!is_pole_cell_[i])
		{
			Vector3D sc = cart_to_sph_coords(tess.GetCellCM(i), origin_);
			Vector3D v_sph = cart_to_sph_vec(new_cells[i].velocity,
				er_[i], etheta_[i], ephi_[i]);
			coord_sph_slope_to_cart(rslopes_[i], sc.x, sc.y, v_sph,
				er_[i], etheta_[i], ephi_[i], temp1);
			coord_sph_slope_to_cart(naive_rslopes_[i], sc.x, sc.y, v_sph,
				er_[i], etheta_[i], ephi_[i], temp1);
		}
	}
	// Convert ghost slopes to Cartesian
#ifdef RICH_MPI
	{
		size_t TotalN = rslopes_.size();
		Slope3D g_temp(cells[0], cells[0], cells[0]);
		for (size_t i = CellNumber; i < TotalN; ++i)
		{
			Vector3D gc = tess.GetCellCM(i);
			Vector3D gsc = cart_to_sph_coords(gc, origin_);
			if (!is_near_pole(gsc.x, gsc.y, tess.GetWidth(i)))
			{
				Vector3D ge_r, ge_t, ge_p;
				sph_basis_at(gsc.y, gsc.z, ge_r, ge_t, ge_p);
				Vector3D gv_sph = cart_to_sph_vec(new_cells[i].velocity,
					ge_r, ge_t, ge_p);
				coord_sph_slope_to_cart(rslopes_[i], gsc.x, gsc.y, gv_sph,
					ge_r, ge_t, ge_p, g_temp);
			}
		}
	}
#endif
}

void SphericalLinearGauss3D::Interp(ComputationalCell3D &res, ComputationalCell3D const& cell,
	size_t cell_index, Vector3D const& cm, Vector3D const& target, EquationOfState const& eos) const
{
	try
	{
		ReplaceComputationalCell(res, cell);
		Vector3D delta = target - cm;
		ComputationalCellAddMult3(res, rslopes_[cell_index].xderivative,
			rslopes_[cell_index].yderivative, rslopes_[cell_index].zderivative,
			delta.x, delta.y, delta.z);
		res.internal_energy = eos.dp2e(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Cell density", cell.density);
		eo.addEntry("cell index", static_cast<double>(cell_index));
		throw eo;
	}
}

void SphericalLinearGauss3D::operator()(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells, double time,
	vector<pair<ComputationalCell3D, ComputationalCell3D> > &res) const
{
	prev_rslopes_ = rslopes_;
	prev_naive_rslopes_ = naive_rslopes_;

	const size_t CellNumber = tess.GetPointNo();
	vector<size_t> boundaryedges;
	boundaryedges.reserve(static_cast<size_t>(std::pow(static_cast<double>(CellNumber), 0.6666)*8.0));

	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);

	vector<ComputationalCell3D> new_cells(cells);
	new_cells.resize(tess.GetTotalPointNumber());
	for (auto it = ghost_cells.begin(); it != ghost_cells.end(); ++it)
		new_cells[it->first] = it->second;
	if (SR_)
	{
		for (size_t j = 0; j < new_cells.size(); ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells[j].velocity, new_cells[j].velocity));
			new_cells[j].velocity *= gamma;
		}
	}

	rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	naive_rslopes_.resize(CellNumber);
	er_.resize(CellNumber);
	etheta_.resize(CellNumber);
	ephi_.resize(CellNumber);
	is_pole_cell_.resize(CellNumber, false);

	Slope3D temp1(cells[0], cells[0], cells[0]);
	ComputationalCell3D temp2(cells[0]), temp3(cells[0]), temp4(cells[0]), temp5(cells[0]);
	ComputationalCell3D neigh_sph_buf(cells[0]);
	vector<ComputationalCell3D> neighbor_list;
	vector<Vector3D> neighbor_cm_list;
	vector<Vector3D> neighbor_mesh_list;
	vector<Vector3D> face_sph_cache;
	vector<Vector3D> face_cart_cache;
	vector<double> face_areas_cache;
	vector<double> face_vel_dr_correction;
	std::vector<Vector3D> c_ij;

	res.resize(tess.GetTotalFacesNumber(), pair<ComputationalCell3D, ComputationalCell3D>(cells[0], cells[0]));
	ComputationalCell3D* cell_ref = nullptr;

	size_t energy_index = ComputationalCell3D::tracerNames.size();
	auto it = binary_find(ComputationalCell3D::tracerNames.begin(),
		ComputationalCell3D::tracerNames.end(), string("Energy"));
	if (it != ComputationalCell3D::tracerNames.end())
		energy_index = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
	bool energy_fix = energy_index < ComputationalCell3D::tracerNames.size();

	Vector3D cell_coords;

	for (size_t i = 0; i < CellNumber; ++i)
	{
		Vector3D cm_i = tess.GetCellCM(i);
		Vector3D sc_i = cart_to_sph_coords(cm_i, origin_);
		bool pole_cell = is_near_pole(sc_i.x, sc_i.y, tess.GetWidth(i));
		is_pole_cell_[i] = pole_cell;

		if (pole_cell)
		{
			calc_slope_cartesian(tess, new_cells, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_,
				eos_, calc_tracers_, origin_, er_[i], etheta_[i], ephi_[i], cell_coords,
				naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
				neighbor_mesh_list, neighbor_cm_list, skip_key_, neighbor_list,
				c_ij, face_cart_cache, face_areas_cache, apply_principal_limit_);

			face_vec const& faces = tess.GetCellFaces(i);
			const size_t nloop = faces.size();

			for (size_t j = 0; j < nloop; ++j)
			{
				bool is_first = (tess.GetFaceNeighbors(faces[j]).first == i);
				cell_ref = is_first ? &res[faces[j]].first : &res[faces[j]].second;

				ReplaceComputationalCell(*cell_ref, new_cells[i]);
				try
				{
					if (pressure_calc_)
						cart_interp23D(*cell_ref, rslopes_[i], face_cart_cache[j], cm_i, eos_, true);
					else
					{
						cart_interp23D(*cell_ref, rslopes_[i], face_cart_cache[j], cm_i, eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
					CheckCell(*cell_ref);
				}
				catch (UniversalError &eo)
				{
					eo.addEntry("SphericalLinearGauss3D face interp error (pole cell)", 0);
					eo.addEntry("Cell", static_cast<double>(i));
					eo.addEntry("Face", static_cast<double>(faces[j]));
					throw eo;
				}

				size_t other = is_first ? tess.GetFaceNeighbors(faces[j]).second : tess.GetFaceNeighbors(faces[j]).first;
				if (is_ghost_index(other, CellNumber))
					boundaryedges.push_back(faces[j]);
			}
		}
		else
		{
				calc_slope_spherical(tess, new_cells, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_,
					eos_, calc_tracers_, origin_, er_[i], etheta_[i], ephi_[i], cell_coords,
					naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
					neigh_sph_buf, neighbor_cm_list, skip_key_, neighbor_list,
					face_sph_cache, face_areas_cache,
					velocity_radial_extrapolation_, face_radius_policy_,
					shell_radius_abs_tol_, shell_radius_rel_tol_, face_vel_dr_correction);

			face_vec const& faces = tess.GetCellFaces(i);
			const size_t nloop = faces.size();

			ComputationalCell3D cell_sph;
			ReplaceComputationalCell(cell_sph, new_cells[i]);
			cell_sph.velocity = cart_to_sph_vec(new_cells[i].velocity, er_[i], etheta_[i], ephi_[i]);

			for (size_t j = 0; j < nloop; ++j)
			{
				double dr = face_sph_cache[j].x - cell_coords.x;
				double dtheta = face_sph_cache[j].y - cell_coords.y;
				double dphi_val = wrap_dphi(face_sph_cache[j].z - cell_coords.z);

				bool is_first = (tess.GetFaceNeighbors(faces[j]).first == i);
				cell_ref = is_first ? &res[faces[j]].first : &res[faces[j]].second;

				ReplaceComputationalCell(*cell_ref, cell_sph);
				try
				{
					if (pressure_calc_)
						interp_coord_eos(*cell_ref, rslopes_[i], dr, dtheta, dphi_val, eos_, true);
					else
					{
						interp_coord_eos(*cell_ref, rslopes_[i], dr, dtheta, dphi_val, eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}

					if (j < face_vel_dr_correction.size())
					{
						double vel_dr_extra = face_vel_dr_correction[j];
						cell_ref->velocity.x += rslopes_[i].xderivative.velocity.x * vel_dr_extra;
						cell_ref->velocity.y += rslopes_[i].xderivative.velocity.y * vel_dr_extra;
						cell_ref->velocity.z += rslopes_[i].xderivative.velocity.z * vel_dr_extra;
					}

					Vector3D face_er, face_et, face_ep;
					sph_basis_at(face_sph_cache[j].y, face_sph_cache[j].z, face_er, face_et, face_ep);
					cell_ref->velocity = sph_to_cart_vec(cell_ref->velocity, face_er, face_et, face_ep);
					CheckCell(*cell_ref);
				}
				catch (UniversalError &eo)
				{
					eo.addEntry("SphericalLinearGauss3D face interp error", 0);
					eo.addEntry("Cell", static_cast<double>(i));
					eo.addEntry("Face", static_cast<double>(faces[j]));
					throw eo;
				}

				size_t other = is_first ? tess.GetFaceNeighbors(faces[j]).second : tess.GetFaceNeighbors(faces[j]).first;
				if (is_ghost_index(other, CellNumber))
					boundaryedges.push_back(faces[j]);
			}
		}
	}

	// Exchange ghost slopes in native form (spherical for non-pole, Cartesian for pole)
#ifdef RICH_MPI
	exchange_ghost_slopes(tess, rslopes_);
#endif

	// Boundary edge interpolation - slopes are still in native form
	Slope3D ghost_slope;
	for (size_t i = 0; i < boundaryedges.size(); ++i)
	{
		size_t N0 = tess.GetFaceNeighbors(boundaryedges[i]).first;
		bool ghost_is_first = is_ghost_index(N0, CellNumber);
		if (!ghost_is_first)
			N0 = tess.GetFaceNeighbors(boundaryedges[i]).second;

		cell_ref = ghost_is_first ? &res[boundaryedges[i]].first : &res[boundaryedges[i]].second;

		Vector3D ghost_cm = tess.GetCellCM(N0);
		ReplaceComputationalCell(*cell_ref, new_cells[N0]);

		try
		{
			bool use_sph_interp = false;
			Vector3D ghost_sc, g_er, g_et, g_ep;

#ifdef RICH_MPI
			if (tess.BoundaryFace(boundaryedges[i]))
			{
				size_t interior = ghost_is_first ?
					tess.GetFaceNeighbors(boundaryedges[i]).second :
					tess.GetFaceNeighbors(boundaryedges[i]).first;
				Slope3D saved = rslopes_[interior];
				if (interior < CellNumber && !is_pole_cell_[interior])
				{
					Vector3D sc_int = cart_to_sph_coords(tess.GetCellCM(interior), origin_);
					Vector3D v_sph_int = cart_to_sph_vec(new_cells[interior].velocity,
						er_[interior], etheta_[interior], ephi_[interior]);
					coord_sph_slope_to_cart(rslopes_[interior], sc_int.x, sc_int.y,
						v_sph_int, er_[interior], etheta_[interior],
						ephi_[interior], temp1);
				}
				ghost_slope = ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]);
				rslopes_[interior] = saved;
			}
			else
				{
					ghost_slope = rslopes_[N0];
					ghost_sc = cart_to_sph_coords(ghost_cm, origin_);
					use_sph_interp = !is_near_pole(ghost_sc.x, ghost_sc.y, tess.GetWidth(N0));
					if (use_sph_interp)
						sph_basis_at(ghost_sc.y, ghost_sc.z, g_er, g_et, g_ep);
			}
#else
			ghost_slope = ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]);
#endif

				if (use_sph_interp)
				{
					cell_ref->velocity = cart_to_sph_vec(cell_ref->velocity, g_er, g_et, g_ep);
					Vector3D physical_face_sc = cart_to_sph_coords(tess.FaceCM(boundaryedges[i]), origin_);
					Vector3D face_sc = effective_face_sph(tess, N0, boundaryedges[i],
						origin_, ghost_sc, physical_face_sc, face_radius_policy_,
						shell_radius_abs_tol_, shell_radius_rel_tol_);
					double dr = face_sc.x - ghost_sc.x;
					double dtheta = face_sc.y - ghost_sc.y;
					double dphi_val = wrap_dphi(face_sc.z - ghost_sc.z);

				if (pressure_calc_)
					interp_coord_eos(*cell_ref, ghost_slope, dr, dtheta, dphi_val, eos_, true);
				else
				{
					interp_coord_eos(*cell_ref, ghost_slope, dr, dtheta, dphi_val, eos_, false);
					if (energy_fix)
						cell_ref->tracers[energy_index] = cell_ref->internal_energy;
				}

				if (velocity_radial_extrapolation_ &&
					face_radius_policy_ == SphericalLinearGauss3D::FaceRadiusPolicy::PhysicalFaceCM)
				{
					auto bneigh = tess.GetFaceNeighbors(boundaryedges[i]);
					double r_avg = 0.5 * (abs(tess.GetMeshPoint(bneigh.first) - origin_)
						+ abs(tess.GetMeshPoint(bneigh.second) - origin_));
					double dr_correction = r_avg - physical_face_sc.x;
					cell_ref->velocity.x += ghost_slope.xderivative.velocity.x * dr_correction;
					cell_ref->velocity.y += ghost_slope.xderivative.velocity.y * dr_correction;
					cell_ref->velocity.z += ghost_slope.xderivative.velocity.z * dr_correction;
				}

				Vector3D bface_er, bface_et, bface_ep;
				sph_basis_at(face_sc.y, face_sc.z, bface_er, bface_et, bface_ep);
				cell_ref->velocity = sph_to_cart_vec(cell_ref->velocity, bface_er, bface_et, bface_ep);
			}
			else
			{
				Vector3D face_cm_cart = tess.FaceCM(boundaryedges[i]);
				if (pressure_calc_)
					cart_interp23D(*cell_ref, ghost_slope, face_cm_cart, ghost_cm, eos_, true);
				else
				{
					cart_interp23D(*cell_ref, ghost_slope, face_cm_cart, ghost_cm, eos_, false);
					if (energy_fix)
						cell_ref->tracers[energy_index] = cell_ref->internal_energy;
				}
			}

			CheckCell(*cell_ref);
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("SphericalLinearGauss3D boundary edge error", 0);
			eo.addEntry("Ghost cell", static_cast<double>(N0));
			eo.addEntry("Face", static_cast<double>(boundaryedges[i]));
			throw eo;
		}
	}

	// Convert all slopes to Cartesian - local cells
	for (size_t i = 0; i < CellNumber; ++i)
	{
		if (!is_pole_cell_[i])
		{
			Vector3D sc = cart_to_sph_coords(tess.GetCellCM(i), origin_);
			Vector3D v_sph = cart_to_sph_vec(new_cells[i].velocity,
				er_[i], etheta_[i], ephi_[i]);
			coord_sph_slope_to_cart(rslopes_[i], sc.x, sc.y, v_sph,
				er_[i], etheta_[i], ephi_[i], temp1);
			coord_sph_slope_to_cart(naive_rslopes_[i], sc.x, sc.y, v_sph,
				er_[i], etheta_[i], ephi_[i], temp1);
		}
	}
#ifdef RICH_MPI
	{
		size_t TotalN = rslopes_.size();
		Slope3D g_temp(cells[0], cells[0], cells[0]);
		for (size_t i = CellNumber; i < TotalN; ++i)
		{
			Vector3D gc = tess.GetCellCM(i);
			Vector3D gsc = cart_to_sph_coords(gc, origin_);
			if (!is_near_pole(gsc.x, gsc.y, tess.GetWidth(i)))
			{
				Vector3D ge_r, ge_t, ge_p;
				sph_basis_at(gsc.y, gsc.z, ge_r, ge_t, ge_p);
				Vector3D gv_sph = cart_to_sph_vec(new_cells[i].velocity,
					ge_r, ge_t, ge_p);
				coord_sph_slope_to_cart(rslopes_[i], gsc.x, gsc.y, gv_sph,
					ge_r, ge_t, ge_p, g_temp);
			}
		}
	}
#endif

	if (SR_)
	{
		size_t N = res.size();
		for (size_t i = 0; i < N; ++i)
		{
			double factor = 1.0 / std::sqrt(1 + ScalarProd(res[i].first.velocity, res[i].first.velocity));
			res[i].first.velocity *= factor;
			factor = 1.0 / std::sqrt(1 + ScalarProd(res[i].second.velocity, res[i].second.velocity));
			res[i].second.velocity *= factor;
		}
	}
}

vector<Slope3D>& SphericalLinearGauss3D::GetSlopes(void)
{
	return rslopes_;
}

vector<Slope3D>& SphericalLinearGauss3D::GetSlopesUnlimited(void) const
{
	return naive_rslopes_;
}

void SphericalLinearGauss3D::GetBasis(vector<Vector3D>& er, vector<Vector3D>& etheta, vector<Vector3D>& ephi) const
{
	er = er_;
	etheta = etheta_;
	ephi = ephi_;
}
