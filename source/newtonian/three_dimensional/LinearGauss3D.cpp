#include "LinearGauss3D.hpp"
#include "../../misc/utils.hpp"
#include <array>
#include <iostream>
#include <cstring>  // for memcpy in fused operations
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
	void CheckCell(ComputationalCell3D const& cell)
	{
		if ((!(cell.density > 0)) || (!(cell.internal_energy > 0)) || (!std::isfinite(cell.velocity.x)) || (!std::isfinite(cell.velocity.y)) || (!std::isfinite(cell.velocity.z)))
			throw UniversalError("Bad cell after interpolation in LinearGauss3D");
	}

	void GetNeighborMesh(Tessellation3D const& tess, size_t cell_index,
		vector<Vector3D> &res, face_vec const& faces)
	{
		res.resize(faces.size());
		const int nloop = static_cast<int>(res.size());
		std::pair<size_t, size_t> neigh;
		for (int i = 0; i < nloop; ++i)
		{
		  neigh = tess.GetFaceNeighbors(faces[static_cast<size_t>(i)]);
			if (neigh.first == cell_index)
			  res[static_cast<size_t>(i)] = tess.GetMeshPoint(neigh.second);
			else
			  res[static_cast<size_t>(i)] = tess.GetMeshPoint(neigh.first);
		}
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
			if(!std::isfinite(cells[other_cell].density))
				throw UniversalError("Bad density getneighborcell");
		}
	}

	// Get neighbor indices instead of copying cells - avoids expensive copies
	void GetNeighborIndices(Tessellation3D const& tess, size_t cell_index,
		face_vec const& faces, vector<size_t> &neighbor_indices)
	{
		const size_t nloop = faces.size();
		neighbor_indices.resize(nloop);
		for (size_t i = 0; i < nloop; ++i)
		{
			const auto& face_neighbors = tess.GetFaceNeighbors(faces[i]);
			neighbor_indices[i] = (face_neighbors.first == cell_index) ?
				face_neighbors.second : face_neighbors.first;
		}
	}

	void calc_naive_slope(ComputationalCell3D const& cell,
		Vector3D const& center, Vector3D const& cell_cm, double cell_volume, vector<ComputationalCell3D> const& neighbors,
		vector<Vector3D> const& neighbor_centers,
		vector<Vector3D> const& neigh_cm, Tessellation3D const& tess,
		Slope3D &res, Slope3D &temp, size_t /*index*/, face_vec const& faces,
		std::vector<Vector3D> &c_ij,
		vector<Vector3D>& face_cms_cache, vector<double>& face_areas_cache)
	{
		size_t n = neighbor_centers.size();
		if (n > 100)
			std::cout << "Cell has too many neighbors in calc naive slope, Cell x cor " << center.x <<
			" Cell y cor " << center.y << " Cell z cor " << center.z << std::endl;
		// Create the matrix to invert and the vector to compare
		std::array<double, 9>  m;
		std::fill_n(m.begin(), 9, 0.0);
		c_ij.resize(n);
		
		// Cache face CMs and areas - avoid repeated tessellation lookups
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
		// Find the det
		const double det = -m[2] * m[4] * m[6] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7] - m[0] * m[5] * m[7] -
			m[1] * m[3] * m[8] + m[0] * m[4] * m[8];
		// Check none singular
		if (std::abs(det) < 1e-10)
		{
			UniversalError eo("Singular matrix");
			eo.addEntry("Cell x cor", center.x);
			eo.addEntry("Cell y cor", center.y);
			eo.addEntry("Cell z cor", center.z);
			eo.addEntry("Cell CMx cor", cell_cm.x);
			eo.addEntry("Cell CMy cor", cell_cm.y);
			eo.addEntry("Cell CMz cor", cell_cm.z);
			eo.addEntry("Cell volume", cell_volume);
			eo.addEntry("Det was", det);
			for (size_t i = 0; i < faces.size(); ++i)
			{
				c_ij[0] = tess.FaceCM(faces[i]) - 0.5 * (neigh_cm[i] + cell_cm);
				eo.addEntry("Neighbor x", neighbor_centers[i].x);
				eo.addEntry("Neighbor y", neighbor_centers[i].y);
				eo.addEntry("Neighbor z", neighbor_centers[i].z);
				eo.addEntry("Face", static_cast<double>(faces[i]));
				eo.addEntry("Neighbor Cx", c_ij[0].x);
				eo.addEntry("Neighbor Cy", c_ij[0].y);
				eo.addEntry("Neighbor Cz", c_ij[0].z);
				eo.addEntry("Face Cx", tess.FaceCM(faces[i]).x);
				eo.addEntry("Face Cy", tess.FaceCM(faces[i]).y);
				eo.addEntry("Face Cz", tess.FaceCM(faces[i]).z);
				eo.addEntry("Face area", tess.GetArea(faces[i]));
			}
			for (size_t i = 0; i < 9; ++i)
				eo.addEntry("M", m[i]);
			throw eo;
		}
		// Invert the matrix
		std::array<double, 9>  m_inv;
		std::fill_n(m_inv.begin(), 9, 0);
		m_inv[0] = m[4] * m[8] - m[5] * m[7];
		m_inv[1] = m[2] * m[7] - m[1] * m[8];
		m_inv[2] = m[1] * m[5] - m[2] * m[4];
		m_inv[3] = m[5] * m[6] - m[3] * m[8];
		m_inv[4] = m[0] * m[8] - m[2] * m[6];
		m_inv[5] = m[2] * m[3] - m[5] * m[0];
		m_inv[6] = m[3] * m[7] - m[6] * m[4];
		m_inv[7] = m[6] * m[1] - m[0] * m[7];
		m_inv[8] = m[4] * m[0] - m[1] * m[3];
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


	double PressureRatio(ComputationalCell3D const& cell, vector<ComputationalCell3D> const& neigh)
	{
		double res = 1.0;
		double p = cell.pressure;
		size_t N = neigh.size();
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma omp simd reduction(min:res)
#endif
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
		ComputationalCell3D const& cell, vector<ComputationalCell3D> const& neighbor_list, double pressure_ratio, double cs)
	{
		double div_v = (naive_slope.xderivative.velocity.x + naive_slope.yderivative.velocity.y +
			naive_slope.zderivative.velocity.z) * cell_width;
		double t_div = std::max(0.0, std::min(1.0, -div_v / (shock_ratio * cs)));
		double w_div = t_div;
		double p_ratio = PressureRatio(cell, neighbor_list);
		double t_pres = std::max(0.0, std::min(1.0, (pressure_ratio - 0.8 * p_ratio) / (0.4 * pressure_ratio)));
		double w_pres = t_pres;
		return std::max(w_div, w_pres);
	}

	bool build_principal_frame(Vector3D const& velocity, Vector3D &e1, Vector3D &e2, Vector3D &e3)
	{
		double vmag = abs(velocity);
		if (vmag < 1e-14)
			return false;
		e1 = velocity / vmag;
		// Smooth orthonormal basis (Duff et al. 2017, "Building an Orthonormal Basis, Revisited")
		double sign_z = std::copysign(1.0, e1.z);
		double a = -1.0 / (sign_z + e1.z);
		double b = e1.x * e1.y * a;
		e2 = Vector3D(1.0 + sign_z * e1.x * e1.x * a, sign_z * b, -sign_z * e1.x);
		e3 = Vector3D(b, sign_z + e1.y * e1.y * a, -e1.y);
		return true;
	}

	void apply_principal_limit(Vector3D &sv, Vector3D const& e1, Vector3D const& e2,
		Vector3D const& e3, double psi1, double psi2, double psi3)
	{
		double c1 = ScalarProd(sv, e1);
		double c2 = ScalarProd(sv, e2);
		double c3 = ScalarProd(sv, e3);
		sv.x = e1.x * c1 * psi1 + e2.x * c2 * psi2 + e3.x * c3 * psi3;
		sv.y = e1.y * c1 * psi1 + e2.y * c2 * psi2 + e3.y * c3 * psi3;
		sv.z = e1.z * c1 * psi1 + e2.z * c2 * psi2 + e3.z * c3 * psi3;
	}

	ComputationalCell3D interp(ComputationalCell3D const& cell, Slope3D const& slope,
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
			//res.pressure = eos.de2p(res.density, res.internal_energy, res.tracers, tsn.tracer_names);
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

	void interp23Dsimple(ComputationalCell3D &res, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm)
	{
		// Use vectorized version - processes all 3 derivatives in one pass
		ComputationalCellAddMult3(res, slope.xderivative, slope.yderivative, slope.zderivative,
			target.x - cm.x, target.y - cm.y, target.z - cm.z);
	}

	void interp23D(ComputationalCell3D &res, Slope3D const& slope,
		Vector3D const& target, Vector3D const& cm, EquationOfState const& eos,
		bool pressure_calc)
	{
		// Use vectorized version - processes all 3 derivatives in one pass
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


	void blended_slope_limit(ComputationalCell3D const& cell, Vector3D const& cm,
		vector<ComputationalCell3D> const& neighbors, Slope3D &slope, ComputationalCell3D &cmax,
		ComputationalCell3D &cmin, ComputationalCell3D &maxdiff, ComputationalCell3D &mindiff,
		double diffusecoeff, double shock_w,
		string const& skip_key, Tessellation3D const& tess,
		size_t cell_index, face_vec const& faces, EquationOfState const& eos,
		vector<Vector3D> const& face_cms_cache, bool apply_principal_limit_flag,
		vector<double> &psi_buf)
	{
		ReplaceComputationalCell(cmax, cell);
		ReplaceComputationalCell(cmin, cell);
		size_t nloop = neighbors.size();
		size_t ntracer = ComputationalCell3D::tracerNames.size();

		Vector3D e1, e2, e3;
		bool has_frame = apply_principal_limit_flag &&
			build_principal_frame(cell.velocity, e1, e2, e3);
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
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma omp simd
#endif
			for (size_t j = 0; j < ntracer; ++j)
			{
				cmax.tracers[j] = std::max(cmax.tracers[j], cell_temp.tracers[j]);
				cmin.tracers[j] = std::min(cmin.tracers[j], cell_temp.tracers[j]);
			}
		}
		// cmax/cmin hold absolute neighbor max/min (used for shocked skip conditions)
		// maxdiff/mindiff hold differences from cell (used for BJ formula and BJ skip)
		ReplaceComputationalCell(maxdiff, cmax);
		maxdiff -= cell;
		ReplaceComputationalCell(mindiff, cmin);
		mindiff -= cell;

		double v1_maxdiff = v1_max - cell_v1, v1_mindiff = v1_min - cell_v1;
		double v2_maxdiff = v2_max - cell_v2, v2_mindiff = v2_min - cell_v2;
		double v3_maxdiff = v3_max - cell_v3, v3_mindiff = v3_min - cell_v3;

		ComputationalCell3D centroid_val = interp(cell, slope, face_cms_cache[0], cm, eos, false);
		ComputationalCell3D dphi = centroid_val - cell;
		psi_buf.assign(6 + cell.tracers.size(), 1);
		vector<double> &psi = psi_buf;
		double psi_v1 = 1.0, psi_v2 = 1.0, psi_v3 = 1.0;
		const size_t nedges = faces.size();
		const double skipfactor = 1e-3;
		const double sf = 1e-9;
		const double w = shock_w;
		const double w1 = 1.0 - w;
		auto blend_psi = [w, w1](double psi_bj, double psi_sh) {
			double psi_mix = w1 * psi_bj + w * psi_sh;
			return psi_mix;
		};

		for (size_t i = 0; i < nedges; i++)
		{
			if (i > 0)
			{
				ReplaceComputationalCell(centroid_val, cell);
				interp23Dsimple(centroid_val, slope, face_cms_cache[i], cm);
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
				// v1
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
				// v2
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
				// v3
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
				// vx
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
				// vy
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
				// vz
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
		if (has_frame)
		{			
			apply_principal_limit(slope.xderivative.velocity, e1, e2, e3, psi_v1, psi_v2, psi_v3);
			apply_principal_limit(slope.yderivative.velocity, e1, e2, e3, psi_v1, psi_v2, psi_v3);
			apply_principal_limit(slope.zderivative.velocity, e1, e2, e3, psi_v1, psi_v2, psi_v3);
		}
		else
		{
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
#if defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
#pragma ivdep
#endif
		for (size_t k = 0; k < N; ++k)
		{
			slope.xderivative.tracers[k] *= psi[counter];
			slope.yderivative.tracers[k] *= psi[counter];
			slope.zderivative.tracers[k] *= psi[counter];
			++counter;
		}
		if (shock_w > 0.9)
		{
			double maxDv = ScalarProd(slope.xderivative.velocity, slope.xderivative.velocity)
				+ ScalarProd(slope.yderivative.velocity, slope.yderivative.velocity) +
				ScalarProd(slope.zderivative.velocity, slope.zderivative.velocity);
			maxDv *= tess.GetWidth(cell_index) * tess.GetWidth(cell_index);
			if (maxDv > 100 * ScalarProd(cell.velocity, cell.velocity))
			{
				double sfactor = fastsqrt(100 * ScalarProd(cell.velocity, cell.velocity) / maxDv);
				slope.xderivative.velocity.x *= sfactor;
				slope.yderivative.velocity.x *= sfactor;
				slope.zderivative.velocity.x *= sfactor;
				slope.xderivative.velocity.y *= sfactor;
				slope.yderivative.velocity.y *= sfactor;
				slope.zderivative.velocity.y *= sfactor;
				slope.xderivative.velocity.z *= sfactor;
				slope.yderivative.velocity.z *= sfactor;
				slope.zderivative.velocity.z *= sfactor;
			}
		}
	}

	void calc_slope(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells, size_t cell_index, bool slf,
		double shockratio, double diffusecoeff, double pressure_ratio, EquationOfState const& eos,
		const vector<string>& calc_tracers, Slope3D &naive_slope_, Slope3D & res, Slope3D &temp1, ComputationalCell3D &temp2,
		ComputationalCell3D &temp3, ComputationalCell3D &temp4, ComputationalCell3D &temp5,
		vector<Vector3D> &neighbor_mesh_list,
		vector<Vector3D> &neighbor_cm_list,
 string const& skip_key,
		std::vector<Vector3D> &c_ij, vector<ComputationalCell3D> &neighbor_list,
		vector<Vector3D>& face_cms_cache, vector<double>& face_areas_cache, bool apply_principal_limit_flag,
		vector<double>& psi_buf)
	{
		face_vec const& faces = tess.GetCellFaces(cell_index);
		GetNeighborMesh(tess, cell_index, neighbor_mesh_list, faces);
		GetNeighborCM(tess, cell_index, neighbor_cm_list, faces);
		GetNeighborCells(tess, cell_index, cells, faces, neighbor_list);
		ComputationalCell3D const& cell = cells[cell_index];
		bool boundary_slope = false;
		size_t Nneigh = faces.size();
		for (size_t i = 0; i < Nneigh; ++i)
			if (tess.BoundaryFace(faces[i]))
			{
				boundary_slope = true;
				break;
			}
		calc_naive_slope(cell, tess.GetMeshPoint(cell_index), tess.GetCellCM(cell_index),
			tess.GetVolume(cell_index), neighbor_list, neighbor_mesh_list, neighbor_cm_list, tess, res, temp1,
			cell_index, faces, c_ij, face_cms_cache, face_areas_cache);
		if(!std::isfinite( res.xderivative.density))
		{
			UniversalError eo("Bad slope");
			eo.addEntry("boundary_slope", boundary_slope);
			throw eo;
		}
		naive_slope_ = res;

		for (size_t i = 0; i < ComputationalCell3D::tracerNames.size(); ++i)
		{
		  if (std::find(calc_tracers.begin(), calc_tracers.end(), ComputationalCell3D::tracerNames[i]) == calc_tracers.end())
			{
				res.xderivative.tracers[i] = 0;
				res.yderivative.tracers[i] = 0;
				res.zderivative.tracers[i] = 0;
			}
		}
		res.xderivative.Erad = 0;
		res.yderivative.Erad = 0;
		res.zderivative.Erad = 0;
		res.xderivative.Erad_dt = 0;
		res.yderivative.Erad_dt = 0;
		res.zderivative.Erad_dt = 0;
		res.xderivative.Erad_dt_dt = 0;
		res.yderivative.Erad_dt_dt = 0;
		res.zderivative.Erad_dt_dt = 0;
		for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.xderivative.Eg[j] = 0;
		for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.yderivative.Eg[j] = 0;
		for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
			res.zderivative.Eg[j] = 0;
		res.xderivative.temperature = 0;
		res.yderivative.temperature = 0;
		res.zderivative.temperature = 0;
		if (slf)
		{
#ifdef RICH_DEBUG
			try
			{
#endif
				{
					double sw = shock_weight(res, tess.GetWidth(cell_index), shockratio, cell,
						neighbor_list, pressure_ratio,
						eos.de2c(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames));
					blended_slope_limit(cell, tess.GetCellCM(cell_index), neighbor_list, res, temp2, temp3, temp4, temp5,
						diffusecoeff, sw, skip_key, tess, cell_index, faces, eos, face_cms_cache, apply_principal_limit_flag, psi_buf);
				}
				if(!std::isfinite( res.xderivative.density))
				{
					UniversalError eo("Bad slope limited");
					eo.addEntry("boundary_slope", boundary_slope);
					throw eo;
				}
#ifdef RICH_DEBUG
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("Error LinearGauss3D", 0);
				eo.addEntry("Cell number", cell_index);
				throw eo;
			}
#endif
		}
	}

#ifdef RICH_MPI

	void exchange_ghost_slopes(Tessellation3D const& tess, vector<Slope3D> & slopes)
	{
		Slope3D sdummy;
		MPI_exchange_data(tess, slopes, true);
	}
#endif//RICH_MPI
}

void LinearGauss3D::Interp(ComputationalCell3D &res, ComputationalCell3D const& cell, size_t cell_index, Vector3D const& cm,
	Vector3D const& target, EquationOfState const& eos)const
{
	try
	{
		res = interp(cell, rslopes_[cell_index], target, cm, eos, true);
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Cell density", cell.density);
		eo.addEntry("Cell internal energy", cell.internal_energy);
		eo.addEntry("cell index", static_cast<double>(cell_index));
		eo.addEntry("CMx", cm.x);
		eo.addEntry("CMy", cm.y);
		eo.addEntry("CMz", cm.z);
		eo.addEntry("Targetx", target.x);
		eo.addEntry("Targety", target.y);
		eo.addEntry("Targetz", target.z);
		throw eo;
	}
}

LinearGauss3D::LinearGauss3D(EquationOfState const& eos, Ghost3D const& ghost, bool slf, double delta_v, double theta,
	double delta_P, bool SR, const vector<string>& calc_tracers, const string& skip_key,
	bool pressure_calc, bool apply_principal_limit) : eos_(eos), ghost_(ghost), rslopes_(),
	naive_rslopes_(), slf_(slf), shockratio_(delta_v), diffusecoeff_(theta), pressure_ratio_(delta_P), SR_(SR),
	calc_tracers_(calc_tracers), skip_key_(skip_key), to_skip_(), pressure_calc_(pressure_calc),
	apply_principal_limit_(apply_principal_limit) {}

void LinearGauss3D::BuildSlopes(Tessellation3D const& tess, std::vector<ComputationalCell3D> const& cells, double time) 
{
	const size_t CellNumber = tess.GetPointNo();
	// Get ghost points
	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);
	// Reuse persistent buffer instead of allocating a new vector each call
	new_cells_.assign(cells.begin(), cells.end());
	new_cells_.resize(tess.GetTotalPointNumber());
	for (boost::container::flat_map<size_t, ComputationalCell3D>::const_iterator it = ghost_cells.begin(); it !=
		ghost_cells.end(); ++it)
		new_cells_[it->first] = it->second;
	if (SR_)
	{
		size_t Nall = new_cells_.size();
		for (size_t j = 0; j < Nall; ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells_[j].velocity, new_cells_[j].velocity));
			new_cells_[j].velocity *= gamma;
		}
	}
	if (CellNumber > 0)
		rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	else
		rslopes_.clear();
	conditional_shrink(rslopes_);
	naive_rslopes_.resize(CellNumber);
	conditional_shrink(naive_rslopes_);
	Slope3D temp1;
	ComputationalCell3D temp2;
	ComputationalCell3D temp3;
	ComputationalCell3D temp4;
	ComputationalCell3D temp5;
	if (CellNumber > 0)
	{
		temp1 = Slope3D(cells[0], cells[0], cells[0]);
		temp2 = cells[0];
		temp3 = cells[0];
		temp4 = cells[0];
		temp5 = cells[0];
	}
	vector<ComputationalCell3D> neighbor_list;
	vector<Vector3D> neighbor_mesh_list;
	vector<Vector3D> neighbor_cm_list;
	std::vector<Vector3D> c_ij;
	vector<Vector3D> face_cms_cache;
	vector<double> face_areas_cache;
	vector<double> psi_buf0;
	for (size_t i = 0; i < CellNumber; ++i)
	{
		calc_slope(tess, new_cells_, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_, eos_,
			calc_tracers_, naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
			neighbor_mesh_list, neighbor_cm_list, skip_key_, c_ij, neighbor_list,
			face_cms_cache, face_areas_cache, apply_principal_limit_, psi_buf0);
	}
#ifdef RICH_MPI
	// communicate ghost slopes
	exchange_ghost_slopes(tess, rslopes_);
#endif //RICH_MPI
}

void LinearGauss3D::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, double time,
	vector<pair<ComputationalCell3D, ComputationalCell3D> > &res) const
{
	const size_t CellNumber = tess.GetPointNo();
	vector<size_t> boundaryedges;
	boundaryedges.reserve(static_cast<size_t>(std::pow(static_cast<double>(CellNumber), 0.6666)*8.0));
	// Get ghost points
	boost::container::flat_map<size_t, ComputationalCell3D> ghost_cells;
	ghost_.operator()(tess, cells, time, ghost_cells);
	// Reuse persistent buffer instead of allocating a new vector each call
	new_cells_.assign(cells.begin(), cells.end());
	new_cells_.resize(tess.GetTotalPointNumber());
	for (boost::container::flat_map<size_t, ComputationalCell3D>::const_iterator it = ghost_cells.begin(); it !=
		ghost_cells.end(); ++it)
		new_cells_[it->first] = it->second;
	if (SR_)
	{
		size_t Nall = new_cells_.size();
		for (size_t j = 0; j < Nall; ++j)
		{
			double gamma = 1.0 / std::sqrt(1 - ScalarProd(new_cells_[j].velocity, new_cells_[j].velocity));
			new_cells_[j].velocity *= gamma;
		}
	}
	if (CellNumber > 0)
		rslopes_.resize(CellNumber, Slope3D(cells[0], cells[0], cells[0]));
	else
		rslopes_.clear();
	conditional_shrink(rslopes_);
	naive_rslopes_.resize(CellNumber);
	conditional_shrink(naive_rslopes_);
	Slope3D temp1;
	ComputationalCell3D temp2;
	ComputationalCell3D temp3;
	ComputationalCell3D temp4;
	ComputationalCell3D temp5;
	if (CellNumber > 0)
	{
		temp1 = Slope3D(cells[0], cells[0], cells[0]);
		temp2 = cells[0];
		temp3 = cells[0];
		temp4 = cells[0];
		temp5 = cells[0];
	}
	vector<ComputationalCell3D> neighbor_list;
	vector<Vector3D> neighbor_mesh_list;
	vector<Vector3D> neighbor_cm_list;
	std::vector<Vector3D> c_ij;
	vector<Vector3D> face_cms_cache;
	vector<double> face_areas_cache;
	vector<double> psi_buf;
	
	// Pre-compute all cell CMs once - avoid repeated lookups
	vector<Vector3D> all_cell_cms(CellNumber);
	for (size_t i = 0; i < CellNumber; ++i)
		all_cell_cms[i] = tess.GetCellCM(i);
	
	if (CellNumber > 0)
		res.resize(tess.GetTotalFacesNumber(), pair<ComputationalCell3D, ComputationalCell3D>(cells[0], cells[0]));
	else
		res.clear();
	ComputationalCell3D* cell_ref = nullptr;
	size_t energy_index = ComputationalCell3D::tracerNames.size();
	vector<string>::const_iterator it = binary_find(ComputationalCell3D::tracerNames.begin(),
							ComputationalCell3D::tracerNames.end(), string("Energy"));
	if (it != ComputationalCell3D::tracerNames.end())
	  energy_index = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
	bool energy_fix = energy_index < ComputationalCell3D::tracerNames.size();
	for (size_t i = 0; i < CellNumber; ++i)
	{
		calc_slope(tess, new_cells_, i, slf_, shockratio_, diffusecoeff_, pressure_ratio_, eos_,
			calc_tracers_, naive_rslopes_[i], rslopes_[i], temp1, temp2, temp3, temp4, temp5,
			neighbor_mesh_list, neighbor_cm_list, skip_key_, c_ij, neighbor_list,
			face_cms_cache, face_areas_cache, apply_principal_limit_, psi_buf);
		face_vec const& faces = tess.GetCellFaces(i);
		const size_t nloop = faces.size();
		// Use pre-computed cell CM - avoid repeated lookups
		const Vector3D& cell_cm = all_cell_cms[i];
		// Get reference to source cell - avoid repeated indexing
		const ComputationalCell3D& source_cell = new_cells_[i];
		const Slope3D& cell_slope = rslopes_[i];
		
		for (size_t j = 0; j < nloop; ++j)
		{
			// Use cached face CM from calc_slope
			const Vector3D& face_cm = face_cms_cache[j];
			
			if (tess.GetFaceNeighbors(faces[j]).first == i)
			{
				cell_ref = &res[faces[j]].first;
				ReplaceComputationalCell(*cell_ref, source_cell);
				try
				{
					if(pressure_calc_)
						interp23D(*cell_ref, cell_slope, face_cm, cell_cm, eos_, true);
					else
					{
						interp23D(*cell_ref, cell_slope, face_cm, cell_cm, eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
					CheckCell(*cell_ref);
				}
				catch (UniversalError &eo)
				{
					eo.addEntry("dslope_x",  cell_slope.xderivative.density);
					eo.addEntry("dslope_y",  cell_slope.yderivative.density);
					eo.addEntry("dslope_z",  cell_slope.zderivative.density);
					eo.addEntry("Old density", source_cell.density);
					eo.addEntry("Old internal energy", source_cell.internal_energy);
					eo.addEntry("Face", static_cast<double>(faces[j]));
					eo.addEntry("Cell", static_cast<double>(i));
					eo.addEntry("Vx", source_cell.velocity.x);
					eo.addEntry("Vy", source_cell.velocity.y);
					eo.addEntry("Vz", source_cell.velocity.z);
					eo.addEntry("Cell id", static_cast<double>(source_cell.ID));
					eo.addEntry("Interpolated density",cell_ref->density);
					eo.addEntry("Interpolated pressure",cell_ref->pressure);
					eo.addEntry("Interpolated internal energy",cell_ref->internal_energy);
					eo.addEntry("Interpolated Vx",cell_ref->velocity.x);
					eo.addEntry("Interpolated Vy",cell_ref->velocity.y);
					eo.addEntry("Interpolated Vz",cell_ref->velocity.z);
					throw eo;
				}
				if (tess.GetFaceNeighbors(faces[j]).second > CellNumber)
					boundaryedges.push_back(faces[j]);
			}
			else
			{
				cell_ref = &res[faces[j]].second;
				ReplaceComputationalCell(*cell_ref, source_cell);
				try
				{
					if (pressure_calc_)
						interp23D(*cell_ref, cell_slope, face_cm, cell_cm, eos_, true);
					else
					{
						interp23D(*cell_ref, cell_slope, face_cm, cell_cm, eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
					CheckCell(*cell_ref);
				}
				catch (UniversalError &eo)
				{
					eo.addEntry("dslope_x1",  cell_slope.xderivative.density);
					eo.addEntry("dslope_y1",  cell_slope.yderivative.density);
					eo.addEntry("dslope_z1",  cell_slope.zderivative.density);
					eo.addEntry("Old density", source_cell.density);
					eo.addEntry("Old internal energy", source_cell.internal_energy);
					eo.addEntry("Face", static_cast<double>(faces[j]));
					eo.addEntry("Cell", static_cast<double>(i));
					eo.addEntry("Vx", source_cell.velocity.x);
					eo.addEntry("Vy", source_cell.velocity.y);
					eo.addEntry("Vz", source_cell.velocity.z);
					eo.addEntry("Cell id", static_cast<double>(source_cell.ID));
					eo.addEntry("Interpolated density",cell_ref->density);
					eo.addEntry("Interpolated pressure",cell_ref->pressure);
					eo.addEntry("Interpolated internal energy",cell_ref->internal_energy);
					eo.addEntry("Interpolated Vx",cell_ref->velocity.x);
					eo.addEntry("Interpolated Vy",cell_ref->velocity.y);
					eo.addEntry("Interpolated Vz",cell_ref->velocity.z);
					throw eo;
				}
				if (tess.GetFaceNeighbors(faces[j]).first > CellNumber)
					boundaryedges.push_back(faces[j]);
			}
		}
	}
#ifdef RICH_MPI
	// communicate ghost slopes
	exchange_ghost_slopes(tess, rslopes_);
#endif //RICH_MPI
	// Interpolate the boundary edges
	size_t Nboundary = boundaryedges.size();
	for (size_t i = 0; i < Nboundary; ++i)
	{
		size_t N0 = tess.GetFaceNeighbors(boundaryedges[i]).first;
		if (N0 > CellNumber)
		{
			cell_ref = &res[boundaryedges[i]].first;
			ReplaceComputationalCell(*cell_ref, new_cells_[N0]);
			try
			{
#ifdef RICH_MPI
				if (tess.BoundaryFace(boundaryedges[i]))
				{ 
					if (pressure_calc_)
						interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
					else
					{
						interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
				}
				else
				{
					if (pressure_calc_)
						interp23D(*cell_ref, rslopes_[N0], tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
					else
					{
						interp23D(*cell_ref, rslopes_[N0], tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
				}
#else
				if (pressure_calc_)
					interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
				else
				{
					interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
					if (energy_fix)
						cell_ref->tracers[energy_index] = cell_ref->internal_energy;
				}
#endif //RICH_MPI

				CheckCell(*cell_ref);
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("old density", new_cells_[N0].density);
				eo.addEntry("old internal energy", new_cells_[N0].internal_energy);
				eo.addEntry("Boundary Face", static_cast<double>(boundaryedges[i]));
				eo.addEntry("Cell", static_cast<double>(N0));
				eo.addEntry("Vx", new_cells_[N0].velocity.x);
				eo.addEntry("Vy", new_cells_[N0].velocity.y);
				eo.addEntry("Vz", new_cells_[N0].velocity.z);
				eo.addEntry("Cell id", static_cast<double>(new_cells_[N0].ID));
				eo.addEntry("Interpolated density",cell_ref->density);
				eo.addEntry("Interpolated pressure",cell_ref->pressure);
				eo.addEntry("Interpolated internal energy",cell_ref->internal_energy);
				eo.addEntry("Interpolated Vx",cell_ref->velocity.x);
				eo.addEntry("Interpolated Vy",cell_ref->velocity.y);
				eo.addEntry("Interpolated Vz",cell_ref->velocity.z);
				eo.addEntry("Face CMx", tess.FaceCM(boundaryedges[i]).x);
				eo.addEntry("Face CMy", tess.FaceCM(boundaryedges[i]).y);
				eo.addEntry("Face CMz", tess.FaceCM(boundaryedges[i]).z);
				eo.addEntry("Cell CMx", tess.GetCellCM(N0).x);
				eo.addEntry("Cell CMy", tess.GetCellCM(N0).y);
				eo.addEntry("Cell CMz", tess.GetCellCM(N0).z);
				size_t N1 = tess.GetFaceNeighbors(boundaryedges[i]).second;
				eo.addEntry("Other cell ID", static_cast<double>(new_cells_[N1].ID));
				eo.addEntry("Other Cell CMx", tess.GetCellCM(N1).x);
				eo.addEntry("Other Cell CMy", tess.GetCellCM(N1).y);
				eo.addEntry("Other Cell CMz", tess.GetCellCM(N1).z);
				eo.addEntry("Other Cell density", new_cells_[N1].density);
				eo.addEntry("Other Cell pressure", new_cells_[N1].pressure);
				eo.addEntry("Slopex", rslopes_[N0].xderivative.density);
				eo.addEntry("Slopey", rslopes_[N0].yderivative.density);
				eo.addEntry("Slopez", rslopes_[N0].zderivative.density);
#ifdef RICH_MPI
				int rank = 0;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				eo.addEntry("Rank", static_cast<double>(rank));
				for (size_t j = 0; j < tess.GetGhostIndeces().size(); ++j)
					for (size_t k = 0; k < tess.GetGhostIndeces()[j].size(); ++k)
						if (tess.GetGhostIndeces()[j][k] == N0)
						{
							eo.addEntry("Point recv from proc", static_cast<double>(tess.GetDuplicatedProcs()[j]));
							eo.addEntry("Point recv index", static_cast<double>(k));
							eo.addEntry("Point proc index", static_cast<double>(j));
						}
#endif
				throw eo;
			}
		}
		else
		{
			N0 = tess.GetFaceNeighbors(boundaryedges[i]).second;
			cell_ref = &res[boundaryedges[i]].second;
			ReplaceComputationalCell(*cell_ref, new_cells_[N0]);
			try
			{
#ifdef RICH_MPI
				if (tess.BoundaryFace(boundaryedges[i]))
				{
					if (pressure_calc_)
						interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
					else
					{
						interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
				}
				else
				{
					if (pressure_calc_)
						interp23D(*cell_ref, rslopes_[N0], tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
					else
					{
						interp23D(*cell_ref, rslopes_[N0], tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
						if (energy_fix)
							cell_ref->tracers[energy_index] = cell_ref->internal_energy;
					}
				}
#else
				if (pressure_calc_)
					interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, true);
				else
				{
					interp23D(*cell_ref, ghost_.GetGhostGradient(tess, cells, rslopes_, N0, time, boundaryedges[i]), tess.FaceCM(boundaryedges[i]), tess.GetCellCM(N0), eos_, false);
					if (energy_fix)
						cell_ref->tracers[energy_index] = cell_ref->internal_energy;
				}
#endif //RICH_MPI

				CheckCell(*cell_ref);
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("old density", new_cells_[N0].density);
				eo.addEntry("old internal energy", new_cells_[N0].internal_energy);
				eo.addEntry("Boundary Face", static_cast<double>(boundaryedges[i]));
				eo.addEntry("Cell", static_cast<double>(N0));
				eo.addEntry("Vx", new_cells_[N0].velocity.x);
				eo.addEntry("Vy", new_cells_[N0].velocity.y);
				eo.addEntry("Vz", new_cells_[N0].velocity.z);
				eo.addEntry("Cell id", static_cast<double>(new_cells_[N0].ID));
				eo.addEntry("Interpolated density",cell_ref->density);
				eo.addEntry("Interpolated pressure",cell_ref->pressure);
				eo.addEntry("Interpolated internal energy",cell_ref->internal_energy);
				eo.addEntry("Interpolated Vx",cell_ref->velocity.x);
				eo.addEntry("Interpolated Vy",cell_ref->velocity.y);
				eo.addEntry("Interpolated Vz",cell_ref->velocity.z);
				eo.addEntry("Face CMx", tess.FaceCM(boundaryedges[i]).x);
				eo.addEntry("Face CMy", tess.FaceCM(boundaryedges[i]).y);
				eo.addEntry("Face CMz", tess.FaceCM(boundaryedges[i]).z);
				eo.addEntry("Cell CMx", tess.GetCellCM(N0).x);
				eo.addEntry("Cell CMy", tess.GetCellCM(N0).y);
				eo.addEntry("Cell CMz", tess.GetCellCM(N0).z);
				size_t N1 = tess.GetFaceNeighbors(boundaryedges[i]).first;
				eo.addEntry("Other cell ID", static_cast<double>(new_cells_[N1].ID));
				eo.addEntry("Other Cell CMx", tess.GetCellCM(N1).x);
				eo.addEntry("Other Cell CMy", tess.GetCellCM(N1).y);
				eo.addEntry("Other Cell CMz", tess.GetCellCM(N1).z);
				eo.addEntry("Other Cell density", new_cells_[N1].density);
				eo.addEntry("Other Cell pressure", new_cells_[N1].pressure);
#ifdef RICH_MPI
				int rank = 0;
				MPI_Comm_rank(MPI_COMM_WORLD, &rank);
				eo.addEntry("Rank", static_cast<double>(rank));
				for (size_t j = 0; j < tess.GetGhostIndeces().size(); ++j)
					for (size_t k = 0; k < tess.GetGhostIndeces()[j].size(); ++k)
						if (tess.GetGhostIndeces()[j][k] == N0)
						{
							eo.addEntry("Point recv from proc", static_cast<double>(tess.GetDuplicatedProcs()[j]));
							eo.addEntry("Point recv index", static_cast<double>(k));
							eo.addEntry("Point proc index", static_cast<double>(j));
						}
#endif
				throw eo;
			}
		}
	}
	//In SR convert back to velocities
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


vector<Slope3D>& LinearGauss3D::GetSlopes(void)
{
	return rslopes_;
}

vector<Slope3D>& LinearGauss3D::GetSlopesUnlimited(void)const
{
	return naive_rslopes_;
}

