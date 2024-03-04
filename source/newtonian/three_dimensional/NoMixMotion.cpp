#include "NoMixMotion.hpp"
#include "../../misc/utils.hpp"
#include <boost/random.hpp>
#include <boost/random/uniform_01.hpp>
#ifdef RICH_MPI
#include "mpi/mpi_commands.hpp"
#endif
#define BOOST_UBLAS_NDEBUG 1
#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/vector.hpp>

NoMixMotion::NoMixMotion(const PointMotion3D& pm, SpatialReconstruction3D const& interpolation, const RiemannSolver3D & rs, const EquationOfState& eos, const vector<std::string>& no_fix = vector<std::string>()) : pm_(pm), interpolation_(interpolation), no_fix_indeces(), rs_(rs), eos_(eos)
{
	size_t const Nstick = ComputationalCell3D::stickerNames.size();
	for (size_t i = 0; i < Nstick; ++i)
	{
		vector<std::string>::const_iterator it = std::find(no_fix.begin(), no_fix.end(), ComputationalCell3D::stickerNames.at(i));
		if(it != no_fix.end())
			no_fix_indeces.push_back(i);
	}
}

namespace
{
	bool DifferentCellType(ComputationalCell3D const& left, ComputationalCell3D const& right, size_t const Ntracers, double& mass_ratio)
	{
		double const allowed_mix = 0.11;
		mass_ratio = 0;
		for(size_t i = 0; i < Ntracers; ++i)
			mass_ratio += left.tracers[i] * right.tracers[i];
		return mass_ratio < allowed_mix;
	}

	void calc_dw(Vector3D &velocity, size_t i, const Tessellation3D & tess, const vector<ComputationalCell3D>& cells, const EquationOfState &eos, double const /*mix_area_ratio*/, double const min_d)
	{
		const Vector3D r = tess.GetMeshPoint(i);
		const Vector3D s = tess.GetCellCM(i);
		const double d = fastabs(s-r);
		const double R = tess.GetWidth(i);
		if(d< R * 0.1 || min_d > 0.25)
			return; 
		double const v_size = std::max(fastabs(velocity), fastabs(cells[i].velocity));
		double const cs = eos.de2c(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
		double const c = std::max(v_size, cs);
		Vector3D dv = 0.25 * c * (s-r) / std::max(R, 2*d);
		velocity += dv;
	}

	void calc_dw_no_fix(Vector3D &velocity, size_t i, const Tessellation3D & tess, const vector<ComputationalCell3D>& cells, const EquationOfState &eos, Vector3D const lagrangian_direction)
	{
		const Vector3D r = tess.GetMeshPoint(i);
		const Vector3D s = tess.GetCellCM(i);
		Vector3D diff = s - r;
		const double d = fastabs(s-r);
		const double R = tess.GetWidth(i);
		double const v_size = std::max(fastabs(velocity), fastabs(cells[i].velocity));
		double const cs = eos.de2c(cells[i].density, cells[i].internal_energy, cells[i].tracers, ComputationalCell3D::tracerNames);
		double const c = std::max(v_size, cs);
		Vector3D dv = 0.25 * c * (s-r) / std::max(R, 2*d);
		velocity += dv;
	}

	void MakeSmoothSurface(std::vector<Vector3D> const & lagrangian_direction, Tessellation3D const& tess, std::vector<Vector3D> &v, std::vector<char> & lagrangian_cell, std::vector<ComputationalCell3D> const& cells, std::vector<double> const & widths)
	{
		size_t const n = tess.GetPointNo();
		std::vector<size_t> neighbors, temp_neighbors;
		boost::numeric::ublas::matrix<double> m(2,2), mtrans(2,2), mprod(6, 6), inverse(6,6), mprod3(3,3), m3(2,2), mtrans3(2, 2), inverse3(3, 3);
		boost::numeric::ublas::identity_matrix<double> Imatrix(6), Imatrix3(3);
		boost::numeric::ublas::vector<double> Zvec(2), vec_result(6), vec_result2(6);
		Vector3D parallel_1;
		for (size_t i = 0; i < n; ++i)
		{
			bool const local_cell_lagrangian = lagrangian_cell[i] == '2';
			if(local_cell_lagrangian)
			{
				boost::numeric::ublas::permutation_matrix<size_t> pmatrix(6), pmatrix3(3);
				Vector3D const point(tess.GetMeshPoint(i));
				tess.GetNeighborNeighbors(temp_neighbors, i);
				tess.GetNeighbors(i, neighbors);
				temp_neighbors.insert(temp_neighbors.end(), neighbors.begin(), neighbors.end());
				neighbors.clear();
				size_t Nneighbors = temp_neighbors.size();
				for(size_t j = 0; j < Nneighbors; ++j)
				{
					size_t const neighbor = temp_neighbors[j];
					if (neighbor < n || not tess.IsPointOutsideBox(neighbor))
					{
						Vector3D temp = tess.GetMeshPoint(temp_neighbors[j]);
						if(lagrangian_cell[neighbor] == '2')
						{
							double sum = 0;
							for(size_t k = 0; k < ComputationalCell3D::tracerNames.size(); ++k)
								sum += cells[i].tracers[k] * cells[neighbor].tracers[k];
							if(sum > 0.75)
								neighbors.push_back(neighbor);
						}
					}
				}
				Nneighbors = neighbors.size();
				if(Nneighbors < 3)
					continue;
				parallel_1 = tess.GetMeshPoint(neighbors[0]) - point;
				parallel_1 -= lagrangian_direction[i] * ScalarProd(lagrangian_direction[i], parallel_1);
				parallel_1 = normalize(parallel_1);
				Vector3D const parallel_2 = CrossProduct(parallel_1, lagrangian_direction[i]);
				Nneighbors > 6 ? m.resize(Nneighbors, 6) : m3.resize(Nneighbors, 3);
				Nneighbors > 6 ? mtrans.resize(6, Nneighbors) : mtrans3.resize(3, Nneighbors);
				Zvec.resize(Nneighbors);
				for(size_t j =0; j < Nneighbors; ++j)
				{
					Vector3D const diff = tess.GetMeshPoint(neighbors[j]) - point;
					double const x = ScalarProd(diff, parallel_1);
					double const y = ScalarProd(diff, parallel_2);
					Zvec(j) = ScalarProd(diff, lagrangian_direction[i]);
					if(Nneighbors > 6)
					{
						m(j, 0) = 1;
						m(j, 1) = x;
						m(j, 2) = y;
						m(j, 3) = x*x;
						m(j, 4) = x*y;
						m(j, 5) = y*y;
					}
					else
					{
						m3(j, 0) = 1;
						m3(j, 1) = x;
						m3(j, 2) = y;
					}
				}
				if(Nneighbors > 6)
				{
					vec_result.resize(6);
					vec_result2.resize(6);
					mtrans = boost::numeric::ublas::trans(m);
					boost::numeric::ublas::prod(mtrans, m, mprod);
					try
					{
						if(boost::numeric::ublas::lu_factorize(mprod, pmatrix) != 0)
						{
							for(size_t k = 0; k < 6; ++k)
							{
								for(size_t kk = 0; kk < 6; ++kk)
									std::cout<<mprod(k, kk)<<" ";
								std::cout<<std::endl;
							}
							std::cout <<"Nneighbors "<<Nneighbors<<std::endl;
							for(size_t kk = 0; kk < 6; ++kk)
							{
								std::cout<<std::endl;
								for(size_t j = 0; j < Nneighbors; ++j)
									std::cout<<m(j, kk)<<" ";
							}
							std::cout<<std::endl;
							throw UniversalError("Bad LU decomposition in NoMixMotion");
						}
					}
					catch(boost::numeric::ublas::external_logic const &eo)
					{
						for(size_t k = 0; k < 6; ++k)
						{
							for(size_t kk = 0; kk < 6; ++kk)
								std::cout<<mprod(k, kk)<<" ";
							std::cout<<std::endl;
						}
						std::cout<<"ID of bad cell "<<cells[i].ID<<std::endl;
						std::cout<<"Number of points "<<n<<std::endl;
						std::cout<<"Nneighbors "<<Nneighbors<<std::endl;
						for(size_t kk = 0; kk < 6; ++kk)
							{
								std::cout<<std::endl;
								for(size_t j = 0; j < Nneighbors; ++j)
									std::cout<<m(j, kk)<<" ";
							}
							std::cout<<std::endl;
							throw UniversalError("Bad LU decomposition in NoMixMotion");
					}
					inverse.assign(Imatrix);
					boost::numeric::ublas::lu_substitute(mprod, pmatrix, inverse);
					boost::numeric::ublas::prod(mtrans, Zvec, vec_result);
					boost::numeric::ublas::prod(inverse, vec_result, vec_result2);
				}
				else
				{
					vec_result.resize(3);
					vec_result2.resize(3);
					boost::numeric::ublas::prod(boost::numeric::ublas::trans(m3), m3, mprod3);
					if(boost::numeric::ublas::lu_factorize(mprod3, pmatrix3) != 0)
						throw UniversalError("Bad LU decomposition in NoMixMotion");
					inverse3.assign(Imatrix3);
					boost::numeric::ublas::lu_substitute(mprod3, pmatrix3, inverse3);
				}
				double fix_magnitude = std::max(-1.0, std::min(1.0, 50*vec_result2[0]/widths[i]));
				v[i] += (fastabs(v[i]) * 0.05 * fix_magnitude) * lagrangian_direction[i];
			}
		}
	}

	void CorrectLagrangianNeighbors(std::vector<Vector3D> & v, Tessellation3D const & tess, std::vector<size_t> const& ID, std::vector<char> & lagrangian_cell,
	boost::random::mt19937_64 & gen, boost::random::uniform_01<double> & dist, std::vector<ComputationalCell3D> const& cells, std::vector<Vector3D> const& lagrangian_direction)
	{
		// check that we don't overshoot lagrangian poinrs
		size_t n = tess.GetPointNo();
		std::vector<size_t> neighbors;
		std::vector<size_t> lag_neigh;
		std::vector<face_vec> const& all_faces = tess.GetAllCellFaces();
		size_t const Ntracers = ComputationalCell3D::tracerNames.size();
		std::vector<double> widths(n);
		for(size_t i= 0; i < n; ++i)
			widths[i] = tess.GetWidth(i);
#ifdef RICH_MPI
		MPI_exchange_data2(tess, widths, true);
#endif
		for(size_t i = 0; i < n; ++i)
		{
			bool const local_cell_lagrangian = lagrangian_cell[i] == '2';
			Vector3D const point(tess.GetMeshPoint(i));
			tess.GetNeighbors(i, neighbors);
			size_t const Nneighbors = neighbors.size();
			if(not local_cell_lagrangian && not (lagrangian_cell[i] == '3'))
			{
				lag_neigh.clear();
				double Area = 0;
				double mix_Area = 0;
				double const R = tess.GetWidth(i);
				double min_d = std::numeric_limits<double>::max();
				for(size_t j = 0; j < Nneighbors; ++j)
				{
					size_t const face_index = all_faces[i][j];
					double const face_area = tess.GetArea(face_index);
					Area += face_area;
					size_t const neighbor = neighbors[j];
					if(neighbor < n || not tess.IsPointOutsideBox(neighbor))
					{ // Are the two neighboring cells from different types - one lagrangian and the second is not?
						double mass_ratio;
						if(lagrangian_cell[neighbor] == '2' && not DifferentCellType(cells[neighbor], cells[i], Ntracers, mass_ratio))
						{
							lag_neigh.push_back(neighbor);
							mix_Area += face_area;
						}
					}
				}
				if(mix_Area > 1e-5 * Area)
				{
					for(size_t j = 0; j < lag_neigh.size(); ++j)
					{
						size_t const lag_index = lag_neigh[j];
						Vector3D const normal = point - tess.GetMeshPoint(lag_index);
						double const d = std::abs(ScalarProd(lagrangian_direction[lag_index], normal));
						min_d = std::min(min_d, d);
						double const maxR = std::max(widths[lag_index], std::max(fastsqrt(tess.GetVolume(i) / fastabs(normal)), R));
						double const v_l = ScalarProd(lagrangian_direction[lag_index], v[lag_index]);
						double const point_v_l = ScalarProd(v[i], lagrangian_direction[lag_index]);
						Vector3D dv;
						double const maxR_factor = 1.6;
						if(d < maxR_factor * maxR)
						{
							if(v_l < 0)
							{
								if(point_v_l > v_l * 1.075)
									dv = ((1 - d/(maxR_factor * maxR)) * (0.04 + 0.1 * dist(gen)) * v_l + v_l - point_v_l) * lagrangian_direction[lag_index];
							}
							else
								if(point_v_l > v_l * 0.925)
									dv = (-(1 - d/(maxR_factor * maxR)) * (0.04 + 0.1 * dist(gen)) * v_l + v_l - point_v_l) * lagrangian_direction[lag_index];
						}
						if(mix_Area < 1e-2 * Area)
							dv = ((v_l < 0 ? 1: -1) * (0.04 + 0.1 * dist(gen)) * v_l + v_l - point_v_l) * lagrangian_direction[lag_index];
						v[i] += dv;
					}
					Vector3D cm_diff = tess.GetCellCM(i) - point;
					double cm_diff_size = fastabs(cm_diff);
					if(cm_diff_size > 0.2 * R)
					{
						cm_diff_size = fastabs(cm_diff); //?????
						double const pre_factor = std::max(0.1, std::min(0.3, 0.3 * min_d / R));
						v[i] += cm_diff * (pre_factor * fastabs(v[i]) / std::max(R, cm_diff_size));
					}
				}
			}
		}
		return;
	}

	void CorrectPointsOverShoot(vector<Vector3D> &v, double dt, Tessellation3D const& tess,std::vector<size_t> const & IDs,std::vector<char>& lagrangian_cell,
	boost::random::mt19937_64 & gen, boost::random::uniform_01<double> & dist,std::vector<ComputationalCell3D> const & cells, std::vector<Vector3D> const & lagrangian_direction)
	{
		// check that we don't go outside grid and that we don't overshoot lagrangian points
		size_t const n = tess.GetPointNo();
		const double inv_dt = 1.0 / dt;
		std::vector<size_t> neighbors;
		double const close_factor = 0.1;
		std::vector<face_vec> const & all_faces = tess.GetAllCellFaces();
		size_t const Ntracers = ComputationalCell3D::tracerNames.size();
		CorrectLagrangianNeighbors(v, tess, IDs, lagrangian_cell, gen, dist, cells, lagrangian_direction);
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess, v, true, &vdummy);
#endif
		for(size_t i = 0; i < n; ++i)
		{
			Vector3D const point(tess.GetMeshPoint(i));
			double const R = tess.GetWidth(i);
			tess.GetNeighbors(i, neighbors);
			size_t const Nneighbors = neighbors.size();
			for(size_t j = 0; j < Nneighbors; ++j)
			{
				size_t const neighbor = neighbors[j];
				if(neighbor >= n)
				{
					if(tess.IsPointOutsideBox(neighbor))
					{
						Vector3D const diff = tess.GetMeshPoint(neighbor) - point;
						double const R_diff = fastabs(diff);
						double const v_diff = ScalarProd(v[i] * dt, diff) / R_diff;
						if(v_diff > close_factor * R_diff || R_diff < 0.5 * R)
							v[i] -= std::abs((std::max(1.1, std::min(1.5, (v_diff / (0.7 * close_factor * R_diff) - 1))) * ScalarProd(v[i], diff) / (R_diff * R_diff))) * diff;
					}
				}
			}
		}
		return;
	}
}


void NoMixMotion::operator()(const Tessellation3D & tess, const vector<ComputationalCell3D>& cells, double time, vector<Vector3D> &velocities) const
{
	pm_(tess, cells, time, velocities);
	const size_t n = tess.GetPointNo();
	ustar_.clear();
	ustar_.resize(n);
	lagrangian_cell_.clear();
	lagrangian_cell_.resize(n);
	lagrangian_direction_.clear();
	lagrangian_direction_.resize(n);
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>> face_values;
	interpolation_(tess, cells, time, face_values);
	interpolation_.slopes_have_been_built = true;
	std::vector<face_vec> const & all_faces = tess.GetAllCellFaces();
	std::vector<std::pair<size_t, size_t>> const & all_face_neigh = tess.GetAllFaceNeighbors();
	size_t const Ntracers = ComputationalCell3D::tracerNames.size();
	size_t const Nstickers = ComputationalCell3D::stickerNames.size();
	double mass_ratio = 0;
	std::vector<double> widths(n), no_fix(n, 0);
	size_t const N_no_fix = no_fix_indeces.size();
	for(size_t i = 0; i < n; ++i)
	{
		widths[i] = tess.GetWidth(i);
		for(size_t j =0 ; j < N_no_fix; ++j)
		{
			if(cells[i].stickers[no_fix_indeces[j]])
			{
				no_fix[i] = 1;
				break;
			}
		}
	}
#ifdef RICH_MPI
	MPI_exchange_data2(tess, widths, true);
	MPI_exchange_data2(tess, no_fix, true);
#endif
	std::vector<Vector3D> v_faces;
	std::vector<double> areas;
	for(size_t i = 0; i < n; ++i)
	{
		v_faces.clear();
		areas.clear();
		bool local_fix = no_fix[i] > 0.5;
		size_t const Nfaces = all_faces[i].size();
		Vector3D cell_velocity;
		double Area = 0, mix_area = 0, pure_mix_area = 0, max_area = 0, max_face = 0;
		size_t max_area_index = 0;
		Vector3D Area_vec;
		bool good_lagrangian = true;
		double min_d = std::numeric_limits<double>::max();
		double const R = tess.GetWidth(i);
		for(size_t j = 0; j < Nfaces; ++j)
		{
			size_t const face_index = all_faces[i][j];
			double const face_area = tess.GetArea(face_index);
			max_face = std::max(max_face, face_area);
			size_t const other_neigh = all_face_neigh[face_index].first == i ? all_face_neigh[face_index].second : all_face_neigh[face_index].first;
			if(other_neigh >= n && tess.IsPointOutsideBox(other_neigh))
			{
				Area += 0.5 * face_area;
				continue;
			}
			Area += face_area;
			double const d = fastabs(tess.GetMeshPoint(i) - tess.GetMeshPoint(other_neigh));
			min_d = std::min(min_d, d);
			if(d < 0.2 * R && no_fix[i] < 0.5)
				good_lagrangian = false;
			if(DifferentCellType(face_values[face_index].first, face_values[face_index].second, Ntracers, mass_ratio));
			{
				Vector3D normal = normalize(tess.Normal(face_index));
				double const vl = ScalarProd(face_values[face_index].first.velocity, normal);
				double const vr = ScalarProd(face_values[face_index].second.velocity, normal);
				Vector3D ustar;
				rs_(face_values[face_index].first, face_values[face_index].second, 0., eos_, normal, ustar);
				mass_ratio = 1 - mass_ratio; //?????????
				mass_ratio = 1;
				pure_mix_area += face_area;
				mix_area += face_area * mass_ratio;
				v_faces.push_back((face_area * mass_ratio) * ustar);
				cell_velocity += (face_area * mass_ratio) * ustar;
				double const first_index = (all_face_neigh[face_index].first == i ? 1 : 0);
				double const second_index = 1 - first_index;
				Vector3D dv;
				areas.push_back(face_area);
				cell_velocity += dv;
				if(face_area > max_area)
				{
					max_area = face_area;
					max_area_index = v_faces.size() - 1;
				}
				if(no_fix[other_neigh] > 0.5)
					local_fix = true;
				dv.Set(0,0,0);
				if(d < 0.2 * (widths[i] + widths[other_neigh]))
					dv = -(0.02 * face_area * mass_ratio * (all_face_neigh[face_index].first == i ? 1 : -1)) * ustar;
				else
					if(d > 2.5 * (widths[i] + widths[other_neigh]))
						dv = (0.02 * face_area * mass_ratio * (all_face_neigh[face_index].first == i ? 1 : -1)) * ustar;
				v_faces.back() += dv;
				cell_velocity += dv;
				Area_vec += (face_area * (all_face_neigh[face_index].first == i ? 1 : -1)) * normal;
			}
		}
		if(mix_area > Area * 1e-4 && mix_area < 0.6 * Area)
		{
			if(good_lagrangian || local_fix)
			{
				double new_area = 0;
				double new_velocity = 0;
				if(max_area > pure_mix_area * 0.98 && abs(cell_velocity) > std::numeric_limits<double>::min() * 100)
				{
					double const Umag = abs(v_faces[max_area_index]);
					if(Umag > std::numeric_limits<double>::min() * 100)
						lagrangian_direction_[i] = normalize(v_faces[max_area_index]);
					else
						lagrangian_direction_[i] = normalize(Area_vec);
					velocities[i] = v_faces[max_area_index] * (1.0 / max_area);
				}
				else
				{
					double const Umag = abs(cell_velocity);
					if(Umag > std::numeric_limits<double>::min() * 1000)
						lagrangian_direction_[i] = ((ScalarProd(cell_velocity, Area_vec) > 0 ? 1 : -1 )/ Umag ) * cell_velocity;
					else
						lagrangian_direction_[i] = normalize(Area_vec);
					Vector3D const v_hat = Umag > std::numeric_limits<double>::min() * 1000 ? cell_velocity * (1.0 / Umag) : lagrangian_direction_[i];
					size_t const Nlag_faces = v_faces.size();
					double max_v = 0;
					for(size_t j = 0; j < Nlag_faces; ++j)
					{
						max_v = std::max(max_v, fastabs(v_faces[j]) / areas[j]);
						new_area += areas[j] * (ScalarProd(v_faces[j], v_hat)) / (std::numeric_limits<double>::min() * 1000 + abs(v_faces[j]));
						new_velocity += ScalarProd(v_faces[j], v_hat);
					}
					velocities[i] = v_hat * (new_velocity / (std::numeric_limits<double>::min() * 1000 + new_area));
					if(fastabs(velocities[i]) > max_v)
					{
						double const Umag2 = abs(v_faces[max_area_index]);
						if(Umag2 > std::numeric_limits<double>::min() * 1000)
							lagrangian_direction_[i] = normalize(v_faces[max_area_index]);
						else
							lagrangian_direction_[i] = normalize(Area_vec);
						velocities[i] = v_faces[max_area_index] * (1.0 / max_area);
					}
				}

				if(fastabs(velocities[i]) > 6e8)
				{
					std::cout<<"fast v Cell "<<cells[i].ID<<std::endl;
					std::cout<<"Did maxarea " << (max_area > pure_mix_area * 0.7)<<" max_area "<<max_area<<" pure_mix_area "<<pure_mix_area<<" new_Area "<<new_area<<" new velocity "<<new_velocity<<std::endl;
					size_t const Nlag_faces = v_faces.size();
					for(size_t j = 0; j < Nlag_faces; ++j)
					{
						std::cout<<"v_faces "<<v_faces[j].x * (1.0 / areas[j])<<","<<v_faces[j].y * (1.0 / areas[j])<<","<<v_faces[j].z * (1.0 / areas[j])<<" area "<<areas[j]<<" norm "<<normalize(v_faces[j]).x<<","<<normalize(v_faces[j]).y<<","<<normalize(v_faces[j]).z<<" dot "<<ScalarProd(v_faces[j], lagrangian_direction_[i]) / (std::numeric_limits<double>::min() * 1000 + abs(v_faces[j]))<<std::endl;
					}
					std::cout<<"lagrangian_direction_ "<<lagrangian_direction_<<" velocities "<<velocities[i]<<std::endl;
				}
				if(fastabs(velocities[i]) > 1e2 &&(not local_fix))
					calc_dw(velocities[i], i, tess, cells, eos_, mix_area / Area, min_d / fastsqrt(max_face));
				else
					if(fastabs(velocities[i]) > 1e2 && local_fix)
						calc_dw_no_fix(velocities[i], i, tess, cells, eos_, normalize(velocities[i]));
				ustar_[i] = velocities[i];
				lagrangian_cell_[i] = '2';
			}
			else
			{
				lagrangian_cell_[i] = '3';
				if(fastabs(velocities[i]) > 1e2)
					calc_dw(velocities[i], i, tess, cells, eos_, mix_area / Area, true);
			}
		}
		else
		lagrangian_cell_[i] = '0';
	}
	int rank = 0;
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess, lagrangian_cell_, true);
	MPI_exchange_data(tess, velocities, true, &vdummy);
	MPI_exchange_data(tess, lagrangian_direction_, true, &vdummy);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	// Fix Lagrangian points that strayed away from the lagrangian surface
	MakeSmoothSurface(lagrangian_direction_, tess, velocities, lagrangian_cell_, cells, widths);
	std::vector<size_t> IDs(n);
	boost::random::mt19937_64 gen(static_cast<size_t>(rank) + static_cast<size_t>(std::pow(10.0, 13 + std::ceil(std::abs(std::log10(time) + 1e-20))) * std::abs(time) + 0.5));
	boost::random::uniform_01<double> dist;
	CorrectLagrangianNeighbors(velocities, tess, IDs, lagrangian_cell_, gen, dist, cells, lagrangian_direction_);
	velocities.resize(n);
}

void NoMixMotion::ApplyFix(Tessellation3D const & tess, vector<ComputationalCell3D> const & cells, double time, double dt,
	vector<Vector3D>& velocities) const
{
	size_t N = tess.GetPointNo();
	pm_.ApplyFix(tess, cells, time, dt, velocities);
	for(size_t i = 0; i < N; ++i)
		if(lagrangian_cell_[i] == '2')
			velocities[i] = ustar_[i];
	int rank = 0;
#ifdef RICH_MPI
	Vector3D vdummy;
	MPI_exchange_data(tess, velocities, true, &vdummy);
	MPI_exchange_data(tess, lagrangian_cell_, true);
#endif
	std::vector<size_t> IDs(N);
	for(size_t i = 0; i < N; ++i)
		IDs[i] = cells[i].ID;
	boost::random::mt19937_64 gen(static_cast<size_t>(rank) + static_cast<size_t>(std::pow(10.0, 13 + std::ceil(std::abs(std::log10(dt)))) * dt));
	boost::random::uniform_01<double> dist;
	CorrectPointsOverShoot(velocities, dt, tess, IDs, lagrangian_cell_, gen, dist, cells, lagrangian_direction_);
	velocities.resize(N);
}
