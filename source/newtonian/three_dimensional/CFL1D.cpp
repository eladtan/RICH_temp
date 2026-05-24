#include "CFL1D.hpp"
#include "../../misc/utils.hpp"
#include <limits>
#ifdef RICH_MPI
#include <mpi.h>
#endif

CFL1D::CFL1D(double cfl, double SourceCFL, SourceTerm3D const& source,
	std::vector<std::string> no_calc, bool debug) :
	cfl_(cfl), sourcecfl_(SourceCFL), source_(source), no_calc_(no_calc), debug_(debug), first_try_(true), dt_first_(-1), last_time_(-10000)
{
	assert(cfl_ < 1 && "cfl number must be smaller than 1");
}

double CFL1D::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
	const EquationOfState& eos, const vector<Vector3D>& face_velocities, const double time)
{
	double res = 0.001 * std::numeric_limits<double>::max();
	size_t const N = tess.GetPointNo();
	size_t loc = 0;
	size_t const N_no_calc = no_calc_.size();
	std::vector<size_t> no_calc_indeces(N_no_calc);
	for (size_t i = 0; i < N_no_calc; ++i)
		no_calc_indeces[i] = binary_index_find(ComputationalCell3D::stickerNames, no_calc_[i]);
	if (N > 0)
	{
		for (size_t i = 0; i < N; ++i)
		{
			const ComputationalCell3D& cell = cells[i];
			if (std::any_of(no_calc_indeces.cbegin(), no_calc_indeces.cend(),
				[&cell](const size_t& idx) { return cell.stickers[idx]; }))
				continue;

			double c = eos.de2c(cell.density, cell.internal_energy, cell.tracers, ComputationalCell3D::tracerNames);

			double max_signal = 0;
			double x_min_neighbor = std::numeric_limits<double>::max();
			double x_max_neighbor = -std::numeric_limits<double>::max();
			face_vec const& faces = tess.GetCellFaces(i);
			size_t const Nloop = faces.size();
			for (size_t j = 0; j < Nloop; ++j)
			{
				Vector3D n = tess.Normal(faces[j]);
				double nx = std::abs(n.x) / fastabs(n);
				if (nx < 0.5)
					continue;
				max_signal = std::max(max_signal, c + std::abs(cell.velocity.x - face_velocities[faces[j]].x));
				auto const& nb = tess.GetFaceNeighbors(faces[j]);
				size_t other = (nb.first == i) ? nb.second : nb.first;
				double xn = tess.GetMeshPoint(other).x;
				x_min_neighbor = std::min(x_min_neighbor, xn);
				x_max_neighbor = std::max(x_max_neighbor, xn);
			}
			double dx = (x_max_neighbor - x_min_neighbor) * 0.5;
			if (dx <= 0 || max_signal <= 0)
				continue;
			double res_temp = dx / max_signal;
			if (res_temp < res)
			{
				res = res_temp;
				loc = i;
			}
		}
	}
	res *= cfl_;
	double old_res = res;
	res = 1.0 / std::max(source_.SuggestInverseTimeStep() / sourcecfl_, 1.0 / res);
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &res, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
#endif
	double hydro_res = res;
	if ((first_try_ && dt_first_ > 0) || (last_time_ == time && dt_first_ > 0))
	{
		res = std::min(res, dt_first_);
		first_try_ = false;
		if (close2zero(last_time_ - time))
			dt_first_ = -1;
	}
	int rank = 0;
	int limiting_rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	struct { double val; int rank; } local_min{old_res, rank}, global_min;
	MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE_INT, MPI_MINLOC, MPI_COMM_WORLD);
	limiting_rank = global_min.rank;
#endif
	if (debug_ && rank == limiting_rank && hydro_res < 0.9999 * dt_suggest_)
	{
		std::cout << "CFL1D: dt=" << res << " cell ID " << cells[loc].ID
			<< " x=" << tess.GetMeshPoint(loc).x
			<< " c=" << eos.dp2c(cells[loc].density, cells[loc].pressure, cells[loc].tracers, ComputationalCell3D::tracerNames)
			<< " vx=" << cells[loc].velocity.x
			<< " dt_org=" << old_res << std::endl;
	}

	last_time_ = time;
	dt_ = res;
	dt_suggest_ = hydro_res;
	return res;
}

double CFL1D::GetTimeStep(void) const
{
	return first_try_ ? dt_first_ : dt_;
}

void CFL1D::SetTimeStep(double dt)
{
	dt_first_ = dt;
	first_try_ = true;
}

double CFL1D::SuggestTimeStep(void) const
{
	return dt_suggest_;
}
