#include "RadiationStep.hpp"
#include "misc/memory_debug.hpp"
#include <cstring>
#include <cstdint>
#include <limits>
#include <sstream>
#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace {

#ifdef RICH_MPI
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int const has_local = matrix_builder.getLastStepFailureReason().empty() ? 0 : 1;
    int const has_cell_id =
        (has_local && matrix_builder.getLastStepFailureCellId() !=
            std::numeric_limits<size_t>::max()) ? 1 : 0;
    int const score = has_cell_id ? 2 : has_local;
    struct {
        int score;
        int rank;
    } local_pick{score, rank};
    struct {
        int score;
        int rank;
    } global_pick{0, std::numeric_limits<int>::max()};
    MPI_Allreduce(&local_pick, &global_pick, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
    int const source_rank = global_pick.score > 0
        ? global_pick.rank
        : std::numeric_limits<int>::max();
    if (source_rank == std::numeric_limits<int>::max()) {
        reason.clear();
        cell_id = std::numeric_limits<size_t>::max();
        return;
    }

    char reason_buf[2048] = {};
    std::uint64_t local_cell_id = static_cast<std::uint64_t>(matrix_builder.getLastStepFailureCellId());
    if (rank == source_rank) {
        std::string const& local_reason = matrix_builder.getLastStepFailureReason();
        std::strncpy(reason_buf, local_reason.c_str(), sizeof(reason_buf) - 1);
    }
    MPI_Bcast(reason_buf, static_cast<int>(sizeof(reason_buf)), MPI_CHAR, source_rank, MPI_COMM_WORLD);
    MPI_Bcast(&local_cell_id, 1, MPI_UINT64_T, source_rank, MPI_COMM_WORLD);

    reason = reason_buf;
    cell_id = static_cast<size_t>(local_cell_id);
}
#else
void broadcast_step_failure(RadiationDriver const& matrix_builder,
                            std::string& reason,
                            size_t& cell_id)
{
    reason = matrix_builder.getLastStepFailureReason();
    cell_id = matrix_builder.getLastStepFailureCellId();
}
#endif

} // namespace
RadiationStep::RadiationStep(Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    ProgressTracker &pt,
                    #ifdef RICH_MPI
                        std::shared_ptr<CostCalculator3D> cost,
                    #endif // RICH_MPI
                    const RadiationDriver &matrix_builder, bool /*no_hydro*/) :
                    tess(tess), cells(cells), extensives(extensives), pt(pt), matrix_builder(matrix_builder)
                        , suggested_dt(std::numeric_limits<double>::max())
                    #ifdef RICH_MPI
                        , cost(cost)
                    #endif // RICH_MPI
{}

#ifdef RICH_MPI
    bool RadiationStep::allowRebalance(void)
    {
        return this->cost != nullptr;
    }

    std::string RadiationStep::getRequiredLB(void) const
    {
        if (!this->cost)
            return "";
        return "hydro";
    }

    std::vector<double> RadiationStep::getLoadBalanceWeights(void)
    {
        if (this->cost)
            return this->cost->CalculateCost(this->tess, this->cells);
        return std::vector<double>(this->tess.GetPointNo(), 1.0);
    }
#endif // RICH_MPI

void RadiationStep::step(double dt)
{
	int total_iters = 0;
	double const CG_eps = 1e-11;
	size_t const N = this->tess.GetPointNo();

#ifdef DEBUG
	if(N == 0) std::cout<<"Zero cells in RadiationTimeStep"<<std::endl;
#endif

	int rank = 0;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

	double total_elapsed_time = 0;
	double dt_try = dt;
	size_t reduce_counter = 0;
	int max_iter_done = 0;


	this->matrix_builder.prestep(this->tess, this->cells);
	MEMORY_DEBUG_PRINT("radiation: after prestep");
	while(total_elapsed_time < dt * 0.9999999)
	{
		dt_try = std::min(dt_try, dt - total_elapsed_time);

		bool step_success = this->matrix_builder.step(CG_eps, total_iters, this->tess, this->cells, this->extensives, dt_try, this->pt.getTime());
		MEMORY_DEBUG_PRINT("radiation: after solver step");

		max_iter_done = std::max(max_iter_done, total_iters);
		
		if(not step_success)
		{
			reduce_counter++;
			dt_try *= 0.5;
			std::string reason;
				size_t cell_id = std::numeric_limits<size_t>::max();
			broadcast_step_failure(this->matrix_builder, reason, cell_id);
			if(rank == 0) {
				std::ostringstream msg;
				msg << "Reducing dt, new dt " << dt_try;
				if (!reason.empty()) {
					msg << " (" << reason;
					if (cell_id != std::numeric_limits<size_t>::max())
						msg << ", example cell ID " << cell_id;
					msg << ")";
				}
				std::cout << msg.str() << std::endl;
			}
			
			if(dt_try < 0.001 * dt) {
				if (rank == 0) {
					std::ostringstream msg;
					msg << "Radiation step failed: dt reduced below 0.1% of target ("
					    << dt_try << " < " << (0.001 * dt) << ")";
					if (!reason.empty()) {
						msg << ", last failure: " << reason;
						if (cell_id != std::numeric_limits<size_t>::max())
							msg << ", example cell ID " << cell_id;
					}
					std::cout << msg.str() << std::endl;
				}
				UniversalError eo("too small dt in RadiationTimeStep");
				if (!reason.empty()) {
					eo.addEntry("last radiation failure", reason);
					if (cell_id != std::numeric_limits<size_t>::max())
						eo.addEntry("example cell ID", cell_id);
				}
				throw eo;
			}
		}
		else {
			total_elapsed_time += dt_try;
		}
	}

	this->suggested_dt = this->matrix_builder.calculate_dt(dt, this->tess, this->cells);

	this->matrix_builder.poststep();
	MEMORY_DEBUG_PRINT("radiation: after poststep");

#ifdef RICH_MPI
	MPI_exchange_data(this->tess, this->cells, true);
#endif
	
    
	// double grow_factor = 1.25;
	// if(max_iter_done > 200)
	// 	grow_factor = 1.02;
	// else
	// 	if(max_iter_done > 125)
	// 		grow_factor = 1.05;

	// new_dt = std::min(new_dt, dt*grow_factor) * std::pow(0.5, std::max(static_cast<double>(reduce_counter), 0.0));
	// if(max_iter_done > 300)
	// 	new_dt = dt * 0.9;
	// return this->radiation_dt_;
}

double RadiationStep::suggestTimeStep(void) const
{
    return this->suggested_dt;
}

std::string RadiationStep::getName(void) const
{
    return "radiation";
}
