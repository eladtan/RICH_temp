#include "RadiationStep.hpp"
#include "misc/memory_debug.hpp"
RadiationStep::RadiationStep(Tessellation3D &tess, std::vector<ComputationalCell3D> &cells,
                    std::vector<Conserved3D> &extensives,
                    ProgressTracker &pt,
                    #ifdef RICH_MPI
                        std::shared_ptr<CostCalculator3D> cost,
                    #endif // RICH_MPI
                    const RadiationDriver &matrix_builder, bool no_hydro) :
                    tess(tess), cells(cells), extensives(extensives), pt(pt), matrix_builder(matrix_builder), no_hydro(no_hydro)
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
			if(rank == 0)
				std::cout<<"Reducing dt, new dt "<<dt_try<<std::endl;
			
			if(dt_try < 0.001 * dt)
				throw UniversalError("too small dt in RadiationTimeStep");
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
	
    
    if(no_hydro)
    {
        this->pt.updateTime(dt);
    }
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