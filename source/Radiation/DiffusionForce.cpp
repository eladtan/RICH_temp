#include "DiffusionForce.hpp"
#include <stdexcept>
#include <boost/math/special_functions/pow.hpp>
#ifdef RICH_MPI
#include "../mpi/IndividualGhostSources.hpp"
#include <map>

namespace
{
template<class Field>
void exchange_individual_diffusion_field(
    Tessellation3D const& tess,
    std::vector<ComputationalCell3D> const& cells,
    std::vector<std::vector<std::size_t> > const& source_indices,
    std::multimap<std::size_t, std::size_t> const& ghost_slots,
    std::vector<Field>& values)
{
    using TaggedField = std::pair<std::size_t, Field>;
    std::vector<std::vector<TaggedField> > outgoing(source_indices.size());
    for(std::size_t peer = 0; peer < source_indices.size(); ++peer)
        for(std::size_t source : source_indices[peer])
            outgoing[peer].emplace_back(cells[source].ID, values[source]);
    auto const incoming = MPI_exchange_data(tess.GetDuplicatedProcs(), outgoing);
    // Several communication paths may supply the same ID. Match the
    // tessellation synchronization rule: the final peer's value wins.
    for(auto const& peer_values : incoming)
        for(auto const& tagged : peer_values)
        {
            auto const slots = ghost_slots.equal_range(tagged.first);
            for(auto slot = slots.first; slot != slots.second; ++slot)
                values[slot->second] = tagged.second;
        }
}
}
#endif
// equations taken from "EQUATIONS AND ALGORITHMS FOR MIXED-FRAME FLUX-LIMITED DIFFUSION RADIATION HYDRODYNAMICS"

void DiffusionForce::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const vector<Conserved3D>& /*fluxes*/,const vector<Vector3D>& /*point_velocities*/, const double /*t*/,double dt,
		vector<Conserved3D> &extensives) const
{
	ApplyImpl(tess, cells, dt, nullptr, IndividualSourcePhase::Full, extensives);
}

void DiffusionForce::ApplyIndividual(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& /*fluxes*/,
	const vector<Vector3D>& /*point_velocities*/,
	double /*time*/,
	const IndividualStepContext& context,
	IndividualSourcePhase phase,
	vector<Conserved3D>& extensives) const
{
	ApplyImpl(tess, cells, 0, &context, phase, extensives);
}

void DiffusionForce::ApplyImpl(const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells,
	double dt,
	const IndividualStepContext* context,
	IndividualSourcePhase phase,
	vector<Conserved3D>& extensives) const
{
	    int rank = 0;
 #ifdef RICH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
	    std::vector<Conserved3D> old_extensives(extensives);
	    std::vector<size_t> neighbors;
	    face_vec faces;
		size_t const N = tess.GetPointNo();
	    if(cells.size() < N || extensives.size() < N ||
	       (context != nullptr && context->active_mask.size() < N))
	        throw std::invalid_argument("DiffusionForce individual cell counts are inconsistent");
		    std::vector<ComputationalCell3D> predicted_cells;
		    const std::vector<ComputationalCell3D>* source_cells = &cells;
#ifdef RICH_MPI
        std::vector<std::vector<std::size_t> > ghost_source_indices;
        std::multimap<std::size_t, std::size_t> ghost_slots;
#endif
	    if(context != nullptr)
	    {
	        predicted_cells = cells;
	        for(size_t i = 0; i < N; ++i)
	        {
	            double const volume = tess.GetVolume(i);
	            double const mass = extensives[i].mass;
	            if(!(std::isfinite(volume) && volume > 0 &&
	                 std::isfinite(mass) && mass > 0))
	                throw std::runtime_error("DiffusionForce cannot predict a non-positive cell");
	            ComputationalCell3D& predicted = predicted_cells[i];
	            predicted.density = mass / volume;
	            predicted.velocity = extensives[i].momentum / mass;
	            predicted.internal_energy = extensives[i].internal_energy / mass;
	            predicted.Erad = extensives[i].Erad / mass;
	            for(size_t tracer = 0;
	                tracer < predicted.tracers.size() && tracer < extensives[i].tracers.size();
	                ++tracer)
	                predicted.tracers[tracer] = extensives[i].tracers[tracer] / mass;
	            predicted.pressure = eos_.de2p(predicted.density,
	                predicted.internal_energy, predicted.tracers,
	                ComputationalCell3D::tracerNames);
	            predicted.temperature = eos_.de2T(predicted.density,
	                predicted.internal_energy, predicted.tracers,
	                ComputationalCell3D::tracerNames);
	        }
		        source_cells = &predicted_cells;
#ifdef RICH_MPI
            ghost_source_indices = CollectIndividualGhostSourceIndices(tess);
            for(std::size_t ghost = N;
                ghost < cells.size() && ghost < tess.GetTotalPointNumber(); ++ghost)
                if(!tess.IsPointOutsideBox(ghost))
                    ghost_slots.emplace(cells[ghost].ID, ghost);
            // The force uses velocity and density as well as radiation energy;
            // all must come from the same owner-predicted state.
            exchange_individual_diffusion_field(tess, cells, ghost_source_indices,
                ghost_slots, predicted_cells);
#endif
		    }
		    const std::vector<ComputationalCell3D>& source = *source_cells;
        const std::size_t field_count = context == nullptr ? N :
            std::min(source.size(), tess.GetTotalPointNumber());
        std::vector<double> flux_limiter(N, 0), R2(field_count,
            std::numeric_limits<double>::quiet_NaN());
        std::fill_n(R2.begin(), N, 0.0);
		    std::vector<double> new_Er(field_count, 0);
		    for(size_t i = 0; i < field_count; ++i)
            if(i < N || !tess.IsPointOutsideBox(i))
		            new_Er[i] = source[i].Erad * source[i].density;
	    std::vector<unsigned char> limiter_needed(N, context == nullptr ? 1 : 0);
	    if(context != nullptr)
	        for(size_t active : context->active_indices)
	        {
	            if(active >= N)
	                throw std::out_of_range("DiffusionForce active cell is out of range");
	            limiter_needed[active] = 1;
	            tess.GetNeighbors(active, neighbors);
	            for(size_t neighbor : neighbors)
	                if(neighbor < N)
		                    limiter_needed[neighbor] = 1;
		        }
#ifdef RICH_MPI
        for(auto const& peer_sources : ghost_source_indices)
            for(std::size_t local : peer_sources)
                limiter_needed[local] = 1;
#endif
			double max_Er = N == 0 ? 0.0 :
				*std::max_element(new_Er.begin(), new_Er.begin() + N);
#ifdef RICH_MPI
    if(context == nullptr)
        MPI_exchange_data(tess, new_Er, true);
    MPI_Allreduce(MPI_IN_PLACE, &max_Er, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    size_t const Nzero = diffusion_.zero_cells_.size();
    std::vector<size_t> zero_indeces;
    for(size_t i = 0; i < Nzero; ++i)
        zero_indeces.push_back(binary_index_find(ComputationalCell3D::stickerNames, diffusion_.zero_cells_[i]));
	    ComputationalCell3D dummy_cell;
	    if(context != nullptr &&
	       (phase != IndividualSourcePhase::SecondHalf || individual_time_step_limits_.size() != N))
	        individual_time_step_limits_.assign(N, std::numeric_limits<double>::infinity());
	    for(size_t i = 0; i < N; ++i)
	    {
	        if(!limiter_needed[i])
	            continue;
	        bool to_calc = true;
        for(size_t j = 0; j < Nzero; ++j)
	            if(source[i].stickers[zero_indeces[j]])
                to_calc = false;
        if(not to_calc)
            continue;
        double const volume = tess.GetVolume(i);
        // Calcualte gradient of radiation field
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
        Vector3D const point = tess.GetMeshPoint(i);
        Vector3D gradE(0, 0, 0);
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
            r_ij *= 1.0 / abs(r_ij);
            double Emid = 0;           
            if(!tess.IsPointOutsideBox(neighbor_j))
                Emid = 0.5 * (new_Er[i] + new_Er[neighbor_j]);
            else
            {
                Vector3D dummy_v;
	                diffusion_.boundary_calc_.GetOutSideValues(tess, source, i, neighbor_j, new_Er, Emid, dummy_v);
                Emid *= 0.5;
                Emid += 0.5 * new_Er[i];
            }
            gradE += r_ij * (tess.GetArea(faces[j]) * Emid);
        }
        gradE *= -1.0 / (diffusion_.length_scale_ * volume);
	        dummy_cell = source[i];
        dummy_cell.density *= diffusion_.mass_scale_ / (diffusion_.length_scale_ * diffusion_.length_scale_ * diffusion_.length_scale_);
	        double const D = diffusion_.D_coefficient_calcualtor.CalcDiffusionCoefficient(dummy_cell);
	        flux_limiter[i] = diffusion_.flux_limiter_ ? CG::CalcSingleFluxLimiter(gradE, D, new_Er[i]) : 1;
	        R2[i] = diffusion_.flux_limiter_ ? flux_limiter[i] / 3 + boost::math::pow<2>(flux_limiter[i] * abs(gradE) * D
	            / (CG::speed_of_light * new_Er[i])) : 1.0 / 3.0;
	        if(diffusion_.flux_limiter_ &&
	           (!std::isfinite(R2[i]) || R2[i] < 0.3))
	        {
	            double const scaled_gradient = abs(gradE) * D /
	                (CG::speed_of_light * new_Er[i] + 1e-200);
	            double const R = 3 * scaled_gradient;
	            double const limiter_times_R = R < 1e-2
	                ? R * (1 - R * R / 15 + 2 * boost::math::pow<4>(R) / 315)
	                : 3 * (1.0 / std::tanh(R) - 1.0 / R);
	            R2[i] = flux_limiter[i] / 3 +
	                boost::math::pow<2>(limiter_times_R / 3);
	        }
        if(not momentum_limit_)
        {
            flux_limiter[i] = 1;
            R2[i] = 1.0 / 3.0;
        }
    }
#ifdef RICH_MPI
    if(context == nullptr)
        MPI_exchange_data(tess, R2, true);
    else
        exchange_individual_diffusion_field(tess, cells, ghost_source_indices,
            ghost_slots, R2);
#endif
    if(context != nullptr)
    {
        // Partial hydro builds close two source layers. Every physical
        // neighbor of an active cell must therefore have a complete owner
        // target whose R2 was computed above, never an inferred ghost gradient.
        int neighbors_ready = 1;
        for(std::size_t active : context->active_indices)
        {
            tess.GetNeighbors(active, neighbors);
            for(std::size_t neighbor : neighbors)
                if(!tess.IsPointOutsideBox(neighbor) &&
                    (neighbor >= R2.size() || !std::isfinite(R2[neighbor])))
                    neighbors_ready = 0;
        }
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &neighbors_ready, 1, MPI_INT, MPI_MIN,
            MPI_COMM_WORLD);
#endif
        if(neighbors_ready == 0)
            throw std::runtime_error("DiffusionForce is missing an owner-computed neighbor R2");
    }
	    for(size_t i = 0; i < N; ++i)
	    {
	        if(context != nullptr && !context->isActive(i))
	            continue;
	        bool to_calc = true;
        for(size_t j = 0; j < Nzero; ++j)
	            if(source[i].stickers[zero_indeces[j]])
                to_calc = false;
        if(not to_calc)
            continue;
        faces = tess.GetCellFaces(i);
        tess.GetNeighbors(i, neighbors);
        size_t const Nneigh = neighbors.size();
	        Vector3D const point = tess.GetMeshPoint(i);
	        double dE = 0;
	        double const full_dt = context == nullptr ? dt : context->cellTimeStep(i);
	        double const fraction = context == nullptr || phase == IndividualSourcePhase::Full ? 1.0 : 0.5;
	        double const applied_dt = fraction * full_dt;
        for(size_t j = 0; j < Nneigh; ++j)
        {
            size_t const neighbor_j = neighbors[j];
            Vector3D r_ij = point - tess.GetMeshPoint(neighbor_j);
            r_ij *= 1.0 / abs(r_ij);
            Vector3D velocity_outside;
            double Er_outside, R2_outside, density_outside;
            // Add enthalpy advection, remember that we already had some advection in the hydro
            if(!tess.IsPointOutsideBox(neighbor_j))
            {
	                velocity_outside = source[neighbor_j].velocity;
                Er_outside = new_Er[neighbor_j];
                R2_outside = R2[neighbor_j];
	                density_outside = source[neighbor_j].density;
            }
            else
            {
	                diffusion_.boundary_calc_.GetOutSideValues(tess, source, i, neighbor_j, new_Er, Er_outside, velocity_outside);
	                R2_outside = R2[i];
	                density_outside = source[i].density;
	            }
	            double const v_cell0 = ScalarProd(r_ij, source[i].velocity);
            double const v_cell1 = ScalarProd(r_ij, velocity_outside);
            if(v_cell0 * v_cell1 > 0)
            {
                if(v_cell1 > 0)
	                    dE += Er_outside * tess.GetArea(faces[j]) * applied_dt * v_cell1 * (0.5 - 0.5 * R2_outside);
	                else
	                    dE += (0.5 - 0.5 * R2[i]) * new_Er[i] * tess.GetArea(faces[j]) * applied_dt * v_cell0;
            }
        }
        extensives[i].Erad += dE ;
        if(extensives[i].Erad < 0 || R2[i] < 0.3 || R2[i] > 1.1)
        {
            UniversalError eo("Negative energy in DiffusionForce2");
            eo.addEntry("Erad", extensives[i].Erad);
	            eo.addEntry("Ecell", source[i].density * source[i].Erad);
            eo.addEntry("R2", R2[i]);
            eo.addEntry("dE", dE);
            eo.addEntry("Volume", tess.GetVolume(i));
	            eo.addEntry("T", source[i].temperature);
	            eo.addEntry("density", source[i].density);
	            eo.addEntry("ID", source[i].ID);
            eo.addEntry("X", tess.GetMeshPoint(i).x);
            eo.addEntry("Y", tess.GetMeshPoint(i).y);
            eo.addEntry("Z", tess.GetMeshPoint(i).z);
            throw eo;
        }
    }

    double max_diff = 0;
    size_t max_loc = 0;
	    for(size_t i = 0; i < N; ++i)
	    {
	        if(context != nullptr && !context->isActive(i))
	            continue;
	        double const denominator = tess.GetVolume(i) * (new_Er[i] + 0.005 * max_Er);
	        double const absolute_change = std::abs(extensives[i].Erad - old_extensives[i].Erad);
	        double diff = denominator > 0 ? absolute_change / denominator :
	            (absolute_change > 0 ? std::numeric_limits<double>::infinity() : 0);
	        if(extensives[i].internal_energy > 10 * extensives[i].Erad)
	            diff *= 0.5;
	        if(context != nullptr)
	        {
	            double const applied_dt = context->cellTimeStep(i);
	            double const nominal_dt =
	                context->nominalCellTimeStep(i);
	            double const fraction = phase == IndividualSourcePhase::Full ? 1.0 : 0.5;
	            double const candidate = diff > 0 ?
	                std::min(
	                    fraction * applied_dt * 0.3 / diff,
	                    nominal_dt * 2.0) :
	                nominal_dt * 2.0;
	            individual_time_step_limits_[i] =
	                std::min(individual_time_step_limits_[i], candidate);
	        }
        if(diff > max_diff)
        {
            max_diff = diff;
            max_loc = i;
        }
    }
    struct
    {
        double val;
        int mpi_id;
    }max_data;
    max_data.mpi_id = rank;
    max_data.val = max_diff;
#ifdef RICH_MPI   
    MPI_Allreduce(MPI_IN_PLACE, &max_data, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
    max_diff = max_data.val;
#endif
	    double representative_dt = dt;
	    if(context != nullptr && max_loc < N)
	        representative_dt = context->cellTimeStep(max_loc);
	    if(rank == max_data.mpi_id && max_diff > 0)
	        std::cout<<"DiffusionForce dt ID "<<source[max_loc].ID<<" new Er "<<extensives[max_loc].Erad / tess.GetVolume(max_loc) <<" old Er "<<new_Er[max_loc]<<" max diff "<<max_diff<<" next dt "<<representative_dt * std::min(0.2 / max_diff, 1.1)<<" density "<<source[max_loc].density<<" T "<<source[max_loc].temperature<<std::endl;
	    if(context == nullptr)
	        next_dt_ = (max_diff > 0) ? dt * std::min(0.3 / max_diff, 1.25) : dt * 1.25;
	    else
	    {
	        next_dt_ = std::numeric_limits<double>::infinity();
	        for(size_t i : context->active_indices)
	            if(i < individual_time_step_limits_.size())
	                next_dt_ = std::min(next_dt_, individual_time_step_limits_[i]);
	    }
	}


double DiffusionForce::SuggestInverseTimeStep(void)const
{
	    return 1.0 / next_dt_;
}

void DiffusionForce::SynchronizedIndividualLimits(const Tessellation3D& /*tess*/,
	const vector<ComputationalCell3D>& /*cells*/,
	const vector<Conserved3D>& /*extensives*/,
	double /*time*/,
	vector<double>& /*limits*/,
	vector<Vector3D>& /*accelerations*/) const
{
	throw std::logic_error(
		"DiffusionForce cannot evaluate its individual limits on a rebuilt mesh (box growth in individual mode)");
}

void DiffusionForce::SuggestIndividualTimeSteps(
	const Tessellation3D& /*tess*/,
	const vector<ComputationalCell3D>& /*cells*/,
	const IndividualStepContext& context,
	vector<double>& time_step_limits) const
{
	for(size_t index : context.active_indices)
	{
		if(index >= time_step_limits.size())
			throw std::out_of_range("DiffusionForce timestep cell is out of range");
		if(index < individual_time_step_limits_.size())
			time_step_limits[index] = std::min(time_step_limits[index],
				individual_time_step_limits_[index]);
	}
}
