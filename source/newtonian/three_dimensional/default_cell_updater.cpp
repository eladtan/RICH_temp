#include "default_cell_updater.hpp"
#include "../../misc/utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
	bool IndividualAllActiveCellUpdateRequested()
	{
		char const* const value =
			std::getenv("RICH_INDIVIDUAL_ALL_ACTIVE_CELL_UPDATE");
		return value != nullptr &&
			(std::strcmp(value, "1") == 0 ||
			 std::strcmp(value, "true") == 0 ||
			 std::strcmp(value, "on") == 0 ||
			 std::strcmp(value, "yes") == 0);
	}

	bool HasCompleteOrderedOwnedActivity(
		IndividualStepContext const& context, size_t const owned_count)
	{
		if(owned_count == 0 ||
		   context.active_indices.size() != owned_count ||
		   context.active_mask.size() < owned_count)
			return false;
		for(size_t owned = 0; owned < owned_count; ++owned)
			if(context.active_indices[owned] != owned ||
			   !context.isActive(owned))
				return false;
		return true;
	}

	void EntropyFix(EquationOfState const& eos, ComputationalCell3D &res, size_t entropy_index, double &energy,
			Conserved3D &extensive, double Et_min)
	{
		if(!(res.density > 0) || !(res.tracers[entropy_index] > 0))
		{
			UniversalError eo("Negative EntropyFix");
			eo.addEntry("entropy", res.tracers[entropy_index]);
			eo.addEntry("density", res.density);
			throw eo;
		}
		double de = 0;
		try
		{
	  		double new_pressure = eos.sd2p(res.tracers[entropy_index], res.density, res.tracers, ComputationalCell3D::tracerNames);
			res.pressure = new_pressure;
			double new_energy = std::max(Et_min, eos.dp2e(res.density, new_pressure));
			if(new_energy == Et_min)
			{
				res.pressure = eos.de2p(res.density, new_energy);
				double new_entropy = eos.dp2s(res.density, res.pressure, res.tracers, ComputationalCell3D::tracerNames);
				res.tracers[entropy_index] = new_entropy;
				extensive.tracers[entropy_index] = new_entropy * extensive.mass;
			}
			de = new_energy - energy;
			energy += de;
			res.internal_energy = energy;
			extensive.energy += de * extensive.mass;
			extensive.internal_energy += de * extensive.mass;
		}
		catch(UniversalError const& eo)
		{
			de = Et_min - energy;
			energy = Et_min;
			res.pressure = eos.de2p(res.density, energy);
			res.internal_energy = energy;
			extensive.energy += de * extensive.mass;
			extensive.internal_energy += de * extensive.mass;
		}
		
		if(!(energy > 0))
		{
			UniversalError eo("negative thermal energu in entropy fix");
			eo.addEntry("de", de);
			eo.addEntry("energy", energy);
			eo.addEntry("entropy", res.tracers[entropy_index]);
			eo.addEntry("density", res.density);
			throw eo;
		}
	}

	void EntropyFixSR(EquationOfState const& eos, ComputationalCell3D &res, size_t entropy_index, Conserved3D &extensive, double vol)
	{
	  double new_pressure = eos.sd2p(res.tracers[entropy_index], res.density, res.tracers, ComputationalCell3D::tracerNames);
		res.pressure = new_pressure;
		double energy = eos.dp2e(res.density, res.pressure);
		double gamma = std::sqrt(1.0 / (1 - ScalarProd(res.velocity, res.velocity)));
		extensive.energy = extensive.mass*(energy*gamma + gamma - 1) - res.pressure*vol;
	}

	// NeighborVelocity(j): the velocity of face neighbour j as the sweep has left it (momentum over mass).
	template<class NeighborVelocityT>
	bool HighRelativeKineticEnergy(Tessellation3D const& tess, size_t index, Conserved3D const& cell, double &maxDV,
		NeighborVelocityT const& NeighborVelocity)
	{
		std::vector<size_t> neigh;
		tess.GetNeighbors(index, neigh);
		size_t N = neigh.size();
		maxDV = 0;
		size_t Norg = tess.GetPointNo();
		Vector3D Vcell = cell.momentum / cell.mass;
		double Et = cell.energy / cell.mass - 0.5*ScalarProd(Vcell, Vcell);
		double div_V = 0;
		Vector3D const r_i = tess.GetMeshPoint(index);
		const auto &faces = tess.GetCellFaces(index);
		for (size_t i = 0; i < N; ++i)
		{
			size_t neighbor_j = neigh[i];
			if (neighbor_j < Norg || !tess.IsPointOutsideBox(neighbor_j))
			{
				auto const r_ij = normalize(r_i - tess.GetMeshPoint(neighbor_j));
				double const A_ij = tess.GetArea(faces[i]);
				Vector3D velocity_j = NeighborVelocity(neighbor_j);
				div_V -= 0.5*ScalarProd(Vcell+velocity_j, r_ij) * A_ij;
				maxDV = std::max(maxDV, abs(Vcell - velocity_j));
			}
		}
		div_V /= tess.GetVolume(index);
		if(div_V < 0 && (div_V * tess.GetWidth(index) < -fastabs(Vcell) * 0.05))
			return false;
		return 0.005*maxDV*maxDV > Et;
	}

	// One cell of regular_update's sweep, in place.  NeighborVelocity is passed to HighRelativeKineticEnergy.
	template<class NeighborVelocityT>
	void RecoverRegularCell(size_t i, std::vector<ComputationalCell3D> &res, std::vector<Conserved3D> & extensives,
		Tessellation3D const& tess, size_t entropy_index, EquationOfState const& eos, bool const includes_temperature,
		const RadiationDriver* diffusion, double const min_temperature, size_t Ntracers,
		NeighborVelocityT const& NeighborVelocity)
	{
		size_t print_id = -1;
		try
		{
			if(res[i].ID == print_id)
				std::cout<<"Entered cell update "<<res[i]<<std::endl;
			Conserved3D& extensive = extensives[i];
			const double vol = tess.GetVolume(i);
			res[i].density = extensive.mass / vol;
			double d_factor = 1;
			if(res[i].density > 0 && res[i].density < 2e-21)
			{
				d_factor = 2e-21 / res[i].density;
				res[i].density = 2e-21;
				extensive.mass *= d_factor;
				extensive.momentum *= d_factor;
				extensive.internal_energy *= d_factor;
				for (size_t j = 0; j < Ntracers; ++j)
					extensive.tracers[j] *= d_factor;
			}
			double const old_etherm = res[i].internal_energy;
			res[i].velocity = extensive.momentum / extensive.mass;
			double energy = extensive.internal_energy / extensive.mass;
			// extensive.energy = extensive.mass*(energy + 0.5*ScalarProd(res[i].velocity, res[i].velocity));
			for (size_t j = 0; j < Ntracers; ++j)
				res[i].tracers[j] = extensive.tracers[j] / extensive.mass;
			// Entropy fix if needed
			if (entropy_index < ComputationalCell3D::tracerNames.size())
			{
				double Et_min = 0;
				try
				{
					if(min_temperature > 0)
					{
						Et_min = eos.dT2e(res[i].density, min_temperature, res[i].tracers, ComputationalCell3D::tracerNames);
						if(energy < Et_min)
						{
							res[i].temperature = min_temperature;
							res[i].internal_energy = Et_min;
							extensive.energy += (res[i].internal_energy - energy) * extensive.mass;
							extensive.internal_energy += (res[i].internal_energy - energy) * extensive.mass;
							energy = Et_min;
							res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
							double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
							res[i].tracers[entropy_index] = new_entropy;
							extensive.tracers[entropy_index] = new_entropy * extensive.mass;
						}
					}
					// Do we have a negative thermal energy?
					if (energy < 0)
					{
						if(print_id == res[i].ID)
							std::cout<<"Entered entropy fix1"<<std::endl;
						EntropyFix(eos, res[i], entropy_index, energy, extensive, Et_min);
					}
					else
					{
						double maxDV;
						// Is the kinetic energy small?
						if ((energy*extensive.mass < 0.005*extensive.energy) &&
							HighRelativeKineticEnergy(tess, i, extensive, maxDV, NeighborVelocity))
						{
							if(print_id == res[i].ID)
								std::cout<<"Entered entropy fix2"<<std::endl;
							EntropyFix(eos, res[i], entropy_index, energy, extensive, Et_min);
						}
						else
						{
							if(print_id == res[i].ID)
								std::cout<<"Regular update "<<maxDV<<std::endl;
							double new_pressure = eos.de2p(res[i].density, energy, res[i].tracers, ComputationalCell3D::tracerNames);
						  	double new_entropy = eos.dp2s(res[i].density, new_pressure, res[i].tracers, ComputationalCell3D::tracerNames);
							// We don't need the entropy fix, update entropy
							res[i].internal_energy = energy;
							res[i].pressure = new_pressure;
							res[i].tracers[entropy_index] = new_entropy;
							extensive.tracers[entropy_index] = new_entropy * extensive.mass;
						}
					}
				}
				catch (UniversalError &eo)
				{
					eo.addEntry("Cell index", static_cast<double>(i));
					eo.addEntry("Cell mass", extensives[i].mass);
					eo.addEntry("Cell x momentum", extensives[i].momentum.x);
					eo.addEntry("Cell y momentum", extensives[i].momentum.y);
					eo.addEntry("Cell z momentum", extensives[i].momentum.z);
					eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
					eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
					eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
					eo.addEntry("cell Temperature", res[i].temperature);
					eo.addEntry("Cell volume", vol);
					eo.addEntry("Cell energy", extensives[i].energy);
					eo.addEntry("Cell thermal energy per unit mass", energy);
					eo.addEntry("Et_min", Et_min);
					eo.addEntry("old_etherm", old_etherm);
					eo.addEntry("Cell id", static_cast<double>(res[i].ID));
					for(size_t j = 0; j <ComputationalCell3D::tracerNames.size(); ++j)
						eo.addEntry(ComputationalCell3D::tracerNames[j], extensives[i].tracers[j]);
					throw eo;
				}
			}
			else
			{
			  	res[i].pressure = eos.de2p(res[i].density, energy, res[i].tracers, ComputationalCell3D::tracerNames);
				res[i].internal_energy = energy;
			}
			if(includes_temperature)
			{
				if(diffusion != nullptr)
				{
					res[i].Erad = extensive.Erad / extensive.mass;
					if(!(res[i].Erad > 0))
					{
						UniversalError eo("Negative quantity in cell update");
						eo.addEntry("Cell index", static_cast<double>(i));
						eo.addEntry("extensive.Erad", extensive.Erad);
						eo.addEntry("Cell mass", extensives[i].mass);
						eo.addEntry("Cell x momentum", extensives[i].momentum.x);
						eo.addEntry("Cell y momentum", extensives[i].momentum.y);
						eo.addEntry("Cell z momentum", extensives[i].momentum.z);
						eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
						eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
						eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
						eo.addEntry("Cell volume", vol);
						eo.addEntry("Cell energy", extensives[i].energy);
						eo.addEntry("Cell thermal energy per unit mass", energy);
						eo.addEntry("Cell id", static_cast<double>(res[i].ID));
						throw eo;
					}
					

					// update group energies
					for(std::size_t g=0; g < ENERGY_GROUPS_NUM; ++g){
						res[i].Eg[g] = extensive.Eg[g] / extensive.mass;
					}
				}
				res[i].temperature = eos.de2T(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
				if(min_temperature > 0 && res[i].temperature < min_temperature)
				{
					res[i].temperature = min_temperature;
					double old_e = res[i].internal_energy;
                        res[i].internal_energy = eos.dT2e(res[i].density, res[i].temperature, res[i].tracers, ComputationalCell3D::tracerNames);
					extensive.energy += (res[i].internal_energy - old_e) * extensive.mass;
					extensive.internal_energy += (res[i].internal_energy - old_e) * extensive.mass;
                        res[i].pressure = eos.de2p(res[i].density, res[i].internal_energy, res[i].tracers, ComputationalCell3D::tracerNames);
                        if (entropy_index < ComputationalCell3D::tracerNames.size())
					{
						double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
						res[i].tracers[entropy_index] = new_entropy;
						extensive.tracers[entropy_index] = new_entropy * extensive.mass;
					}
				}
			}
			if (!(res[i].density > 0) || !(res[i].pressure > 0) || (!std::isfinite(fastabs(extensives[i].momentum))))
			{
				UniversalError eo("Negative quantity in cell update");
				eo.addEntry("Cell index", static_cast<double>(i));
				eo.addEntry("Cell mass", extensives[i].mass);
				eo.addEntry("Cell x momentum", extensives[i].momentum.x);
				eo.addEntry("Cell y momentum", extensives[i].momentum.y);
				eo.addEntry("Cell z momentum", extensives[i].momentum.z);
				eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
				eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
				eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
				eo.addEntry("Cell volume", vol);
				eo.addEntry("Cell energy", extensives[i].energy);
				eo.addEntry("Cell thermal energy per unit mass", energy);
				eo.addEntry("Cell id", static_cast<double>(res[i].ID));
				throw eo;
			}
			if(print_id == res[i].ID)
				std::cout<<"Exited cell update "<<res[i]<<std::endl;
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("Cell index", static_cast<double>(i));
			eo.addEntry("Cell mass", extensives[i].mass);
			eo.addEntry("Cell x momentum", extensives[i].momentum.x);
			eo.addEntry("Cell y momentum", extensives[i].momentum.y);
			eo.addEntry("Cell z momentum", extensives[i].momentum.z);
			eo.addEntry("Cell energy", extensives[i].energy);
			throw eo;
		}
	}

	void regular_update(std::vector<ComputationalCell3D> &res, std::vector<Conserved3D> & extensives,
		Tessellation3D const& tess, size_t entropy_index,
		EquationOfState const& eos, bool const includes_temperature, const RadiationDriver* diffusion, double const min_temperature)
	{
		size_t Nloop = tess.GetPointNo();
		size_t Ntracers = ComputationalCell3D::tracerNames.size();
		// The full sweep reads every neighbour from the swept array itself.
		auto const SweptVelocity = [&extensives](size_t j) {return extensives.at(j).momentum / extensives[j].mass;};
		for (size_t i = 0; i < Nloop; ++i)
			RecoverRegularCell(i, res, extensives, tess, entropy_index, eos, includes_temperature, diffusion,
				min_temperature, Ntracers, SweptVelocity);
	}

	void regular_updateSR(std::vector<ComputationalCell3D> &res, std::vector<Conserved3D> & extensives,
		Tessellation3D const& tess, size_t entropy_index,
		EquationOfState const& eos, double G)
	{
		size_t Nloop = tess.GetPointNo();
		size_t Ntracers = ComputationalCell3D::tracerNames.size();
		for (size_t i = 0; i < Nloop; ++i)
		{
			try
			{
				double v = GetVelocity(extensives[i], G);
				const double volume = 1.0 / tess.GetVolume(i);
				if (fastabs(extensives[i].momentum)*1e8 < extensives[i].mass)
				{
					res[i].velocity = extensives[i].momentum / extensives[i].mass;
					//v = abs(res[i].velocity);
				}
				else
					res[i].velocity = v * extensives[i].momentum / abs(extensives[i].momentum);
				double gamma_1 = std::sqrt(1 - ScalarProd(res[i].velocity, res[i].velocity));
				res[i].density = extensives[i].mass *gamma_1*volume;
				//				double r = fastabs(tess.GetMeshPoint(i));
				if (res[i].density < 0)
					throw UniversalError("Negative density");
				for (size_t j = 0; j < extensives[i].tracers.size(); ++j)
					res[i].tracers[j] = extensives[i].tracers[j] / extensives[i].mass;
				if (fastabs(res[i].velocity) < 1e-5)
					res[i].pressure = (G - 1)*((extensives[i].energy - ScalarProd(extensives[i].momentum, res[i].velocity))*volume
						+ (0.5*ScalarProd(res[i].velocity, res[i].velocity))*res[i].density);
				else
					res[i].pressure = (G - 1)*(extensives[i].energy*volume - ScalarProd(extensives[i].momentum, res[i].velocity)*volume
						+ (1.0 / gamma_1 - 1)*res[i].density);
				// Entropy fix if needed
				if (entropy_index < Ntracers)
				{
					if (!(extensives[i].tracers[entropy_index] > 0))
					{
						UniversalError eo("Negative entropy");
						eo.addEntry("Extensive Entropy", extensives[i].tracers[entropy_index]);
						throw eo;
					}
					// Do we have a negative thermal energy?
					if (res[i].pressure < 0)
					{
						EntropyFixSR(eos, res[i], entropy_index, extensives[i], 1.0 / volume);
					}
					else
					{
					  double new_entropy = eos.dp2s(res[i].density, res[i].pressure, res[i].tracers, ComputationalCell3D::tracerNames);
						// We don't need the entropy fix, update entropy
						res[i].tracers[entropy_index] = new_entropy;
						extensives[i].tracers[entropy_index] = new_entropy * extensives[i].mass;
					}
				}
				if (!(res[i].density > 0) || !(res[i].pressure > 0)||(!std::isfinite(fastabs(extensives[i].momentum))))
				{
					UniversalError eo("Negative quantity in cell update");
					throw eo;
				}
				res[i].internal_energy = eos.dp2e(res[i].density, res[i].pressure,res[i].tracers,ComputationalCell3D::tracerNames);
			}
			catch (UniversalError &eo)
			{
			  eo.addEntry("Cell ID", static_cast<double>(res[i].ID));
				eo.addEntry("Cell volume", tess.GetVolume(i));
				eo.addEntry("Cell index", static_cast<double>(i));
				eo.addEntry("Cell mass", extensives[i].mass);
				eo.addEntry("Cell x momentum", extensives[i].momentum.x);
				eo.addEntry("Cell y momentum", extensives[i].momentum.y);
				eo.addEntry("Cell z momentum", extensives[i].momentum.z);
				eo.addEntry("Cell x location", tess.GetMeshPoint(i).x);
				eo.addEntry("Cell y location", tess.GetMeshPoint(i).y);
				eo.addEntry("Cell z location", tess.GetMeshPoint(i).z);
				eo.addEntry("Cell energy", extensives[i].energy);
				eo.addEntry("Number of tracers", static_cast<double>(Ntracers));
				for (size_t j = 0; j < Ntracers; ++j)
					eo.addEntry("Tracer extensive value", extensives[i].tracers[j]);
				eo.addEntry("Entropy index", static_cast<double>(entropy_index));
				throw;
			}
		}
	}

    // RICH_INDIVIDUAL_CELL_UPDATE_SPARSE: strict (unset means 0), read once by its caller's static and agreed across
    // ranks in one reduction.
    int AgreedSparseCellUpdateSwitch(void)
    {
        char const* const Name = "RICH_INDIVIDUAL_CELL_UPDATE_SPARSE";
        char const* const Value = std::getenv(Name);
        std::string const Text = Value == nullptr ? "" : Value;
        int Local = -1;
        if(Text.empty())
            Local = 0;
        else if(Text.size() == 1 && Text[0] >= '0' && Text[0] <= '2')
            Local = Text[0] - '0';
        int Extrema[2] = {Local, -Local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, Extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(Extrema[0] != -Extrema[1])
            throw std::invalid_argument(std::string(Name) + " differs across MPI ranks or is invalid on some rank");
        if(Extrema[0] < 0)
            throw std::invalid_argument(std::string(Name) + " must be 0, 1 or 2");
        return Extrema[0];
    }

    // The sparse sweep needs the active targets owned, strictly ascending (the order of the full sweep), and arrays
    // that hold every owned target; otherwise the full path runs.
    bool SparseRecoveryApplies(vector<ComputationalCell3D> const& Cells, vector<Conserved3D> const& Extensives,
        size_t OwnedCount, std::vector<size_t> const& ActiveIndices)
    {
        if(Cells.size() < OwnedCount || Extensives.size() < OwnedCount)
            return false;
        for(size_t Position = 0; Position < ActiveIndices.size(); ++Position)
            if(ActiveIndices[Position] >= OwnedCount ||
                (Position > 0 && ActiveIndices[Position - 1] >= ActiveIndices[Position]))
                return false;
        return true;
    }

    bool SameBits(double First, double Second)
    {
        return std::memcmp(&First, &Second, sizeof(double)) == 0;
    }

    bool SameVector(Vector3D const& First, Vector3D const& Second)
    {
        return SameBits(First.x, Second.x) && SameBits(First.y, Second.y) && SameBits(First.z, Second.z);
    }

    bool SameCell(ComputationalCell3D const& First, ComputationalCell3D const& Second)
    {
        if(!(SameBits(First.density, Second.density) && SameBits(First.pressure, Second.pressure) &&
            SameBits(First.internal_energy, Second.internal_energy) && SameBits(First.temperature, Second.temperature) &&
            First.ID == Second.ID && SameVector(First.velocity, Second.velocity) && SameBits(First.dt, Second.dt) &&
            SameBits(First.Erad, Second.Erad) && SameBits(First.Erad_dt, Second.Erad_dt) &&
            SameBits(First.Erad_dt_dt, Second.Erad_dt_dt) && SameBits(First.cs, Second.cs) &&
            First.Eg.size() == Second.Eg.size() && First.stickers == Second.stickers))
            return false;
        for(size_t Group = 0; Group < First.Eg.size(); ++Group)
            if(!SameBits(First.Eg[Group], Second.Eg[Group]))
                return false;
        for(size_t Tracer = 0; Tracer < First.tracers.size(); ++Tracer)
            if(!SameBits(First.tracers[Tracer], Second.tracers[Tracer]))
                return false;
        return true;
    }

    bool SameConserved(Conserved3D const& First, Conserved3D const& Second)
    {
        if(!(SameBits(First.mass, Second.mass) && SameVector(First.momentum, Second.momentum) &&
            SameBits(First.energy, Second.energy) && SameBits(First.internal_energy, Second.internal_energy) &&
            SameBits(First.Erad, Second.Erad) && SameBits(First.Erad_dt, Second.Erad_dt) &&
            SameBits(First.Erad_dt_dt, Second.Erad_dt_dt) && First.Eg.size() == Second.Eg.size()))
            return false;
        for(size_t Group = 0; Group < First.Eg.size(); ++Group)
            if(!SameBits(First.Eg[Group], Second.Eg[Group]))
                return false;
        for(size_t Tracer = 0; Tracer < First.tracers.size(); ++Tracer)
            if(!SameBits(First.tracers[Tracer], Second.tracers[Tracer]))
                return false;
        return true;
    }

    // Mode 2 of RICH_INDIVIDUAL_CELL_UPDATE_SPARSE: every active target's recovered primitive and extensive state from
    // the sparse sweep against the kept full path, bit for bit.  One reduction; rank 0 prints the counts; on a
    // difference the lowest rank that found one prints an example and every rank throws.
    void VerifySparseCellUpdate(vector<ComputationalCell3D> const& KeptCells, vector<Conserved3D> const& KeptExtensives,
        vector<ComputationalCell3D> const& SparseCells, vector<Conserved3D> const& SparseExtensives,
        IndividualStepContext const& Context, bool Applied, std::string const& SparseFailure)
    {
        unsigned long long Counts[3] = {0, 0, Applied ? 0ULL : 1ULL};
        std::ostringstream Example;
        Example.precision(17);
        if(Applied && !SparseFailure.empty()) {
            Counts[1] = 1;
            Example << " sparse_exception=\"" << SparseFailure << '"';
        }
        else if(Applied) {
            for(size_t const Index : Context.active_indices) {
                ++Counts[0];
                bool const Same = Index < KeptCells.size() && Index < SparseCells.size() &&
                    Index < KeptExtensives.size() && Index < SparseExtensives.size() &&
                    SameCell(KeptCells[Index], SparseCells[Index]) &&
                    SameConserved(KeptExtensives[Index], SparseExtensives[Index]);
                if(Same || Counts[1]++ != 0)
                    continue;
                Example << " index=" << Index;
                if(Index < KeptCells.size() && Index < SparseCells.size() && Index < KeptExtensives.size() &&
                    Index < SparseExtensives.size())
                    Example << " kept=" << KeptCells[Index] << " sparse=" << SparseCells[Index] << " kept_mass="
                        << KeptExtensives[Index].mass << " sparse_mass=" << SparseExtensives[Index].mass
                        << " kept_energy=" << KeptExtensives[Index].energy << " sparse_energy="
                        << SparseExtensives[Index].energy;
            }
        }
        unsigned long long const LocalMismatches = Counts[1];
        int Rank = 0;
        int RankCount = 1;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
        MPI_Allreduce(MPI_IN_PLACE, Counts, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(Rank == 0)
            std::cout << "RICH_B10_CELL_VERIFY event_tick=" << Context.event_tick << " checks=" << Counts[0]
                << " mismatches=" << Counts[1] << " full_path_ranks=" << Counts[2] << std::endl;
        if(Counts[1] == 0)
            return;
        int FirstRank = LocalMismatches != 0 ? Rank : RankCount;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &FirstRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        if(Rank == FirstRank)
            std::cerr << "RICH_B10_CELL_VERIFY_MISMATCH rank=" << Rank << " event_tick=" << Context.event_tick
                << " local=" << LocalMismatches << Example.str() << std::endl;
        throw std::runtime_error("RICH_INDIVIDUAL_CELL_UPDATE_SPARSE=2: the sparse cell update differs from the full "
            "update (" + std::to_string(Counts[1]) + " mismatches)");
    }
}

int IndividualCellUpdateSparseMode(void)
{
    static int const Mode = AgreedSparseCellUpdateSwitch();
    return Mode;
}

DefaultCellUpdater::DefaultCellUpdater(bool SR, double G, bool const includes_temperature, double const min_temperature, const RadiationDriver* diffusion) :SR_(SR), G_(G), includes_temperature_(includes_temperature), min_temperature_(min_temperature), diffusion_(diffusion), entropy_index_(9999999) {}

void DefaultCellUpdater::UpdateIndividual(
	vector<ComputationalCell3D> &res,
	EquationOfState const& eos,
	const Tessellation3D& tess,
	vector<Conserved3D>& extensives,
	const IndividualStepContext& context) const
{
	size_t const owned_count = tess.GetPointNo();
	bool const replace_whole_vectors =
		IndividualAllActiveCellUpdateRequested() &&
		HasCompleteOrderedOwnedActivity(context, owned_count) &&
		res.size() >= owned_count && extensives.size() >= owned_count;
    // RICH_INDIVIDUAL_CELL_UPDATE_SPARSE: agreed on every rank (collective on first use; the hydro step calls this on
    // every rank, twice per event).  The sparse sweep replaces the copy-sweep-select path below; with whole-vector
    // replacement or special relativity this rank keeps that path.
    int const SparseMode = IndividualCellUpdateSparseMode();
    bool const SparseApplies = SparseMode != 0 && !SR_ && !replace_whole_vectors &&
        SparseRecoveryApplies(res, extensives, owned_count, context.active_indices);
    if(SparseMode == 1 && SparseApplies) {
        UpdateActiveCells(res, eos, tess, extensives, context.active_indices);
        return;
    }
    // Verify: the sparse sweep on copies of the incoming state, then the path below, whose result is kept.
    vector<ComputationalCell3D> SparseCells;
    vector<Conserved3D> SparseExtensives;
    std::string SparseFailure;
    if(SparseMode == 2 && SparseApplies) {
        SparseCells = res;
        SparseExtensives = extensives;
        try {
            UpdateActiveCells(SparseCells, eos, tess, SparseExtensives, context.active_indices);
        }
        catch(UniversalError const& Error) {
            SparseFailure = "UniversalError: " + Error.getErrorMessage();
        }
        catch(std::exception const& Error) {
            SparseFailure = Error.what();
        }
    }

	// When the entropy fix is enabled, the individual hydro path refreshes every
	// compact ghost cell through SyncPartialBuildData before primitive recovery.
	// MadVoro's duplicate send indices address the much larger all-points array
	// during a sparse build, so repeating the legacy MPI exchange here would
	// index the compact event vectors with all-point indices.
	vector<ComputationalCell3D> candidate_cells = res;
	vector<Conserved3D> candidate_extensives = extensives;
	UpdateCells(candidate_cells, eos, tess, candidate_extensives, false);
	if(replace_whole_vectors && candidate_cells.size() == res.size() &&
	   candidate_extensives.size() == extensives.size())
	{
		res.swap(candidate_cells);
		extensives.swap(candidate_extensives);
        if(SparseMode == 2)
            VerifySparseCellUpdate(res, extensives, SparseCells, SparseExtensives, context, SparseApplies, SparseFailure);
		return;
	}

	for(size_t index : context.active_indices)
	{
		if(index >= res.size() || index >= candidate_cells.size() ||
		   index >= extensives.size() || index >= candidate_extensives.size())
			throw std::out_of_range(
				"Individual cell update index is out of range");
		res[index] = candidate_cells[index];
		extensives[index] = candidate_extensives[index];
	}
    if(SparseMode == 2)
        VerifySparseCellUpdate(res, extensives, SparseCells, SparseExtensives, context, SparseApplies, SparseFailure);
}

void DefaultCellUpdater::UpdateActiveCells(vector<ComputationalCell3D>& Cells, EquationOfState const& Eos,
    const Tessellation3D& Tess, vector<Conserved3D>& Extensives, std::vector<size_t> const& ActiveIndices) const
{
    // The full path recovers every owned target in index order on copies and keeps the active ones.  An active
    // target's recovery reads its own state and, in HighRelativeKineticEnergy, the momentum and mass of its face
    // neighbours as the sweep has left them: a lower-index owned target after its density floor (the only step that
    // changes either), any other cell as it was.  This sweep recovers the active targets in the same order, in place,
    // with the same per-cell operations, and gives a passive lower-index neighbour the result of its density floor.
    entropy_index_ = ComputationalCell3D::tracerNames.size();
    vector<string>::const_iterator it = binary_find(ComputationalCell3D::tracerNames.begin(),
        ComputationalCell3D::tracerNames.end(), string("Entropy"));
    if (it != ComputationalCell3D::tracerNames.end())
        entropy_index_ = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
    size_t const Ntracers = ComputationalCell3D::tracerNames.size();
    size_t Current = 0;
    auto const SweptVelocity = [&Extensives, &Tess, &ActiveIndices, &Current](size_t j) {
        if(j < Current && !std::binary_search(ActiveIndices.begin(), ActiveIndices.end(), j)) {
            // regular_update's density floor of target j, as the full sweep applied it before reaching Current.
            Conserved3D const& Passive = Extensives.at(j);
            double const Density = Passive.mass / Tess.GetVolume(j);
            if(Density > 0 && Density < 2e-21) {
                double const DFactor = 2e-21 / Density;
                double Mass = Passive.mass;
                Mass *= DFactor;
                Vector3D Momentum = Passive.momentum;
                Momentum *= DFactor;
                return Momentum / Mass;
            }
        }
        return Extensives.at(j).momentum / Extensives[j].mass;
    };
    // The full path leaves both arrays untouched when it throws, so the active targets are restored on an exception.
    vector<ComputationalCell3D> SavedCells;
    vector<Conserved3D> SavedExtensives;
    SavedCells.reserve(ActiveIndices.size());
    SavedExtensives.reserve(ActiveIndices.size());
    for(size_t const Index : ActiveIndices) {
        SavedCells.push_back(Cells[Index]);
        SavedExtensives.push_back(Extensives[Index]);
    }
    try {
        for(size_t const Index : ActiveIndices) {
            Current = Index;
            RecoverRegularCell(Index, Cells, Extensives, Tess, entropy_index_, Eos, includes_temperature_, diffusion_,
                min_temperature_, Ntracers, SweptVelocity);
        }
    }
    catch(...) {
        for(size_t Position = 0; Position < ActiveIndices.size(); ++Position) {
            Cells[ActiveIndices[Position]] = SavedCells[Position];
            Extensives[ActiveIndices[Position]] = SavedExtensives[Position];
        }
        throw;
    }
}

void DefaultCellUpdater::operator()(vector<ComputationalCell3D> &res, EquationOfState const& eos,
	const Tessellation3D& tess, vector<Conserved3D>& extensives) const
{
	UpdateCells(res, eos, tess, extensives, true);
}

void DefaultCellUpdater::UpdateCells(vector<ComputationalCell3D> &res,
	EquationOfState const& eos, const Tessellation3D& tess,
	vector<Conserved3D>& extensives,
	bool const synchronize_entropy_ghosts) const
{
#ifndef RICH_MPI
	(void)synchronize_entropy_ghosts;
#endif
  entropy_index_ = ComputationalCell3D::tracerNames.size();
  vector<string>::const_iterator it = binary_find(ComputationalCell3D::tracerNames.begin(),
						  ComputationalCell3D::tracerNames.end(), string("Entropy"));
  if (it != ComputationalCell3D::tracerNames.end())
    entropy_index_ = static_cast<size_t>(it - ComputationalCell3D::tracerNames.begin());
#ifdef RICH_MPI
	if (synchronize_entropy_ghosts &&
		entropy_index_ < ComputationalCell3D::tracerNames.size())
	{
		MPI_exchange_data(tess, extensives, true);
	}
#endif
	if (!SR_)
		regular_update(res, extensives, tess, entropy_index_, eos, includes_temperature_, diffusion_, min_temperature_);
	else
		regular_updateSR(res, extensives, tess, entropy_index_, eos, G_);
#ifdef RICH_MPI
	if (entropy_index_ < ComputationalCell3D::tracerNames.size())
		extensives.resize(tess.GetPointNo());
#endif

}
