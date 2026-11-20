#include "ConditionActionFlux1.hpp"
#include "LinearGauss3D.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

ConditionActionFlux1::ConditionActionFlux1(const vector<pair<const Condition3D*, const Action3D*> >& sequence,
	SpatialReconstruction3D const& interp) :
	sequence_(sequence),interp_(interp), VerifyFaceValues_() {}

ConditionActionFlux1::~ConditionActionFlux1(void)
{}

namespace
{
	double directional_derivative(double vx, double vy, double vz,
		double dx, double dy, double dz)
	{
		return vx * dx + vy * dy + vz * dz;
	}

	void predict_primitive(ComputationalCell3D &face_value,
		const ComputationalCell3D &cell,
		const Slope3D &slope,
		const EquationOfState &eos,
		const Vector3D &face_velocity,
		const Vector3D &acceleration,
		double time_offset)
	{
		const double vx = cell.velocity.x - face_velocity.x;
		const double vy = cell.velocity.y - face_velocity.y;
		const double vz = cell.velocity.z - face_velocity.z;
		const double div_v = slope.xderivative.velocity.x +
			slope.yderivative.velocity.y + slope.zderivative.velocity.z;
		const double drho = -directional_derivative(vx, vy, vz,
			slope.xderivative.density, slope.yderivative.density,
			slope.zderivative.density) - cell.density * div_v;
		const Vector3D pressure_gradient(slope.xderivative.pressure,
			slope.yderivative.pressure, slope.zderivative.pressure);
		Vector3D dv;
		dv.x = -directional_derivative(vx, vy, vz,
			slope.xderivative.velocity.x, slope.yderivative.velocity.x,
			slope.zderivative.velocity.x) - pressure_gradient.x / cell.density;
		dv.y = -directional_derivative(vx, vy, vz,
			slope.xderivative.velocity.y, slope.yderivative.velocity.y,
			slope.zderivative.velocity.y) - pressure_gradient.y / cell.density;
		dv.z = -directional_derivative(vx, vy, vz,
			slope.xderivative.velocity.z, slope.yderivative.velocity.z,
			slope.zderivative.velocity.z) - pressure_gradient.z / cell.density;
		dv += acceleration;
		const double de = -directional_derivative(vx, vy, vz,
			slope.xderivative.internal_energy,
			slope.yderivative.internal_energy,
			slope.zderivative.internal_energy) -
			cell.pressure * div_v / cell.density;

		ComputationalCell3D candidate = face_value;
		double predictor_factor = 1;
		for(std::size_t attempt = 0; attempt < 20; ++attempt)
		{
			candidate.density = face_value.density +
				predictor_factor * time_offset * drho;
			candidate.velocity = face_value.velocity +
				predictor_factor * time_offset * dv;
			candidate.internal_energy = face_value.internal_energy +
				predictor_factor * time_offset * de;
			for(std::size_t tracer = 0; tracer < candidate.tracers.size(); ++tracer)
			{
				const double dq = -directional_derivative(vx, vy, vz,
					slope.xderivative.tracers[tracer],
					slope.yderivative.tracers[tracer],
					slope.zderivative.tracers[tracer]);
				candidate.tracers[tracer] = face_value.tracers[tracer] +
					predictor_factor * time_offset * dq;
			}

			bool const primitive_valid = candidate.density > 0 &&
				candidate.internal_energy > 0 &&
				std::isfinite(candidate.density) &&
				std::isfinite(candidate.internal_energy) &&
				std::isfinite(candidate.velocity.x) &&
				std::isfinite(candidate.velocity.y) &&
				std::isfinite(candidate.velocity.z);
			if(primitive_valid)
			{
				try
				{
					candidate.pressure = eos.de2p(candidate.density,
						candidate.internal_energy, candidate.tracers,
						ComputationalCell3D::tracerNames);
					if(candidate.pressure > 0 &&
					   std::isfinite(candidate.pressure))
					{
						face_value = candidate;
						return;
					}
				}
				catch(UniversalError&)
					{}
			}
			predictor_factor *= 0.5;
		}
		// The spatially reconstructed face state was already EOS-closed.  Keep it
		// when no admissible temporal correction can be found.
	}

	void choose_action(size_t face, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
		const EquationOfState& eos, const Vector3D& face_velocity,
		const vector<pair<const ConditionActionFlux1::Condition3D*, const ConditionActionFlux1::Action3D*> >& sequence,
		Conserved3D &res, double time, std::pair<ComputationalCell3D, ComputationalCell3D>
		const& face_values)
	{
		for (size_t i = 0; i < sequence.size(); ++i)
		{
			const pair<bool, bool> flag_aux = (*sequence[i].first)
				(face, tess, cells);
#ifdef RICH_DEBUG
			try
			{
#endif
				if (flag_aux.first)
				{
					(*sequence[i].second)(
						face, tess, face_velocity, cells, eos,
						flag_aux.second, res, time, face_values);
					return;
				}
#ifdef RICH_DEBUG
			}
			catch (UniversalError &eo)
			{
				eo.addEntry("Face number", face);
				size_t n0 = tess.GetFaceNeighbors(face).first;
				size_t n1 = tess.GetFaceNeighbors(face).second;
				eo.addEntry("Left cell density", cells[n0].density);
				eo.addEntry("Left cell pressure", cells[n0].pressure);
				eo.addEntry("Left cell Vx", cells[n0].velocity.x);
				eo.addEntry("Left cell Vy", cells[n0].velocity.y);
				eo.addEntry("Left cell Vz", cells[n0].velocity.z);
				eo.addEntry("Left cell internal energy", cells[n0].internal_energy);
				eo.addEntry("Left cell ID", cells[n0].ID);
				eo.addEntry("Right cell density", cells[n1].density);
				eo.addEntry("Right cell pressure", cells[n1].pressure);
				eo.addEntry("Right cell Vx", cells[n1].velocity.x);
				eo.addEntry("Right cell Vy", cells[n1].velocity.y);
				eo.addEntry("Right cell Vz", cells[n1].velocity.z);
				eo.addEntry("Right cell internal energy", cells[n1].internal_energy);
				eo.addEntry("Right cell ID", cells[n1].ID);
				throw;
			}
#endif
		}
		throw UniversalError("Error in ConditionActionFlux1");
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
        for(std::size_t Group = 0; Group < First.Eg.size(); ++Group)
            if(!SameBits(First.Eg[Group], Second.Eg[Group]))
                return false;
        for(std::size_t Tracer = 0; Tracer < First.tracers.size(); ++Tracer)
            if(!SameBits(First.tracers[Tracer], Second.tracers[Tracer]))
                return false;
        return true;
    }

    bool SameSlope(Slope3D const& First, Slope3D const& Second)
    {
        return SameCell(First.xderivative, Second.xderivative) && SameCell(First.yderivative, Second.yderivative) &&
            SameCell(First.zderivative, Second.zderivative);
    }

    bool SameConserved(Conserved3D const& First, Conserved3D const& Second)
    {
        if(!(SameBits(First.mass, Second.mass) && SameVector(First.momentum, Second.momentum) &&
            SameBits(First.energy, Second.energy) && SameBits(First.internal_energy, Second.internal_energy) &&
            SameBits(First.Erad, Second.Erad) && SameBits(First.Erad_dt, Second.Erad_dt) &&
            SameBits(First.Erad_dt_dt, Second.Erad_dt_dt) && First.Eg.size() == Second.Eg.size()))
            return false;
        for(std::size_t Group = 0; Group < First.Eg.size(); ++Group)
            if(!SameBits(First.Eg[Group], Second.Eg[Group]))
                return false;
        for(std::size_t Tracer = 0; Tracer < First.tracers.size(); ++Tracer)
            if(!SameBits(First.tracers[Tracer], Second.tracers[Tracer]))
                return false;
        return true;
    }

    // The faces the individual flux loop reads: those with an active endpoint.
    bool IndividualFaceRead(Tessellation3D const& Tess, IndividualStepContext const& Context, std::size_t Face)
    {
        auto const Neighbors = Tess.GetFaceNeighbors(Face);
        std::size_t const Norg = Context.active_mask.size();
        return (Neighbors.first < Norg && Context.isActive(Neighbors.first)) ||
            (Neighbors.second < Norg && Context.isActive(Neighbors.second));
    }

    // The slopes the individual flux loop reads: both endpoints of a read face, except a physical boundary ghost.  They
    // include every ghost slope exchanged from a peer.
    std::vector<std::size_t> IndividualReadSlopes(Tessellation3D const& Tess, IndividualStepContext const& Context,
        std::size_t SlopeCount)
    {
        std::vector<unsigned char> Marked(SlopeCount, 0);
        std::vector<std::size_t> Indices;
        std::size_t const Owned = Tess.GetPointNo();
        for(std::size_t Face = 0; Face < Tess.GetTotalFacesNumber(); ++Face) {
            if(!IndividualFaceRead(Tess, Context, Face))
                continue;
            auto const Neighbors = Tess.GetFaceNeighbors(Face);
            bool const Boundary = Tess.BoundaryFace(Face);
            for(std::size_t const Index : {Neighbors.first, Neighbors.second}) {
                if(Index >= SlopeCount || (Boundary && Index >= Owned) || Marked[Index] != 0)
                    continue;
                Marked[Index] = 1;
                Indices.push_back(Index);
            }
        }
        return Indices;
    }

    struct IndividualFluxVerifyInput {
        int FluxMode;
        int SlopeMode;
        std::vector<Conserved3D> const* NewFluxes;
        std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const* NewFaceValues;
        std::vector<std::size_t> const* SlopeIndices;
        std::vector<Slope3D> const* NewSlopes;
        std::vector<Slope3D> const* NewNaiveSlopes;
        IndividualInterpolationCounts NewCounts;
        IndividualInterpolationCounts KeptCounts;
    };

    // Verify mode of RICH_INDIVIDUAL_FLUX_PASSES and RICH_INDIVIDUAL_GHOST_SLOPE_REQUEST: everything a reader of the event
    // sees (the read face states after the temporal predictor, every flux, every slope the flux loop reads and the
    // unlimited slopes of its owned endpoints) from the new paths against the kept paths, bit for bit.  One reduction;
    // rank 0 prints one line per switch in verify mode; on a difference the lowest rank that found one prints an example
    // and every rank throws.
    void VerifyIndividualFluxes(IndividualFluxVerifyInput const& Input, Tessellation3D const& Tess,
        IndividualStepContext const& Context, std::vector<Conserved3D> const& Fluxes,
        std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& FaceValues,
        std::vector<Slope3D> const* Slopes, std::vector<Slope3D> const* NaiveSlopes)
    {
        enum { FaceChecks, FaceMismatches, FluxChecks, FluxMismatches, SlopeChecks, SlopeMismatches, GhostSources,
            SentKept, SentNew, ReconstructedKept, ReconstructedNew, CountSize };
        unsigned long long Counts[CountSize] = {};
        std::ostringstream Example;
        Example.precision(17);
        char const* ExampleKind = "";
        std::size_t const FaceCount = Tess.GetTotalFacesNumber();
        for(std::size_t Face = 0; Face < FaceCount; ++Face) {
            if(!IndividualFaceRead(Tess, Context, Face))
                continue;
            ++Counts[FaceChecks];
            bool const Same = Face < FaceValues.size() && Face < Input.NewFaceValues->size() &&
                SameCell(FaceValues[Face].first, (*Input.NewFaceValues)[Face].first) &&
                SameCell(FaceValues[Face].second, (*Input.NewFaceValues)[Face].second);
            if(Same || Counts[FaceMismatches]++ != 0 || Counts[FluxMismatches] + Counts[SlopeMismatches] != 0)
                continue;
            ExampleKind = "face";
            Example << " face=" << Face;
            if(Face < FaceValues.size() && Face < Input.NewFaceValues->size())
                Example << " kept_first=" << FaceValues[Face].first << " new_first=" << (*Input.NewFaceValues)[Face].first
                    << " kept_second=" << FaceValues[Face].second << " new_second="
                    << (*Input.NewFaceValues)[Face].second;
        }
        for(std::size_t Face = 0; Face < FaceCount; ++Face) {
            ++Counts[FluxChecks];
            bool const Same = Face < Fluxes.size() && Face < Input.NewFluxes->size() &&
                SameConserved(Fluxes[Face], (*Input.NewFluxes)[Face]);
            if(Same || Counts[FluxMismatches]++ != 0 || Counts[FaceMismatches] + Counts[SlopeMismatches] != 0)
                continue;
            ExampleKind = "flux";
            Example << " face=" << Face << " read=" << (IndividualFaceRead(Tess, Context, Face) ? 1 : 0);
            if(Face < Fluxes.size() && Face < Input.NewFluxes->size())
                Example << " kept_mass=" << Fluxes[Face].mass << " new_mass=" << (*Input.NewFluxes)[Face].mass
                    << " kept_energy=" << Fluxes[Face].energy << " new_energy=" << (*Input.NewFluxes)[Face].energy;
        }
        std::size_t const Owned = Tess.GetPointNo();
        for(std::size_t Position = 0; Position < Input.SlopeIndices->size(); ++Position) {
            std::size_t const Index = (*Input.SlopeIndices)[Position];
            ++Counts[SlopeChecks];
            bool Same = Slopes != nullptr && Index < Slopes->size() && Position < Input.NewSlopes->size() &&
                SameSlope((*Slopes)[Index], (*Input.NewSlopes)[Position]);
            if(Same && Index < Owned && NaiveSlopes != nullptr && Input.NewNaiveSlopes != nullptr) {
                ++Counts[SlopeChecks];
                Same = Index < NaiveSlopes->size() && Position < Input.NewNaiveSlopes->size() &&
                    SameSlope((*NaiveSlopes)[Index], (*Input.NewNaiveSlopes)[Position]);
            }
            if(Same || Counts[SlopeMismatches]++ != 0 || Counts[FaceMismatches] + Counts[FluxMismatches] != 0)
                continue;
            ExampleKind = "slope";
            Example << " slope_index=" << Index << " owned=" << (Index < Owned ? 1 : 0);
            if(Slopes != nullptr && Index < Slopes->size() && Position < Input.NewSlopes->size())
                Example << " kept_x=" << (*Slopes)[Index].xderivative << " new_x="
                    << (*Input.NewSlopes)[Position].xderivative;
        }
        Counts[GhostSources] = Input.KeptCounts.GhostSources;
        Counts[SentKept] = Input.KeptCounts.SentGhostSources;
        Counts[SentNew] = Input.NewCounts.SentGhostSources;
        Counts[ReconstructedKept] = Input.KeptCounts.Reconstructed;
        Counts[ReconstructedNew] = Input.NewCounts.Reconstructed;
        unsigned long long const LocalMismatches = Counts[FaceMismatches] + Counts[FluxMismatches] +
            Counts[SlopeMismatches];
        int Rank = 0;
        int RankCount = 1;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
        MPI_Allreduce(MPI_IN_PLACE, Counts, CountSize, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        unsigned long long const Checks = Counts[FaceChecks] + Counts[FluxChecks] + Counts[SlopeChecks];
        unsigned long long const Mismatches = Counts[FaceMismatches] + Counts[FluxMismatches] + Counts[SlopeMismatches];
        if(Rank == 0 && Input.FluxMode == 2)
            std::cout << "RICH_B10_FLUX_VERIFY event_tick=" << Context.event_tick << " checks=" << Checks
                << " mismatches=" << Mismatches << " face_checks=" << Counts[FaceChecks] << " face_mismatches="
                << Counts[FaceMismatches] << " flux_checks=" << Counts[FluxChecks] << " flux_mismatches="
                << Counts[FluxMismatches] << " slope_checks=" << Counts[SlopeChecks] << " slope_mismatches="
                << Counts[SlopeMismatches] << std::endl;
        if(Rank == 0 && Input.SlopeMode == 2)
            std::cout << "RICH_B10_SLOPE_VERIFY event_tick=" << Context.event_tick << " checks=" << Checks
                << " mismatches=" << Mismatches << " slope_checks=" << Counts[SlopeChecks] << " slope_mismatches="
                << Counts[SlopeMismatches] << " ghost_sources=" << Counts[GhostSources] << " sent_full="
                << Counts[SentKept] << " sent_requested=" << Counts[SentNew] << " reconstructed_full="
                << Counts[ReconstructedKept] << " reconstructed_requested=" << Counts[ReconstructedNew] << std::endl;
        if(Mismatches == 0)
            return;
        int FirstRank = LocalMismatches != 0 ? Rank : RankCount;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &FirstRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        bool const SlopeTag = Input.FluxMode != 2 || (Input.SlopeMode == 2 && std::strcmp(ExampleKind, "slope") == 0);
        if(Rank == FirstRank)
            std::cerr << (SlopeTag ? "RICH_B10_SLOPE_VERIFY_MISMATCH" : "RICH_B10_FLUX_VERIFY_MISMATCH") << " rank="
                << Rank << " event_tick=" << Context.event_tick << " local=" << LocalMismatches << " kind="
                << ExampleKind << Example.str() << std::endl;
        throw std::runtime_error(std::string(SlopeTag ? "RICH_INDIVIDUAL_GHOST_SLOPE_REQUEST" :
            "RICH_INDIVIDUAL_FLUX_PASSES") + "=2: the new individual flux paths differ from the kept paths (" +
            std::to_string(Mismatches) + " mismatches)");
    }
}

void ConditionActionFlux1::CalculateIndividual(
	vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const vector<Vector3D>& face_velocities,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& /*extensives*/,
	const EquationOfState& eos,
	const IndividualStepContext& context,
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& face_values) const
{
    // RICH_INDIVIDUAL_FLUX_PASSES and RICH_INDIVIDUAL_GHOST_SLOPE_REQUEST: agreed on every rank (collective on first use,
    // which is this call, made by every rank in every individual event).
    int const FluxMode = IndividualFluxPassesMode();
    int const SlopeMode = IndividualGhostSlopeRequestMode();
    if(FluxMode != 2 && SlopeMode != 2) {
        CalculateIndividualPass(fluxes, tess, face_velocities, cells, eos, context, face_values, FluxMode == 1,
            SlopeMode == 1);
        return;
    }
    // Verify: the new paths first, on a copy of the incoming flux array and on a face-state buffer that keeps the
    // previous event's entries; then the configured paths, whose results are kept and leave the reconstruction in the
    // state those paths alone leave it (they recompute every slope the new paths wrote).
    IndividualFluxVerifyInput Input = {FluxMode, SlopeMode, nullptr, nullptr, nullptr, nullptr, nullptr,
        IndividualInterpolationCounts(), IndividualInterpolationCounts()};
    std::vector<Conserved3D> NewFluxes(fluxes);
    CalculateIndividualPass(NewFluxes, tess, face_velocities, cells, eos, context, VerifyFaceValues_, FluxMode >= 1,
        SlopeMode >= 1);
    LinearGauss3D const* const Linear = dynamic_cast<LinearGauss3D const*>(&interp_);
    if(Linear != nullptr)
        Input.NewCounts = Linear->GetLastIndividualCounts();
    std::vector<Slope3D> const* const Slopes = interp_.GetSlopesForTimePrediction();
    std::vector<std::size_t> const SlopeIndices = Slopes != nullptr ?
        IndividualReadSlopes(tess, context, Slopes->size()) : std::vector<std::size_t>();
    std::vector<Slope3D> NewSlopes;
    std::vector<Slope3D> NewNaiveSlopes;
    NewSlopes.reserve(SlopeIndices.size());
    for(std::size_t const Index : SlopeIndices)
        NewSlopes.push_back((*Slopes)[Index]);
    std::vector<Slope3D> const* const NaiveSlopes = Linear != nullptr ? &Linear->GetSlopesUnlimited() : nullptr;
    if(NaiveSlopes != nullptr) {
        NewNaiveSlopes.resize(SlopeIndices.size());
        for(std::size_t Position = 0; Position < SlopeIndices.size(); ++Position)
            if(SlopeIndices[Position] < NaiveSlopes->size())
                NewNaiveSlopes[Position] = (*NaiveSlopes)[SlopeIndices[Position]];
    }
    CalculateIndividualPass(fluxes, tess, face_velocities, cells, eos, context, face_values, FluxMode == 1,
        SlopeMode == 1);
    if(Linear != nullptr)
        Input.KeptCounts = Linear->GetLastIndividualCounts();
    Input.NewFluxes = &NewFluxes;
    Input.NewFaceValues = &VerifyFaceValues_;
    Input.SlopeIndices = &SlopeIndices;
    Input.NewSlopes = &NewSlopes;
    Input.NewNaiveSlopes = NaiveSlopes != nullptr ? &NewNaiveSlopes : nullptr;
    VerifyIndividualFluxes(Input, tess, context, fluxes, face_values, interp_.GetSlopesForTimePrediction(),
        NaiveSlopes);
}

void ConditionActionFlux1::CalculateIndividualPass(vector<Conserved3D>& fluxes, const Tessellation3D& tess,
    const vector<Vector3D>& face_velocities, const vector<ComputationalCell3D>& cells, const EquationOfState& eos,
    const IndividualStepContext& context,
    std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& face_values, bool FluxPasses,
    bool RequestSlopes) const
{
	for(std::size_t i = 0; i < sequence_.size(); ++i)
		sequence_[i].second->Reset();
    // The new paths need LinearGauss3D (the same type on every rank); another reconstruction takes today's path, which
    // starts from a cleared face-state array.
    LinearGauss3D const* const Linear = FluxPasses || RequestSlopes ?
        dynamic_cast<LinearGauss3D const*>(&interp_) : nullptr;
    if(Linear == nullptr || !FluxPasses)
        face_values.clear();
    if(Linear != nullptr) {
        IndividualInterpolationOptions Options;
        Options.KeepInactiveFaceStates = FluxPasses;
        Options.ReadCellsInPlace = FluxPasses;
        Options.RequestGhostSlopes = RequestSlopes;
        Linear->InterpolateIndividualWithOptions(tess, cells, context.event_time, context.active_mask, face_values,
            Options);
    }
    else
        interp_.InterpolateIndividual(tess, cells, context.event_time, context.active_mask, face_values);
    // RICH_INDIVIDUAL_FLUX_PASSES: the hydro step assigns this array of the event mesh with the same values just
    // before the call (hdsim_3d.cpp, phase 7), and nothing writes it in between.
    if(!FluxPasses || fluxes.size() != tess.GetTotalFacesNumber()) {
        fluxes.assign(tess.GetTotalFacesNumber(), Conserved3D());
    }
	const std::vector<Slope3D>* slopes = interp_.GetSlopesForTimePrediction();
	const std::size_t norg = context.active_mask.size();
	const std::size_t owned_cell_count = tess.GetPointNo();

	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		const bool first_active = neighbors.first < norg && context.isActive(neighbors.first);
		const bool second_active = neighbors.second < norg && context.isActive(neighbors.second);
		if(!first_active && !second_active)
			continue;

		double face_dt = 0;
		if(neighbors.first < norg && neighbors.second < norg)
			face_dt = context.hydroFaceTimeStep(neighbors.first, neighbors.second);
		else if(neighbors.first < norg)
			face_dt = context.cellTimeStep(neighbors.first);
		else
			face_dt = context.cellTimeStep(neighbors.second);

		if(slopes)
		{
			const bool boundary_face = tess.BoundaryFace(face);
			const double midpoint_time = context.event_time - 0.5 * face_dt;
			auto prediction_offset = [&](std::size_t index,
				std::size_t fallback_index)
			{
				std::size_t const source = index < norg ? index : fallback_index;
				if(source < context.primitive_ticks.size())
					return midpoint_time - (context.time_origin +
						context.time_quantum *
						static_cast<double>(context.primitive_ticks[source]));
				return 0.5 * face_dt;
			};
			auto prediction_acceleration = [&](std::size_t index,
				std::size_t fallback_index)
			{
				std::size_t const source = index < norg ? index : fallback_index;
				return source < context.cached_accelerations.size()
					? context.cached_accelerations[source]
					: Vector3D();
			};
			// Physical boundary ghosts have no cell-centred primitive state.
			// Their face state is supplied by the boundary reconstruction/action.
			if(neighbors.first < slopes->size() &&
				(!boundary_face || neighbors.first < owned_cell_count))
				predict_primitive(face_values[face].first, cells[neighbors.first],
					(*slopes)[neighbors.first], eos, face_velocities[face],
					prediction_acceleration(neighbors.first, neighbors.second),
					prediction_offset(neighbors.first, neighbors.second));
			if(neighbors.second < slopes->size() &&
				(!boundary_face || neighbors.second < owned_cell_count))
				predict_primitive(face_values[face].second, cells[neighbors.second],
					(*slopes)[neighbors.second], eos, face_velocities[face],
					prediction_acceleration(neighbors.second, neighbors.first),
					prediction_offset(neighbors.second, neighbors.first));
		}

		if(!(face_values[face].first.density > 0) ||
			!(face_values[face].first.pressure > 0) ||
			!(face_values[face].first.internal_energy > 0) ||
			!(face_values[face].second.density > 0) ||
			!(face_values[face].second.pressure > 0) ||
			!(face_values[face].second.internal_energy > 0) ||
			!std::isfinite(face_values[face].first.density) ||
			!std::isfinite(face_values[face].first.pressure) ||
			!std::isfinite(face_values[face].first.internal_energy) ||
			!std::isfinite(face_values[face].second.density) ||
			!std::isfinite(face_values[face].second.pressure) ||
			!std::isfinite(face_values[face].second.internal_energy))
			throw UniversalError("Bad input to individual flux calculator");
		choose_action(face, tess, cells, eos, face_velocities[face], sequence_,
			fluxes[face], context.event_time - 0.5 * face_dt, face_values[face]);
	}
}

std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > ConditionActionFlux1::operator()(vector<Conserved3D> &fluxes, const Tessellation3D& tess, const vector<Vector3D>& face_velocities,
													   const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& /*extensives*/, const EquationOfState& eos,
	const double time, const double /*dt*/) const
{
	vector<std::pair<ComputationalCell3D, ComputationalCell3D> > face_values;
	this->Calculate(fluxes, tess, face_velocities, cells, vector<Conserved3D>(), eos, time, 0, face_values);
	return face_values;
}

void ConditionActionFlux1::Calculate(vector<Conserved3D> &fluxes, const Tessellation3D& tess, const vector<Vector3D>& face_velocities,
													   const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& /*extensives*/, const EquationOfState& eos,
	const double time, const double /*dt*/, std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > &face_values) const
{
	for (size_t i = 0; i < sequence_.size(); ++i)
		sequence_[i].second->Reset();
	face_values.clear();
	interp_(tess, cells, time, face_values);
	fluxes.resize(tess.GetTotalFacesNumber());
	size_t Nloop = fluxes.size();
	for (size_t i = 0; i < Nloop; ++i)
	{
		if (face_values[i].first.density < 0 ||  face_values[i].first.internal_energy < 0
			|| face_values[i].second.density < 0  || face_values[i].second.internal_energy < 0)
		{
			UniversalError eo("Bad input to flux calculator");
			eo.addEntry("Face", static_cast<double>(i));
			eo.addEntry("Face neigh 0", static_cast<double>(tess.GetFaceNeighbors(i).first));
			eo.addEntry("Face neigh 1", static_cast<double>(tess.GetFaceNeighbors(i).second));
			eo.addEntry("First input Density", face_values[i].first.density);
			eo.addEntry("First input pressure", face_values[i].first.pressure);
			eo.addEntry("First input internal energy", face_values[i].first.internal_energy);
			eo.addEntry("First input vx", face_values[i].first.velocity.x);
			eo.addEntry("First input vy", face_values[i].first.velocity.y);
			eo.addEntry("First input vz", face_values[i].first.velocity.z);
			eo.addEntry("Second input Density", face_values[i].second.density);
			eo.addEntry("Second input pressure", face_values[i].second.pressure);
			eo.addEntry("Second input internal energy", face_values[i].second.internal_energy);
			eo.addEntry("Second input vx", face_values[i].second.velocity.x);
			eo.addEntry("Second input vy", face_values[i].second.velocity.y);
			eo.addEntry("Second input vz", face_values[i].second.velocity.z);
			eo.addEntry("First cell Density", cells[tess.GetFaceNeighbors(i).first].density);
			eo.addEntry("First cell pressure", cells[tess.GetFaceNeighbors(i).first].pressure);
			eo.addEntry("First cell internal energy", cells[tess.GetFaceNeighbors(i).first].internal_energy);
			eo.addEntry("First cell vx", cells[tess.GetFaceNeighbors(i).first].velocity.x);
			eo.addEntry("First cell vy", cells[tess.GetFaceNeighbors(i).first].velocity.y);
			eo.addEntry("First cell vz", cells[tess.GetFaceNeighbors(i).first].velocity.z);
			eo.addEntry("Second cell Density", cells[tess.GetFaceNeighbors(i).second].density);
			eo.addEntry("Second cell pressure", cells[tess.GetFaceNeighbors(i).second].pressure);
			eo.addEntry("Second cell internal energy", cells[tess.GetFaceNeighbors(i).second].internal_energy);
			eo.addEntry("Second cell vx", cells[tess.GetFaceNeighbors(i).second].velocity.x);
			eo.addEntry("Second cell vy", cells[tess.GetFaceNeighbors(i).second].velocity.y);
			eo.addEntry("Second cell vz", cells[tess.GetFaceNeighbors(i).second].velocity.z);
			eo.addEntry("Face vx", face_velocities[i].x);
			eo.addEntry("Face vy", face_velocities[i].y);
			eo.addEntry("Face vz", face_velocities[i].z);
			throw eo;
		}
		choose_action(i, tess, cells, eos, face_velocities[i], sequence_, fluxes[i], time, face_values[i]);
	}
}

ConditionActionFlux1::Condition3D::~Condition3D(void) {}

ConditionActionFlux1::Action3D::~Action3D(void) {}

RegularFlux3D::RegularFlux3D(const RiemannSolver3D& rs) :
	rs_(rs) {}

void RegularFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool /*aux*/, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs_, res, eos);
}

RigidWallFlux3D::RigidWallFlux3D(const RiemannSolver3D& rs) : rs_(rs) {}

namespace
{
	void reflect(ComputationalCell3D & cell, Vector3D const& normal)
	{
		cell.velocity -= 2 * ScalarProd(cell.velocity, normal)*normal;
	}

	void rigid_wall_states(std::pair<ComputationalCell3D, ComputationalCell3D> &res,
		Vector3D const& normal, const bool aux)
	{
		if (aux)
		{
			res.second = res.first;
			reflect(res.second, normal);
		}
		else
		{
			res.first = res.second;
			reflect(res.first, normal);
		}
	}
}

void RigidWallFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool aux, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	std::pair<ComputationalCell3D, ComputationalCell3D>	rigid_states(face_values);
	rigid_wall_states(rigid_states, normal, aux);
	RotateSolveBack3D(normal, rigid_states.first, rigid_states.second, face_velocity, rs_, res, eos);
}

FreeFlowFlux3D::FreeFlowFlux3D(const RiemannSolver3D& rs) : rs_(rs) {}

void FreeFlowFlux3D::operator()(size_t face_index, const Tessellation3D& tess, const Vector3D& face_velocity,
	const vector<ComputationalCell3D>& /*cells*/, const EquationOfState& eos, const bool aux, Conserved3D &res,
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D>
	const& face_values) const
{
	const Vector3D normal = normalize(tess.Normal(face_index));
	std::pair<ComputationalCell3D, ComputationalCell3D>	states(face_values);
	if (aux)
		states.second = states.first;
	else
		states.first = states.second;
	RotateSolveBack3D(normal, states.first, states.second, face_velocity, rs_, res, eos);
}

IsBoundaryFace3D::IsBoundaryFace3D(void) {}

pair<bool, bool> IsBoundaryFace3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& /*cells*/) const
{
	if (!tess.BoundaryFace(face_index))
		return pair<bool, bool>(false, false);
	if (tess.GetFaceNeighbors(face_index).first > tess.GetPointNo())
		return pair<bool, bool>(true, false);
	else
		return pair<bool, bool>(true, true);
}

IsBulkFace3D::IsBulkFace3D(void) {}

pair<bool, bool> IsBulkFace3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& /*cells*/)const
{
	if (tess.BoundaryFace(face_index))
		return pair<bool, bool>(false, false);
	else
		return pair<bool, bool>(true, false);
}

RegularSpecialEdge3D::RegularSpecialEdge3D(const string& sticker_name) :
	sticker_name_(sticker_name) {}

pair<bool, bool> RegularSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess,
	const vector<ComputationalCell3D>& cells)const
{
  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
		     ComputationalCell3D::stickerNames.end(), sticker_name_))
	{
	  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
			     ComputationalCell3D::stickerNames.end(), sticker_name_))
			return pair<bool, bool>(false, false);
		return pair<bool, bool>(true, false);
	}
  if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), ComputationalCell3D::stickerNames.begin(),
		     ComputationalCell3D::stickerNames.end(), sticker_name_))
		return pair<bool, bool>(true, true);
	return pair<bool, bool>(false, false);
}

LagrangianFlux3D::LagrangianFlux3D(const LagrangianHLLC3D & rs, const LagrangianHLLC3D & rs2, 
	LagrangianCriteria3D const & criteria):ws_(std::vector<double>()), edge_vel_(std::vector<double>()),
	Lag_calc_(std::vector<bool>()), rs_(rs), rs2_(rs2), criteria_(criteria) {}

void LagrangianFlux3D::operator()(size_t face_index, const Tessellation3D & tess, const Vector3D & face_velocity, 
	const vector<ComputationalCell3D>& cells, const EquationOfState & eos, const bool aux, Conserved3D & res, double time, 
	 std::pair<ComputationalCell3D, ComputationalCell3D> const & face_values) const
{
	size_t N = tess.GetTotalFacesNumber();
	ws_.resize(N, 0.0);
	edge_vel_.resize(N, 0.0);
	Lag_calc_.resize(N, false);
	const Vector3D normal = normalize(tess.Normal(face_index));
	if (criteria_(face_index, tess, face_velocity, cells, eos, aux, face_values, time))
	{
		RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs_, res, eos);
		ws_[face_index] = rs_.ws;
		Lag_calc_[face_index] = true;
	}
	else
	{
		RotateSolveBack3D(normal, face_values.first, face_values.second, face_velocity, rs2_, res, eos);
		ws_[face_index] = 0;
		Lag_calc_[face_index] = false;
	}
	edge_vel_[face_index] = ScalarProd(normal, face_velocity);
}

void LagrangianFlux3D::Reset(void) const
{
	ws_.assign(ws_.size(), 0);
	edge_vel_.assign(edge_vel_.size(), 0);
	Lag_calc_.assign(Lag_calc_.size(), false);
}

LagrangianFlux3D::LagrangianCriteria3D::~LagrangianCriteria3D(){}

BothSpecialEdge3D::BothSpecialEdge3D(const string& sticker_name):sticker_name_(sticker_name)
{}

pair<bool, bool> BothSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const
{
	if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
		ComputationalCell3D::stickerNames.begin(),
		ComputationalCell3D::stickerNames.end(), sticker_name_) &&
		*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name_))
			return pair<bool, bool>(true, false);
	else
		return pair<bool, bool>(false, false);
}

TwoSpecialEdge3D::TwoSpecialEdge3D(const string& sticker_name1, const string& sticker_name2):
	sticker_name1_(sticker_name1), sticker_name2_(sticker_name2){}

pair<bool, bool> TwoSpecialEdge3D::operator()(size_t face_index, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells) const
{
	if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name1_) &&
		*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
			ComputationalCell3D::stickerNames.begin(),
			ComputationalCell3D::stickerNames.end(), sticker_name2_))
		return pair<bool, bool>(true, false);
	else
		if (*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).first).stickers.begin(), 
				ComputationalCell3D::stickerNames.begin(),
				ComputationalCell3D::stickerNames.end(), sticker_name2_) &&
			*safe_retrieve(cells.at(tess.GetFaceNeighbors(face_index).second).stickers.begin(), 
				ComputationalCell3D::stickerNames.begin(),
				ComputationalCell3D::stickerNames.end(), sticker_name1_))
			return pair<bool, bool>(true, true);
		else
			return pair<bool, bool>(false, false);
}

void ZeroFlux3D::operator()(size_t /*face_index*/, const Tessellation3D& /*tess*/, 
	const Vector3D& /*face_velocity*/, const vector<ComputationalCell3D>& 
	/*cells*/, const EquationOfState& /*eos*/, const bool /*aux*/, Conserved3D& res, 
	double /*time*/, std::pair<ComputationalCell3D, ComputationalCell3D> const& /*face_values*/) const
{
	res.energy = 0;
	res.internal_energy = 0;
	res.mass = 0;
	res.momentum = Vector3D();
	size_t const Ntracers = ComputationalCell3D::tracerNames.size();
	for (size_t i = 0; i < Ntracers; ++i)
		res.tracers[i] = 0;
}
