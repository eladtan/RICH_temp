#include "default_extensive_updater.hpp"
#include "newtonian/three_dimensional/simulation/ActiveMeshView.hpp"
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unistd.h>

#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
bool write_all(int descriptor, const std::string& message)
{
	std::size_t offset = 0;
	while(offset < message.size())
	{
		const ssize_t written = ::write(descriptor,
			message.data() + offset, message.size() - offset);
		if(written > 0)
		{
			offset += static_cast<std::size_t>(written);
			continue;
		}
		if(written < 0 && errno == EINTR)
			continue;
		if(written == 0)
			errno = EIO;
		return false;
	}
	return true;
}

struct PendingIndividualHydroDelta
{
	std::size_t cell_index = 0;
	std::size_t counterpart_cell_id = 0;
	int counterpart_owner = -1;
	Conserved3D delta;
};

#ifdef RICH_MPI
struct IndividualHydroDelta : public Serializable
{
	size_t cell_id = 0;
	size_t counterpart_cell_id = 0;
	int counterpart_owner = -1;
	Conserved3D delta;

	force_inline size_t dump(Serializer* serializer) const override
	{
		size_t bytes = 0;
		bytes += serializer->insert(cell_id);
		bytes += serializer->insert(counterpart_cell_id);
		bytes += serializer->insert(counterpart_owner);
		bytes += serializer->insert(delta);
		return bytes;
	}

	force_inline size_t load(const Serializer* serializer, size_t byte_offset) override
	{
		size_t bytes = 0;
		bytes += serializer->extract(cell_id, byte_offset);
		bytes += serializer->extract(counterpart_cell_id, byte_offset + bytes);
		bytes += serializer->extract(counterpart_owner, byte_offset + bytes);
		bytes += serializer->extract(delta, byte_offset + bytes);
		return bytes;
	}
};

struct IndividualHydroScale : public Serializable
{
	size_t cell_id = 0;
	size_t counterpart_cell_id = 0;
	double scale = 1;

	force_inline size_t dump(Serializer* serializer) const override
	{
		size_t bytes = 0;
		bytes += serializer->insert(cell_id);
		bytes += serializer->insert(counterpart_cell_id);
		bytes += serializer->insert(scale);
		return bytes;
	}

	force_inline size_t load(const Serializer* serializer,
		size_t byte_offset) override
	{
		size_t bytes = 0;
		bytes += serializer->extract(cell_id, byte_offset);
		bytes += serializer->extract(counterpart_cell_id, byte_offset + bytes);
		bytes += serializer->extract(scale, byte_offset + bytes);
		return bytes;
	}
};

struct StableFaceEndpoint
{
	size_t cell_id = 0;
	size_t counterpart_cell_id = 0;

	bool operator==(StableFaceEndpoint const& other) const
	{
		return cell_id == other.cell_id &&
			counterpart_cell_id == other.counterpart_cell_id;
	}
};

struct StableFaceEndpointHash
{
	size_t operator()(StableFaceEndpoint const& endpoint) const
	{
		size_t const first = std::hash<size_t>()(endpoint.cell_id);
		size_t const second =
			std::hash<size_t>()(endpoint.counterpart_cell_id);
		return first ^ (second + 0x9e3779b9 + (first << 6) + (first >> 2));
	}
};

// B9: the all-to-all over the duplicated-point peers only.  Outgoing and the result are indexed by rank, as for
// MPI_Exchange_all_to_all, and the serialization is the same, so callers see the same packets in the same order.
// Lost is set when a packet is addressed to a rank that is not a peer (the caller folds it into its error reduction).
template <class T>
std::vector<std::vector<T> > ExchangeWithDuplicatedPeers(std::vector<std::vector<T> > const& Outgoing,
    std::vector<int> const& Peers, bool& Lost)
{
    std::vector<unsigned char> IsPeer(Outgoing.size(), 0);
    std::vector<std::vector<T> > PeerData(Peers.size());
    for(std::size_t Index = 0; Index < Peers.size(); ++Index) {
        std::size_t const Peer = static_cast<std::size_t>(Peers[Index]);
        if(Peer < Outgoing.size()) {
            IsPeer[Peer] = 1;
            PeerData[Index] = Outgoing[Peer];
        }
    }
    for(std::size_t Rank = 0; Rank < Outgoing.size(); ++Rank)
        if(IsPeer[Rank] == 0 && !Outgoing[Rank].empty())
            Lost = true;
    std::vector<std::vector<T> > Received = MPI_exchange_data(Peers, PeerData);
    std::vector<std::vector<T> > Incoming(Outgoing.size());
    for(std::size_t Index = 0; Index < Peers.size() && Index < Received.size(); ++Index)
        if(static_cast<std::size_t>(Peers[Index]) < Incoming.size())
            Incoming[static_cast<std::size_t>(Peers[Index])] = std::move(Received[Index]);
    return Incoming;
}

bool SameDoubleBits(double First, double Second)
{
    return std::memcmp(&First, &Second, sizeof(double)) == 0;
}

// True when every conserved component of Own is the exact negation of Other's (thermal energy excluded: each side
// splits it with its own cell velocity).
bool OppositeConserved(Conserved3D const& Own, Conserved3D const& Other)
{
    bool Opposite = SameDoubleBits(Own.mass, -Other.mass) && SameDoubleBits(Own.energy, -Other.energy) &&
        SameDoubleBits(Own.momentum.x, -Other.momentum.x) && SameDoubleBits(Own.momentum.y, -Other.momentum.y) &&
        SameDoubleBits(Own.momentum.z, -Other.momentum.z) && SameDoubleBits(Own.Erad, -Other.Erad) &&
        SameDoubleBits(Own.Erad_dt, -Other.Erad_dt) && SameDoubleBits(Own.Erad_dt_dt, -Other.Erad_dt_dt) &&
        Own.Eg.size() == Other.Eg.size();
    for(std::size_t Group = 0; Opposite && Group < Own.Eg.size(); ++Group)
        Opposite = SameDoubleBits(Own.Eg[Group], -Other.Eg[Group]);
    for(std::size_t Tracer = 0; Opposite && Tracer < Own.tracers.size(); ++Tracer)
        Opposite = SameDoubleBits(Own.tracers[Tracer], -Other.tracers[Tracer]);
    return Opposite;
}

// B9 mode 2: cross-rank face conservation at 0 ULP.  After the radiation scales are applied, each rank sends every
// delta whose counterpart cell lives on another rank to that rank, which finds its own delta of the same face (stable
// endpoint IDs) and requires exactly opposite conserved components.  Missing or repeated partners count as
// mismatches.  One peer exchange and one reduction; rank 0 prints RICH_B9_CONSERVATION; on a mismatch the lowest
// rank that found one prints an example and every rank throws.
void VerifyCrossRankFaceConservation(std::vector<PendingIndividualHydroDelta> const& Deltas,
    std::vector<std::size_t> const& WorkCellIds, int Rank, int RankCount, std::vector<int> const& Peers,
    std::uint64_t EventTick)
{
    std::vector<std::vector<IndividualHydroDelta> > Outgoing(static_cast<std::size_t>(RankCount));
    std::unordered_map<StableFaceEndpoint, std::size_t, StableFaceEndpointHash> OwnByEndpoint;
    unsigned long long Counts[2] = {0, 0};
    std::ostringstream Example;
    Example << std::setprecision(17);
    auto const NoteMismatch = [&](char const* Kind, std::size_t CellId, std::size_t CounterpartId)
    {
        if(Counts[1]++ == 0)
            Example << " kind=" << Kind << " cell_id=" << CellId << " counterpart_id=" << CounterpartId;
    };
    for(std::size_t Index = 0; Index < Deltas.size(); ++Index) {
        PendingIndividualHydroDelta const& Delta = Deltas[Index];
        if(Delta.counterpart_owner < 0 || Delta.counterpart_owner == Rank)
            continue;
        std::size_t const CellId = WorkCellIds[Delta.cell_index];
        if(Delta.counterpart_owner >= RankCount) {
            NoteMismatch("bad_owner", CellId, Delta.counterpart_cell_id);
            continue;
        }
        if(!OwnByEndpoint.emplace(StableFaceEndpoint{CellId, Delta.counterpart_cell_id}, Index).second)
            NoteMismatch("repeated_face", CellId, Delta.counterpart_cell_id);
        IndividualHydroDelta Packet;
        Packet.cell_id = Delta.counterpart_cell_id;
        Packet.counterpart_cell_id = CellId;
        Packet.counterpart_owner = Rank;
        Packet.delta = Delta.delta;
        Outgoing[static_cast<std::size_t>(Delta.counterpart_owner)].push_back(Packet);
    }
    bool Lost = false;
    std::vector<std::vector<IndividualHydroDelta> > const Incoming = ExchangeWithDuplicatedPeers(Outgoing, Peers, Lost);
    if(Lost)
        NoteMismatch("lost_packet", 0, 0);
    std::size_t Matched = 0;
    for(std::size_t Source = 0; Source < Incoming.size(); ++Source)
        for(IndividualHydroDelta const& Packet : Incoming[Source]) {
            ++Counts[0];
            auto const Found = OwnByEndpoint.find(StableFaceEndpoint{Packet.cell_id, Packet.counterpart_cell_id});
            if(Found == OwnByEndpoint.end() ||
                Deltas[Found->second].counterpart_owner != static_cast<int>(Source)) {
                NoteMismatch("unmatched", Packet.cell_id, Packet.counterpart_cell_id);
                continue;
            }
            ++Matched;
            if(!OppositeConserved(Deltas[Found->second].delta, Packet.delta)) {
                if(Counts[1] == 0)
                    Example << " own_mass=" << Deltas[Found->second].delta.mass << " other_mass="
                        << Packet.delta.mass;
                NoteMismatch("not_opposite", Packet.cell_id, Packet.counterpart_cell_id);
            }
        }
    if(Matched < OwnByEndpoint.size()) {
        NoteMismatch("unanswered", 0, 0);
        Counts[1] += OwnByEndpoint.size() - Matched - 1;
    }
    unsigned long long const LocalMismatches = Counts[1];
    MPI_Allreduce(MPI_IN_PLACE, Counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if(Rank == 0)
        std::cout << "RICH_B9_CONSERVATION event_tick=" << EventTick << " checked=" << Counts[0] << " mismatches="
            << Counts[1] << std::endl;
    if(Counts[1] == 0)
        return;
    int FirstRank = LocalMismatches != 0 ? Rank : RankCount;
    MPI_Allreduce(MPI_IN_PLACE, &FirstRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if(Rank == FirstRank)
        std::cerr << "RICH_B9_CONSERVATION_MISMATCH event_tick=" << EventTick << " rank=" << Rank << " local="
            << LocalMismatches << Example.str() << std::endl;
    throw std::runtime_error("RICH_INDIVIDUAL_ALL_ACTIVE_IDENTITY=2: cross-rank face deltas are not exactly opposite (" +
        std::to_string(Counts[1]) + " mismatches)");
}

#endif

void report_applied_individual_hydro_deltas(
	const std::vector<PendingIndividualHydroDelta>& deltas,
	std::size_t locally_constructed_delta_count,
	const vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const IndividualStepContext& context,
	const vector<ComputationalCell3D>& cells,
	const vector<Conserved3D>& extensives,
	const std::vector<std::size_t>& work_cell_ids,
	const vector<Vector3D>& face_velocities,
	const vector<Vector3D>& point_velocities,
	const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& face_states,
	std::size_t invalid_index,
	const char* phase,
	int rank)
{
	const std::size_t cell_id = work_cell_ids[invalid_index];
	const std::string record_id = std::to_string(rank) + ":" +
		std::to_string(context.event_tick) + ":" + std::to_string(cell_id);
	double applied_mass_sum = 0;
	std::size_t correction_count = 0;
	std::size_t local_correction_count = 0;
	for(std::size_t i = 0; i < deltas.size(); ++i)
		if(deltas[i].cell_index == invalid_index)
		{
			applied_mass_sum += deltas[i].delta.mass;
			++correction_count;
			if(i < locally_constructed_delta_count)
				++local_correction_count;
		}

	const double post_mass = extensives[invalid_index].mass;
	const double reconstructed_pre_mass = post_mass - applied_mass_sum;
	std::ostringstream record;
	record << std::setprecision(17)
		<< "INDIVIDUAL_HYDRO_INVALID_MASS_APPLIED"
		<< " record_id=" << record_id
		<< " rank=" << rank
		<< " work_index=" << invalid_index
		<< " cell_id=" << cell_id
		<< " phase=" << phase
		<< " active=" <<
			(invalid_index < context.active_mask.size() &&
			 context.isActive(invalid_index) ? 1 : 0)
		<< " previous_event_tick=" << context.previous_event_tick
		<< " event_tick=" << context.event_tick
		<< " previous_event_time=" << context.previous_event_time
		<< " event_time=" << context.event_time;
	if(invalid_index < context.cell_time_steps.size())
		record << " cell_dt=" << context.cellTimeStep(invalid_index);
	if(invalid_index < context.cell_time_bins.size())
		record << " cell_bin=" << static_cast<unsigned>(context.cell_time_bins[invalid_index]);
	if(invalid_index < context.primitive_ticks.size())
		record << " primitive_tick=" << context.primitive_ticks[invalid_index];
	record << " reconstructed_pre_mass=" << reconstructed_pre_mass
		<< " post_mass=" << post_mass
		<< " applied_mass_sum=" << applied_mass_sum
		<< " correction_count=" << correction_count
		<< " local_correction_count=" << local_correction_count
		<< " received_correction_count=" <<
			correction_count - local_correction_count << '\n';

	double replayed_mass = reconstructed_pre_mass;
	std::size_t correction_index = 0;
	for(std::size_t i = 0; i < deltas.size(); ++i)
	{
		const PendingIndividualHydroDelta& pending = deltas[i];
		if(pending.cell_index != invalid_index)
			continue;
		const double mass_before = replayed_mass;
		replayed_mass += pending.delta.mass;
		record << std::setprecision(17)
			<< "INDIVIDUAL_HYDRO_INVALID_MASS_DELTA"
			<< " record_id=" << record_id
			<< " correction_index=" << correction_index
			<< " origin=" <<
				(i < locally_constructed_delta_count ? "local" : "received")
			<< " target_work_index=" << pending.cell_index
			<< " target_cell_id=" << cell_id
			<< " counterpart_cell_id=" << pending.counterpart_cell_id
			<< " counterpart_owner=" << pending.counterpart_owner
			<< " running_mass_before=" << mass_before
			<< " delta_mass=" << pending.delta.mass
			<< " running_mass_after=" << replayed_mass
			<< " delta=" << pending.delta;

		bool flux_found = false;
		if(i < locally_constructed_delta_count &&
			pending.counterpart_owner >= 0)
			for(std::size_t face = 0;
				face < fluxes.size() && face < tess.GetTotalFacesNumber(); ++face)
			{
				const auto neighbors = tess.GetFaceNeighbors(face);
				if(neighbors.first >= cells.size() ||
					neighbors.second >= cells.size())
					continue;
				const std::size_t first_id = cells[neighbors.first].ID;
				const std::size_t second_id = cells[neighbors.second].ID;
				if(!((first_id == cell_id &&
						second_id == pending.counterpart_cell_id) ||
					(second_id == cell_id &&
						first_id == pending.counterpart_cell_id)))
					continue;
				double face_dt = 0;
				const bool first_physical =
					neighbors.first < context.cell_time_steps.size() &&
					!tess.IsPointOutsideBox(neighbors.first);
				const bool second_physical =
					neighbors.second < context.cell_time_steps.size() &&
					!tess.IsPointOutsideBox(neighbors.second);
				if(first_physical && second_physical)
					face_dt = context.hydroFaceTimeStep(
						neighbors.first, neighbors.second);
				else if(first_physical)
					face_dt = context.cellTimeStep(neighbors.first);
				else if(second_physical)
					face_dt = context.cellTimeStep(neighbors.second);
				record << " flux_available=1"
					<< " face=" << face
					<< " area=" << tess.GetArea(face)
					<< " face_dt=" << face_dt
					<< " dt_area=" << face_dt * tess.GetArea(face)
					<< " flux=" << fluxes[face];
				record << " face_velocity_available=" << (face < face_velocities.size());
				if(face < face_velocities.size())
					record << " face_velocity=" << face_velocities[face];
				if(neighbors.first < point_velocities.size())
					record << " left_point_velocity=" << point_velocities[neighbors.first];
				if(neighbors.second < point_velocities.size())
					record << " right_point_velocity=" << point_velocities[neighbors.second];
				record << " reconstructed_states_available=" << (face < face_states.size());
				if(face < face_states.size())
				{
					auto const& left = face_states[face].first;
					auto const& right = face_states[face].second;
					record << " left_density=" << left.density
						<< " left_pressure=" << left.pressure
						<< " left_internal_energy=" << left.internal_energy
						<< " left_velocity=" << left.velocity
						<< " right_density=" << right.density
						<< " right_pressure=" << right.pressure
						<< " right_internal_energy=" << right.internal_energy
						<< " right_velocity=" << right.velocity;
				}
				flux_found = true;
				break;
			}
		if(!flux_found)
			record << " flux_available=0 flux_unavailable_reason=" <<
				(i >= locally_constructed_delta_count ?
				 "received_packet_does_not_retain_sender_flux" :
				 "no_physical_counterpart");
		record << '\n';
		++correction_index;
	}
	record << std::setprecision(17)
		<< "INDIVIDUAL_HYDRO_INVALID_MASS_APPLIED_SUMMARY"
		<< " record_id=" << record_id
		<< " replayed_post_mass=" << replayed_mass
		<< " actual_post_mass=" << post_mass << '\n';
	PersistIndividualHydroDiagnosticRecord(record_id, record.str());
}
}

namespace
{
	bool IndividualDualEnergyTraceEnabled()
	{
		static bool const enabled = []()
		{
			char const* const value = std::getenv("RICH_INDIVIDUAL_DUAL_ENERGY_TRACE");
			return value != nullptr && value[0] == '1';
		}();
		return enabled;
	}

    // A strict switch: unset or empty is 0, otherwise exactly one digit from 0 to Highest.  Agreed in one reduction;
    // every rank throws when any rank holds another value or the ranks differ.
    int AgreedHydroStrictSwitch(char const* Name, int Highest)
    {
        char const* const Value = std::getenv(Name);
        std::string const Text = Value == nullptr ? "" : Value;
        int Local = -1;
        if(Text.empty())
            Local = 0;
        else if(Text.size() == 1 && Text[0] >= '0' && Text[0] <= static_cast<char>('0' + Highest))
            Local = Text[0] - '0';
        int Extrema[2] = {Local, -Local};
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, Extrema, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
        if(Extrema[0] != -Extrema[1])
            throw std::invalid_argument(std::string(Name) + " differs across MPI ranks or is invalid on some rank");
        if(Extrema[0] < 0)
            throw std::invalid_argument(std::string(Name) + " must be one of 0.." + std::to_string(Highest));
        return Extrema[0];
    }
}

int IndividualExtensiveLogForwardMode(void)
{
    static int const Mode = AgreedHydroStrictSwitch("RICH_INDIVIDUAL_EXTENSIVE_LOG_FORWARD", 2);
    return Mode;
}

bool HydroD2TimersEnabled(void)
{
    static bool const Enabled = AgreedHydroStrictSwitch("RICH_D2_TIMERS", 1) == 1;
    return Enabled;
}

int IndividualAllActiveIdentityMode(void)
{
    static int const Mode = AgreedHydroStrictSwitch("RICH_INDIVIDUAL_ALL_ACTIVE_IDENTITY", 2);
    return Mode;
}

std::uint64_t IndividualHydroInvalidComponentMask(Conserved3D const& state)
{
	std::uint64_t mask = 0;
	if(!std::isfinite(state.Erad) || state.Erad < 0) mask |= 1;
	if(!std::isfinite(state.Erad_dt)) mask |= 2;
	if(!std::isfinite(state.Erad_dt_dt)) mask |= 4;
	for(double const group_energy : state.Eg)
		if(!std::isfinite(group_energy) || group_energy < 0) mask |= 8;
	if(!std::isfinite(state.mass) || !(state.mass > 0)) mask |= 16;
	if(!std::isfinite(state.energy)) mask |= 32;
	if(!std::isfinite(state.internal_energy)) mask |= 64;
	if(!std::isfinite(state.momentum.x) || !std::isfinite(state.momentum.y) ||
		!std::isfinite(state.momentum.z)) mask |= 128;
	return mask;
}

void PersistIndividualHydroDiagnosticRecord(const std::string& record_id,
	const std::string& record)
{
	const std::string path = "individual_hydro_invalid_mass." + record_id +
		".pid" + std::to_string(static_cast<long long>(::getpid())) + ".log";
	const std::string partial_path = path + ".partial";
	const int descriptor = ::open(partial_path.c_str(),
		O_WRONLY | O_CREAT | O_EXCL, 0644);
	bool written = false;
	int diagnostic_errno = 0;
	if(descriptor >= 0)
	{
		const bool write_succeeded = write_all(descriptor, record);
		diagnostic_errno = write_succeeded ? 0 : errno;
		const bool close_succeeded = ::close(descriptor) == 0;
		if(!close_succeeded && write_succeeded)
			diagnostic_errno = errno;
		if(write_succeeded && close_succeeded)
		{
			written = ::link(partial_path.c_str(), path.c_str()) == 0;
			if(!written)
				diagnostic_errno = errno;
		}
		::unlink(partial_path.c_str());
	}
	else
		diagnostic_errno = errno;

	std::ostringstream notice;
	notice << "INDIVIDUAL_HYDRO_DIAGNOSTIC_FILE"
		<< " record_id=" << record_id
		<< " path=" << path
		<< " bytes=" << record.size()
		<< " status=" << (written ? "written" : "failed");
	if(!written)
		notice << " errno=" << diagnostic_errno
			<< " full_record=not_persisted";
	notice << '\n';
	write_all(STDERR_FILENO, notice.str());
}

DefaultExtensiveUpdater::DefaultExtensiveUpdater(void){}

void DefaultExtensiveUpdater::UpdateIndividual(
	const vector<Conserved3D>& fluxes,
	const Tessellation3D& tess,
	const IndividualStepContext& context,
	const vector<ComputationalCell3D>& cells,
	vector<Conserved3D>& extensives,
	double /*time*/,
	const vector<Vector3D>& edge_velocities,
	const vector<Vector3D>& point_velocities,
	const std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> >& interp_values,
	const vector<ComputationalCell3D>* canonical_cells,
	vector<Conserved3D>* canonical_extensives) const
{
	LastCanonicalWrites_.clear();
	LastCanonicalWritesComplete_ = false;
	const std::size_t norg = tess.GetPointNo();
	if(extensives.size() < norg || cells.size() < norg ||
		context.active_mask.size() < norg ||
		context.cell_time_steps.size() < norg)
		throw std::invalid_argument("Individual extensive update has inconsistent cell counts");
	std::vector<PendingIndividualHydroDelta> local_deltas;
	std::unordered_map<std::size_t, std::size_t> owned_index_by_id;
	unsigned local_mapping_issues = 0;
	if((canonical_cells == nullptr) != (canonical_extensives == nullptr))
		throw std::invalid_argument(
			"Individual hydro canonical cell and extent arrays must be supplied together");
	const std::size_t canonical_count = canonical_cells == nullptr ? 0 :
		canonical_cells->size();
	if(canonical_extensives != nullptr &&
		canonical_extensives->size() != canonical_count)
		throw std::invalid_argument(
			"Individual hydro canonical cell and extent counts differ");
    // B9 (RICH_INDIVIDUAL_ALL_ACTIVE_IDENTITY): every canonical cell is a target at its own index, so the persistent
    // canonical index answers every lookup the per-event target map would, and no cell is appended.
    int const IdentityMode = AllActiveIdentity_;
    bool const Identity = IdentityMode != 0;
    constexpr std::size_t NoWorkIndex = std::numeric_limits<std::size_t>::max();
    if(!Identity) {
	owned_index_by_id.reserve(norg);
	for(std::size_t i = 0; i < norg; ++i)
		if(!owned_index_by_id.emplace(cells[i].ID, i).second)
			local_mapping_issues |= 128u;
    }
    else {
        bool IdentityTargets = canonical_cells != nullptr && canonical_count == norg;
        for(std::size_t Index = 0; IdentityTargets && Index < norg; ++Index)
            IdentityTargets = cells[Index].ID == (*canonical_cells)[Index].ID;
        if(!IdentityTargets)
            local_mapping_issues |= 1024u;
    }

	// A partial target set need not contain an inactive owned recipient of a
	// remote face flux.  Append such canonical extents to this transaction so
	// conservation and the radiation positivity limiter remain cell-local,
	// while the passive primitive stays untouched.  A recipient is appended
	// when its first delta arrives.  Every other canonical cell is checked in
	// place below; appending it too (every owned cell, each event) only gave
	// it the same validation and a donor scale of exactly 1.  Its stable-ID
	// lookup is kept across events and rehashed only when the IDs change.
	std::vector<std::pair<std::size_t, std::size_t> > canonical_work_indices;
	std::unordered_map<std::size_t, std::size_t> const* canonical_index_by_id = nullptr;
	if(canonical_cells != nullptr)
	{
		CanonicalIdIndex& CanonicalIds = SharedCanonicalIdIndex();
		CanonicalIds.Refresh(*canonical_cells, canonical_count);
		canonical_index_by_id = &CanonicalIds.Map();
		if(!CanonicalIds.Unique())
			local_mapping_issues |= 256u;
        if(Identity) {
            // The targets are the canonical cells, so repeated canonical IDs are repeated target IDs.
            if(!CanonicalIds.Unique())
                local_mapping_issues |= 128u;
            canonical_work_indices.reserve(norg);
            for(std::size_t Index = 0; Index < norg && Index < canonical_count; ++Index)
                canonical_work_indices.emplace_back(Index, Index);
        }
        else {
		canonical_work_indices.reserve(owned_index_by_id.size());
		for(auto const& entry : owned_index_by_id)
		{
			auto const canonical = canonical_index_by_id->find(entry.first);
			if(canonical == canonical_index_by_id->end())
			{
				local_mapping_issues |= 512u;
				continue;
			}
			canonical_work_indices.emplace_back(entry.second,
				canonical->second);
		}
        }
	}
    // Work index of an owned (or appended) cell by stable ID, or NoWorkIndex.
    auto const FindOwned = [&](std::size_t CellId) -> std::size_t {
        if(Identity) {
            if(canonical_index_by_id == nullptr)
                return NoWorkIndex;
            auto const Found = canonical_index_by_id->find(CellId);
            return Found == canonical_index_by_id->end() ? NoWorkIndex : Found->second;
        }
        auto const Found = owned_index_by_id.find(CellId);
        return Found == owned_index_by_id.end() ? NoWorkIndex : Found->second;
    };

	// A canonical cell that is not in the transaction; its radiation donor
	// scale is exactly 1 (no outflow, validated state).
	auto const IsPassiveCanonicalId = [&](std::size_t CellId)
	{
		return FindOwned(CellId) == NoWorkIndex &&
			canonical_index_by_id != nullptr && canonical_index_by_id->count(CellId) != 0;
	};
#ifndef RICH_MPI
	if(local_mapping_issues != 0)
		throw std::runtime_error(
			"Individual hydro stable cell IDs are not unique");
#endif

	int rank = 0;
	int rank_count = 1;
#ifdef RICH_MPI
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_count);

	const std::size_t point_count = tess.getMeshPoints().size();
	std::vector<int> point_owner(point_count, -1);
	Tessellation3D::AllPointsMap const& local_to_global =
		tess.GetIndicesInAllPoints();
	if(local_to_global.size() < norg)
		local_mapping_issues |= 1u;
	for(auto const& mapping : local_to_global)
	{
		if(mapping.first >= point_owner.size())
		{
			local_mapping_issues |= 2u;
			continue;
		}
		point_owner[mapping.first] = rank;
	}
	const std::vector<int> duplicated_procs = tess.GetDuplicatedProcs();
	const std::vector<std::vector<std::size_t> >& ghost_indices = tess.GetGhostIndeces();
	if(duplicated_procs.size() != ghost_indices.size())
		local_mapping_issues |= 4u;
    // The identity exchange goes to each duplicated-point peer once.
    if(Identity) {
        std::vector<int> SortedPeers(duplicated_procs);
        std::sort(SortedPeers.begin(), SortedPeers.end());
        if(std::adjacent_find(SortedPeers.begin(), SortedPeers.end()) != SortedPeers.end() ||
            std::binary_search(SortedPeers.begin(), SortedPeers.end(), rank))
            local_mapping_issues |= 2048u;
    }
	for(std::size_t i = 0; i < duplicated_procs.size() && i < ghost_indices.size(); ++i)
	{
		if(duplicated_procs[i] < 0 || duplicated_procs[i] >= rank_count)
			local_mapping_issues |= 8u;
		for(std::size_t ghost : ghost_indices[i])
		{
			if(ghost >= point_owner.size())
			{
				local_mapping_issues |= 16u;
				continue;
			}
			// SyncPartialBuildData uses the same last-peer-wins order for
			// duplicate mesh slots after many-rank rebuilds.
			point_owner[ghost] = duplicated_procs[i];
		}
	}
	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		if(!context.isActive(neighbors.first) &&
			!context.isActive(neighbors.second))
			continue;
		if(neighbors.first >= norg && neighbors.second >= norg)
			continue;
		const std::size_t endpoints[2] = {neighbors.first, neighbors.second};
		for(std::size_t endpoint : endpoints)
		{
			if(endpoint >= point_owner.size())
			{
				local_mapping_issues |= 32u;
				continue;
			}
			// Box-boundary generator points deliberately have neither a hydro
			// state nor an MPI owner.  They cannot receive a physical update.
			if(tess.IsPointOutsideBox(endpoint))
				continue;
			if(endpoint >= cells.size() ||
				endpoint >= context.active_mask.size() ||
				endpoint >= context.cell_time_steps.size())
			{
				local_mapping_issues |= 32u;
				continue;
			}
			if(point_owner[endpoint] < 0)
				local_mapping_issues |= 64u;
		}
	}
	unsigned mapping_issues = local_mapping_issues;
	MPI_Allreduce(MPI_IN_PLACE, &mapping_issues, 1, MPI_UNSIGNED, MPI_BOR,
		MPI_COMM_WORLD);
	if(mapping_issues != 0)
		throw std::runtime_error(
			"Individual hydro MPI ownership mapping is inconsistent (issue mask " +
			std::to_string(mapping_issues) + ")");
	std::vector<std::vector<IndividualHydroDelta> > outgoing(rank_count);
#endif

	// The kinetic/thermal split uses the velocity of the primitive the flux
	// was reconstructed from: the energy carried by the delta is kinetic at
	// that velocity.  Splitting with the cell's running momentum instead
	// (tried 2026-09-18) charged the velocity mismatch to the thermal energy
	// and multiplied radiation negative-energy retries eightfold.
	auto make_applied_delta = [&cells](Conserved3D const& face_delta,
		std::size_t index, double sign)
	{
		Conserved3D applied = face_delta * sign;
		applied.internal_energy = sign * (face_delta.energy -
			ScalarProd(cells[index].velocity, face_delta.momentum) +
			0.5 * ScalarProd(cells[index].velocity, cells[index].velocity) * face_delta.mass);
		return applied;
	};

	for(std::size_t face = 0; face < fluxes.size(); ++face)
	{
		const auto neighbors = tess.GetFaceNeighbors(face);
		const bool first_active = context.isActive(neighbors.first);
		const bool second_active = context.isActive(neighbors.second);
		if(!first_active && !second_active)
			continue;

		double face_dt = 0;
		const bool first_physical =
			neighbors.first < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.first);
		const bool second_physical =
			neighbors.second < context.cell_time_steps.size() &&
			!tess.IsPointOutsideBox(neighbors.second);
		if(first_physical && second_physical)
			face_dt = context.hydroFaceTimeStep(neighbors.first, neighbors.second);
		else if(first_physical)
			face_dt = context.cellTimeStep(neighbors.first);
		else if(second_physical)
			face_dt = context.cellTimeStep(neighbors.second);
		else
			continue;

		const bool first_owned = neighbors.first < norg;
		const bool second_owned = neighbors.second < norg;
		if(!first_owned && !second_owned)
			continue;

#ifdef RICH_MPI
		if(first_physical && second_physical && first_owned != second_owned)
		{
			const int first_rank = point_owner[neighbors.first];
			const int second_rank = point_owner[neighbors.second];
			// A lone active endpoint owns its face.  This guarantees that a rank
			// with no locally active cells need not build a passive target cell.
			// Active-active faces use the stable endpoint key as the tie-breaker.
			const bool first_is_owner = first_active != second_active ? first_active :
				(cells[neighbors.first].ID < cells[neighbors.second].ID ||
				 (cells[neighbors.first].ID == cells[neighbors.second].ID && first_rank < second_rank));
			const int face_owner = first_is_owner ? first_rank : second_rank;
			if(face_owner != rank)
				continue;
		}
#endif

		Conserved3D delta = fluxes[face] * (face_dt * tess.GetArea(face));
		delta.internal_energy = 0;
		Conserved3D first_delta;
		Conserved3D second_delta;
		if(first_physical)
			first_delta = make_applied_delta(delta, neighbors.first, -1);
		if(second_physical)
			second_delta = make_applied_delta(delta, neighbors.second, 1);
		if(first_owned)
		{
			PendingIndividualHydroDelta pending;
			pending.cell_index = neighbors.first;
			pending.delta = first_delta;
			if(second_physical)
			{
				pending.counterpart_cell_id = cells[neighbors.second].ID;
				pending.counterpart_owner = rank;
#ifdef RICH_MPI
				pending.counterpart_owner = point_owner[neighbors.second];
#endif
			}
			local_deltas.push_back(pending);
		}
#ifdef RICH_MPI
		else if(first_physical)
		{
			outgoing[point_owner[neighbors.first]].emplace_back();
			IndividualHydroDelta& packet = outgoing[point_owner[neighbors.first]].back();
			packet.cell_id = cells[neighbors.first].ID;
			if(second_physical)
			{
				packet.counterpart_cell_id = cells[neighbors.second].ID;
				packet.counterpart_owner = point_owner[neighbors.second];
			}
			packet.delta = first_delta;
		}
#endif
		if(second_owned)
		{
			PendingIndividualHydroDelta pending;
			pending.cell_index = neighbors.second;
			pending.delta = second_delta;
			if(first_physical)
			{
				pending.counterpart_cell_id = cells[neighbors.first].ID;
				pending.counterpart_owner = rank;
#ifdef RICH_MPI
				pending.counterpart_owner = point_owner[neighbors.first];
#endif
			}
			local_deltas.push_back(pending);
		}
#ifdef RICH_MPI
		else if(second_physical)
		{
			outgoing[point_owner[neighbors.second]].emplace_back();
			IndividualHydroDelta& packet = outgoing[point_owner[neighbors.second]].back();
			packet.cell_id = cells[neighbors.second].ID;
			if(first_physical)
			{
				packet.counterpart_cell_id = cells[neighbors.first].ID;
				packet.counterpart_owner = point_owner[neighbors.first];
			}
			packet.delta = second_delta;
		}
#endif
	}
	// This scalar separates locally constructed and received corrections in a
	// failure record without adding per-correction state to the normal path.
	const std::size_t locally_constructed_delta_count = local_deltas.size();
#ifdef RICH_MPI
    // Identity: only duplicated-point peers can receive deltas; the result is indexed by source rank, as the
    // all-to-all's, so the deltas are appended in the same ascending rank order.
    bool LostIdentityPackets = false;
	std::vector<std::vector<IndividualHydroDelta> > incoming = Identity ?
        ExchangeWithDuplicatedPeers(outgoing, duplicated_procs, LostIdentityPackets) :
		MPI_Exchange_all_to_all(outgoing, MPI_COMM_WORLD);
	bool local_packet_error = LostIdentityPackets;
	for(const auto& rank_packets : incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
		{
			if(FindOwned(packet.cell_id) != NoWorkIndex)
				continue;
			if(canonical_index_by_id == nullptr)
			{
				local_packet_error = true;
				continue;
			}
			auto const canonical = canonical_index_by_id->find(packet.cell_id);
			if(canonical == canonical_index_by_id->end())
			{
				local_packet_error = true;
				continue;
			}
			// First delta for a passive canonical recipient: append its extent.
			std::size_t const WorkIndex = extensives.size();
			extensives.push_back((*canonical_extensives)[canonical->second]);
			owned_index_by_id.emplace(packet.cell_id, WorkIndex);
			canonical_work_indices.emplace_back(WorkIndex, canonical->second);
		}
	int packet_error = local_packet_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &packet_error, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
	if(packet_error != 0)
		throw std::runtime_error("Individual hydro received a delta for an unknown stable cell ID");
	for(const auto& rank_packets : incoming)
		for(const IndividualHydroDelta& packet : rank_packets)
		{
			PendingIndividualHydroDelta pending;
            pending.cell_index = FindOwned(packet.cell_id);
            if(pending.cell_index == NoWorkIndex)
                throw std::out_of_range("Individual hydro delta for a cell outside the transaction");
			pending.counterpart_cell_id = packet.counterpart_cell_id;
			pending.counterpart_owner = packet.counterpart_owner;
			pending.delta = packet.delta;
			local_deltas.push_back(pending);
		}
	std::vector<std::vector<IndividualHydroDelta> >().swap(outgoing);
	std::vector<std::vector<IndividualHydroDelta> >().swap(incoming);
#endif
	// The transaction is complete: owned targets, mesh slots, then the
	// appended recipients.
	const std::size_t work_count = extensives.size();
	oldEk_.assign(work_count, 0);
	oldEtherm_.assign(work_count, 0);
	oldE_.assign(work_count, 0);
	std::vector<unsigned char> touched(work_count, 0);
	std::vector<std::size_t> work_cell_ids(work_count, 0);
	for(std::size_t i = 0; i < cells.size() && i < work_count; ++i)
		work_cell_ids[i] = cells[i].ID;
	if(canonical_cells != nullptr)
		for(auto const& mapping : canonical_work_indices)
			work_cell_ids[mapping.first] =
				(*canonical_cells)[mapping.second].ID;
	constexpr std::size_t radiation_group_count = ENERGY_GROUPS_NUM;
	static_assert(radiation_group_count > 0,
		"Individual hydro radiation limiting requires at least one group");
	if(work_count > std::numeric_limits<std::size_t>::max() /
		radiation_group_count)
		throw std::overflow_error("Individual hydro radiation limiter size overflow");
	const std::size_t radiation_value_count =
		work_count * radiation_group_count;
	// For each donor, use one scale constrained by both Erad and every Eg.
	// Applying that scale only to the full radiation vector preserves
	// Erad=sum(Eg) in multigroup transport while gray transport is constrained
	// by Erad even when Eg[0] is zero.  Both face endpoints use the donor scale
	// directly, so internal and MPI faces remain conservative without applying
	// a large subtractive correction to the receiving cell.
	std::vector<double> erad_loss(work_count, 0);
	std::vector<double> group_loss(radiation_value_count, 0);
	std::vector<double> radiation_scale(work_count, 1);
	std::vector<std::vector<std::size_t> >
		radiation_outflow_delta_indices(work_count);
	constexpr std::uint64_t nonfinite_delta_issue = 1;
	constexpr std::uint64_t mixed_direction_issue = 2;
	constexpr std::uint64_t invalid_erad_issue = 4;
	constexpr std::uint64_t invalid_group_issue = 8;
	constexpr std::uint64_t invalid_derivative_issue = 16;
	constexpr std::uint64_t overflowed_loss_issue = 32;
	constexpr std::uint64_t invalid_material_issue = 64;
	const std::uint64_t no_radiation_issue_cell =
		std::numeric_limits<std::uint64_t>::max();
	const std::size_t no_radiation_issue_group =
		std::numeric_limits<std::size_t>::max();
	std::uint64_t local_radiation_issue_cell = no_radiation_issue_cell;
	std::uint64_t local_radiation_issue_mask = 0;
	std::size_t local_radiation_issue_group = no_radiation_issue_group;
	auto RecordRadiationIssueForId =
		[&local_radiation_issue_cell, &local_radiation_issue_mask,
			&local_radiation_issue_group]
		(std::uint64_t cell_id, std::uint64_t issue, std::size_t group)
	{
		if(cell_id < local_radiation_issue_cell)
		{
			local_radiation_issue_cell = cell_id;
			local_radiation_issue_mask = issue;
			local_radiation_issue_group = group;
		}
		else if(cell_id == local_radiation_issue_cell)
		{
			local_radiation_issue_mask |= issue;
			local_radiation_issue_group =
				std::min(local_radiation_issue_group, group);
		}
	};
	auto record_radiation_issue =
		[&work_cell_ids, &RecordRadiationIssueForId]
		(std::size_t cell, std::uint64_t issue,
			std::size_t group = std::numeric_limits<std::size_t>::max())
	{
		RecordRadiationIssueForId(
			static_cast<std::uint64_t>(work_cell_ids[cell]), issue, group);
	};
	for(std::size_t pending_index = 0;
		pending_index < local_deltas.size(); ++pending_index)
	{
		const PendingIndividualHydroDelta& pending =
			local_deltas[pending_index];
		const std::size_t cell = pending.cell_index;
		bool has_positive_radiation = pending.delta.Erad > 0;
		bool has_negative_radiation = pending.delta.Erad < 0;
		if(!std::isfinite(pending.delta.Erad) ||
			!std::isfinite(pending.delta.Erad_dt) ||
			!std::isfinite(pending.delta.Erad_dt_dt))
			record_radiation_issue(cell, nonfinite_delta_issue);
		if(pending.delta.Erad < 0)
		{
			erad_loss[cell] = std::nextafter(
				erad_loss[cell] - pending.delta.Erad,
				std::numeric_limits<double>::infinity());
			if(!std::isfinite(erad_loss[cell]))
				record_radiation_issue(cell, overflowed_loss_issue);
		}
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const double proposed = pending.delta.Eg[group];
			if(!std::isfinite(proposed))
				record_radiation_issue(cell, nonfinite_delta_issue, group);
			has_positive_radiation = has_positive_radiation || proposed > 0;
			has_negative_radiation = has_negative_radiation || proposed < 0;
			if(proposed < 0)
			{
				const std::size_t key = cell * radiation_group_count + group;
				group_loss[key] = std::nextafter(
					group_loss[key] - proposed,
					std::numeric_limits<double>::infinity());
				if(!std::isfinite(group_loss[key]))
					record_radiation_issue(
						cell, overflowed_loss_issue, group);
			}
		}
		if(has_positive_radiation && has_negative_radiation)
			record_radiation_issue(cell, mixed_direction_issue);
		if(has_negative_radiation)
			radiation_outflow_delta_indices[cell].push_back(pending_index);
	}
	auto const RecordStateIssues = [&](Conserved3D const& State, std::uint64_t CellId)
	{
		const std::size_t NoGroup = std::numeric_limits<std::size_t>::max();
		const std::uint64_t state_mask = IndividualHydroInvalidComponentMask(State);
		if((state_mask & (16 | 32 | 64 | 128)) != 0)
			RecordRadiationIssueForId(CellId, invalid_material_issue, NoGroup);
		if((state_mask & 1) != 0)
			RecordRadiationIssueForId(CellId, invalid_erad_issue, NoGroup);
		if((state_mask & (2 | 4)) != 0)
			RecordRadiationIssueForId(CellId, invalid_derivative_issue, NoGroup);
		if((state_mask & 8) != 0)
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const double available = State.Eg[group];
			if(!std::isfinite(available) || available < 0)
				RecordRadiationIssueForId(CellId, invalid_group_issue, group);
		}
	};
	if(Identity) {
        for(std::size_t Owned = 0; Owned < norg; ++Owned)
            RecordStateIssues(extensives[Owned], static_cast<std::uint64_t>(work_cell_ids[Owned]));
    }
    else
	for(auto const& owned : owned_index_by_id)
		RecordStateIssues(extensives[owned.second], static_cast<std::uint64_t>(work_cell_ids[owned.second]));
	// Canonical cells outside the transaction get the same checks in place.
	// The result is a minimum over cell IDs with OR-ed masks, so it does not
	// depend on the order in which cells are checked.
	if(canonical_cells != nullptr)
	{
		std::vector<unsigned char> InTransaction(canonical_count, 0);
		for(auto const& mapping : canonical_work_indices)
			InTransaction[mapping.second] = 1;
		for(std::size_t canonical = 0; canonical < canonical_count; ++canonical)
			if(InTransaction[canonical] == 0)
				RecordStateIssues((*canonical_extensives)[canonical],
					static_cast<std::uint64_t>((*canonical_cells)[canonical].ID));
	}

	std::uint64_t radiation_issue_cell = local_radiation_issue_cell;
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, &radiation_issue_cell, 1, MPI_UINT64_T,
		MPI_MIN, MPI_COMM_WORLD);
#endif
	if(radiation_issue_cell != no_radiation_issue_cell)
	{
		std::uint64_t radiation_issue_mask =
			local_radiation_issue_cell == radiation_issue_cell ?
			local_radiation_issue_mask : 0;
		std::uint64_t radiation_issue_group =
			local_radiation_issue_cell == radiation_issue_cell ?
			static_cast<std::uint64_t>(local_radiation_issue_group) :
			std::numeric_limits<std::uint64_t>::max();
#ifdef RICH_MPI
		MPI_Allreduce(MPI_IN_PLACE, &radiation_issue_mask, 1, MPI_UINT64_T,
			MPI_BOR, MPI_COMM_WORLD);
		MPI_Allreduce(MPI_IN_PLACE, &radiation_issue_group, 1, MPI_UINT64_T,
			MPI_MIN, MPI_COMM_WORLD);
#endif
		const std::string group_detail =
			radiation_issue_group != std::numeric_limits<std::uint64_t>::max() ?
			" group " + std::to_string(radiation_issue_group) : "";
		if((radiation_issue_mask & invalid_material_issue) != 0)
		{
			std::uint64_t component_mask = 0;
			Conserved3D const* InvalidState = nullptr;
			std::size_t const InvalidWork = FindOwned(radiation_issue_cell);
			if(InvalidWork != NoWorkIndex)
				InvalidState = &extensives[InvalidWork];
			else if(canonical_index_by_id != nullptr)
			{
				auto const canonical = canonical_index_by_id->find(radiation_issue_cell);
				if(canonical != canonical_index_by_id->end())
					InvalidState = &(*canonical_extensives)[canonical->second];
			}
			if(InvalidState != nullptr)
			{
				component_mask = IndividualHydroInvalidComponentMask(*InvalidState);
				const std::string record_id = std::to_string(rank) + ":" +
					std::to_string(context.event_tick) + ":" + std::to_string(radiation_issue_cell);
				std::ostringstream record;
				record << std::setprecision(17) << "INDIVIDUAL_HYDRO_INVALID_INPUT"
					<< " record_id=" << record_id << " phase=before_flux_application"
					<< " component_mask=" << component_mask
					<< " state=" << *InvalidState << '\n';
				PersistIndividualHydroDiagnosticRecord(record_id, record.str());
			}
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &component_mask, 1, MPI_UINT64_T,
				MPI_BOR, MPI_COMM_WORLD);
#endif
			throw std::runtime_error("Individual hydro invalid input before_flux_application for cell ID " +
				std::to_string(radiation_issue_cell) + " component mask " + std::to_string(component_mask));
		}
		throw std::runtime_error(
			"Individual hydro radiation limiter rejected cell ID " +
			std::to_string(radiation_issue_cell) + group_detail +
			" issue mask " + std::to_string(radiation_issue_mask));
	}

	// Upward-rounded losses make the budget conservative.  This reserve keeps
	// accepted states away from cancellation; the ordered replay below still
	// verifies the actual double-precision additions before deltas are changed.
	constexpr double radiation_reserve_fraction = 1e-12;
    // Identity visits the targets in index order; otherwise the map's own order, as before.
    auto OwnedIterator = owned_index_by_id.begin();
    for(std::size_t OwnedStep = 0; Identity ? OwnedStep < norg : OwnedIterator != owned_index_by_id.end(); ++OwnedStep)
	{
        const std::size_t cell = Identity ? OwnedStep : (OwnedIterator++)->second;
		const double safe_erad =
			std::nextafter(extensives[cell].Erad *
				(1 - radiation_reserve_fraction), 0.0);
		if(erad_loss[cell] > safe_erad)
		{
			const double ratio =
				safe_erad > 0 ? safe_erad / erad_loss[cell] : 0;
			radiation_scale[cell] = ratio > 0 ?
				std::nextafter(std::min(1.0, ratio), 0.0) : 0;
		}
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			const std::size_t key = cell * radiation_group_count + group;
			const double safe_group =
				std::nextafter(extensives[cell].Eg[group] *
					(1 - radiation_reserve_fraction), 0.0);
			if(group_loss[key] > safe_group)
			{
				const double ratio =
					safe_group > 0 ? safe_group / group_loss[key] : 0;
				const double group_scale = ratio > 0 ?
					std::nextafter(std::min(1.0, ratio), 0.0) : 0;
				radiation_scale[cell] =
					std::min(radiation_scale[cell], group_scale);
			}
		}
	}

	std::vector<double> predicted_groups(radiation_group_count, 0);
	auto donor_application_is_safe =
		[&extensives, &local_deltas, &radiation_outflow_delta_indices,
			&predicted_groups, radiation_group_count](
			std::size_t cell, double scale)
	{
		double predicted_erad = extensives[cell].Erad;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
			predicted_groups[group] = extensives[cell].Eg[group];
		for(const std::size_t pending_index :
			radiation_outflow_delta_indices[cell])
		{
			const PendingIndividualHydroDelta& pending =
				local_deltas[pending_index];
			predicted_erad += pending.delta.Erad * scale;
			for(std::size_t group = 0;
				group < radiation_group_count; ++group)
				predicted_groups[group] +=
					pending.delta.Eg[group] * scale;
		}
		if(predicted_erad < 0 || !std::isfinite(predicted_erad))
			return false;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
			if(predicted_groups[group] < 0 ||
				!std::isfinite(predicted_groups[group]))
				return false;
		return true;
	};
    OwnedIterator = owned_index_by_id.begin();
    for(std::size_t OwnedStep = 0; Identity ? OwnedStep < norg : OwnedIterator != owned_index_by_id.end(); ++OwnedStep)
	{
        const std::size_t cell = Identity ? OwnedStep : (OwnedIterator++)->second;
		if(donor_application_is_safe(cell, radiation_scale[cell]))
			continue;
		radiation_scale[cell] =
			std::nextafter(radiation_scale[cell], 0.0);
		if(!donor_application_is_safe(cell, radiation_scale[cell]))
			radiation_scale[cell] = 0;
		if(!donor_application_is_safe(cell, radiation_scale[cell]))
			throw std::logic_error(
				"Individual hydro radiation limiter could not preserve "
				"the pre-event radiation state");
	}

	std::vector<double> radiation_counterpart_scale(local_deltas.size(), 1);
#ifdef RICH_MPI
	std::vector<std::vector<IndividualHydroScale> >
		radiation_scale_outgoing(rank_count);
	std::unordered_map<StableFaceEndpoint, std::size_t,
		StableFaceEndpointHash> radiation_pending_by_endpoint;
	radiation_pending_by_endpoint.reserve(local_deltas.size());
	bool local_radiation_scale_mapping_error = false;
	for(std::size_t pending_index = 0;
		pending_index < local_deltas.size(); ++pending_index)
	{
		const PendingIndividualHydroDelta& pending =
			local_deltas[pending_index];
		if(pending.counterpart_owner < 0)
			continue;
		if(pending.counterpart_owner == rank)
		{
			std::size_t const counterpart = FindOwned(pending.counterpart_cell_id);
			if(counterpart != NoWorkIndex)
				radiation_counterpart_scale[pending_index] =
					radiation_scale[counterpart];
			else if(!IsPassiveCanonicalId(pending.counterpart_cell_id))
				local_radiation_scale_mapping_error = true;
		}
		else if(pending.counterpart_owner < rank_count)
		{
			const StableFaceEndpoint endpoint = {
				work_cell_ids[pending.cell_index],
				pending.counterpart_cell_id};
			if(!radiation_pending_by_endpoint.emplace(
				endpoint, pending_index).second)
				local_radiation_scale_mapping_error = true;
			IndividualHydroScale packet;
			packet.cell_id = pending.counterpart_cell_id;
			packet.counterpart_cell_id =
				work_cell_ids[pending.cell_index];
			packet.scale = radiation_scale[pending.cell_index];
			radiation_scale_outgoing[pending.counterpart_owner].push_back(
				packet);
		}
		else
			local_radiation_scale_mapping_error = true;
	}
	int radiation_scale_mapping_error =
		local_radiation_scale_mapping_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &radiation_scale_mapping_error, 1, MPI_INT,
		MPI_MAX, MPI_COMM_WORLD);
	if(radiation_scale_mapping_error != 0)
		throw std::runtime_error(
			"Individual hydro radiation limiter could not map a face endpoint");
    bool LostIdentityScales = false;
	const std::vector<std::vector<IndividualHydroScale> >
		radiation_scale_incoming = Identity ?
        ExchangeWithDuplicatedPeers(radiation_scale_outgoing, duplicated_procs, LostIdentityScales) :
        MPI_Exchange_all_to_all(radiation_scale_outgoing, MPI_COMM_WORLD);
	std::vector<unsigned char> received_radiation_scale(
		local_deltas.size(), 0);
	bool local_radiation_scale_packet_error = LostIdentityScales;
	for(const auto& rank_packets : radiation_scale_incoming)
		for(const IndividualHydroScale& packet : rank_packets)
		{
			const StableFaceEndpoint endpoint = {
				packet.cell_id, packet.counterpart_cell_id};
			const auto found =
				radiation_pending_by_endpoint.find(endpoint);
			if(found == radiation_pending_by_endpoint.end() ||
				!std::isfinite(packet.scale) ||
				packet.scale < 0 ||
				packet.scale > 1 ||
				received_radiation_scale[found->second] != 0)
			{
				local_radiation_scale_packet_error = true;
				continue;
			}
			radiation_counterpart_scale[found->second] = packet.scale;
			received_radiation_scale[found->second] = 1;
		}
	for(std::size_t pending_index = 0;
		pending_index < local_deltas.size(); ++pending_index)
		if(local_deltas[pending_index].counterpart_owner >= 0 &&
			local_deltas[pending_index].counterpart_owner != rank &&
			received_radiation_scale[pending_index] == 0)
			local_radiation_scale_packet_error = true;
	int radiation_scale_packet_error =
		local_radiation_scale_packet_error ? 1 : 0;
	MPI_Allreduce(MPI_IN_PLACE, &radiation_scale_packet_error, 1, MPI_INT,
		MPI_MAX, MPI_COMM_WORLD);
	if(radiation_scale_packet_error != 0)
		throw std::runtime_error(
			"Individual hydro radiation limiter received inconsistent face "
			"scales");
#else
	for(std::size_t pending_index = 0;
		pending_index < local_deltas.size(); ++pending_index)
	{
		const PendingIndividualHydroDelta& pending =
			local_deltas[pending_index];
		if(pending.counterpart_owner < 0)
			continue;
		std::size_t const counterpart = FindOwned(pending.counterpart_cell_id);
		if(counterpart != NoWorkIndex)
			radiation_counterpart_scale[pending_index] =
				radiation_scale[counterpart];
		else if(!IsPassiveCanonicalId(pending.counterpart_cell_id))
			throw std::runtime_error(
				"Individual hydro radiation limiter could not map a face "
				"endpoint");
	}
#endif
	for(std::size_t pending_index = 0;
		pending_index < local_deltas.size(); ++pending_index)
	{
		PendingIndividualHydroDelta& pending = local_deltas[pending_index];
		bool radiation_outflow = pending.delta.Erad < 0;
		bool radiation_inflow = pending.delta.Erad > 0;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
		{
			radiation_outflow =
				radiation_outflow || pending.delta.Eg[group] < 0;
			radiation_inflow =
				radiation_inflow || pending.delta.Eg[group] > 0;
		}
		double face_scale = 1;
		if(radiation_outflow)
			face_scale = radiation_scale[pending.cell_index];
		else if(radiation_inflow)
			face_scale = radiation_counterpart_scale[pending_index];
		else
			continue;
		if(!(face_scale < 1))
			continue;
		pending.delta.Erad *= face_scale;
		for(std::size_t group = 0; group < radiation_group_count; ++group)
			pending.delta.Eg[group] *= face_scale;
		pending.delta.Erad_dt *= face_scale;
		pending.delta.Erad_dt_dt *= face_scale;
	}
#ifdef RICH_MPI
    // B9 mode 2: both sides of every cross-rank face must hold exactly opposite conserved deltas.
    if(IdentityMode == 2)
        VerifyCrossRankFaceConservation(local_deltas, work_cell_ids, rank, rank_count, duplicated_procs,
            context.event_tick);
#endif

	for(const PendingIndividualHydroDelta& indexed_delta : local_deltas)
	{
		const std::size_t index = indexed_delta.cell_index;
		if(!touched[index])
		{
			touched[index] = 1;
			oldEk_[index] = 0.5 * ScalarProd(extensives[index].momentum,
				extensives[index].momentum) / extensives[index].mass;
			oldEtherm_[index] = extensives[index].internal_energy;
			oldE_[index] = extensives[index].energy;
		}
		extensives[index] += indexed_delta.delta;
	}

	auto require_valid_updated_state =
		[&extensives, &touched, &work_cell_ids, &local_deltas,
			locally_constructed_delta_count, &fluxes, &tess, &context, &cells,
			&edge_velocities, &point_velocities, &interp_values,
			radiation_group_count, rank](char const* phase)
	{
		std::uint64_t invalid_cell_id =
			std::numeric_limits<std::uint64_t>::max();
		std::uint64_t invalid_mass_cell_id =
			std::numeric_limits<std::uint64_t>::max();
		std::size_t invalid_mass_index =
			std::numeric_limits<std::size_t>::max();
		std::uint64_t invalid_component_mask = 0;
		std::uint64_t invalid_group =
			std::numeric_limits<std::uint64_t>::max();
		for(std::size_t i = 0; i < touched.size(); ++i)
		{
			if(!touched[i])
				continue;
			std::uint64_t const component_mask = IndividualHydroInvalidComponentMask(extensives[i]);
			std::uint64_t cell_invalid_group =
				std::numeric_limits<std::uint64_t>::max();
			if((component_mask & 8) != 0)
			for(std::size_t group = 0; group < radiation_group_count; ++group)
				if(!std::isfinite(extensives[i].Eg[group]) ||
					extensives[i].Eg[group] < 0)
				{
					cell_invalid_group =
						std::min(cell_invalid_group,
							static_cast<std::uint64_t>(group));
				}
			if(component_mask != 0)
			{
				const std::uint64_t cell_id =
					static_cast<std::uint64_t>(work_cell_ids[i]);
				if((component_mask & 16) != 0 &&
					cell_id < invalid_mass_cell_id)
				{
					invalid_mass_cell_id = cell_id;
					invalid_mass_index = i;
				}
				if(cell_id < invalid_cell_id)
				{
					invalid_cell_id = cell_id;
					invalid_component_mask = component_mask;
					invalid_group = cell_invalid_group;
				}
				else if(cell_id == invalid_cell_id)
				{
					invalid_component_mask |= component_mask;
					invalid_group =
						std::min(invalid_group, cell_invalid_group);
				}
			}
		}
		const std::uint64_t local_invalid_mass_cell_id =
			invalid_mass_cell_id;
#ifdef RICH_MPI
		const std::uint64_t local_invalid_cell_id = invalid_cell_id;
		MPI_Allreduce(MPI_IN_PLACE, &invalid_cell_id, 1, MPI_UINT64_T,
			MPI_MIN, MPI_COMM_WORLD);
		if(invalid_cell_id != std::numeric_limits<std::uint64_t>::max())
		{
			if(local_invalid_cell_id != invalid_cell_id)
			{
				invalid_component_mask = 0;
				invalid_group = std::numeric_limits<std::uint64_t>::max();
			}
			MPI_Allreduce(MPI_IN_PLACE, &invalid_component_mask, 1,
				MPI_UINT64_T, MPI_BOR, MPI_COMM_WORLD);
			MPI_Allreduce(MPI_IN_PLACE, &invalid_group, 1, MPI_UINT64_T,
				MPI_MIN, MPI_COMM_WORLD);
		}
#endif
		if(invalid_cell_id != std::numeric_limits<std::uint64_t>::max())
		{
#ifdef RICH_MPI
			MPI_Allreduce(MPI_IN_PLACE, &invalid_mass_cell_id, 1,
				MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
#endif
			if(local_invalid_mass_cell_id == invalid_mass_cell_id &&
				invalid_mass_index < extensives.size())
			{
				report_applied_individual_hydro_deltas(local_deltas,
					locally_constructed_delta_count, fluxes, tess, context,
					cells, extensives, work_cell_ids, edge_velocities,
					point_velocities, interp_values, invalid_mass_index,
					phase, rank);
			}
			if(invalid_mass_cell_id != invalid_cell_id)
			{
				for(std::size_t i = 0; i < work_cell_ids.size(); ++i)
					if(work_cell_ids[i] == invalid_cell_id && touched[i])
					{
						const std::string record_id = std::to_string(rank) + ":" +
							std::to_string(context.event_tick) + ":" + std::to_string(invalid_cell_id);
						std::ostringstream record;
						record << std::setprecision(17) << "INDIVIDUAL_HYDRO_INVALID_STATE"
							<< " record_id=" << record_id << " phase=" << phase
							<< " component_mask=" << invalid_component_mask
							<< " state=" << extensives[i] << '\n';
						PersistIndividualHydroDiagnosticRecord(record_id, record.str());
						break;
					}
			}
#ifdef RICH_MPI
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			const std::string group_detail =
				invalid_group != std::numeric_limits<std::uint64_t>::max() ?
				" group " + std::to_string(invalid_group) : "";
			throw std::runtime_error(
				"Individual hydro update produced an invalid "
				"state " + std::string(phase) + " for cell ID " +
				std::to_string(invalid_cell_id) + group_detail +
				" component mask " +
				std::to_string(invalid_component_mask));
		}
	};
	require_valid_updated_state("before_dual_energy_synchronization");

	// Dual-energy trace (RICH_INDIVIDUAL_DUAL_ENERGY_TRACE=1): how often the
	// thermal energy is overwritten by total minus kinetic, how often the two
	// estimates disagree so the overwrite is skipped, and how often the
	// synchronized value would have been non-positive.
	unsigned long long dual_energy_counts[4] = {0, 0, 0, 0};
	for(std::size_t i = 0; i < touched.size(); ++i)
	{
		if(!touched[i])
			continue;
		++dual_energy_counts[0];
		const double dEtherm = extensives[i].internal_energy - oldEtherm_[i];
		const double Eknew = 0.5 * ScalarProd(extensives[i].momentum,
			extensives[i].momentum) / extensives[i].mass;
		const double dEk = Eknew - oldEk_[i];
		const double dE = extensives[i].energy - oldE_[i];
		if(dEtherm * (dE - dEk) > 0 &&
			std::abs(dEtherm) > 0.95 * std::abs(dE - dEk) &&
			std::abs(dEtherm) < 1.05 * std::abs(dE - dEk))
		{
			const double synchronized_internal_energy =
				extensives[i].energy - Eknew;
			if(synchronized_internal_energy > 0 &&
				std::isfinite(synchronized_internal_energy))
			{
				extensives[i].internal_energy =
					synchronized_internal_energy;
				++dual_energy_counts[1];
			}
			else
				++dual_energy_counts[3];
		}
		else
			++dual_energy_counts[2];
	}
	if(IndividualDualEnergyTraceEnabled())
	{
		int trace_rank = 0;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &trace_rank);
		MPI_Allreduce(MPI_IN_PLACE, dual_energy_counts, 4,
			MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
		if(trace_rank == 0)
			std::cout << "INDIVIDUAL_DUAL_ENERGY event_tick=" << context.event_tick
				<< " touched=" << dual_energy_counts[0]
				<< " synchronized=" << dual_energy_counts[1]
				<< " disagreed=" << dual_energy_counts[2]
				<< " nonpositive_sync=" << dual_energy_counts[3] << std::endl;
	}
	require_valid_updated_state("after_dual_energy_synchronization");
	if(canonical_extensives != nullptr)
	{
		for(auto const& mapping : canonical_work_indices)
			if(touched[mapping.first])
			{
				Conserved3D& Canonical = (*canonical_extensives)[mapping.second];
				LastCanonicalWrites_.push_back({mapping.second, Canonical.mass, Canonical.energy,
					Canonical.internal_energy});
				Canonical = extensives[mapping.first];
			}
		LastCanonicalWritesComplete_ = true;
	}
	extensives.resize(norg);
}

void DefaultExtensiveUpdater::operator()(const vector<Conserved3D>& fluxes, const Tessellation3D& tess,
	const double dt, const vector<ComputationalCell3D>& cells, vector<Conserved3D>& extensives, double /*time*/,
				 std::vector<Vector3D> const& /*face_vel*/,
				 const vector<Vector3D>& /*point_velocities*/,
				 std::vector<std::pair<ComputationalCell3D, ComputationalCell3D> > const& /*interp_values*/) const
{
	size_t N = tess.GetPointNo();
	size_t Nfluxes = fluxes.size();
	Conserved3D delta;
	oldEk_.assign(N, 0);
	oldEtherm_.assign(N, 0);
	oldE_.assign(N, 0);
	for (size_t i = 0; i < N; ++i)
	{
		oldEk_[i] = 0.5*ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		oldEtherm_[i] = extensives[i].internal_energy;
		oldE_[i] = extensives[i].energy;
	}
	for (size_t i = 0; i < Nfluxes; ++i)
	{
		delta = fluxes[i] * dt*tess.GetArea(i);
		size_t n0 = tess.GetFaceNeighbors(i).first;
		size_t n1 = tess.GetFaceNeighbors(i).second;
#ifdef RICH_DEBUG
		bool good = true;
		if (!std::isfinite(fluxes[i].energy))
			good = false;
		if (!std::isfinite(fluxes[i].internal_energy))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.x))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.y))
			good = false;
		if (!std::isfinite(fluxes[i].momentum.z))
			good = false;
		if (!std::isfinite(fluxes[i].mass))
			good = false;
		for (size_t j = 0; j < delta.tracers.size(); ++j)
		{
			if (!std::isfinite(fluxes[i].tracers[j]))
				good = false;
		}
		if (!good)
		{
			UniversalError eo("Bad flux");
			eo.addEntry("Face index", i);
			eo.addEntry("Area", tess.GetArea(i));
			eo.addEntry("First neigh", n0);
			eo.addEntry("Second neigh", n1);
			eo.addEntry("Norg", N);
			eo.addEntry("Energy flux", fluxes[i].energy);
			eo.addEntry("Internal Energy flux", fluxes[i].internal_energy);
			eo.addEntry("Mass flux", fluxes[i].mass);
			eo.addEntry("Momentum x flux", fluxes[i].momentum.x);
			eo.addEntry("Momentum y flux", fluxes[i].momentum.y);
			eo.addEntry("Momentum z flux", fluxes[i].momentum.z);
			for (size_t j = 0; j < delta.tracers.size(); ++j)
				eo.addEntry("Tracer flux", fluxes[i].tracers[j]);
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
			throw eo;
		}
#endif
		delta.internal_energy = 0;
		if (n0 < N)
		{
			extensives[n0] -= delta;
			extensives[n0].internal_energy -= delta.energy - ScalarProd(cells[n0].velocity, delta.momentum) +
				0.5*ScalarProd(cells[n0].velocity, cells[n0].velocity)*delta.mass;
		}
		if (n1 < N)
		{
			extensives[n1] += delta;
			extensives[n1].internal_energy += delta.energy - ScalarProd(cells[n1].velocity, delta.momentum) +
				0.5*ScalarProd(cells[n1].velocity, cells[n1].velocity)*delta.mass;
		}
	}
	for (size_t i = 0; i < N; ++i)
	{
		double dEtherm = extensives[i].internal_energy - oldEtherm_[i];
		double Eknew = 0.5*ScalarProd(extensives[i].momentum, extensives[i].momentum) / extensives[i].mass;
		double dEk = Eknew - oldEk_[i];
		double dE = extensives[i].energy - oldE_[i];
		if (dEtherm*(dE - dEk) > 0)
		{
			if (std::abs(dEtherm) > 0.95 *std::abs(dE - dEk) && std::abs(dEtherm) < 1.05*std::abs(dE - dEk))
				extensives[i].internal_energy = extensives[i].energy - Eknew;
			else
				extensives[i].energy = extensives[i].internal_energy + Eknew;
		}
		else
			extensives[i].energy = extensives[i].internal_energy + Eknew;
	}
	extensives.resize(tess.GetPointNo());
}
