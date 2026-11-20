#include "RoundCells3D.hpp"
#include "../../misc/utils.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#ifdef RICH_MPI
#include "../../mpi/mpi_commands.hpp"
#endif

namespace
{
    // RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS (B4; unset, 0, 1 or 2; default 0), strict, read once and agreed across ranks
    // in one reduction on first use (the first ApplyFix, which every rank enters).  See ApplyFixImpl.
    int RoundCellsSkipEmptyStickersMode(void)
    {
        static int const Mode = []()
        {
            char const* const Name = "RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS";
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
        }();
        return Mode;
    }

    // Mode 2 of RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS: where mode 1 would skip, the s48 path must leave nomove all zero
    // with the size the skip gives it and must stop no canonical cell (all_nomove all zero).  One reduction; rank 0
    // prints the counts; on a difference the lowest rank that found one prints an example and every rank throws.
    void VerifyNoMoveSkip(vector<char> const& NoMove, vector<char> const& AllNoMove, size_t ExpectedSize,
        bool Individual)
    {
        unsigned long long Counts[2] = {static_cast<unsigned long long>(NoMove.size() + AllNoMove.size()), 0};
        std::ostringstream Example;
        if(NoMove.size() != ExpectedSize)
        {
            ++Counts[1];
            Example << " nomove_size=" << NoMove.size() << " expected_size=" << ExpectedSize;
        }
        for(size_t Index = 0; Index < NoMove.size(); ++Index)
            if(NoMove[Index] != 0 && Counts[1]++ == 0)
                Example << " stuck_mesh_index=" << Index;
        for(size_t Index = 0; Index < AllNoMove.size(); ++Index)
            if(AllNoMove[Index] != 0 && Counts[1]++ == 0)
                Example << " stuck_canonical_index=" << Index;
        unsigned long long const LocalMismatches = Counts[1];
        int Rank = 0;
        int RankCount = 1;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
        MPI_Allreduce(MPI_IN_PLACE, Counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(Rank == 0)
            std::cout << "RICH_B4_ROUNDCELLS_VERIFY path=" << (Individual ? "individual" : "global") << " checked="
                << Counts[0] << " mismatches=" << Counts[1] << std::endl;
        if(Counts[1] == 0)
            return;
        int FirstRank = LocalMismatches != 0 ? Rank : RankCount;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &FirstRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        if(Rank == FirstRank)
            std::cerr << "RICH_B4_ROUNDCELLS_VERIFY_MISMATCH rank=" << Rank << " local=" << LocalMismatches
                << Example.str() << std::endl;
        throw std::runtime_error("RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS=2: skipping the no-move scan and sync would "
            "change the result (" + std::to_string(Counts[1]) + " mismatches)");
    }

    // Marks every owned target whose velocity the velocity sync after the kicks sends to a peer.  Individual path:
    // SyncPartialBuildData sends the canonical entries of the duplicated all-points, and a target's entry holds the
    // target's velocity after the write-back.  Full path: MPI_exchange_data sends the duplicated points' entries.
    void MarkSentTargets(Tessellation3D const& tess, bool Individual, size_t CanonicalSize, size_t n,
        std::vector<unsigned char>& Mark)
    {
#ifdef RICH_MPI
        if(Individual) {
            std::vector<size_t> SentCanonical;
            for(auto const& Peer : tess.GetDuplicatedAllPointIndices())
                for(size_t const AllPoint : Peer)
                    if(AllPoint < tess.GetAllPointsNo())
                        SentCanonical.push_back(tess.GetInputIndexForAllPoint(AllPoint));
            size_t Size = CanonicalSize;
            for(size_t const Canonical : SentCanonical)
                if(Canonical != std::numeric_limits<size_t>::max())
                    Size = std::max(Size, Canonical + 1);
            std::vector<unsigned char> Sent(Size, 0);
            for(size_t const Canonical : SentCanonical)
                if(Canonical < Size)
                    Sent[Canonical] = 1;
            for(auto const& Entry : tess.GetIndicesInAllPoints())
                if(Entry.first < n && Entry.second < Size && Sent[Entry.second] != 0)
                    Mark[Entry.first] = 1;
        }
        else
            for(auto const& Peer : tess.GetDuplicatedPoints())
                for(size_t const Local : Peer)
                    if(Local < n)
                        Mark[Local] = 1;
#else
        (void)tess;
        (void)Individual;
        (void)CanonicalSize;
        (void)n;
        (void)Mark;
#endif
    }

    bool SameBits(double First, double Second)
    {
        return std::memcmp(&First, &Second, sizeof(double)) == 0;
    }

    // Mode 2 of RICH_INDIVIDUAL_HYDRO_TARGET_PASSES: the active targets' velocities from the restricted and the full
    // passes must agree bit for bit.  One reduction; rank 0 prints the counts; on a difference the lowest rank that
    // found one prints an example and every rank throws.
    void VerifyTargetPasses(vector<Vector3D> const& Full, vector<Vector3D> const& Restricted,
        std::vector<size_t> const& Active)
    {
        unsigned long long Counts[2] = {static_cast<unsigned long long>(Active.size()), 0};
        std::ostringstream Example;
        Example.precision(17);
        for(size_t const Target : Active) {
            bool const Same = Target < Full.size() && Target < Restricted.size() &&
                SameBits(Full[Target].x, Restricted[Target].x) && SameBits(Full[Target].y, Restricted[Target].y) &&
                SameBits(Full[Target].z, Restricted[Target].z);
            if(Same || Counts[1]++ != 0)
                continue;
            Example << " target=" << Target;
            if(Target < Full.size() && Target < Restricted.size())
                Example << " full=" << Full[Target].x << ',' << Full[Target].y << ',' << Full[Target].z
                    << " restricted=" << Restricted[Target].x << ',' << Restricted[Target].y << ','
                    << Restricted[Target].z;
        }
        unsigned long long const LocalMismatches = Counts[1];
        int Rank = 0;
        int RankCount = 1;
#ifdef RICH_MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &Rank);
        MPI_Comm_size(MPI_COMM_WORLD, &RankCount);
        MPI_Allreduce(MPI_IN_PLACE, Counts, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
        if(Rank == 0)
            std::cout << "RICH_B10_FIX_VERIFY checked=" << Counts[0] << " mismatches=" << Counts[1] << std::endl;
        if(Counts[1] == 0)
            return;
        int FirstRank = LocalMismatches != 0 ? Rank : RankCount;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &FirstRank, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        if(Rank == FirstRank)
            std::cerr << "RICH_B10_FIX_VERIFY_MISMATCH rank=" << Rank << " local=" << LocalMismatches
                << Example.str() << std::endl;
        throw std::runtime_error("RICH_INDIVIDUAL_HYDRO_TARGET_PASSES=2: the restricted point-motion passes differ "
            "from the full passes (" + std::to_string(Counts[1]) + " mismatches)");
    }
}

int IndividualHydroTargetPassesMode(void)
{
    static int const Mode = []()
    {
        char const* const Name = "RICH_INDIVIDUAL_HYDRO_TARGET_PASSES";
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
    }();
    return Mode;
}

RoundCells3D::RoundCells3D(const PointMotion3D& pm, const EquationOfState& eos,
	double chi, double eta, bool cold, double min_dw, double dt_speed, const vector<std::string>& no_move, double max_velocity) : pm_(pm), eos_(eos), chi_(chi),
	eta_(eta), cold_(cold), min_dw_(min_dw),dt_speed_(dt_speed),no_move_(no_move),max_velocity_(max_velocity) {}

namespace
{
	void SlowDown(Vector3D &velocity, Tessellation3D const& tess, double R, size_t index, const vector<Vector3D> & velocities,
		vector<char> const& nomove, vector<size_t> &neigh_buf)
	{
		if (nomove[index] == 1)
			return;
		Vector3D const& point = tess.GetMeshPoint(index);
		tess.GetNeighbors(index, neigh_buf);
		vector<size_t> &neigh = neigh_buf;
		size_t N = neigh.size();
		size_t min_loc = 0;
		double min_d = fastabs(point - tess.GetMeshPoint(neigh[0]));
		for (size_t i = 1; i < N; ++i)
		{
			double d = fastabs(point - tess.GetMeshPoint(neigh[i]));
			if (d < min_d)
			{
				min_d = d;
				min_loc = i;
			}
		}
		// A neighbour past the velocity and no-move arrays (MPI ghosts are
		// inside them once synchronized) is, on a serial partial
		// (individual-event) mesh, an unbuilt support point: it has no cell
		// geometry or velocity here, so it slows nothing down.
		size_t const nearest = neigh[min_loc];
		double other_width = tess.IsPointOutsideBox(nearest) ||
			nearest >= velocities.size() || nearest >= nomove.size() ?
			0.0 : tess.GetWidth(nearest);
		if (min_d < 0.4 * std::min(R, other_width))
		{
			size_t face = tess.GetCellFaces(index)[min_loc];
			size_t other = tess.GetFaceNeighbors(face).first == index ? tess.GetFaceNeighbors(face).second :
				tess.GetFaceNeighbors(face).first;
			Vector3D parallel = tess.FaceCM(face) - 0.5*(tess.GetMeshPoint(index) + tess.GetMeshPoint(other));
			parallel *= (1.0 / (fastabs(parallel) + 1e-20));
			Vector3D v_par = parallel * ScalarProd(velocity, parallel);
			Vector3D v_norm = velocity - parallel * ScalarProd(velocity, parallel);
			if (!tess.IsPointOutsideBox(other))
			{
				if (nomove.at(other) == 1)
				{
					v_par = Vector3D();
				}
				else
				{
					double d_over_R_self = fastabs(tess.GetCellCM(index) - tess.GetMeshPoint(index)) / R;
					double R_other = tess.GetWidth(other);
					double d_over_R_other = fastabs(tess.GetCellCM(other) - tess.GetMeshPoint(other)) / R_other;
					double d_over_R_max = std::max(d_over_R_self, d_over_R_other);
					if (d_over_R_max < 0.5)
					{
						if (d_over_R_self < d_over_R_other)
							v_par = parallel * ScalarProd(velocities[other], parallel);
					}
				}
			}
			velocity = v_norm + v_par;
		}
	}

    // The overshoot limit of one target; it reads and writes v[i] only.
    void CorrectPointOverShoot(vector<Vector3D>& v, double dt, Tessellation3D const& tess, size_t n, size_t i,
        std::vector<size_t>& neighbors)
    {
		Vector3D const point(tess.GetMeshPoint(i));
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
					if(v_diff > 0.3 * R_diff)
						v[i] *= R_diff * 0.3 / v_diff;
				}
			}
		}
    }

	void CorrectPointsOverShoot(vector<Vector3D> &v, double dt, Tessellation3D const& tess)
	{
		// check that we don't go outside grid
		size_t const n = tess.GetPointNo();
		const double inv_dt = 1.0 / dt;
		std::vector<size_t> neighbors;
		for (size_t i = 0; i < n; ++i)
            CorrectPointOverShoot(v, dt, tess, n, i, neighbors);
		return;
	}

    // The only point whose velocity SlowDown(index) can read: the face neighbour of its nearest neighbouring
    // generator, chosen as SlowDown chooses it, or SIZE_MAX when SlowDown returns before reading.  Geometry and the
    // no-move flags only; the distance test that gates the read is left out, so the dependency set is a superset.
    size_t SlowDownPartner(Tessellation3D const& tess, size_t index, vector<char> const& nomove,
        vector<size_t>& neigh)
    {
        if(nomove[index] == 1)
            return std::numeric_limits<size_t>::max();
        Vector3D const& point = tess.GetMeshPoint(index);
        tess.GetNeighbors(index, neigh);
        if(neigh.empty())
            return std::numeric_limits<size_t>::max();
        size_t min_loc = 0;
        double min_d = fastabs(point - tess.GetMeshPoint(neigh[0]));
        for(size_t i = 1; i < neigh.size(); ++i) {
            double const d = fastabs(point - tess.GetMeshPoint(neigh[i]));
            if(d < min_d) {
                min_d = d;
                min_loc = i;
            }
        }
        size_t const face = tess.GetCellFaces(index)[min_loc];
        return tess.GetFaceNeighbors(face).first == index ? tess.GetFaceNeighbors(face).second :
            tess.GetFaceNeighbors(face).first;
    }
}

void RoundCells3D::calc_dw(Vector3D &velocity, size_t i, const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Vector3D> & velocities, vector<char> const& nomove) const
{
	const Vector3D r = tess.GetMeshPoint(i);
	const Vector3D s = tess.GetCellCM(i);
	const double d = fastabs(s - r);
	const double R = tess.GetWidth(i);
	if (d < 0.9*eta_*R)
		return;
	double cs = 0;
#ifdef RICH_DEBUG
	try
	{
#endif
	tess.GetNeighbors(i, calc_dw_neigh_buf_);
	vector<size_t> &neigh = calc_dw_neigh_buf_;
	size_t N = neigh.size();
	for (size_t j = 0; j < N; ++j)
	{
		if (tess.IsPointOutsideBox(neigh[j]))
			continue;
#ifdef RICH_DEBUG
		try
		{
#endif
			cs = std::max(cs, eos_.dp2c(cells[neigh[j]].density, cells[neigh[j]].pressure,
					    cells[static_cast<size_t>(neigh[j])].tracers, ComputationalCell3D::tracerNames));
			cs = std::max(cs, fastabs(cells[neigh[j]].velocity));
#ifdef RICH_DEBUG
			if (!std::isfinite(cs))
				throw UniversalError("Bad cs in roundcells");
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("Error RoundCells3D::calc_dw", 0);
			eo.addEntry("Neigh Cell number", neigh[j]);
			eo.addEntry("Neigh index", j);
			throw eo;
	}
#endif
		
}
	  cs =std::max(cs, std::max(fastabs(cells[i].velocity), std::max(eos_.dp2c(cells[i].density, cells[i].pressure,cells[i].tracers, ComputationalCell3D::tracerNames), min_dw_)));
#ifdef RICH_DEBUG
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Error RoundCells3D::calc_dw", 0);
		eo.addEntry("Cell number", i);
		throw eo;
	}
#endif
	Vector3D dw = chi_ * cs * (s - r) / std::max(R, d);
	const double dw_mag = fastabs(dw);
	if (dw_mag > max_velocity_)
		dw *= max_velocity_ / dw_mag;
	velocity += dw;
	// SlowDown(velocity, tess, R, i, velocities, nomove, slowdown_neigh_buf_);
}

void RoundCells3D::calc_dw(Vector3D &velocity, size_t i, const Tessellation3D& tess, double dt, vector<ComputationalCell3D> const& cells,
 vector<Vector3D> & velocities, vector<char> const& nomove)const
{
	const Vector3D r = tess.GetMeshPoint(i);
	const Vector3D s = tess.GetCellCM(i);
	const double d = fastabs(s - r);
	const double R = tess.GetWidth(i);
	if (d < 0.9*eta_*R)
		return;
	tess.GetNeighbors(i, calc_dw2_neigh_buf_);
	vector<size_t> &neigh = calc_dw2_neigh_buf_;
	size_t N = neigh.size();
	double cs = 0;
	double min_R = R;
#ifdef RICH_DEBUG
	try
	{
#endif
	  cs = std::max(min_dw_, eos_.dp2c(cells[i].density, cells[i].pressure,	cells[i].tracers, ComputationalCell3D::tracerNames));
#ifdef RICH_DEBUG
	}
	catch (UniversalError &eo)
	{
		eo.addEntry("Error RoundCells3D::calc_dw", 0);
		eo.addEntry("Cell number", i);
		throw eo;
	}
#endif
	for (size_t j = 0; j < N; ++j)
	{
		if (tess.IsPointOutsideBox(neigh[j]))
			continue;
#ifdef RICH_DEBUG
		try
		{
#endif
			cs = std::max(cs, eos_.dp2c(cells[neigh[j]].density, cells[neigh[j]].pressure, cells[neigh[j]].tracers, ComputationalCell3D::tracerNames));
			cs = std::max(cs, fastabs(cells[neigh[j]].velocity));
#ifndef RICH_MPI
			// A serial partial (individual-event) mesh has no cell geometry for
			// its support points past the built cells; their primitives count.
			if(neigh[j] < tess.GetPointNo())
#endif
			min_R = std::min(min_R, tess.GetWidth(neigh[j]));
#ifdef RICH_DEBUG
			if (!std::isfinite(cs))
				throw UniversalError("Bad cs in roundcells");
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("Error RoundCells3D::calc_dw", 0);
			eo.addEntry("Neigh Cell number", neigh[j]);
			eo.addEntry("Neigh index", j);
			throw eo;
	}
#endif
		
}
	const double c_dt = std::max(std::max(dt_speed_*std::min(d, min_R) / dt, cs), min_dw_);
	Vector3D dw = chi_ * c_dt*(s - r) / std::max(R, d);
	const double dw_mag = fastabs(dw);
	if (dw_mag > max_velocity_)
		dw *= max_velocity_ / dw_mag;
	velocity += dw;
	// SlowDown(velocity, tess, R, i, velocities, nomove, slowdown_neigh_buf_);
}

void RoundCells3D::operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells,
	double time, vector<Vector3D> &res) const
{
	pm_(tess, cells, time, res);
	const size_t n = tess.GetPointNo();
	if (n == 0)
		return;
	size_t Nstick = ComputationalCell3D::stickerNames.size();
	vector<char> nomove(n, 0);
	vector<size_t> no_move_indeces;
	for (size_t i = 0; i < Nstick; ++i)
	{
		vector<std::string>::const_iterator it = std::find(no_move_.begin(), no_move_.end(),
								   ComputationalCell3D::stickerNames.at(i));
		if (it != no_move_.end())
			no_move_indeces.push_back(i);
	}
	Nstick = no_move_indeces.size();
	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < Nstick; ++j)
		{
			if (cells[i].stickers[no_move_indeces[j]])
			{
				res[i] = Vector3D();
				nomove[i] = 1;
				break;
			}
		}
	}
#ifndef RICH_MPI
	for (size_t i = 0; i < n; ++i)
		SlowDown(res[i], tess, tess.GetWidth(i), i, res, nomove, slowdown_neigh_buf_);
#endif
}

void RoundCells3D::ApplyFix(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells, double time,
	double dt, vector<Vector3D> &velocities)const
{
	ApplyFixImpl(tess, cells, nullptr, time, dt, velocities, nullptr);
}

void RoundCells3D::ApplyFixIndividual(
	Tessellation3D const& tess,
	vector<ComputationalCell3D> const& cells,
	vector<ComputationalCell3D> const& all_cells, double time, double dt,
	vector<Vector3D>& velocities,
	vector<Vector3D>& all_velocities) const
{
	ApplyFixImpl(tess, cells, &all_cells, time, dt, velocities,
		&all_velocities);
}

bool RoundCells3D::NoMoveStickerOnNoRank(size_t LocalStickerCount) const
{
    if(NoStickerAgreed_ < 0)
    {
        int NoSticker = LocalStickerCount == 0 ? 1 : 0;
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &NoSticker, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        NoStickerAgreed_ = NoSticker;
        AgreedLocalStickerCount_ = LocalStickerCount;
    }
    else if(LocalStickerCount != AgreedLocalStickerCount_)
        throw std::logic_error("RoundCells3D: the no-move sticker configuration changed after it was agreed");
    return NoStickerAgreed_ == 1;
}

void RoundCells3D::ApplyFixImpl(
	Tessellation3D const& tess,
	vector<ComputationalCell3D> const& cells,
	vector<ComputationalCell3D> const* all_cells, double time, double dt,
	vector<Vector3D>& velocities,
	vector<Vector3D>* all_velocities) const
{
    // RICH_INDIVIDUAL_HYDRO_TARGET_PASSES (B10): an active list is set only by the individual hydro step, on every
    // rank, when the mode is 1 or 2.
    std::vector<size_t> const* const Active = ActiveTargets_;
    int const TargetPassesMode = Active != nullptr ? IndividualHydroTargetPassesMode() : 0;
    if(TargetPassesMode != 2) {
        ApplyFixPasses(tess, cells, all_cells, time, dt, velocities, all_velocities,
            TargetPassesMode == 1 ? Active : nullptr);
        return;
    }
    // Mode 2: the restricted passes on copies, then the s48 passes on the caller's arrays (their result is kept).
    vector<Vector3D> RestrictedVelocities = velocities;
    vector<Vector3D> RestrictedAllVelocities;
    if(all_velocities != nullptr)
        RestrictedAllVelocities = *all_velocities;
    ApplyFixPasses(tess, cells, all_cells, time, dt, RestrictedVelocities,
        all_velocities != nullptr ? &RestrictedAllVelocities : nullptr, Active);
    ApplyFixPasses(tess, cells, all_cells, time, dt, velocities, all_velocities, nullptr);
    VerifyTargetPasses(velocities, RestrictedVelocities, *Active);
}

void RoundCells3D::ApplyFixPasses(Tessellation3D const& tess, vector<ComputationalCell3D> const& cells,
    vector<ComputationalCell3D> const* all_cells, double time, double dt, vector<Vector3D>& velocities,
    vector<Vector3D>* all_velocities, std::vector<size_t> const* Active) const
{
	if(all_cells != nullptr)
		pm_.ApplyFixIndividual(tess, cells, *all_cells, time, dt,
			velocities, *all_velocities);
	else
	{
		pm_.ApplyFix(tess, cells, time, dt, velocities);
#ifdef RICH_MPI
		MPI_exchange_data(tess, velocities, true);
#endif
	}
	const size_t n = tess.GetPointNo();
	/*if (n == 0)
		return;*/
	vector<char> nomove(n, 0);
	size_t Nstick = ComputationalCell3D::stickerNames.size();
	vector<size_t> no_move_indeces;
	for (size_t i = 0; i < Nstick; ++i)
	{
		vector<std::string>::const_iterator it = std::find(no_move_.begin(), no_move_.end(),
								   ComputationalCell3D::stickerNames.at(i));
		if (it != no_move_.end())
			no_move_indeces.push_back(i);
	}
	Nstick = no_move_indeces.size();
    // RICH_ROUNDCELLS_SKIP_EMPTY_STICKERS (B4), agreed on every rank:
    //   0  the s48 path;
    //   1  when no rank configures a no-move sticker, no cell can be stuck: skip both sticker scans and the no-move
    //      sync and give nomove the size the sync gives it (the sync would only add zeros).  With stickers
    //      configured, an individual event scans as before and one reduction, which replaces the sync, tells
    //      whether any cell is stuck on any rank; when none is, the sync is skipped the same way;
    //   2  the s48 path, checked against that result wherever mode 1 would skip.
    // The checks the skipped sync makes on the mesh maps are made again by the velocity sync below.
    int const SkipMode = RoundCellsSkipEmptyStickersMode();
    bool const NoStickers = SkipMode != 0 && NoMoveStickerOnNoRank(Nstick);
#ifdef RICH_MPI
    size_t const SyncedSize = all_cells != nullptr ? tess.getMeshPoints().size() : tess.GetTotalPointNumber();
#else
    size_t const SyncedSize = all_cells != nullptr ? tess.getMeshPoints().size() : n;
#endif
    if(SkipMode == 1 && NoStickers)
        nomove.resize(SyncedSize, 0);
    else {
		for (size_t i = 0; i < n; ++i)
		{
			for (size_t j = 0; j < Nstick; ++j)
			{
				if (cells[i].stickers[no_move_indeces[j]])
				{
					velocities[i] = Vector3D();
					nomove[i] = 1;
					break;
				}
			}
		}
		if(all_cells != nullptr)
		{
			vector<char> all_nomove(all_cells->size(), 0);
			for(size_t i = 0; i < all_cells->size(); ++i)
				for(size_t j = 0; j < Nstick; ++j)
					if(all_cells->at(i).stickers[no_move_indeces[j]])
					{
						all_nomove[i] = 1;
						all_velocities->at(i) = Vector3D();
						break;
					}
            int AnyStuck = 1;
            if(SkipMode != 0 && !NoStickers) {
                auto const Stuck = [](char Value) { return Value != 0; };
                AnyStuck = std::any_of(nomove.begin(), nomove.end(), Stuck) ||
                    std::any_of(all_nomove.begin(), all_nomove.end(), Stuck) ? 1 : 0;
#ifdef RICH_MPI
                MPI_Allreduce(MPI_IN_PLACE, &AnyStuck, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
#endif
            }
            if(SkipMode == 1 && AnyStuck == 0)
                nomove.resize(SyncedSize, 0);
            else {
				tess.SyncPartialBuildData(nomove, all_nomove);
				for(size_t i = 0; i < std::min(nomove.size(), velocities.size()); ++i)
					if(nomove[i] != 0)
						velocities[i] = Vector3D();
            }
            if(SkipMode == 2 && (NoStickers || AnyStuck == 0))
                VerifyNoMoveSkip(nomove, all_nomove, SyncedSize, true);
		}
		else
		{
#ifdef RICH_MPI
			MPI_exchange_data(tess, nomove, true);
#endif
            if(SkipMode == 2 && NoStickers)
                VerifyNoMoveSkip(nomove, vector<char>(), SyncedSize, false);
		}
    }

    // B10 restriction.  The caller reads only the active targets' velocities.  Such a velocity is the kick, then
    // SlowDown, then the overshoot limit of that target.  SlowDown(c) reads the velocity of one partner p: the kicked
    // one when p > c, the slowed one when p < c (an ordered in-place sweep), the synced one when p is not a target.
    // So the slowed set S is the active targets closed under "partner with a lower index"; the kicked set K is S, the
    // targets partnered by S, and every target the velocity sync sends to a peer (whose SlowDown may read it).  S is
    // swept in ascending order, as the full sweep visits it.
    // A rank whose targets are all active runs the full passes; both paths make the same exchanges.
    if(Active != nullptr && Active->size() >= n) {
        LastTargetPassCounts_ = TargetPassCounts();
        LastTargetPassCounts_.Targets = n;
        LastTargetPassCounts_.Active = Active->size();
        LastTargetPassCounts_.Kicked = n;
        LastTargetPassCounts_.Slowed = n;
        Active = nullptr;
    }
    std::vector<unsigned char> Kick;
    std::vector<size_t> SlowedCells;
    if(Active != nullptr) {
        Kick.assign(n, 0);
        MarkSentTargets(tess, all_cells != nullptr, all_velocities != nullptr ? all_velocities->size() : 0, n, Kick);
        size_t SentCount = static_cast<size_t>(std::count(Kick.begin(), Kick.end(), 1));
        std::vector<unsigned char> Slow(n, 0);
        std::vector<size_t> Work;
        Work.reserve(Active->size());
        for(size_t const Target : *Active) {
            if(Target >= n)
                throw std::logic_error("RoundCells3D: an active target is outside the owned targets");
            if(Slow[Target] == 0) {
                Slow[Target] = 1;
                Work.push_back(Target);
            }
        }
        while(!Work.empty()) {
            size_t const Cell = Work.back();
            Work.pop_back();
            Kick[Cell] = 1;
            size_t const Partner = SlowDownPartner(tess, Cell, nomove, slowdown_neigh_buf_);
            if(Partner >= n)
                continue;
            Kick[Partner] = 1;
            if(Partner < Cell && Slow[Partner] == 0) {
                Slow[Partner] = 1;
                Work.push_back(Partner);
            }
        }
        for(size_t Cell = 0; Cell < n; ++Cell)
            if(Slow[Cell] != 0)
                SlowedCells.push_back(Cell);
        LastTargetPassCounts_.Targets = n;
        LastTargetPassCounts_.Active = Active->size();
        LastTargetPassCounts_.Kicked = static_cast<size_t>(std::count(Kick.begin(), Kick.end(), 1));
        LastTargetPassCounts_.Slowed = SlowedCells.size();
        LastTargetPassCounts_.Sent = SentCount;
    }
	if (cold_)
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (nomove[i] == 0) {
                if(Active == nullptr || Kick[i] != 0)
				    calc_dw(velocities.at(i), i, tess, dt, cells, velocities, nomove);
            }
			else
				velocities.at(i) = Vector3D();
		}
	}
	else
	{
		for (size_t i = 0; i < n; ++i)
		{
			if (nomove[i] == 0) {
                if(Active == nullptr || Kick[i] != 0)
				    calc_dw(velocities.at(i), i, tess, cells, velocities, nomove);
            }
			else
				velocities.at(i) = Vector3D();
		}
	}
	if(all_velocities != nullptr)
		tess.SyncPartialBuildData(velocities, *all_velocities);
#ifdef RICH_MPI
	else
		MPI_exchange_data(tess, velocities, true);
    if(Active == nullptr) {
	    for (size_t i = 0; i < n; ++i)
		    SlowDown(velocities[i], tess, tess.GetWidth(i), i, velocities, nomove, slowdown_neigh_buf_);
    }
    else {
        for(size_t const Cell : SlowedCells)
            SlowDown(velocities[Cell], tess, tess.GetWidth(Cell), Cell, velocities, nomove, slowdown_neigh_buf_);
    }
#endif
	velocities.resize(n);
    if(Active == nullptr)
	    CorrectPointsOverShoot(velocities, dt, tess);
    else {
        std::vector<size_t> Neighbors;
        for(size_t const Target : *Active)
            CorrectPointOverShoot(velocities, dt, tess, n, Target, Neighbors);
    }
}
