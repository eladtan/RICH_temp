#ifndef ACTIVE_MESH_VIEW_HPP
#define ACTIVE_MESH_VIEW_HPP

#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>

#include "3D/tessellation/Tessellation3D.hpp"
#include "misc/universal_error.hpp"
#include "newtonian/three_dimensional/simulation/IndividualTimeStep.hpp"

/**
 * Immutable index translation for a full or partial tessellation build.
 * Tessellation indices are always local; scheduler/canonical arrays are always
 * indexed by the original owned-cell order.
 */
class ActiveMeshView
{
public:
    static std::size_t invalidIndex()
    {
        return std::numeric_limits<std::size_t>::max();
    }

    ActiveMeshView(Tessellation3D const& tess, std::size_t global_cell_count)
        : tess_(&tess),
          local_to_global_(tess.GetPointNo(), invalidIndex()),
          global_to_local_(global_cell_count, invalidIndex()),
          mesh_local_to_global_(tess.getMeshPoints().size(), invalidIndex())
	{
		Tessellation3D::AllPointsMap const& map = tess.GetIndicesInAllPoints();
		std::vector<unsigned char> remote_ghost(mesh_local_to_global_.size(), 0);
#ifdef RICH_MPI
		for(auto const& rank_ghosts : tess.GetGhostIndeces())
			for(std::size_t ghost : rank_ghosts)
				if(ghost < remote_ghost.size())
					remote_ghost[ghost] = 1;
#endif
		for(auto const& entry : map) {
			std::size_t const local = entry.first;
			std::size_t const global = entry.second;
			if(local < mesh_local_to_global_.size() &&
			   remote_ghost[local] == 0 &&
			   global < global_cell_count &&
			   !tess.IsPointOutsideBox(local))
                mesh_local_to_global_[local] = global;
        }
		for(std::size_t local = 0; local < local_to_global_.size(); ++local) {
			if(remote_ghost[local] != 0) {
				UniversalError error("ActiveMeshView: remote ghost appears in owned target range");
				error.addEntry("Local index", local);
				throw error;
			}
			auto const entry = map.find(local);
            if(entry == map.end()) {
                UniversalError error("ActiveMeshView: missing local-to-global map entry");
                error.addEntry("Local index", local);
                throw error;
            }
            std::size_t const global = entry->second;
            if(global >= global_cell_count) {
                UniversalError error("ActiveMeshView: mapped global index is out of range");
                error.addEntry("Local index", local);
                error.addEntry("Global index", global);
                error.addEntry("Global cell count", global_cell_count);
                throw error;
            }
            if(global_to_local_[global] != invalidIndex()) {
                UniversalError error("ActiveMeshView: duplicate global index in partial build");
                error.addEntry("Global index", global);
                throw error;
            }
            local_to_global_[local] = global;
            global_to_local_[global] = local;
            mesh_local_to_global_[local] = global;
        }
    }

    std::size_t localSize() const { return local_to_global_.size(); }
    std::size_t globalSize() const { return global_to_local_.size(); }

    std::size_t localToGlobal(std::size_t local) const
    {
        return local_to_global_.at(local);
    }

    std::vector<std::size_t> const& localToGlobalMapping() const
    {
        return local_to_global_;
    }

    std::size_t globalToLocal(std::size_t global) const
    {
        return global_to_local_.at(global);
    }

    bool containsGlobal(std::size_t global) const
    {
        return global < global_to_local_.size() &&
               global_to_local_[global] != invalidIndex();
    }

    std::vector<std::size_t> const& localToGlobalMap() const
    {
        return local_to_global_;
    }

    std::vector<std::size_t> const& globalToLocalMap() const
    {
        return global_to_local_;
    }

    IndividualStepContext remapContext(IndividualStepContext const& global,
        bool copy_gravity_sources = true) const
    {
        IndividualStepContext local;
        local.previous_event_tick = global.previous_event_tick;
        local.event_tick = global.event_tick;
        local.time_origin = global.time_origin;
        local.previous_event_time = global.previous_event_time;
        local.event_time = global.event_time;
        local.time_quantum = global.time_quantum;
        local.mesh_build_policy = global.mesh_build_policy;
        local.partial_build_fraction = global.partial_build_fraction;
        local.verify_partial_build = global.verify_partial_build;
        if(copy_gravity_sources) {
            local.gravity_source_points = global.gravity_source_points;
            local.gravity_source_masses = global.gravity_source_masses;
            local.gravity_source_ids = global.gravity_source_ids;
        }
        local.radiation_repair_accounting =
            global.radiation_repair_accounting;
        local.radiation_defect_accounting =
            global.radiation_defect_accounting;
        local.active_mask.assign(meshLocalSize(), 0);
        local.cell_time_steps.assign(meshLocalSize(),
            std::numeric_limits<double>::infinity());
        // Whether bins are synced must be one decision on every rank: an empty
        // owner (no time steps) votes neutral, otherwise it would take the bin
        // sync (0 == 0) while populated owners without bins skip it.
        int BinsVote = global.cell_time_steps.empty() ? 1 :
            (global.cell_time_bins.size() == global.cell_time_steps.size() ? 1 : 0);
#ifdef RICH_MPI
        MPI_Allreduce(MPI_IN_PLACE, &BinsVote, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif
        bool const have_time_bins = BinsVote != 0;
        if(have_time_bins)
            local.cell_time_bins.assign(meshLocalSize(), 0);
        local.primitive_ticks.resize(meshLocalSize());
        local.point_velocities.resize(meshLocalSize());
        local.cached_accelerations.resize(meshLocalSize());
        local.gravity_half_kick_pending.resize(meshLocalSize());
        for(std::size_t local_index = 0; local_index < meshLocalSize(); ++local_index) {
            std::size_t const global_index = mesh_local_to_global_[local_index];
            if(global_index == invalidIndex())
                continue;
            local.cell_time_steps[local_index] = global.cellTimeStep(global_index);
            if(have_time_bins)
                local.cell_time_bins[local_index] =
                    global.cell_time_bins[global_index];
            if(global_index < global.primitive_ticks.size())
                local.primitive_ticks[local_index] = global.primitive_ticks[global_index];
            if(global_index < global.point_velocities.size())
                local.point_velocities[local_index] = global.point_velocities[global_index];
            if(global_index < global.cached_accelerations.size())
                local.cached_accelerations[local_index] = global.cached_accelerations[global_index];
            if(global_index < global.gravity_half_kick_pending.size())
                local.gravity_half_kick_pending[local_index] =
                    global.gravity_half_kick_pending[global_index];
            if(local_index < localSize() && global.isActive(global_index)) {
                local.active_mask[local_index] = 1;
                local.active_indices.push_back(local_index);
            }
        }
#ifdef RICH_MPI
        // The immutable maps cover this rank's canonical owned cells.  Fill
        // inactive owned halo values and remote ghosts through the exact
        // partial-build exchange metadata so face operators can see remote
        // activity and remote cell intervals without extending the canonical
        // scheduler arrays.
        //
        // Every owned target (local index < localSize()) was filled above with
        // its canonical value, so the write-back half of SyncPartialBuildData
        // would only rewrite equal values: the arrays are synced read-only
        // against the canonical ones instead of being copied first (seven
        // owned-size copies per call).  The activity mask keeps the copy: a
        // target holds 1 for any nonzero canonical entry.
        {
            std::vector<unsigned char> all_active = global.active_mask;
            tess_->SyncPartialBuildData(local.active_mask, all_active);
        }
        auto sync = [this](auto& local_data, auto const& global_data) {
            tess_->SyncPartialBuildDataReadOnly(local_data, global_data);
        };
        sync(local.cell_time_steps, global.cell_time_steps);
        if(have_time_bins)
            sync(local.cell_time_bins, global.cell_time_bins);
        sync(local.primitive_ticks, global.primitive_ticks);
        sync(local.point_velocities, global.point_velocities);
        sync(local.cached_accelerations, global.cached_accelerations);
        sync(local.gravity_half_kick_pending,
             global.gravity_half_kick_pending);
#endif
        return local;
    }

    /**
     * remapContext(global).active_mask and .active_indices alone, built the
     * same way, for callers that only ask isActive() of the mesh points: one
     * collective sync instead of seven and no other per-cell arrays.  The
     * remaining fields keep their defaults.
     */
    IndividualStepContext RemapActivity(IndividualStepContext const& global) const
    {
        IndividualStepContext local;
        local.active_mask.assign(meshLocalSize(), 0);
        for(std::size_t LocalIndex = 0; LocalIndex < localSize(); ++LocalIndex) {
            std::size_t const GlobalIndex = mesh_local_to_global_[LocalIndex];
            if(GlobalIndex != invalidIndex() && global.isActive(GlobalIndex)) {
                local.active_mask[LocalIndex] = 1;
                local.active_indices.push_back(LocalIndex);
            }
        }
#ifdef RICH_MPI
        std::vector<unsigned char> AllActive = global.active_mask;
        tess_->SyncPartialBuildData(local.active_mask, AllActive);
#endif
        return local;
    }

    template<typename T>
    std::vector<T> gatherOwned(std::vector<T> const& global) const
    {
        std::vector<T> local;
        this->gatherOwnedInto(global, local);
        return local;
    }

    template<typename T>
    void gatherOwnedInto(std::vector<T> const& global,
                         std::vector<T>& local) const
    {
        if(global.size() < globalSize())
            throw UniversalError(
                "ActiveMeshView::gatherOwnedInto: global data is too short");
        local.resize(localSize());
        for(std::size_t i = 0; i < localSize(); ++i)
            local[i] = global[local_to_global_[i]];
    }

    template<typename T>
    void scatterOwned(std::vector<T> const& local, std::vector<T>& global) const
    {
        if(local.size() < localSize() || global.size() < globalSize())
            throw UniversalError("ActiveMeshView::scatterOwned: data size does not match mapping");
        for(std::size_t i = 0; i < localSize(); ++i)
            global[local_to_global_[i]] = local[i];
    }

	std::size_t meshLocalSize() const
	{
		return mesh_local_to_global_.size();
	}

	std::size_t meshLocalToGlobal(std::size_t local) const
	{
		return local < mesh_local_to_global_.size() ?
			mesh_local_to_global_[local] : invalidIndex();
	}

    /** Scatter target cells plus mapped passive halo recipients. */
    template<typename T>
    void scatterMeshOwned(std::vector<T> const& local, std::vector<T>& global) const
    {
        if(local.size() < meshLocalSize() || global.size() < globalSize())
            throw UniversalError("ActiveMeshView::scatterMeshOwned: data size does not match mapping");
        scatterOwned(local, global);
        for(std::size_t i = localSize(); i < meshLocalSize(); ++i) {
            std::size_t const global_index = mesh_local_to_global_[i];
            if(global_index != invalidIndex() && !containsGlobal(global_index))
                global[global_index] = local[i];
        }
    }

private:
    Tessellation3D const* tess_;
    std::vector<std::size_t> local_to_global_;
    std::vector<std::size_t> global_to_local_;
    std::vector<std::size_t> mesh_local_to_global_;
};

/**
 * Stable cell ID -> canonical index of a canonical cell array, kept across
 * events.  Map() answers every lookup exactly as an unordered_map filled by
 * emplace(cells[i].ID, i) for i = 0, 1, ... would (the first index wins for a
 * repeated ID; Unique() reports whether any ID repeated).  Refresh compares the
 * IDs with the stored ones in one sequential pass and hashes them again only
 * after they changed (AMR, rebalancing, box growth), instead of hashing every
 * owned cell on every event.
 */
class CanonicalIdIndex
{
public:
    template<typename CellT>
    void Refresh(std::vector<CellT> const& Cells, std::size_t Count)
    {
        if(Count > Cells.size())
            throw UniversalError("CanonicalIdIndex::Refresh: count exceeds the cell array");
        bool Unchanged = Valid_ && Ids_.size() == Count;
        for(std::size_t Index = 0; Unchanged && Index < Count; ++Index)
            Unchanged = Ids_[Index] == static_cast<std::size_t>(Cells[Index].ID);
        if(Unchanged)
            return;
        Valid_ = false;
        Ids_.resize(Count);
        IndexById_.clear();
        IndexById_.reserve(Count);
        Unique_ = true;
        for(std::size_t Index = 0; Index < Count; ++Index) {
            Ids_[Index] = static_cast<std::size_t>(Cells[Index].ID);
            if(!IndexById_.emplace(Ids_[Index], Index).second)
                Unique_ = false;
        }
        Valid_ = true;
    }

    std::unordered_map<std::size_t, std::size_t> const& Map() const { return IndexById_; }

    bool Unique() const { return Unique_; }

    void Release() noexcept
    {
        std::vector<std::size_t>().swap(Ids_);
        std::unordered_map<std::size_t, std::size_t>().swap(IndexById_);
        Valid_ = false;
        Unique_ = true;
    }

private:
    std::vector<std::size_t> Ids_;
    std::unordered_map<std::size_t, std::size_t> IndexById_;
    bool Valid_ = false;
    bool Unique_ = true;
};

/**
 * The calling thread's CanonicalIdIndex.  The hydro step and its extensive
 * updater index the same canonical cells, and ConditionExtensiveUpdater3D
 * builds a new DefaultExtensiveUpdater for every event, so the index is kept
 * here rather than in either object.  A reference from Map() stays valid until
 * the next Refresh.
 */
inline CanonicalIdIndex& SharedCanonicalIdIndex()
{
    thread_local CanonicalIdIndex Index;
    return Index;
}

#endif // ACTIVE_MESH_VIEW_HPP
