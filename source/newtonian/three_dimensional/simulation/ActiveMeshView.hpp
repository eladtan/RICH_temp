#ifndef ACTIVE_MESH_VIEW_HPP
#define ACTIVE_MESH_VIEW_HPP

#include <cstddef>
#include <limits>
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
        bool const have_time_bins =
            global.cell_time_bins.size() == global.cell_time_steps.size();
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
        auto sync = [this](auto& local_data, auto const& global_data) {
            auto all_data = global_data;
            tess_->SyncPartialBuildData(local_data, all_data);
        };
        sync(local.active_mask, global.active_mask);
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

#endif // ACTIVE_MESH_VIEW_HPP
