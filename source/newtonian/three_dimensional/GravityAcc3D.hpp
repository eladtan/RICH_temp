#ifndef GRAVITY_ACC_3D_HPP
#define GRAVITY_ACC_3D_HPP

#ifdef RICH_MPI
	    #include "3D/gravity/DistributedGravityCalculator.hpp"
#endif // RICH_MPI
#include "3D/gravity/GravityTree.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "newtonian/three_dimensional/ConservativeForce3D.hpp"
#include "misc/memory_profile.hpp"

class GravityAcceleration3D : public Acceleration3D
{
public:
    GravityAcceleration3D(double theta, bool quadrupole = false, double G = 1): theta(theta), quadrupole(quadrupole), G(G), lastWalkTime_(0){};

    double getLastWalkTime() const { return lastWalkTime_; }

    bool SupportsIndividualTargetEvaluation(void) const override
    {
        return true;
    }

    void operator()(const Tessellation3D& tess, const vector<ComputationalCell3D>& cells, const vector<Conserved3D>& fluxes, const double time, vector<Vector3D> &acc) const
    {
        (void) fluxes;
        (void) time;
        MEMORY_PROFILE_SCOPE("gravity source");

        size_t N = tess.GetPointNo();
        this->points_.resize(N);
        this->masses_.resize(N);
        for(size_t cellIdx = 0; cellIdx < N; cellIdx++)
        {
            this->points_[cellIdx] = tess.GetCellCM(cellIdx);
            this->masses_[cellIdx] = (cells[cellIdx].density) * (tess.GetVolume(cellIdx));
        }
        #ifdef RICH_MPI
            DistributedGravityCalculator agent(tess, this->masses_, this->theta, this->quadrupole);
            acc = agent.getAcceleration(this->points_);
            lastWalkTime_ = agent.getWalkTime();
        #else // RICH_MPI
            std::pair<Vector3D, Vector3D> boundaries = tess.GetBoxCoordinates();
            GravityTree<Vector3D> gravTree(boundaries.first, boundaries.second, this->theta, this->quadrupole);
            this->massed_points_.clear();
            this->massed_points_.reserve(N);
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                this->massed_points_.emplace_back(MassedPoint<Vector3D>(this->points_[pointIdx], this->masses_[pointIdx]));
            }
            {
                MEMORY_PROFILE_SCOPE("gravity tree build");
                gravTree.build(this->massed_points_);
            }
            
            double wt0 = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            acc.resize(N);
            for(size_t pointIdx = 0; pointIdx < N; pointIdx++)
            {
                acc[pointIdx] = gravTree.gravity(this->points_[pointIdx]);
            }
            lastWalkTime_ = std::chrono::duration<double>(std::chrono::high_resolution_clock::now().time_since_epoch()).count() - wt0;
        #endif // RICH_MPI

        for(size_t i = 0; i < N; ++i)
            acc[i] *= this->G;
    }

    void EvaluateIndividualTargets(
        std::pair<Vector3D, Vector3D> const& bounds,
        vector<Vector3D> const& source_points,
        vector<double> const& source_masses,
        vector<std::uint64_t> const& /*source_ids*/,
        vector<Vector3D> const& target_points,
        vector<ComputationalCell3D> const& /*target_cells*/,
        double /*time*/,
        vector<Vector3D>& acc) const override
    {
#ifdef RICH_MPI
        MEMORY_PROFILE_SCOPE("individual distributed gravity source");
	        auto const total_start = std::chrono::steady_clock::now();
	        DistributedGravityCalculator agent(source_points, source_masses,
	            bounds.first, bounds.second, this->theta, this->quadrupole,
	            MPI_COMM_WORLD);
	        auto const tree_end = std::chrono::steady_clock::now();
	        acc = agent.getAcceleration(target_points);
	        auto const acceleration_end = std::chrono::steady_clock::now();
	        lastWalkTime_ = agent.getWalkTime();
	        DistributedGravityCalculator::SolveTiming const solve_timing =
	            agent.getLastSolveTiming();
	        for(Vector3D& acceleration : acc)
	            acceleration *= this->G;
	        auto const total_end = std::chrono::steady_clock::now();
	        double const tree_seconds = std::chrono::duration<double>(
	            tree_end - total_start).count();
	        double const scale_seconds = std::chrono::duration<double>(
	            total_end - acceleration_end).count();
		        double const total_seconds = std::chrono::duration<double>(
		            total_end - total_start).count();
		        double const accounted_seconds = tree_seconds +
	            solve_timing.exchangePlan + solve_timing.countPostAndRelease +
	            solve_timing.localFlatBuild + solve_timing.localWalk +
	            solve_timing.payloadWaitAndUnpack +
	            solve_timing.remoteTreeBuild + solve_timing.remoteFlatBuild +
	            solve_timing.remoteWalk + scale_seconds;
	        ReportIndividualTiming({{tree_seconds,
	            solve_timing.exchangePlan, solve_timing.countPostAndRelease,
	            solve_timing.localFlatBuild, solve_timing.localWalk,
	            solve_timing.payloadWaitAndUnpack,
	            solve_timing.remoteTreeBuild, solve_timing.remoteFlatBuild,
	            solve_timing.remoteWalk, scale_seconds,
	            std::max(total_seconds - accounted_seconds, 0.0), total_seconds,
	            static_cast<double>(source_points.size()),
	            static_cast<double>(target_points.size()),
	            solve_timing.sentValues, solve_timing.receivedValues,
	            solve_timing.remoteFlatNodes,
	            solve_timing.activeExchangeNeighbors}});
#else
        MEMORY_PROFILE_SCOPE("individual gravity source");
        if(source_points.size() != source_masses.size())
            throw std::invalid_argument(
                "Individual gravity source point/mass count mismatch");

        GravityTree<Vector3D> gravTree(bounds.first, bounds.second,
            this->theta, this->quadrupole);
        this->massed_points_.clear();
        this->massed_points_.reserve(source_points.size());
        for(size_t source = 0; source < source_points.size(); ++source)
            this->massed_points_.emplace_back(MassedPoint<Vector3D>(
                source_points[source], source_masses[source]));
        {
            MEMORY_PROFILE_SCOPE("individual gravity tree build");
            gravTree.build(this->massed_points_);
        }

        double const walk_start = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        acc.resize(target_points.size());
        for(size_t target = 0; target < target_points.size(); ++target)
        {
            acc[target] = gravTree.gravity(target_points[target]);
            acc[target] *= this->G;
        }
        lastWalkTime_ = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count() -
            walk_start;
#endif
    }

private:
	static bool IndividualTimingEnabled(void)
	{
		char const* const value = std::getenv("RICH_INDIVIDUAL_PERF_TRACE");
		return value != nullptr && value[0] != '\0' &&
			std::strcmp(value, "0") != 0 &&
			std::strcmp(value, "false") != 0 &&
			std::strcmp(value, "off") != 0 &&
			std::strcmp(value, "no") != 0;
	}

	static void ReportIndividualTiming(std::array<double, 18> const& local)
	{
		if(!IndividualTimingEnabled())
			return;
		int rank = 0;
		int rank_count = 1;
#ifdef RICH_MPI
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
#endif
		std::vector<double> gathered;
		if(rank == 0)
			gathered.resize(static_cast<std::size_t>(rank_count) * local.size());
#ifdef RICH_MPI
		MPI_Gather(local.data(), static_cast<int>(local.size()), MPI_DOUBLE,
			rank == 0 ? gathered.data() : nullptr,
			static_cast<int>(local.size()), MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
		gathered.assign(local.begin(), local.end());
#endif
		if(rank != 0)
			return;
		auto const statistic = [&](std::size_t const field,
			double const quantile) {
			std::vector<double> values(static_cast<std::size_t>(rank_count));
			for(int source_rank = 0; source_rank < rank_count; ++source_rank)
				values[static_cast<std::size_t>(source_rank)] = gathered[
					static_cast<std::size_t>(source_rank) * local.size() + field];
			std::sort(values.begin(), values.end());
			if(quantile == 0.5 && values.size() % 2 == 0)
				return 0.5 * (values[values.size() / 2 - 1] +
					values[values.size() / 2]);
			std::size_t const index = std::min(values.size() - 1,
				static_cast<std::size_t>(std::ceil(quantile * values.size())) - 1);
			return values[index];
		};
		static std::array<char const*, 11> const names = {{
			"tree_setup", "exchange_plan", "count_post_release",
			"local_flat_build", "local_tree_walk", "payload_wait_unpack",
			"remote_tree_build", "remote_flat_build", "remote_tree_walk",
			"scale", "unaccounted"}};
		static std::array<char const*, 4> const count_names = {{
			"sent_values", "received_values", "remote_flat_nodes",
			"active_exchange_neighbors"}};
		std::size_t const total_index = names.size();
		std::size_t const source_index = total_index + 1;
		std::size_t const target_index = total_index + 2;
		std::size_t const counts_index = total_index + 3;
		double source_cells_global = 0.0;
		double target_cells_global = 0.0;
		std::size_t critical_rank = 0;
		for(int source_rank = 0; source_rank < rank_count; ++source_rank) {
			std::size_t const offset =
				static_cast<std::size_t>(source_rank) * local.size();
			source_cells_global += gathered[offset + source_index];
			target_cells_global += gathered[offset + target_index];
			std::size_t const critical_offset = critical_rank * local.size();
			if(gathered[offset + total_index] >
				gathered[critical_offset + total_index])
				critical_rank = static_cast<std::size_t>(source_rank);
		}
		std::cout << "INDIVIDUAL_GRAVITY_PHASE_TIMING"
			<< " ranks=" << rank_count
			<< " source_cells_global=" << source_cells_global
			<< " target_cells_global=" << target_cells_global;
		for(std::size_t field = 0; field < names.size(); ++field)
			std::cout << " " << names[field] << "_median="
				<< statistic(field, 0.5)
				<< " " << names[field] << "_p95=" << statistic(field, 0.95)
				<< " " << names[field] << "_max=" << statistic(field, 1.0);
		std::cout << " total_median=" << statistic(total_index, 0.5)
			<< " total_p95=" << statistic(total_index, 0.95)
			<< " total_max=" << statistic(total_index, 1.0)
			<< " source_cells_median=" << statistic(source_index, 0.5)
			<< " source_cells_max=" << statistic(source_index, 1.0)
			<< " target_cells_median=" << statistic(target_index, 0.5)
			<< " target_cells_max=" << statistic(target_index, 1.0);
		for(std::size_t field = 0; field < count_names.size(); ++field)
			std::cout << " " << count_names[field] << "_median="
				<< statistic(counts_index + field, 0.5)
				<< " " << count_names[field] << "_p95="
				<< statistic(counts_index + field, 0.95)
				<< " " << count_names[field] << "_max="
				<< statistic(counts_index + field, 1.0);
		std::cout
			<< " critical_rank=" << critical_rank;
		std::size_t const critical_offset = critical_rank * local.size();
		for(std::size_t field = 0; field < names.size(); ++field)
			std::cout << " " << names[field] << "_critical="
				<< gathered[critical_offset + field];
		std::cout << " total_critical="
				<< gathered[critical_offset + total_index]
			<< " source_cells_critical="
				<< gathered[critical_offset + source_index]
			<< " target_cells_critical="
				<< gathered[critical_offset + target_index];
		for(std::size_t field = 0; field < count_names.size(); ++field)
			std::cout << " " << count_names[field] << "_critical="
				<< gathered[critical_offset + counts_index + field];
		std::cout << std::endl;
	}

    double theta, G;
    bool quadrupole;
    mutable double lastWalkTime_;
    mutable std::vector<Vector3D> points_;
    mutable std::vector<gravity_result_t> masses_;
	    mutable std::vector<MassedPoint<Vector3D>> massed_points_;
};

#endif // GRAVITY_ACC_3D_HPP
