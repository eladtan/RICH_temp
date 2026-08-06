#include "SphericalShellGeometry3D.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
std::uint64_t mix_bits(std::uint64_t value)
{
	value += 0x9e3779b97f4a7c15ULL;
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
	value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
	return value ^ (value >> 31);
}

std::uint64_t radius_hash(double radius)
{
	static_assert(sizeof(double) == sizeof(std::uint64_t),
		"Spherical shell mesh hashing requires 64-bit doubles");
	std::uint64_t bits = 0;
	std::memcpy(&bits, &radius, sizeof(bits));
	return mix_bits(bits);
}
}

SphericalShellGeometry3D::SphericalShellGeometry3D(Vector3D center,
	std::vector<double> shell_radii,
	double relative_radius_tolerance,
	double absolute_radius_tolerance)
	: center_(center),
	  shell_radii_(std::move(shell_radii)),
	  relative_radius_tolerance_(relative_radius_tolerance),
	  absolute_radius_tolerance_(absolute_radius_tolerance)
{
	NormalizeShellRadii();
}

void SphericalShellGeometry3D::NormalizeShellRadii()
{
	shell_radii_.erase(std::remove_if(shell_radii_.begin(),
		shell_radii_.end(), [](double radius) {
			return !(radius > 0) || !std::isfinite(radius);
		}), shell_radii_.end());
	std::sort(shell_radii_.begin(), shell_radii_.end());
	shell_radii_.erase(std::unique(shell_radii_.begin(),
		shell_radii_.end(), [this](double left, double right) {
			return std::abs(left - right) <= RadiusTolerance(left, right);
		}), shell_radii_.end());
}

double SphericalShellGeometry3D::RadiusTolerance(double left, double right) const
{
	return std::max(absolute_radius_tolerance_, relative_radius_tolerance_ * std::max({1.0, std::abs(left), std::abs(right)}));
}

bool SphericalShellGeometry3D::MeshChanged(Tessellation3D const& tess) const
{
	unsigned long long signature[3] = {0, 0, 0};
	unsigned long long signature_xor = 0;
	for (size_t i = 0; i < tess.GetPointNo(); ++i) {
		double const radius = abs(tess.GetMeshPoint(i) - center_);
		std::uint64_t const hash = radius_hash(radius);
		++signature[0];
		signature[1] += hash;
		signature[2] += hash * hash;
		signature_xor ^= hash;
	}
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, signature, 3, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
	MPI_Allreduce(MPI_IN_PLACE, &signature_xor, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
#endif
	bool const changed = !mesh_signature_initialized_ || signature[0] != mesh_point_count_ || signature[1] != mesh_hash_sum_ || signature[2] != mesh_hash_square_sum_ || signature_xor != mesh_hash_xor_;
	mesh_signature_initialized_ = true;
	mesh_point_count_ = signature[0];
	mesh_hash_sum_ = signature[1];
	mesh_hash_square_sum_ = signature[2];
	mesh_hash_xor_ = signature_xor;
	return changed;
}

void SphericalShellGeometry3D::InitializeReferencePopulation(Tessellation3D const& tess) const
{
	reference_population_initialized_ = true;
	if (shell_radii_.empty())
		return;

	std::vector<unsigned long long> populations(shell_radii_.size(), 0);
	for (size_t i = 0; i < tess.GetPointNo(); ++i) {
		int const shell = FindShell(abs(tess.GetMeshPoint(i) - center_));
		if (shell >= 0)
			++populations[static_cast<size_t>(shell)];
	}
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, populations.data(), static_cast<int>(populations.size()), MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
#endif
	populations.erase(std::remove(populations.begin(), populations.end(), 0), populations.end());
	if (populations.empty())
		return;
	std::sort(populations.begin(), populations.end());
	reference_shell_population_ = static_cast<double>(populations[populations.size() / 2]);
}

std::vector<SphericalShellGeometry3D::RadiusCluster> SphericalShellGeometry3D::BuildGlobalRadiusClusters(Tessellation3D const& tess) const
{
	std::vector<double> radii;
	radii.reserve(tess.GetPointNo());
	for (size_t i = 0; i < tess.GetPointNo(); ++i) {
		double const radius = abs(tess.GetMeshPoint(i) - center_);
		if (radius > 0 && std::isfinite(radius))
			radii.push_back(radius);
	}
	std::sort(radii.begin(), radii.end());

	std::vector<RadiusCluster> local_clusters;
	for (double const radius : radii) {
		if (local_clusters.empty() || std::abs(radius - local_clusters.back().radius) > RadiusTolerance(radius, local_clusters.back().radius)) {
			local_clusters.push_back({radius, 1});
			continue;
		}
		RadiusCluster& cluster = local_clusters.back();
		cluster.radius = (cluster.population * cluster.radius + radius) / (cluster.population + 1);
		cluster.population += 1;
	}

	int world_size = 1;
#ifdef RICH_MPI
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
	double const expected_local_population = reference_shell_population_ / static_cast<double>(world_size);
	double const local_population_threshold = std::max(1.0, std::ceil(0.25 * expected_local_population));

	std::vector<double> local_payload;
	for (RadiusCluster const& cluster : local_clusters) {
		if (cluster.population < local_population_threshold)
			continue;
		local_payload.push_back(cluster.radius);
		local_payload.push_back(cluster.population);
	}

	std::vector<double> global_payload;
#ifdef RICH_MPI
	int const local_size = static_cast<int>(local_payload.size());
	std::vector<int> counts(static_cast<size_t>(world_size), 0);
	MPI_Allgather(&local_size, 1, MPI_INT, counts.data(), 1, MPI_INT,
		MPI_COMM_WORLD);
	std::vector<int> displacements(static_cast<size_t>(world_size), 0);
	for (int rank = 1; rank < world_size; ++rank)
		displacements[static_cast<size_t>(rank)] =
			displacements[static_cast<size_t>(rank - 1)] +
			counts[static_cast<size_t>(rank - 1)];
	int const global_size = displacements.back() + counts.back();
	global_payload.resize(static_cast<size_t>(global_size));
	MPI_Allgatherv(local_payload.data(), local_size, MPI_DOUBLE,
		global_payload.data(), counts.data(), displacements.data(),
		MPI_DOUBLE, MPI_COMM_WORLD);
#else
	global_payload = local_payload;
#endif

	std::vector<RadiusCluster> entries;
	entries.reserve(global_payload.size() / 2);
	for (size_t i = 0; i + 1 < global_payload.size(); i += 2)
		entries.push_back({global_payload[i], global_payload[i + 1]});
	std::sort(entries.begin(), entries.end(),
		[](RadiusCluster const& left, RadiusCluster const& right) {
			return left.radius < right.radius;
		});

	std::vector<RadiusCluster> result;
	for (RadiusCluster const& entry : entries) {
		if (result.empty() ||
			std::abs(entry.radius - result.back().radius) >
				RadiusTolerance(entry.radius, result.back().radius)) {
			result.push_back(entry);
			continue;
		}
		RadiusCluster& cluster = result.back();
		double const total_population =
			cluster.population + entry.population;
		cluster.radius = (cluster.population * cluster.radius +
			entry.population * entry.radius) / total_population;
		cluster.population = total_population;
	}
	return result;
}

bool SphericalShellGeometry3D::Update(Tessellation3D const& tess) const
{
	if (!MeshChanged(tess))
		return false;
	if (!reference_population_initialized_)
		InitializeReferencePopulation(tess);
	if (!(reference_shell_population_ > 0) || shell_radii_.empty())
		return true;

	std::vector<RadiusCluster> const clusters = BuildGlobalRadiusClusters(tess);
	double const population_threshold = std::max(2.0, 0.5 * reference_shell_population_);
	double inner_margin = 0.5 * shell_radii_.front();
	double outer_margin = inner_margin;
	if (shell_radii_.size() > 1) {
		inner_margin = 2.0 * (shell_radii_[1] - shell_radii_[0]);
		outer_margin = 2.0 * (shell_radii_.back() -
			shell_radii_[shell_radii_.size() - 2]);
	}
	double const lower_bound = std::max(0.0,
		shell_radii_.front() - inner_margin);
	double const upper_bound = shell_radii_.back() + outer_margin;

	std::vector<double> discovered;
	for (RadiusCluster const& cluster : clusters) {
		if (cluster.population >= population_threshold &&
			cluster.radius >= lower_bound && cluster.radius <= upper_bound)
			discovered.push_back(cluster.radius);
	}
	if (!discovered.empty())
		shell_radii_ = std::move(discovered);
	return true;
}

int SphericalShellGeometry3D::FindShell(double radius) const
{
	auto const upper = std::lower_bound(shell_radii_.begin(),
		shell_radii_.end(), radius);
	int best = -1;
	double best_distance = std::numeric_limits<double>::infinity();
	if (upper != shell_radii_.end()) {
		best = static_cast<int>(upper - shell_radii_.begin());
		best_distance = std::abs(*upper - radius);
	}
	if (upper != shell_radii_.begin()) {
		auto const lower = upper - 1;
		double const distance = std::abs(*lower - radius);
		if (distance < best_distance) {
			best = static_cast<int>(lower - shell_radii_.begin());
			best_distance = distance;
		}
	}
	if (best < 0)
		return -1;
	double const shell_radius = shell_radii_[static_cast<size_t>(best)];
	return best_distance <= RadiusTolerance(radius, shell_radius) ? best : -1;
}

Vector3D const& SphericalShellGeometry3D::GetCenter() const
{
	return center_;
}

std::vector<double> const& SphericalShellGeometry3D::GetShellRadii() const
{
	return shell_radii_;
}
