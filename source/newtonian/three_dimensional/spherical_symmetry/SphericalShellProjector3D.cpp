#include "SphericalShellProjector3D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "misc/universal_error.hpp"

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
size_t const VOLUME_OFFSET = 0;
size_t const MASS_OFFSET = 1;
size_t const RADIAL_MOMENTUM_OFFSET = 2;
size_t const ENERGY_OFFSET = 3;
size_t const INTERNAL_ENERGY_OFFSET = 4;
size_t const TRACER_OFFSET = 5;
size_t const SHELL_SUM_STRIDE = TRACER_OFFSET + MAX_TRACERS;
}

SphericalShellProjector3D::SphericalShellProjector3D(Vector3D center,
	std::vector<double> shell_radii,
	double relative_radius_tolerance,
	double absolute_radius_tolerance)
	: center_(center),
	  shell_radii_(std::move(shell_radii)),
	  relative_radius_tolerance_(relative_radius_tolerance),
	  absolute_radius_tolerance_(absolute_radius_tolerance)
{
	shell_radii_.erase(std::remove_if(shell_radii_.begin(), shell_radii_.end(),
		[](double radius) { return !(radius > 0.0) || !std::isfinite(radius); }),
		shell_radii_.end());
	std::sort(shell_radii_.begin(), shell_radii_.end());
	shell_radii_.erase(std::unique(shell_radii_.begin(), shell_radii_.end(),
		[this](double left, double right) {
			double const tolerance = std::max(absolute_radius_tolerance_,
				relative_radius_tolerance_ * std::max(left, right));
			return std::abs(left - right) <= tolerance;
		}), shell_radii_.end());
}

void SphericalShellProjector3D::ProjectExtensives(
	Tessellation3D const& tess,
	std::vector<Conserved3D> const& input,
	std::vector<Conserved3D>& output) const
{
	ProjectLinear(tess, input, output);
	for (size_t i = 0; i < tess.GetPointNo(); ++i) {
		if (!(output[i].mass > 0))
			continue;
		output[i].internal_energy = output[i].energy -
			ScalarProd(output[i].momentum, output[i].momentum) /
			(2.0 * output[i].mass);
	}
}

void SphericalShellProjector3D::ProjectRates(
	Tessellation3D const& tess,
	std::vector<Conserved3D> const& input,
	std::vector<Conserved3D>& output) const
{
	ProjectLinear(tess, input, output);
}

void SphericalShellProjector3D::ProjectLinear(
	Tessellation3D const& tess,
	std::vector<Conserved3D> const& input,
	std::vector<Conserved3D>& output) const
{
	size_t const local_count = tess.GetPointNo();
	if (input.size() < local_count) {
		UniversalError error("SphericalShellProjector3D input is smaller than local mesh");
		error.addEntry("input_size", input.size());
		error.addEntry("local_cell_count", local_count);
		throw error;
	}

	output = input;
	if (shell_radii_.empty())
		return;

	std::vector<int> shell_ids(local_count, -1);
	std::vector<Vector3D> radial_directions(local_count);
	std::vector<double> shell_sums(shell_radii_.size() * SHELL_SUM_STRIDE, 0.0);
	for (size_t i = 0; i < local_count; ++i) {
		Vector3D const generator_offset = tess.GetMeshPoint(i) - center_;
		int const shell = FindShell(abs(generator_offset));
		if (shell < 0)
			continue;

		Vector3D const cm_offset = tess.GetCellCM(i) - center_;
		double const cm_radius = abs(cm_offset);
		if (!(cm_radius > std::numeric_limits<double>::min()))
			continue;

		shell_ids[i] = shell;
		radial_directions[i] = cm_offset / cm_radius;
		size_t const offset = static_cast<size_t>(shell) * SHELL_SUM_STRIDE;
		double const volume = tess.GetVolume(i);
		shell_sums[offset + VOLUME_OFFSET] += volume;
		shell_sums[offset + MASS_OFFSET] += input[i].mass;
		shell_sums[offset + RADIAL_MOMENTUM_OFFSET] +=
			ScalarProd(input[i].momentum, radial_directions[i]);
		shell_sums[offset + ENERGY_OFFSET] += input[i].energy;
		shell_sums[offset + INTERNAL_ENERGY_OFFSET] += input[i].internal_energy;
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			shell_sums[offset + TRACER_OFFSET + tracer] +=
				input[i].tracers[tracer];
	}

#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, shell_sums.data(),
		static_cast<int>(shell_sums.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	for (size_t i = 0; i < local_count; ++i) {
		int const shell = shell_ids[i];
		if (shell < 0)
			continue;
		size_t const offset = static_cast<size_t>(shell) * SHELL_SUM_STRIDE;
		double const shell_volume = shell_sums[offset + VOLUME_OFFSET];
		if (!(shell_volume > 0.0))
			continue;
		double const volume_fraction = tess.GetVolume(i) / shell_volume;
		output[i].mass = volume_fraction * shell_sums[offset + MASS_OFFSET];
		output[i].momentum =
			(volume_fraction * shell_sums[offset + RADIAL_MOMENTUM_OFFSET]) *
			radial_directions[i];
		output[i].energy = volume_fraction * shell_sums[offset + ENERGY_OFFSET];
		output[i].internal_energy =
			volume_fraction * shell_sums[offset + INTERNAL_ENERGY_OFFSET];
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			output[i].tracers[tracer] =
				volume_fraction * shell_sums[offset + TRACER_OFFSET + tracer];
	}
}

Vector3D const& SphericalShellProjector3D::GetCenter() const
{
	return center_;
}

std::vector<double> const& SphericalShellProjector3D::GetShellRadii() const
{
	return shell_radii_;
}

int SphericalShellProjector3D::FindShell(double radius) const
{
	auto const upper = std::lower_bound(shell_radii_.begin(), shell_radii_.end(),
		radius);
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
	double const tolerance = std::max(absolute_radius_tolerance_,
		relative_radius_tolerance_ * shell_radii_[static_cast<size_t>(best)]);
	return best_distance <= tolerance ? best : -1;
}
