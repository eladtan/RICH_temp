#include "SphericalShellProjector3D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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
	: SphericalShellProjector3D(
		std::make_shared<SphericalShellGeometry3D>(center,
			std::move(shell_radii), relative_radius_tolerance,
			absolute_radius_tolerance))
{}

SphericalShellProjector3D::SphericalShellProjector3D(
	std::shared_ptr<SphericalShellGeometry3D> shell_geometry)
	: shell_geometry_(std::move(shell_geometry))
{
	if (!shell_geometry_)
		throw UniversalError(
			"SphericalShellProjector3D requires shell geometry");
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
	shell_geometry_->Update(tess);
	std::vector<double> const& shell_radii =
		shell_geometry_->GetShellRadii();
	if (shell_radii.empty())
		return;

	std::vector<int> shell_ids(local_count, -1);
	std::vector<Vector3D> radial_directions(local_count);
	std::vector<double> shell_sums(shell_radii.size() * SHELL_SUM_STRIDE, 0.0);
	for (size_t i = 0; i < local_count; ++i) {
		Vector3D const generator_offset = tess.GetMeshPoint(i) -
			shell_geometry_->GetCenter();
		int const shell = shell_geometry_->FindShell(abs(generator_offset));
		if (shell < 0)
			continue;

		Vector3D const cm_offset = tess.GetCellCM(i) -
			shell_geometry_->GetCenter();
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
	return shell_geometry_->GetCenter();
}

std::vector<double> const& SphericalShellProjector3D::GetShellRadii() const
{
	return shell_geometry_->GetShellRadii();
}

std::shared_ptr<SphericalShellGeometry3D>
SphericalShellProjector3D::GetShellGeometry() const
{
	return shell_geometry_;
}
