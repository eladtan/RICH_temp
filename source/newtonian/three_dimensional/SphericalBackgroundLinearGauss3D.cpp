#include "SphericalBackgroundLinearGauss3D.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#ifdef RICH_MPI
#include <mpi.h>
#endif

namespace
{
size_t const VOLUME_OFFSET = 0;
size_t const MASS_OFFSET = 1;
size_t const RADIAL_MOMENTUM_OFFSET = 2;
size_t const PRESSURE_OFFSET = 3;
size_t const TRACER_OFFSET = 4;
size_t const SHELL_STRIDE = TRACER_OFFSET + MAX_TRACERS;

double minmod(double first, double second)
{
	if (first * second <= 0)
		return 0;
	return std::copysign(std::min(std::abs(first), std::abs(second)),
		first);
}

double limited_centered_slope(double left_value, double center_value,
	double right_value, double left_radius, double center_radius,
	double right_radius)
{
	double const left = (center_value - left_value) /
		(center_radius - left_radius);
	double const right = (right_value - center_value) /
		(right_radius - center_radius);
	double const centered = (right_value - left_value) /
		(right_radius - left_radius);
	return minmod(centered, minmod(2.0 * left, 2.0 * right));
}
}

SphericalBackgroundLinearGauss3D::SphericalBackgroundLinearGauss3D(
	EquationOfState const& eos,
	Ghost3D const& ghost,
	Vector3D center,
	std::vector<double> shell_radii,
	bool slope_limiter,
	double delta_v,
	double theta,
	double delta_pressure)
	: eos_(eos),
	  center_(center),
	  shell_radii_(std::move(shell_radii)),
	  full_reconstruction_(eos, ghost, slope_limiter, delta_v, theta,
		  delta_pressure),
	  background_reconstruction_(eos, ghost, slope_limiter, delta_v,
		  theta, delta_pressure)
{
	std::sort(shell_radii_.begin(), shell_radii_.end());
	shell_radii_.erase(std::unique(shell_radii_.begin(),
		shell_radii_.end(), [](double left, double right) {
			return std::abs(left - right) <=
				1e-12 * std::max({1.0, left, right});
		}), shell_radii_.end());
}

int SphericalBackgroundLinearGauss3D::FindShell(double radius) const
{
	auto const upper = std::lower_bound(shell_radii_.begin(),
		shell_radii_.end(), radius);
	int best = -1;
	double distance = std::numeric_limits<double>::infinity();
	if (upper != shell_radii_.end()) {
		best = static_cast<int>(upper - shell_radii_.begin());
		distance = std::abs(*upper - radius);
	}
	if (upper != shell_radii_.begin()) {
		auto const lower = upper - 1;
		double const candidate = std::abs(*lower - radius);
		if (candidate < distance) {
			best = static_cast<int>(lower - shell_radii_.begin());
			distance = candidate;
		}
	}
	if (best < 0)
		return -1;
	double const tolerance = 1e-9 *
		std::max(1.0, shell_radii_[static_cast<size_t>(best)]);
	return distance <= tolerance ? best : -1;
}

void SphericalBackgroundLinearGauss3D::PrepareBackground(
	Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells) const
{
	size_t const local_count = tess.GetPointNo();
	size_t const available_count = std::min(cells.size(),
		tess.GetTotalPointNumber());
	std::vector<double> sums(shell_radii_.size() * SHELL_STRIDE, 0);
	for (size_t i = 0; i < local_count; ++i) {
		int const shell = FindShell(abs(tess.GetMeshPoint(i) - center_));
		if (shell < 0)
			continue;
		Vector3D const cm_offset = tess.GetCellCM(i) - center_;
		double const cm_radius = abs(cm_offset);
		if (!(cm_radius > 0))
			continue;
		Vector3D const direction = cm_offset / cm_radius;
		double const volume = tess.GetVolume(i);
		double const mass = volume * cells[i].density;
		size_t const offset = static_cast<size_t>(shell) * SHELL_STRIDE;
		sums[offset + VOLUME_OFFSET] += volume;
		sums[offset + MASS_OFFSET] += mass;
		sums[offset + RADIAL_MOMENTUM_OFFSET] += mass *
			ScalarProd(cells[i].velocity, direction);
		sums[offset + PRESSURE_OFFSET] += volume * cells[i].pressure;
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			sums[offset + TRACER_OFFSET + tracer] +=
				mass * cells[i].tracers[tracer];
	}
#ifdef RICH_MPI
	MPI_Allreduce(MPI_IN_PLACE, sums.data(), static_cast<int>(sums.size()),
		MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

	shell_states_.assign(shell_radii_.size(), ShellPrimitive());
	for (size_t shell = 0; shell < shell_radii_.size(); ++shell) {
		size_t const offset = shell * SHELL_STRIDE;
		double const volume = sums[offset + VOLUME_OFFSET];
		shell_states_[shell].volume = volume;
		if (!(volume > 0))
			continue;
		double const mass = sums[offset + MASS_OFFSET];
		if (!(mass > 0))
			continue;
		shell_states_[shell].density =
			mass / volume;
		shell_states_[shell].radial_velocity =
			sums[offset + RADIAL_MOMENTUM_OFFSET] / mass;
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			shell_states_[shell].tracers[tracer] =
				sums[offset + TRACER_OFFSET + tracer] / mass;
		// Pressure is a primitive reconstruction variable.  Averaging total
		// energy here would fold resolved tangential-velocity variance into the
		// radial thermal background and contaminate the Cartesian perturbation.
		shell_states_[shell].pressure =
			sums[offset + PRESSURE_OFFSET] / volume;
	}

	shell_slopes_.assign(shell_radii_.size(), ShellPrimitive());
	for (size_t shell = 1; shell + 1 < shell_radii_.size(); ++shell) {
		ShellPrimitive const& left = shell_states_[shell - 1];
		ShellPrimitive const& center = shell_states_[shell];
		ShellPrimitive const& right = shell_states_[shell + 1];
		if (!(left.volume > 0 && center.volume > 0 && right.volume > 0))
			continue;
		double const left_radius = shell_radii_[shell - 1];
		double const center_radius = shell_radii_[shell];
		double const right_radius = shell_radii_[shell + 1];
		shell_slopes_[shell].density = limited_centered_slope(
			left.density, center.density, right.density,
			left_radius, center_radius, right_radius);
		shell_slopes_[shell].pressure = limited_centered_slope(
			left.pressure, center.pressure, right.pressure,
			left_radius, center_radius, right_radius);
		shell_slopes_[shell].radial_velocity = limited_centered_slope(
			left.radial_velocity, center.radial_velocity,
			right.radial_velocity, left_radius, center_radius,
			right_radius);
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			shell_slopes_[shell].tracers[tracer] =
				limited_centered_slope(left.tracers[tracer],
					center.tracers[tracer], right.tracers[tracer],
					left_radius, center_radius, right_radius);
	}

	background_cells_ = cells;
	background_cells_.resize(available_count);
	shell_ids_.assign(available_count, -1);
	for (size_t i = 0; i < available_count; ++i) {
		int const shell = FindShell(abs(tess.GetMeshPoint(i) - center_));
		shell_ids_[i] = shell;
		if (shell < 0 || !(shell_states_[static_cast<size_t>(shell)].volume > 0))
			continue;
		Vector3D const cm_offset = tess.GetCellCM(i) - center_;
		double const cm_radius = abs(cm_offset);
		if (!(cm_radius > 0))
			continue;
		ShellPrimitive const& state =
			shell_states_[static_cast<size_t>(shell)];
		background_cells_[i].density = state.density;
		background_cells_[i].pressure = state.pressure;
		background_cells_[i].velocity =
			state.radial_velocity * cm_offset / cm_radius;
		for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
			background_cells_[i].tracers[tracer] =
				state.tracers[tracer];
		background_cells_[i].internal_energy = eos_.dp2e(
			background_cells_[i].density,
			background_cells_[i].pressure,
			background_cells_[i].tracers,
			ComputationalCell3D::tracerNames);
	}
}

SphericalBackgroundLinearGauss3D::ShellPrimitive
SphericalBackgroundLinearGauss3D::RadialState(
	int shell, double target_radius) const
{
	size_t const index = static_cast<size_t>(shell);
	ShellPrimitive result = shell_states_[index];
	double const delta = target_radius - shell_radii_[index];
	double const density_change = delta * shell_slopes_[index].density;
	double const pressure_change = delta * shell_slopes_[index].pressure;
	double factor = 1.0;
	if (density_change < 0)
		factor = std::min(factor, (1.0 - 1e-12) *
			result.density / (-density_change));
	if (pressure_change < 0)
		factor = std::min(factor, (1.0 - 1e-12) *
			result.pressure / (-pressure_change));
	factor = std::max(0.0, std::min(1.0, factor));
	result.density += factor * density_change;
	result.pressure += factor * pressure_change;
	result.radial_velocity +=
		factor * delta * shell_slopes_[index].radial_velocity;
	for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
		result.tracers[tracer] +=
			factor * delta * shell_slopes_[index].tracers[tracer];
	return result;
}

ComputationalCell3D SphericalBackgroundLinearGauss3D::Combine(
	ComputationalCell3D const& full,
	ComputationalCell3D const& cartesian_background,
	int shell,
	Vector3D const& direction,
	double target_radius) const
{
	if (shell < 0)
		return full;
	ShellPrimitive const radial = RadialState(shell, target_radius);
	double const density_delta =
		full.density - cartesian_background.density;
	double const pressure_delta =
		full.pressure - cartesian_background.pressure;
	Vector3D const velocity_delta =
		full.velocity - cartesian_background.velocity;
	double factor = 1.0;
	if (density_delta < 0)
		factor = std::min(factor,
			(1.0 - 1e-12) * radial.density / (-density_delta));
	if (pressure_delta < 0)
		factor = std::min(factor,
			(1.0 - 1e-12) * radial.pressure / (-pressure_delta));
	factor = std::max(0.0, std::min(1.0, factor));

	ComputationalCell3D result = full;
	result.density = radial.density + factor * density_delta;
	result.pressure = radial.pressure + factor * pressure_delta;
	result.velocity = radial.radial_velocity * direction +
		factor * velocity_delta;
	for (size_t tracer = 0; tracer < MAX_TRACERS; ++tracer)
		result.tracers[tracer] = radial.tracers[tracer] + factor *
			(full.tracers[tracer] -
				cartesian_background.tracers[tracer]);
	result.internal_energy = eos_.dp2e(result.density, result.pressure,
		result.tracers, ComputationalCell3D::tracerNames);
	return result;
}

void SphericalBackgroundLinearGauss3D::operator()(
	Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells,
	double time,
	std::vector<std::pair<ComputationalCell3D,
		ComputationalCell3D>>& result) const
{
	PrepareBackground(tess, cells);
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
		full_faces;
	std::vector<std::pair<ComputationalCell3D, ComputationalCell3D>>
		background_faces;
	full_reconstruction_(tess, cells, time, full_faces);
	background_reconstruction_(tess, background_cells_, time,
		background_faces);
	result = full_faces;

	for (size_t face = 0; face < result.size(); ++face) {
		auto const neighbors = tess.GetFaceNeighbors(face);
		Vector3D const face_offset = tess.FaceCM(face) - center_;
		double const face_radius = abs(face_offset);
		if (!(face_radius > 0))
			continue;
		Vector3D const direction = face_offset / face_radius;
		int const first_shell = neighbors.first < shell_ids_.size() ?
			shell_ids_[neighbors.first] : -1;
		int const second_shell = neighbors.second < shell_ids_.size() ?
			shell_ids_[neighbors.second] : -1;
		double const first_target =
			first_shell >= 0 && first_shell == second_shell ?
			shell_radii_[static_cast<size_t>(first_shell)] :
			face_radius;
		double const second_target =
			second_shell >= 0 && second_shell == first_shell ?
			shell_radii_[static_cast<size_t>(second_shell)] :
			face_radius;
		result[face].first = Combine(full_faces[face].first,
			background_faces[face].first, first_shell, direction,
			first_target);
		result[face].second = Combine(full_faces[face].second,
			background_faces[face].second, second_shell, direction,
			second_target);
	}
}

void SphericalBackgroundLinearGauss3D::BuildSlopes(
	Tessellation3D const& tess,
	std::vector<ComputationalCell3D> const& cells,
	double time)
{
	PrepareBackground(tess, cells);
	full_reconstruction_.BuildSlopes(tess, cells, time);
	background_reconstruction_.BuildSlopes(tess, background_cells_, time);
}

std::vector<Slope3D>& SphericalBackgroundLinearGauss3D::GetSlopes()
{
	return full_reconstruction_.GetSlopes();
}

void SphericalBackgroundLinearGauss3D::Interp(
	ComputationalCell3D& result,
	ComputationalCell3D const& cell,
	size_t cell_index,
	Vector3D const& cell_cm,
	Vector3D const& target,
	EquationOfState const& eos) const
{
	ComputationalCell3D full = cell;
	ComputationalCell3D background = background_cells_[cell_index];
	full_reconstruction_.Interp(full, cell, cell_index, cell_cm, target,
		eos);
	background_reconstruction_.Interp(background,
		background_cells_[cell_index], cell_index, cell_cm, target, eos);
	Vector3D const target_offset = target - center_;
	double const target_radius = abs(target_offset);
	if (!(target_radius > 0)) {
		result = full;
		return;
	}
	result = Combine(full, background, shell_ids_[cell_index],
		target_offset / target_radius, target_radius);
}
