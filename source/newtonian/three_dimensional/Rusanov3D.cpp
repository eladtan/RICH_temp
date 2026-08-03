#include "Rusanov3D.hpp"

#include <algorithm>
#include <cmath>

namespace
{
Conserved3D ale_flux(ComputationalCell3D const& cell,
	Conserved3D const& conserved,
	double normal_velocity,
	double face_velocity,
	Vector3D const& normal)
{
	double const relative_velocity = normal_velocity - face_velocity;
	Conserved3D result;
	result.mass = cell.density * relative_velocity;
	result.momentum =
		cell.density * relative_velocity * cell.velocity +
		cell.pressure * normal;
	result.energy =
		conserved.energy * relative_velocity +
		cell.pressure * normal_velocity;
	result.Erad = conserved.Erad * relative_velocity;
	result.Erad_dt = conserved.Erad_dt * relative_velocity;
	result.Erad_dt_dt = conserved.Erad_dt_dt * relative_velocity;
	for (size_t i = 0; i < MAX_TRACERS; ++i)
		result.tracers[i] = conserved.tracers[i] * relative_velocity;
	for (size_t i = 0; i < ENERGY_GROUPS_NUM; ++i)
		result.Eg[i] = conserved.Eg[i] * relative_velocity;
	return result;
}
}

Conserved3D Rusanov3D::operator()(ComputationalCell3D const& left,
	ComputationalCell3D const& right,
	double face_velocity,
	EquationOfState const& eos,
	Vector3D const& normal) const
{
	Conserved3D left_conserved;
	Conserved3D right_conserved;
	PrimitiveToConserved(left, 1.0, left_conserved);
	PrimitiveToConserved(right, 1.0, right_conserved);

	double const left_normal_velocity = ScalarProd(left.velocity, normal);
	double const right_normal_velocity = ScalarProd(right.velocity, normal);
	double const left_sound_speed = eos.de2c(left.density,
		left.internal_energy, left.tracers,
		ComputationalCell3D::tracerNames);
	double const right_sound_speed = eos.de2c(right.density,
		right.internal_energy, right.tracers,
		ComputationalCell3D::tracerNames);
	double const signal_speed = std::max(
		std::abs(left_normal_velocity - face_velocity) + left_sound_speed,
		std::abs(right_normal_velocity - face_velocity) + right_sound_speed);

	Conserved3D const left_flux = ale_flux(left, left_conserved,
		left_normal_velocity, face_velocity, normal);
	Conserved3D const right_flux = ale_flux(right, right_conserved,
		right_normal_velocity, face_velocity, normal);
	Conserved3D result = 0.5 * (left_flux + right_flux) -
		0.5 * signal_speed * (right_conserved - left_conserved);
	result.internal_energy = 0;
	return result;
}
