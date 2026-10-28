#include "Hllc3D.hpp"
#include "../../misc/universal_error.hpp"
#include "../../misc/utils.hpp"
#include "HllcKernels.hpp"
#include <cassert>
#include <iostream>

using namespace std;

namespace
{
	using hllc::WaveSpeeds;

	double sound_speed(ComputationalCell3D const& cell, EquationOfState const &eos, const char *side)
	{
#ifdef RICH_DEBUG
		try
		{
#endif
		  return eos.dp2c(cell.density, cell.pressure, cell.tracers, ComputationalCell3D::tracerNames);
#ifdef RICH_DEBUG
		}
		catch (UniversalError &eo)
		{
			eo.addEntry(string("Error in HLLC3D ") + side + " sound speed", 0);
			throw eo;
		}
#endif
	}

	void print_many_iterations(hllc::Primitive const& left, hllc::Primitive const& right)
	{
		std::cout << "Too many iterations in HLLC" << std::endl;
		std::cout << " Left density = " << left.density << " pressure = " << left.pressure << " internal_energy = " << left.internal_energy << " vx = " << left.vx <<
			" vy = " << left.vy << " vz = " << left.vz << std::endl;
		std::cout << " Right density = " << right.density << " pressure = " << right.pressure << " internal_energy = " << right.internal_energy << " vx = " << right.vx <<
			" vy = " << right.vy << " vz = " << right.vz << std::endl;
	}

	hllc::WaveSpeeds estimate_wave_speeds(ComputationalCell3D const& left, ComputationalCell3D const& right,
		EquationOfState const &eos, double gamma)
	{
		const double cl = sound_speed(left, eos, "left");
		const double cr = sound_speed(right, eos, "right");
		int iterations = 0;
		const hllc::WaveSpeeds ws = hllc::EstimateWaveSpeeds(left.density, left.pressure, left.velocity.x, cl,
			right.density, right.pressure, right.velocity.x, cr, gamma, iterations);
		if (iterations > 54)
		{
			print_many_iterations(hllc::Primitive{left.density, left.pressure, left.internal_energy, left.velocity.x, left.velocity.y, left.velocity.z},
				hllc::Primitive{right.density, right.pressure, right.internal_energy, right.velocity.x, right.velocity.y, right.velocity.z});
		}
		assert(iterations < 55);
		return ws;
	}

	UniversalError invalid_wave_speeds(hllc::Primitive const& left, hllc::Primitive const& right,
		double velocity, double left_wave_speed, double center_wave_speed, double right_wave_speed)
	{
		UniversalError res("Invalid wave speeds in hllc solver");
		res.addEntry("left density", left.density);
		res.addEntry("left pressure", left.pressure);
		res.addEntry("left x velocity", left.vx);
		res.addEntry("left y velocity", left.vy);
		res.addEntry("right density", right.density);
		res.addEntry("right pressure", right.pressure);
		res.addEntry("right x velocity", right.vx);
		res.addEntry("right y velocity", right.vy);
		res.addEntry("interface velocity", velocity);
		res.addEntry("left wave speed", left_wave_speed);
		res.addEntry("center wave speed", center_wave_speed);
		res.addEntry("right wave speed", right_wave_speed);
		return res;
	}
}

Hllc3D::Hllc3D(double gamma) :gamma_((gamma + 1) / (2 * gamma))
{}

std::pair<double, double> Hllc3D::GetUstarPstar(ComputationalCell3D const& left,	ComputationalCell3D const& right,
		EquationOfState const& eos, Vector3D const& normaldir) const
{
	ComputationalCell3D local_left = left;
	ComputationalCell3D local_right = right;

	local_left.velocity.x = ScalarProd(local_left.velocity, normaldir);
	local_right.velocity.x = ScalarProd(local_right.velocity, normaldir);
	WaveSpeeds ws = estimate_wave_speeds(local_left, local_right, eos, -1);
	double ps = std::max(0.0, 0.5 * (left.pressure + right.pressure + left.density * (ws.left - local_left.velocity.x) * (ws.center - local_left.velocity.x)
		+ right.density * (ws.right - local_right.velocity.x) * (ws.center - local_right.velocity.x)));
	return std::make_pair(ws.center, ps);
}

Conserved3D Hllc3D::operator()(ComputationalCell3D const& left, ComputationalCell3D const& right, double velocity,
	EquationOfState const& eos, Vector3D const& normaldir) const
{
	const double cl = sound_speed(left, eos, "left");
	const double cr = sound_speed(right, eos, "right");
	hllc::Primitive local_left, local_right;
	hllc::WaveSpeeds ws;
	hllc::Flux flux;
	double frame_velocity = 0;
	const std::uint8_t status = hllc::Solve(
		hllc::Primitive{left.density, left.pressure, left.internal_energy, left.velocity.x, left.velocity.y, left.velocity.z},
		hllc::Primitive{right.density, right.pressure, right.internal_energy, right.velocity.x, right.velocity.y, right.velocity.z},
		velocity, normaldir.x, normaldir.y, normaldir.z, cl, cr, gamma_, flux, local_left, local_right, ws, frame_velocity);
	if (status == hllc::InvalidWaveSpeeds)
		throw invalid_wave_speeds(local_left, local_right, frame_velocity, ws.left, ws.center, ws.right);
	if (status == hllc::ManyIterations)
		print_many_iterations(local_left, local_right);
	assert(status != hllc::ManyIterations);
	Conserved3D f_gr;
	f_gr.mass = flux.mass;
	f_gr.momentum = Vector3D(flux.mx, flux.my, flux.mz);
	f_gr.energy = flux.energy;
	f_gr.internal_energy = 0;
	return f_gr;
}
