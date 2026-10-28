/*! \file HllcKernels.hpp
  \brief The HLLC Riemann solver on plain numbers: the only implementation, used by Hllc3D on the host and by the
  flux kernels on the host's threads or a device. The arithmetic follows the original Conserved3D and Vector3D code
  term by term, so that results are identical bit for bit. The equation of state is not called here: the callers
  pass the sound speeds.
*/
#ifndef HLLC_KERNELS_HPP
#define HLLC_KERNELS_HPP 1

#include <cstdint>
#include "utils/PortableFunction.hpp"

namespace hllc
{
//! \brief The primitive variables of a side of a face
struct Primitive
{
	double density = 0;
	double pressure = 0;
	double internal_energy = 0;
	double vx = 0;
	double vy = 0;
	double vz = 0;
};

//! \brief The fluxes the solver computes (the rest of a Conserved3D is filled by the caller)
struct Flux
{
	double mass = 0;
	double mx = 0;
	double my = 0;
	double mz = 0;
	double energy = 0;
};

struct WaveSpeeds
{
	double left = 0;
	double center = 0;
	double right = 0;
};

//! \brief How a solution went: Ok, or why the caller must report it
enum Status : std::uint8_t
{
	Ok = 0,
	ManyIterations, // the wave speed estimate took more than 54 iterations (a warning)
	InvalidWaveSpeeds
};

//! \brief The velocity of a face along its normal n (RotateSolveBack3D)
RICH_INLINE_FUNCTION double NormalVelocity(double nx, double ny, double nz, double ux, double uy, double uz)
{
	return nx * ux + ny * uy + nz * uz;
}

//! \brief Reflects a velocity on the plane with normal n, v -= 2 (v . n) n (RigidWallFlux3D)
RICH_INLINE_FUNCTION void Reflect(double &vx, double &vy, double &vz, double nx, double ny, double nz)
{
	const double s = 2 * (vx * nx + vy * ny + vz * nz);
	vx -= nx * s;
	vy -= ny * s;
	vz -= nz * s;
}

// Conserved3D's arithmetic on its mass, momentum and energy
RICH_INLINE_FUNCTION Flux Add(const Flux &a, const Flux &b)
{
	return Flux{a.mass + b.mass, a.mx + b.mx, a.my + b.my, a.mz + b.mz, a.energy + b.energy};
}

RICH_INLINE_FUNCTION Flux Subtract(const Flux &a, const Flux &b)
{
	return Flux{a.mass - b.mass, a.mx - b.mx, a.my - b.my, a.mz - b.mz, a.energy - b.energy};
}

RICH_INLINE_FUNCTION Flux Scale(double s, const Flux &c)
{
	return Flux{s * c.mass, c.mx * s, c.my * s, c.mz * s, s * c.energy};
}

RICH_INLINE_FUNCTION double TotalEnergyDensity(const Primitive &p)
{
	return p.density * (0.5 * (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) + p.internal_energy);
}

// PrimitiveToConserved(p, 1, res)
RICH_INLINE_FUNCTION Flux Conserved(const Primitive &p)
{
	const double mass = p.density * 1;
	const double internal_energy = mass * p.internal_energy;
	return Flux{mass, p.vx * mass, p.vy * mass, p.vz * mass,
		mass * 0.5 * (p.vx * p.vx + p.vy * p.vy + p.vz * p.vz) + internal_energy};
}

// The flux of a state through a face normal to x
RICH_INLINE_FUNCTION Flux PrimitiveToFlux(const Primitive &p)
{
	const double dvx = p.density * p.vx;
	return Flux{p.density * p.vx, 1.0 * p.pressure + p.vx * dvx, 0.0 * p.pressure + p.vy * dvx,
		0.0 * p.pressure + p.vz * dvx, (TotalEnergyDensity(p) + p.pressure) * p.vx};
}

RICH_INLINE_FUNCTION Flux StarredState(const Primitive &state, double sk, double ss)
{
	const double dk = state.density;
	const double pk = state.pressure;
	const double uk = state.vx;
	const double vk = state.vy;
	const double wk = state.vz;
	const double ds = dk * (sk - uk) / (sk - ss);
	const double ek = TotalEnergyDensity(state);
	return Flux{ds, ds * ss, ds * vk, ds * wk, ek * ds / dk + ds * (ss - uk) * (ss + pk / dk / (sk - uk))};
}

RICH_INLINE_FUNCTION Flux HllStarredState(const Primitive &left, const Primitive &right, double sl, double sr)
{
	const Flux Fl = PrimitiveToFlux(left), Fr = PrimitiveToFlux(right);
	const double denom = 1.0 / (sr - sl);
	const double rd = sr * right.density, ld = sl * left.density;
	return Flux{(sr * right.density - sl * left.density + Fl.mass - Fr.mass) * denom,
		(right.vx * rd - left.vx * ld + Fl.mx - Fr.mx) * denom,
		(right.vy * rd - left.vy * ld + Fl.my - Fr.my) * denom,
		(right.vz * rd - left.vz * ld + Fl.mz - Fr.mz) * denom,
		(sr * TotalEnergyDensity(right) - sl * TotalEnergyDensity(left) + Fl.energy - Fr.energy) * denom};
}

/* The wave speeds of a Riemann problem along x (cl, cr the sound speeds). With gamma > 0.001, the estimate is
   iterated; iterations counts them. */
RICH_INLINE_FUNCTION WaveSpeeds EstimateWaveSpeeds(double dl, double pl, double vl, double cl, double dr, double pr,
	double vr, double cr, double gamma, int &iterations)
{
	double sl = portable::Min(vr - cr, vl - cl);
	double sr = portable::Max(vr + cr, vl + cl);
	double pstar = 0;
	double denom = 1.0 / (dl * (sl - vl) - dr * (sr - vr));
	double ps = portable::Max(0.0, dl * (sl - vl) * (pr - dr * (vr - vl) * (sr - vr)) * denom - pl * dr * (sr - vr) * denom);
	iterations = 0;
	while((ps > 1.1 * pstar || pstar > 1.1 * ps) && gamma > 0.001)
	{
		pstar = ps;
		const double al = (pstar > pl ? portable::Sqrt(1 + gamma * (pstar / pl - 1)) : 1);
		const double ar = (pstar > pr ? portable::Sqrt(1 + gamma * (pstar / pr - 1)) : 1);
		sl = vl - cl * al;
		sr = vr + cr * ar;
		if(sl > sr)
		{
			sl = portable::Min(vr - cr * ar, vl - cl * al);
			sr = portable::Max(vr + cr * ar, vl + cl * al);
		}
		denom = 1.0 / (dl * (sl - vl) - dr * (sr - vr));
		ps = portable::Max(0.0, dl * (sl - vl) * (pr - dr * (vr - vl) * (sr - vr)) * denom - pl * dr * (sr - vr) * denom);
		++iterations;
	}
	const double ss = (pr - pl + dl * vl * (sl - vl) - dr * vr * (sr - vr)) * denom;
	return WaveSpeeds{sl, ss, sr};
}

// The flux's momentum along the normal, plus each side's tangential velocity times the mass flux (upwind)
RICH_INLINE_FUNCTION void AddTangential(Flux &f_gr, double nx, double ny, double nz, const Primitive &left,
	const Primitive &right)
{
	const Primitive &upwind = f_gr.mass > 0 ? left : right;
	const double dot = upwind.vx * nx + upwind.vy * ny + upwind.vz * nz;
	f_gr.mx += (upwind.vx - nx * dot) * f_gr.mass;
	f_gr.my += (upwind.vy - ny * dot) * f_gr.mass;
	f_gr.mz += (upwind.vz - nz * dot) * f_gr.mass;
}

/*! \brief Hllc3D's flux of left | right through a face with normal n moving at `velocity` along n (gamma: Hllc3D's
  iteration factor, cl and cr: the sound speeds). The local states, wave speeds and the velocity of the frame the
  problem is solved in are also returned, for errors.
*/
RICH_INLINE_FUNCTION std::uint8_t Solve(const Primitive &left, const Primitive &right, double velocity, double nx,
	double ny, double nz, double cl, double cr, double gamma, Flux &f_gr, Primitive &local_left,
	Primitive &local_right, WaveSpeeds &ws, double &frame_velocity)
{
	double face_v = 0;
	const double minv = portable::Min(portable::Sqrt(left.vx * left.vx + left.vy * left.vy + left.vz * left.vz),
		portable::Sqrt(right.vx * right.vx + right.vy * right.vy + right.vz * right.vz));
	const bool fast_flow = minv < portable::Abs(velocity) * 1e-3;
	if(fast_flow)
	{
		face_v = velocity;
		velocity = 0;
	}
	frame_velocity = velocity;

	local_left = left;
	local_right = right;
	local_left.vx -= velocity * nx;
	local_left.vy -= velocity * ny;
	local_left.vz -= velocity * nz;
	local_right.vx -= velocity * nx;
	local_right.vy -= velocity * ny;
	local_right.vz -= velocity * nz;

	Primitive *const locals[2] = {&local_left, &local_right};
	for(Primitive *local : locals)
	{
		const double dot = local->vx * nx + local->vy * ny + local->vz * nz;
		const double px = local->vx - nx * dot, py = local->vy - ny * dot, pz = local->vz - nz * dot;
		const double par = portable::Sqrt(px * px + py * py + pz * pz);
		local->vx = dot;
		local->vy = par;
		local->vz = 0;
	}

	const Flux ul = Conserved(local_left);
	const Flux ur = Conserved(local_right);

	int iterations = 0;
	ws = EstimateWaveSpeeds(local_left.density, local_left.pressure, local_left.vx, cl, local_right.density,
		local_right.pressure, local_right.vx, cr, gamma, iterations);

	// check if bad wavespeed
	bool HLL = false;
	if((ws.center < ws.left || ws.center > ws.right))
	{
		ws.center = 0.5 * (ws.left + ws.right);
		HLL = true;
	}

	if(ws.left > face_v)
	{
		f_gr = PrimitiveToFlux(local_left);
		if(fast_flow)
		{
			f_gr = Subtract(f_gr, Scale(face_v, Conserved(local_left)));
		}
	}
	else if(ws.left <= face_v && ws.center >= face_v)
	{
		const Flux usl = StarredState(local_left, ws.left, ws.center);
		f_gr = Add(PrimitiveToFlux(local_left), Scale(ws.left, Subtract(usl, ul)));
		if(fast_flow)
		{
			f_gr = Subtract(f_gr, Scale(face_v, usl));
		}
	}
	else if(ws.center < face_v && ws.right >= face_v)
	{
		const Flux usr = StarredState(local_right, ws.right, ws.center);
		f_gr = Add(PrimitiveToFlux(local_right), Scale(ws.right, Subtract(usr, ur)));
		if(fast_flow)
		{
			f_gr = Subtract(f_gr, Scale(face_v, usr));
		}
	}
	else if(ws.right < face_v)
	{
		f_gr = PrimitiveToFlux(local_right);
		if(fast_flow)
		{
			f_gr = Subtract(f_gr, Scale(face_v, Conserved(local_right)));
		}
	}
	else
	{
		return InvalidWaveSpeeds;
	}

	// check if bad wavespeed
	if(HLL && ws.left < face_v && ws.right > face_v)
	{
		const Flux fr = PrimitiveToFlux(local_right);
		const Flux fl = PrimitiveToFlux(local_left);
		f_gr = Scale(1.0 / (ws.right - ws.left), Add(Subtract(Scale(ws.right, fl), Scale(ws.left, fr)),
			Scale(ws.left * ws.right, Subtract(ur, ul)))); // HLL flux
		if(fast_flow)
		{
			f_gr = Subtract(f_gr, Scale(face_v, HllStarredState(local_left, local_right, ws.left, ws.right)));
		}
	}
	if(fast_flow)
	{
		// RotateBack
		const double mx = f_gr.mx;
		f_gr.mx = nx * mx;
		f_gr.my = ny * mx;
		f_gr.mz = nz * mx;
	}
	else
	{
		// BoostBack
		f_gr.energy += f_gr.mx * velocity + 0.5 * f_gr.mass * velocity * velocity;
		const double normal = f_gr.mx + f_gr.mass * velocity;
		f_gr.mx = nx * normal;
		f_gr.my = ny * normal;
		f_gr.mz = nz * normal;
	}
	AddTangential(f_gr, nx, ny, nz, left, right);
	return iterations > 54 ? ManyIterations : Ok;
}
} // namespace hllc

#endif // HLLC_KERNELS_HPP
