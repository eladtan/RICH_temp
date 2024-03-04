#include "Hllc3D.hpp"
#include "../../misc/universal_error.hpp"
#include "../../misc/utils.hpp"

using namespace std;

namespace
{
	class WaveSpeeds
	{
	public:

		WaveSpeeds(double left_i, Vector3D center_i, double right_i) :left(left_i), center(center_i), right(right_i) {}
		double left;
		Vector3D center;
		double right;
	};

	WaveSpeeds estimate_wave_speeds(ComputationalCell3D const& left, ComputationalCell3D const& right, EquationOfState const &eos)
	{
		double cl = 0, cr = 0;
		const double dl = left.density;
		const double plx = left.pressure - left.stress.at(0,0);
		const double ply = -left.stress.at(0,1);
		const double plz = -left.stress.at(0,2);
		const double ul = left.velocity.x;
		const double vl = left.velocity.y;
		const double wl = left.velocity.z;
#ifdef RICH_DEBUG
		try
		{
#endif
		  cl = eos.dp2c(dl, left.pressure, left.tracers, ComputationalCell3D::tracerNames);
		  cl = std::max(std::numeric_limits<double>::epsilon()*1e2 * std::abs(ul), std::max(std::sqrt(cl*cl + 4*left.G/(3*dl)), std::numeric_limits<double>::min() * 1e6));
#ifdef RICH_DEBUG
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("Error in HLLC3D left sound speed", 0);
			throw eo;
		}
#endif
		const double dr = right.density;
		const double prx = right.pressure - right.stress.at(0,0);
		const double pry = -right.stress.at(0,1);
		const double prz = -right.stress.at(0,2);
		const double ur = right.velocity.x;
		const double vr = right.velocity.y;
		const double wr = right.velocity.z;
#ifdef RICH_DEBUG
		try
		{
#endif
		  cr = eos.dp2c(dr, right.pressure, right.tracers, ComputationalCell3D::tracerNames);
#ifdef RICH_DEBUG
		}
		catch (UniversalError &eo)
		{
			eo.addEntry("Error in HLLC3D right sound speed", 0);
			throw eo;
		}
#endif
		double sl = std::min(vr - cr, vl - cl);
		double sr = std::max(vr + cr, vl + cl);
		double denom = 1.0 / (dl*(sl - ul) - dr * (sr - ur));
		double us = (prx - plx + dl*ul*(sl-ul)-dr*ur*(sr-ur))*denom;
		double vs = (pry - ply + dl*vl*(sl-ul)-dr*vr*(sr-ur))*denom;
		double ws = (prz - plz + dl*wl*(sl-ul)-dr*wr*(sr-ur))*denom;
		return WaveSpeeds(sl, Vector3D(us, vs, ws), sr);
	}

	UniversalError invalid_wave_speeds(ComputationalCell3D const& left, ComputationalCell3D const& right,
		double velocity, double left_wave_speed, double center_wave_speed, double right_wave_speed)
	{
		UniversalError res("Invalid wave speeds in hllc solver");
		res.addEntry("left density", left.density);
		res.addEntry("left pressure", left.pressure);
		res.addEntry("left x velocity", left.velocity.x);
		res.addEntry("left y velocity", left.velocity.y);
		res.addEntry("right density", right.density);
		res.addEntry("right pressure", right.pressure);
		res.addEntry("right x velocity", right.velocity.x);
		res.addEntry("right y velocity", right.velocity.y);
		res.addEntry("interface velocity", velocity);
		res.addEntry("left wave speed", left_wave_speed);
		res.addEntry("center wave speed", center_wave_speed);
		res.addEntry("right wave speed", right_wave_speed);
		return res;
	}

	double TotalEnergyDensity3D(ComputationalCell3D const& p)
	{
		double Eelast = 0.25 * p.stress.J2()/p.G;
		return p.density*(0.5*ScalarProd(p.velocity, p.velocity) + p.internal_energy) + Eelast;
	}

	Conserved3D starred_state(ComputationalCell3D const& state, double sk, Vector3D ss, Vector3D& pstar)
	{
		const double dk = state.density;
		const double pkx = state.pressure - state.stress.at(0,0);
		const double pky = -state.stress.at(0,1);
		const double pkz = -state.stress.at(0,2);
		const double uk = state.velocity.x;
		const double vk = state.velocity.y;
		const double wk = state.velocity.z;
		const double ds = dk * (sk - uk) / (sk - ss.x);
		const double ek = TotalEnergyDensity3D(state);
		const double psx = pkx + dk*(uk-ss.x)*(uk-sk);
		const double psy = pky + dk*(vk-ss.y)*(uk-sk);
		const double psz = pkz + dk*(wk-ss.z)*(uk-sk);
		Conserved3D res;
		res.mass = ds;
		res.momentum = ds * ss;
		res.energy = ek * ds / dk - ds * (psx*ss.x + psy*ss.y + psz*ss.z - pkx*uk - pky*vk - pkz*wk)/(dk*(uk-sk));
		res.internal_energy = 0;
		res.mass_eps = 0;
		res.mass_eps_dt = 0;
		pstar = Vector3D(psx, psy, psz);
		return res;
	}

	Conserved3D PrimitiveToFlux(ComputationalCell3D const& p)
	{
		return Conserved3D(p.density*p.velocity.x,
			Vector3D(p.pressure - p.stress.at(0,0), p.stress.at(0,1), p.stress.at(0,2)) + p.density*p.velocity.x*p.velocity,
			(TotalEnergyDensity3D(p) + p.pressure)*	p.velocity.x - (p.stress.at(0,0)*p.velocity.x + p.stress.at(0,1)*p.velocity.y + p.stress.at(0,2)*p.velocity.z), 0);
	}

	void BoostBack(Conserved3D &f_gr, double velocity, Vector3D const& normaldir, ComputationalCell3D const& left,
		ComputationalCell3D const& right, Mat33<double> const& R)
	{
		f_gr.energy += f_gr.momentum.x * velocity + 0.5*f_gr.mass*velocity*velocity;
		f_gr.momentum.x += f_gr.mass*velocity;
		f_gr.momentum = R.transpose() * f_gr.momentum;
		f_gr.internal_energy = 0;
	}

	void RotateBack(Conserved3D& f_gr, Vector3D const& normaldir, ComputationalCell3D const& left,
		ComputationalCell3D const& right)
	{
		f_gr.momentum = f_gr.momentum.x * normaldir;
		if (f_gr.mass > 0)
			f_gr.momentum += (left.velocity - normaldir * ScalarProd(left.velocity, normaldir)) * f_gr.mass;
		else
			f_gr.momentum += (right.velocity - normaldir * ScalarProd(right.velocity, normaldir)) * f_gr.mass;
		f_gr.internal_energy = 0;
	}

	Conserved3D HLLstarred_state(ComputationalCell3D const& left, ComputationalCell3D const& right,
		double sl, double sr)
	{
		Conserved3D res;
		Conserved3D Fl = PrimitiveToFlux(left), Fr = PrimitiveToFlux(right);
		double denom = 1.0 / (sr - sl);
		res.mass = (sr * right.density - sl * left.density + Fl.mass - Fr.mass) * denom;
		res.momentum = (sr * right.density * right.velocity - sl * left.density * left.velocity
			+ Fl.momentum - Fr.momentum) * denom;
		res.energy = (sr * TotalEnergyDensity3D(right) - sl * TotalEnergyDensity3D(left)
			+ Fl.energy - Fr.energy) * denom;
		return res;
	}	
}


Conserved3D Hllc3D::operator()(ComputationalCell3D const& left, ComputationalCell3D const& right, double velocity,
	EquationOfState const& eos, Vector3D const& normaldir, Vector3D & ustar_vec, Vector3D & pstar_vec) const
{
	double face_v = 0;	
	double minv = std::min(fastabs(left.velocity), fastabs(right.velocity));
	bool fast_flow = minv < std::abs(velocity) * 1e-3;
	if (fast_flow)
	{
		face_v = velocity;
		velocity = 0;
	}

	ComputationalCell3D local_left = left;
	ComputationalCell3D local_right = right;
	Vector3D normal_local = normaldir;

	if(normal_local.x*normal_local.x < 5*std::numeric_limits<double>::epsilon())
		normal_local.x = 0;
	if(normal_local.y*normal_local.y < 5*std::numeric_limits<double>::epsilon())
		normal_local.y = 0;
	if(normal_local.z*normal_local.z < 5*std::numeric_limits<double>::epsilon())
		normal_local.z = 0;

	double const small_eps = 1e-15;
	double const nx = normal_local.x;
	double const ny = normal_local.y + small_eps;
	double const nz = normal_local.z + small_eps;
	double const nyz = std::sqrt(ny*ny+nz*nz);
	double _1_nyz= 1/nyz;
	const Mat33<double> R(nx, ny, nz, -nyz, nx*ny*_1_nyz, nx*nz*_1_nyz, 0., -nz*_1_nyz, ny*_1_nyz);


	local_left.velocity = R * local_left.velocity;
	local_right.velocity = R * local_right.velocity;
	
	local_left.velocity.x -= velocity;
	local_right.velocity.x -= velocity;

	local_left.stress = R * (local_left.stress * R.transpose());
	local_right.stress = R * (local_right.stress * R.transpose());

	Conserved3D ul, ur;
	PrimitiveToConserved(local_left, 1, ul);
	PrimitiveToConserved(local_right, 1, ur);

	WaveSpeeds ws = estimate_wave_speeds(local_left, local_right, eos);

	Conserved3D f_gr;
	// check if bad wavespeed
	bool HLL = false;
	if ((ws.center.x<ws.left || ws.center.x>ws.right))
	{
		ws.center.x = 0.5*(ws.left + ws.right);
		HLL = true;
	}

	if (ws.left > face_v)
	{
		const Conserved3D fl = PrimitiveToFlux(local_left);
		f_gr = fl;
		if (fast_flow)
		{
			Conserved3D ustate;
			PrimitiveToConserved(local_left, 1, ustate);
			f_gr -= ustate * face_v;			
		}
	}
	else if (ws.left <= face_v && ws.center.x >= face_v)
	{
		const Conserved3D fl = PrimitiveToFlux(local_left);
		const Conserved3D usl = starred_state(local_left, ws.left, ws.center, pstar_vec);
		f_gr = fl + ws.left*(usl - ul);
		if (fast_flow)
			f_gr -= usl * face_v;
	}
	else if (ws.center.x < face_v && ws.right >= face_v)
	{
		const Conserved3D fr = PrimitiveToFlux(local_right);
		const Conserved3D usr = starred_state(local_right, ws.right, ws.center, pstar_vec));
		f_gr = fr + ws.right*(usr - ur);
		if (fast_flow)
			f_gr -= usr * face_v;
	}
	else if (ws.right < face_v)
	{
		const Conserved3D fr = PrimitiveToFlux(local_right);
		f_gr = fr;
		if (fast_flow)
		{
			Conserved3D ustate;
			PrimitiveToConserved(local_right, 1, ustate);
			f_gr -= ustate * face_v;
		}
	}
	else
		throw invalid_wave_speeds(local_left, local_right, velocity, ws.left, ws.center, ws.right);

	// check if bad wavespeed
	if (HLL && ws.left < face_v && ws.right>face_v)
	{
		const Conserved3D fr = PrimitiveToFlux(local_right);
		const Conserved3D fl = PrimitiveToFlux(local_left);
		f_gr = (ws.right*fl - ws.left*fr + ws.left*ws.right*(ur - ul))*(1.0 / (ws.right - ws.left)); // HLL flux
		if (fast_flow)
			f_gr -= face_v * HLLstarred_state(local_left, local_right, ws.left, ws.right);
	}
	if (fast_flow)
		RotateBack(f_gr, normaldir, left, right);
	else
		BoostBack(f_gr, velocity, normaldir, left, right, R);
	ustar_vec = R.transpose() * (ws.center + velocity*Vector3D(1, 0, 0));
	starred_state(local_left, ws.left, ws.center, pstar_vec);
	pstar_vec = R.transpose() * pstar_vec;
	return f_gr;
}
