#include "conserved_3d.hpp"

using std::size_t;

Conserved3D::Conserved3D(void) :
	mass(0), momentum(), energy(0), internal_energy(0), Erad(0), Erad_dt(0), Erad_dt_dt(0), tracers() {}

Conserved3D::Conserved3D(double mass_i,
	const Vector3D& momentum_i,
	double energy_i, double internal_energy_i) :
	mass(mass_i), momentum(momentum_i), energy(energy_i), internal_energy(internal_energy_i), 
	Erad(0), Erad_dt(0), Erad_dt_dt(0),  tracers() {}

Conserved3D::Conserved3D(double mass_i,
	const Vector3D& momentum_i,
	double energy_i, double internal_energy_i,
	const std::array<double, MAX_TRACERS >& tracers_i) :
	mass(mass_i), momentum(momentum_i),
	energy(energy_i), internal_energy(internal_energy_i), Erad(0), Erad_dt(0), Erad_dt_dt(0),  tracers(tracers_i) {}

namespace
{
	std::array<double, MAX_TRACERS> operator*(double s, const std::array<double, MAX_TRACERS>& v)
	{
		std::array<double, MAX_TRACERS> res;
		for (size_t i = 0; i < v.size(); ++i)
			res[i] = s * v[i];
		return res;
	}

  /*
	std::array<double, MAX_TRACERS> operator/(const std::array<double, MAX_TRACERS>& v, double s)
	{
		std::array<double, MAX_TRACERS> res;
		double s_1 = 1.0 / s;
		for (size_t i = 0; i < v.size(); ++i)
			res[i] = v[i] * s_1;
		return res;
	}
  */
}

Conserved3D& Conserved3D::operator-=(const Conserved3D& diff)
{
	mass -= diff.mass;
	momentum -= diff.momentum;
	energy -= diff.energy;
	internal_energy -= diff.internal_energy;
	Erad -= diff.Erad;
	Erad_dt -= diff.Erad_dt;
	Erad_dt_dt -= diff.Erad_dt_dt;
	for (size_t i = 0; i < MAX_TRACERS; ++i)
		tracers[i] -= diff.tracers[i];
	return *this;
}

Conserved3D& Conserved3D::operator+=(const Conserved3D& diff)
{
	mass += diff.mass;
	momentum += diff.momentum;
	energy += diff.energy;
	internal_energy += diff.internal_energy;
	Erad += diff.Erad;
	Erad_dt += diff.Erad_dt;
	Erad_dt_dt += diff.Erad_dt_dt;
	for (size_t i = 0; i < tracers.size(); ++i)
		tracers[i] += diff.tracers[i];
	return *this;
}

#ifdef RICH_MPI
	size_t Conserved3D::dump(Serializer *serializer) const
	{
		size_t bytes = 0;
		bytes += serializer->insert(this->mass);
		bytes += serializer->insert(this->energy);
		bytes += serializer->insert(this->momentum);
		bytes += serializer->insert(this->internal_energy);
		bytes += serializer->insert(this->Erad);
		bytes += serializer->insert(this->Erad_dt);
		bytes += serializer->insert(this->Erad_dt_dt);
		for (size_t j = 0; j < MAX_TRACERS; ++j)
		{
			bytes += serializer->insert(this->tracers[j]);
		}
		return bytes;
	}

	size_t Conserved3D::load(const Serializer *serializer, std::size_t byteOffset)
	{
		size_t bytes = 0;
		bytes += serializer->extract(this->mass, byteOffset);
		bytes += serializer->extract(this->energy, byteOffset + bytes);
		bytes += serializer->extract(this->momentum, byteOffset + bytes);
		bytes += serializer->extract(this->internal_energy, byteOffset + bytes);
		bytes += serializer->extract(this->Erad, byteOffset + bytes);
		bytes += serializer->extract(this->Erad_dt, byteOffset + bytes);
		bytes += serializer->extract(this->Erad_dt_dt, byteOffset + bytes);
		for (size_t j = 0; j < MAX_TRACERS; ++j)
		{
			bytes += serializer->extract(this->tracers[j], byteOffset + bytes);
		}
		return bytes;
	}
#endif

Conserved3D operator*(double s, const Conserved3D& c)
{
	Conserved3D res(s*c.mass,
		s*c.momentum,
		s*c.energy, s*c.internal_energy,
		s*c.tracers);
	res.Erad = s * c.Erad;
	res.Erad_dt = s * c.Erad_dt;
	res.Erad_dt_dt = s * c.Erad_dt_dt;
	return res;
}

Conserved3D operator*(const Conserved3D& c, double s)
{
	return s * c;
}


Conserved3D operator/(const Conserved3D& c, double s)
{
	double s_1 = 1.0 / s;
	Conserved3D res(c.mass * s_1,
		c.momentum * s_1,
		c.energy * s_1, c.internal_energy * s_1,
		s_1 * c.tracers);
	res.Erad = c.Erad * s_1;
	res.Erad_dt = c.Erad_dt * s_1;
	res.Erad_dt_dt = c.Erad_dt_dt * s_1;
	return res;
}

void PrimitiveToConserved(ComputationalCell3D const& cell, double vol, Conserved3D &res)
{
	res.mass = cell.density*vol;
	res.momentum = cell.velocity;
	res.momentum *= res.mass;
	res.internal_energy = res.mass*cell.internal_energy;
	res.energy = res.mass*0.5*ScalarProd(cell.velocity, cell.velocity) + res.internal_energy;
	res.Erad = cell.Erad * res.mass;
	res.Erad_dt = cell.Erad_dt * res.mass;
	res.Erad_dt_dt = cell.Erad_dt_dt * res.mass;
	//size_t N = cell.tracers.size();
	//res.tracers.resize(N);
	for (size_t i = 0; i < MAX_TRACERS; ++i)
		res.tracers[i] = cell.tracers[i] * res.mass;
}

void PrimitiveToConservedSR(ComputationalCell3D const& cell, double vol, Conserved3D &res, EquationOfState const& eos)
{
	double gamma = 1 / std::sqrt(1 - ScalarProd(cell.velocity, cell.velocity));
	res.mass = cell.density*vol*gamma;
	const double enthalpy = eos.dp2e(cell.density, cell.pressure, cell.tracers, ComputationalCell3D::tracerNames);
	res.internal_energy = enthalpy * res.mass;
	res.Erad = res.mass * cell.Erad;
	if (fastabs(cell.velocity) < 1e-5)
		res.energy = (gamma*enthalpy + 0.5*ScalarProd(cell.velocity, cell.velocity))* res.mass - cell.pressure*vol;
	else
		res.energy = (gamma*enthalpy + (gamma - 1))* res.mass - cell.pressure*vol;
	res.momentum = res.mass * (enthalpy + 1)*gamma*cell.velocity;
	size_t N = cell.tracers.size();
	//res.tracers.resize(N);
	for (size_t i = 0; i < N; ++i)
		res.tracers[i] = cell.tracers[i] * res.mass;
}

Conserved3D operator+(Conserved3D const& p1, Conserved3D const& p2)
{
	Conserved3D res(p1);
	res += p2;
	return res;
}

Conserved3D operator-(Conserved3D const& p1, Conserved3D const& p2)
{
	Conserved3D res(p1);
	res -= p2;
	return res;
}

Conserved3D& Conserved3D::operator*=(double s)
{
	this->mass *= s;
	this->momentum *= s;
	this->energy *= s;
	this->internal_energy *= s;
	this->Erad *= s;
	this->Erad_dt *= s;
	this->Erad_dt_dt *= s;
	size_t N = this->tracers.size();
	for (size_t j = 0; j < N; ++j)
		this->tracers[j] *= s;
	return *this;
}
