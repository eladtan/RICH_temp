#include "conserved_3d.hpp"

using std::size_t;

Conserved3D::Conserved3D(void) :
	mass(0), momentum(), energy(0), internal_energy(0), Erad(0), Erad_dt(0), Erad_dt_dt(0), tracers(), mass_stress(), mass_eps(0), mass_eps_dt(0), Eelast(0) {}

Conserved3D::Conserved3D(double mass_i,
	const Vector3D& momentum_i,
	double energy_i, double internal_energy_i) :
	mass(mass_i), momentum(momentum_i), energy(energy_i), internal_energy(internal_energy_i), 
	Erad(0), Erad_dt(0), Erad_dt_dt(0),  tracers(), mass_stress(), mass_eps(0), mass_eps_dt(0), Eelast(0) {}

Conserved3D::Conserved3D(double mass_i,
	const Vector3D& momentum_i,
	double energy_i, double internal_energy_i,
	const std::array<double, MAX_TRACERS >& tracers_i) :
	mass(mass_i), momentum(momentum_i),
	energy(energy_i), internal_energy(internal_energy_i), Erad(0), Erad_dt(0), Erad_dt_dt(0),  tracers(tracers_i), mass_stress(), mass_eps(0), mass_eps_dt(0), Eelast(0) {}

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
	mass_stress -= diff.mass_stress;
	mass_eps -= diff.mass_eps;
	mass_eps_dt -= diff.mass_eps_dt;
	Eelast -= diff.Eelast;
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
	mass_stress += diff.mass_stress;
	mass_eps += diff.mass_eps;
	mass_eps_dt += diff.mass_eps_dt;
	Eelast += diff.Eelast;
	for (size_t i = 0; i < tracers.size(); ++i)
		tracers[i] += diff.tracers[i];
	return *this;
}

#ifdef RICH_MPI
size_t Conserved3D::getChunkSize(void) const
{
	return 21 + tracers.size();
}

vector<double> Conserved3D::serialize(void) const
{
	vector<double> res(getChunkSize());
	res.at(0) = mass;
	res.at(1) = energy;
	res.at(2) = momentum.x;
	res.at(3) = momentum.y;
	res.at(4) = momentum.z;
	res.at(5) = internal_energy;
	res.at(6) = Erad;
	res.at(7) = Erad_dt;
	res.at(8) = Erad_dt_dt;
	res.at(9) = Eelast;
	res.at(10) = mass_eps;
	res.at(11) = mass_eps_dt;
	size_t counter = 11;
	for(size_t i=0; i<3; ++i)
		for(size_t j=0; j<3; ++j)
			res.at(counter+3*i+j) = mass_stress.at(i,j);
	counter = 20;
	//size_t N = tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res[j + counter] = tracers[j];
	return res;
}

void Conserved3D::unserialize(const vector<double>& data)
{
	assert(data.size() == getChunkSize());
	mass = data.at(0);
	energy = data.at(1);
	momentum.x = data.at(2);
	momentum.y = data.at(3);
	momentum.z = data.at(4);
	internal_energy = data.at(5);
	Erad = data.at(6);
	Erad_dt = data.at(7);
	Erad_dt_dt = data.at(8);
	Eelast = data.at(9);
	mass_eps = data.at(10);
	mass_eps_dt = data.at(11);
	size_t counter = 11;
	for(size_t i=0; i<3; ++i)
		for(size_t j=0; j<3; ++j)
			mass_stress.at(i,j) = data.at(counter+3*i+j);
	counter = 20;
	//size_t N = tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		tracers[j] = data.at(counter + j);
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
	res.mass_stress = c.mass_stress*s;
	res.mass_eps  = c.mass_eps*s;
	res.mass_eps_dt = c.mass_eps_dt*s;
	res.Eelast = c.Eelast*s;
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
	res.mass_stress = c.mass_stress*s_1;
	res.mass_eps  = c.mass_eps*s_1;
	res.mass_eps_dt = c.mass_eps_dt*s_1;
	res.Eelast = c.Eelast*s_1;
	return res;
}

void PrimitiveToConserved(ComputationalCell3D const& cell, double vol, Conserved3D &res)
{
	res.mass = cell.density*vol;
	res.momentum = cell.velocity;
	res.momentum *= res.mass;
	res.internal_energy = res.mass*cell.internal_energy;
	res.Eelast = cell.elastic_energy*res.mass;
	res.energy = res.mass*0.5*ScalarProd(cell.velocity, cell.velocity) + res.internal_energy + res.Eelast;
	res.Erad = cell.Erad * res.mass;
	res.Erad_dt = cell.Erad_dt * res.mass;
	res.Erad_dt_dt = cell.Erad_dt_dt * res.mass;
	res.mass_stress = cell.stress*res.mass;
	res.mass_eps  = cell.strain_plastic*res.mass;
	res.mass_eps_dt = cell.strain_plastic_dt*res.mass;
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
	this->mass_stress *= s;
	this->mass_eps  *= s;
	this->mass_eps_dt *= s;
	this->Eelast *= s;
	size_t N = this->tracers.size();
	for (size_t j = 0; j < N; ++j)
		this->tracers[j] *= s;
	return *this;
}
