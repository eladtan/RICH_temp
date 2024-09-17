#include "computational_cell.hpp"

ComputationalCell3D::ComputationalCell3D(void):
  density(0), pressure(0),internal_energy(0),temperature(0),ID(0), velocity(), Erad(0), Erad_dt(0),
  	Erad_dt_dt(0), cs(0), tracers(),stickers(), dt(0) {}

ComputationalCell3D::ComputationalCell3D(double density_i,
				     double pressure_i,double internal_energy_i,size_t ID_i,
				     const Vector3D& velocity_i):
  density(density_i), pressure(pressure_i),internal_energy(internal_energy_i),temperature(0),ID(ID_i),
  velocity(velocity_i), Erad(0), Erad_dt(0), Erad_dt_dt(0), cs(0), tracers(),stickers(), dt(0) {}

ComputationalCell3D::ComputationalCell3D(double density_i,
				     double pressure_i, double internal_energy_i,size_t ID_i,
				     const Vector3D& velocity_i,
				     const std::array<double,MAX_TRACERS>& tracers_i,
					 const std::array<bool,MAX_STICKERS>& stickers_i):
  density(density_i), pressure(pressure_i),internal_energy(internal_energy_i),temperature(0),ID(ID_i),
  velocity(velocity_i), Erad(0), Erad_dt(0), Erad_dt_dt(0), cs(0), tracers(tracers_i),stickers(stickers_i), dt(0) {}

ComputationalCell3D::ComputationalCell3D(const ComputationalCell3D& other):
density(other.density),
pressure(other.pressure),
internal_energy(other.internal_energy),
temperature(other.temperature),
ID(other.ID),
velocity(other.velocity),
dt(other.dt),
Erad(other.Erad),
Erad_dt(other.Erad_dt),
Erad_dt_dt(other.Erad_dt_dt),
cs(other.cs),
tracers(other.tracers),
stickers(other.stickers) {}


ComputationalCell3D& ComputationalCell3D::operator=(ComputationalCell3D const& other)
{
	density = other.density;
	pressure = other.pressure;
	internal_energy = other.internal_energy;
	temperature = other.temperature;
	velocity = other.velocity;
	dt = other.dt;
	Erad = other.Erad;
	Erad_dt = other.Erad_dt;
	Erad_dt_dt = other.Erad_dt_dt;
	cs = other.cs;
	tracers = other.tracers;
	stickers = other.stickers;
	ID = other.ID;
	return *this;
}

ComputationalCell3D& ComputationalCell3D::operator+=(ComputationalCell3D const& other)
{
	this->density += other.density;
	this->pressure += other.pressure;
	this->internal_energy += other.internal_energy;
	this->velocity += other.velocity;
	this->temperature += other.temperature;
	this->dt = std::min<double>(this->dt, other.dt); // todo: correct?
	this->Erad += other.Erad;
	this->Erad_dt += other.Erad_dt;
	this->Erad_dt_dt += other.Erad_dt_dt;
	this->cs = std::max<double>(this->cs, other.cs); // todo: correct?
	//assert(this->tracers.size() == other.tracers.size());
	//size_t N = this->tracers.size();
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		this->tracers[j] += other.tracers[j];
	return *this;
}

ComputationalCell3D& ComputationalCell3D::operator-=(ComputationalCell3D const& other)
{
	this->density -= other.density;
	this->pressure -= other.pressure;
	this->internal_energy -= other.internal_energy;
	this->velocity -= other.velocity;
	this->temperature -= other.temperature;
	this->dt = std::min<double>(this->dt, other.dt); // todo: correct?
	this->Erad -= other.Erad;
	this->Erad_dt += other.Erad_dt;
	this->Erad_dt_dt += other.Erad_dt_dt;	
	this->cs = std::max<double>(this->cs, other.cs); // todo: correct?
	//assert(this->tracers.size() == other.tracers.size());
	//size_t N = this->tracers.size();
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		this->tracers[j] -= other.tracers[j];
	return *this;
}

ComputationalCell3D& ComputationalCell3D::operator*=(double s)
{
	this->density *= s;
	this->pressure *= s;
	this->internal_energy *= s;
	this->velocity *= s;
	this->temperature *= s;
	this->Erad *= s;
	this->Erad_dt *= s;
	this->Erad_dt_dt *= s;
	this->cs *= s; // todo: correct?
	//size_t N = this->tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		this->tracers[j] *= s;
	return *this;
}

vector<string> ComputationalCell3D::tracerNames;
vector<string> ComputationalCell3D::stickerNames;

#ifdef RICH_MPI
size_t ComputationalCell3D::dump(Serializer *serializer) const
{
	size_t bytes = 0;
	bytes += serializer->insert(this->density);
	bytes += serializer->insert(this->pressure);
	bytes += serializer->insert(this->velocity);
	bytes += serializer->insert(this->internal_energy);
	bytes += serializer->insert(this->temperature);
	bytes += serializer->insert(this->ID);
	bytes += serializer->insert(this->Erad);
	bytes += serializer->insert(this->Erad_dt);
	bytes += serializer->insert(this->Erad_dt_dt);
	bytes += serializer->insert(this->cs);
	for(size_t j = 0; j < MAX_TRACERS; ++j)
	{
		bytes += serializer->insert(this->tracers[j]);
	}
	for(size_t i = 0; i < MAX_STICKERS; ++i)
	{
		bytes += serializer->insert(this->stickers[i]);
	}
	return bytes;
}

size_t ComputationalCell3D::load(const Serializer *serializer, std::size_t byteOffset)
{
	size_t bytesRead = 0;
	bytesRead += serializer->extract(this->density, byteOffset);
	bytesRead += serializer->extract(this->pressure, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->velocity, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->internal_energy, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->temperature, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->ID, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->Erad, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->Erad_dt, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->Erad_dt_dt, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->cs, byteOffset + bytesRead);
	for(size_t j = 0; j < MAX_TRACERS; ++j)
	{
		bytesRead += serializer->extract(this->tracers[j], byteOffset + bytesRead);
	}
	for(size_t i = 0; i < MAX_STICKERS; ++i)
	{
		bytesRead += serializer->extract(this->stickers[i], byteOffset + bytesRead);
	}
	return bytesRead;
}

size_t Slope3D::dump(Serializer *serializer) const
{
	size_t bytes = 0;
	bytes += serializer->insert(this->xderivative);
	bytes += serializer->insert(this->yderivative);
	bytes += serializer->insert(this->zderivative);
	return bytes;
}

size_t Slope3D::load(const Serializer *serializer, std::size_t byteOffset)
{
	size_t bytesRead = 0;
	bytesRead += serializer->extract(this->xderivative, byteOffset);
	bytesRead += serializer->extract(this->yderivative, byteOffset + bytesRead);
	bytesRead += serializer->extract(this->zderivative, byteOffset + bytesRead);
	return bytesRead;
}

#endif // RICH_MPI


void ComputationalCellAddMult(ComputationalCell3D &res, ComputationalCell3D const& other, double scalar)
{
	res.density += other.density*scalar;
	res.pressure += other.pressure*scalar;
	res.internal_energy += other.internal_energy*scalar;
	res.velocity += other.velocity*scalar;
	res.temperature += other.temperature*scalar;
	res.Erad += other.Erad*scalar;
	res.Erad_dt += other.Erad_dt*scalar;
	res.Erad_dt_dt += other.Erad_dt_dt*scalar;
	//assert(res.tracers.size() == other.tracers.size());
	//size_t N = res.tracers.size();
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res.tracers[j] += other.tracers[j] * scalar;
}

ComputationalCell3D operator+(ComputationalCell3D const& p1, ComputationalCell3D const& p2)
{
	ComputationalCell3D res(p1);
	res += p2;
	return res;
}

ComputationalCell3D operator-(ComputationalCell3D const& p1, ComputationalCell3D const& p2)
{
	ComputationalCell3D res(p1);
	res -= p2;
	return res;
}

ComputationalCell3D operator/(ComputationalCell3D const& p, double s)
{
	ComputationalCell3D res(p);
	double const s_1 = 1.0 / s;
	res.density *= s_1;
	res.pressure *= s_1;
	res.internal_energy *= s_1;
	res.temperature *= s_1;
	res.Erad *= s_1;
	res.Erad_dt *= s_1;
	res.Erad_dt_dt *= s_1;
	//size_t N = res.tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res.tracers[j] *= s_1;
	res.velocity = res.velocity * s_1;
	res.cs = res.cs * s_1; // todo: correct?
	return res;
}

ComputationalCell3D operator*(ComputationalCell3D const& p, double s)
{
	ComputationalCell3D res(p);
	res.density *= s;
	res.pressure *= s;
	res.internal_energy *= s;
	res.temperature *= s;
	res.Erad *= s;
	res.Erad_dt *= s;
	res.Erad_dt_dt *= s;
	//size_t N = res.tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res.tracers[j] *= s;
	res.velocity = res.velocity * s;
	res.cs = res.cs * s; // todo: correct?
	return res;
}

ComputationalCell3D operator*(double s, ComputationalCell3D const& p)
{
	return p*s;
}

void ReplaceComputationalCell(ComputationalCell3D & cell, ComputationalCell3D const& other)
{
	cell.density = other.density;
	cell.pressure = other.pressure;
	cell.internal_energy = other.internal_energy;
	cell.ID = other.ID;
	cell.velocity = other.velocity;
	cell.dt = other.dt;
	cell.temperature = other.temperature;
	cell.Erad = other.Erad;
	cell.Erad_dt = other.Erad_dt;
	cell.Erad_dt_dt = other.Erad_dt_dt;
	cell.cs = other.cs;
	//size_t N = other.tracers.size();
	//cell.tracers.resize(N);
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		cell.tracers[j] = other.tracers[j];
	//N = other.stickers.size();
	//cell.stickers.resize(N);
#ifdef __INTEL_COMPILER
#pragma ivdep
#endif
	for (size_t i = 0; i < MAX_STICKERS; ++i)
		cell.stickers[i] = other.stickers[i];
}



Slope3D::Slope3D(void) : xderivative(ComputationalCell3D()), yderivative(ComputationalCell3D()), zderivative(ComputationalCell3D()) {}

Slope3D::Slope3D(ComputationalCell3D const & x, ComputationalCell3D const & y,ComputationalCell3D const & z) : xderivative(x), yderivative(y),zderivative(z)
{}
