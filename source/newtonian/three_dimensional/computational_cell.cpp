#include "computational_cell.hpp"
#include <cstring>  // for memcpy

ComputationalCell3D::ComputationalCell3D(void):
  density(0), pressure(0),internal_energy(0),temperature(0),ID(0), velocity(), Erad(0), Eg(ENERGY_GROUPS_NUM, 0), Erad_dt(0),
  	Erad_dt_dt(0), cs(0), tracers({}),stickers(),dt(0) {}

ComputationalCell3D::ComputationalCell3D(double density_i, double pressure_i, double internal_energy_i, size_t ID_i, const Vector3D& velocity_i):
  density(density_i), pressure(pressure_i),internal_energy(internal_energy_i),temperature(0),ID(ID_i),
  velocity(velocity_i), Erad(0), Eg(ENERGY_GROUPS_NUM, 0), Erad_dt(0), Erad_dt_dt(0), cs(0), tracers({}),stickers(),dt(0) {}

ComputationalCell3D::ComputationalCell3D(double density_i, double pressure_i, double internal_energy_i,size_t ID_i, const Vector3D& velocity_i, 
										const std::array<double,MAX_TRACERS>& tracers_i, const std::array<bool,MAX_STICKERS>& stickers_i):
  density(density_i), pressure(pressure_i),internal_energy(internal_energy_i),temperature(0),ID(ID_i),
  velocity(velocity_i), Erad(0), Eg(ENERGY_GROUPS_NUM, 0), Erad_dt(0), Erad_dt_dt(0), cs(0), tracers(tracers_i),stickers(stickers_i),dt(0) {}

ComputationalCell3D::ComputationalCell3D(const ComputationalCell3D& other):
density(other.density),
pressure(other.pressure),
internal_energy(other.internal_energy),
temperature(other.temperature),
ID(other.ID),
velocity(other.velocity),
dt(other.dt),
Erad(other.Erad),
Eg(other.Eg),
Erad_dt(other.Erad_dt),
Erad_dt_dt(other.Erad_dt_dt),
cs(other.cs),
tracers(other.tracers),
stickers(other.stickers) {}

std::ostream& operator<<(std::ostream &stream, const ComputationalCell3D &cell)
{
	stream << "(density=" << cell.density<<" pressure="<<cell.pressure<<" temperature="<<cell.temperature<<" sie="<<cell.internal_energy<<" ID="<<cell.ID<<" velocity="<<cell.velocity
		<<" Erad="<<cell.Erad;
	for(size_t i = 0; i < ENERGY_GROUPS_NUM; ++i)
		stream<<" Eg["<<i<<"]="<<cell.Eg[i];
	for(size_t i = 0; i < ComputationalCell3D::tracerNames.size(); ++i)
		stream<<" "<<ComputationalCell3D::tracerNames[i]<<"="<<cell.tracers[i];
	for(size_t i = 0; i < ComputationalCell3D::stickerNames.size(); ++i)
		stream<<" "<<ComputationalCell3D::stickerNames[i]<<"="<<cell.stickers[i];
	stream<< ")";
	return stream;
}

ComputationalCell3D& ComputationalCell3D::operator=(ComputationalCell3D const& other)
{
	density = other.density;
	pressure = other.pressure;
	internal_energy = other.internal_energy;
	temperature = other.temperature;
	velocity = other.velocity;
	dt = other.dt;
	Erad = other.Erad;
	Eg = other.Eg;
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
	this->cs += other.cs;
	
	double* __restrict__ t = this->tracers.data();
	const double* __restrict__ ot = other.tracers.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		t[j] += ot[j];
	
	double* __restrict__ eg = this->Eg.data();
	const double* __restrict__ oeg = other.Eg.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
	    eg[j] += oeg[j];
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
	this->Erad_dt -= other.Erad_dt;
	this->Erad_dt_dt -= other.Erad_dt_dt;	
	this->cs -= other.cs;
	
	double* __restrict__ t = this->tracers.data();
	const double* __restrict__ ot = other.tracers.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		t[j] -= ot[j];
	
	double* __restrict__ eg = this->Eg.data();
	const double* __restrict__ oeg = other.Eg.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
	    eg[j] -= oeg[j];
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
	this->cs *= s;
	
	double* __restrict__ t = this->tracers.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		t[j] *= s;
	
	double* __restrict__ eg = this->Eg.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
	    eg[j] *= s;
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
	bytes += serializer->insert_array(this->tracers);
	bytes += serializer->insert_array(this->stickers);
	bytes += serializer->insert(this->Eg.size());
	bytes += serializer->insert_array(this->Eg.data(), this->Eg.size());
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
	bytesRead += serializer->extract_array(this->tracers, byteOffset + bytesRead);
	bytesRead += serializer->extract_array(this->stickers, byteOffset + bytesRead);
	size_t EgSize;
	bytesRead += serializer->extract(EgSize, byteOffset + bytesRead);
	this->Eg.resize(EgSize);
	bytesRead += serializer->extract_array(this->Eg.data(), EgSize, byteOffset + bytesRead);
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
	// Process scalar fields - group for better cache utilization
	res.density += other.density * scalar;
	res.pressure += other.pressure * scalar;
	res.internal_energy += other.internal_energy * scalar;
	res.temperature += other.temperature * scalar;
	res.Erad += other.Erad * scalar;
	res.Erad_dt += other.Erad_dt * scalar;
	res.Erad_dt_dt += other.Erad_dt_dt * scalar;
	res.cs += other.cs * scalar;
	res.dt += other.dt * scalar;
	
	// Velocity - 3 components
	res.velocity.x += other.velocity.x * scalar;
	res.velocity.y += other.velocity.y * scalar;
	res.velocity.z += other.velocity.z * scalar;
	
	// Tracers - use restrict pointers and explicit vectorization hints
	double* __restrict__ res_tracers = res.tracers.data();
	const double* __restrict__ other_tracers = other.tracers.data();
	
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res_tracers[j] += other_tracers[j] * scalar;
	
	// Energy groups - use restrict pointers
	double* __restrict__ res_eg = res.Eg.data();
	const double* __restrict__ other_eg = other.Eg.data();
	const size_t eg_size = res.Eg.size();
	
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < eg_size; ++j)
		res_eg[j] += other_eg[j] * scalar;
}

// Vectorized version that processes all 3 derivatives at once
// More cache-friendly: one pass through data instead of 3
void ComputationalCellAddMult3(ComputationalCell3D &res, 
	ComputationalCell3D const& dx, ComputationalCell3D const& dy, ComputationalCell3D const& dz,
	double sx, double sy, double sz)
{
	// Scalar fields - process all 3 at once for better instruction pipelining
	res.density += dx.density * sx + dy.density * sy + dz.density * sz;
	res.pressure += dx.pressure * sx + dy.pressure * sy + dz.pressure * sz;
	res.internal_energy += dx.internal_energy * sx + dy.internal_energy * sy + dz.internal_energy * sz;
	res.temperature += dx.temperature * sx + dy.temperature * sy + dz.temperature * sz;
	res.Erad += dx.Erad * sx + dy.Erad * sy + dz.Erad * sz;
	res.Erad_dt += dx.Erad_dt * sx + dy.Erad_dt * sy + dz.Erad_dt * sz;
	res.Erad_dt_dt += dx.Erad_dt_dt * sx + dy.Erad_dt_dt * sy + dz.Erad_dt_dt * sz;
	res.cs += dx.cs * sx + dy.cs * sy + dz.cs * sz;
	res.dt += dx.dt * sx + dy.dt * sy + dz.dt * sz;
	
	// Velocity
	res.velocity.x += dx.velocity.x * sx + dy.velocity.x * sy + dz.velocity.x * sz;
	res.velocity.y += dx.velocity.y * sx + dy.velocity.y * sy + dz.velocity.y * sz;
	res.velocity.z += dx.velocity.z * sx + dy.velocity.z * sy + dz.velocity.z * sz;
	
	// Tracers - vectorized loop with restrict pointers
	double* __restrict__ res_t = res.tracers.data();
	const double* __restrict__ dx_t = dx.tracers.data();
	const double* __restrict__ dy_t = dy.tracers.data();
	const double* __restrict__ dz_t = dz.tracers.data();
	
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res_t[j] += dx_t[j] * sx + dy_t[j] * sy + dz_t[j] * sz;
	
	// Energy groups - vectorized
	double* __restrict__ res_eg = res.Eg.data();
	const double* __restrict__ dx_eg = dx.Eg.data();
	const double* __restrict__ dy_eg = dy.Eg.data();
	const double* __restrict__ dz_eg = dz.Eg.data();
	const size_t eg_size = res.Eg.size();
	
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < eg_size; ++j)
		res_eg[j] += dx_eg[j] * sx + dy_eg[j] * sy + dz_eg[j] * sz;
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
	res.cs *= s_1;
	res.velocity = res.velocity * s_1;
	
	double* __restrict__ t = res.tracers.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		t[j] *= s_1;
	
	double* __restrict__ eg = res.Eg.data();
#if defined(__GNUC__) && !defined(__INTEL_COMPILER)
	#pragma GCC ivdep
#elif defined(__INTEL_COMPILER)
	#pragma omp simd
#endif
	for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
		eg[j] *= s_1;
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
	res.cs *= s;
	//size_t N = res.tracers.size();
	for (size_t j = 0; j < MAX_TRACERS; ++j)
		res.tracers[j] *= s;
	for(size_t j = 0; j < ENERGY_GROUPS_NUM; ++j)
		res.Eg[j] *= s;
	res.velocity = res.velocity * s;
	return res;
}

ComputationalCell3D operator*(double s, ComputationalCell3D const& p)
{
	return p*s;
}

void ReplaceComputationalCell(ComputationalCell3D & cell, ComputationalCell3D const& other)
{
	// Copy scalar fields in bulk - these are contiguous in memory
	cell.density = other.density;
	cell.pressure = other.pressure;
	cell.internal_energy = other.internal_energy;
	cell.temperature = other.temperature;
	cell.ID = other.ID;
	cell.velocity = other.velocity;
	cell.dt = other.dt;
	cell.Erad = other.Erad;
	cell.Erad_dt = other.Erad_dt;
	cell.Erad_dt_dt = other.Erad_dt_dt;
	cell.cs = other.cs;
	
	// Use memcpy for arrays - much faster than element-by-element
	std::memcpy(cell.tracers.data(), other.tracers.data(), MAX_TRACERS * sizeof(double));
	std::memcpy(cell.stickers.data(), other.stickers.data(), MAX_STICKERS * sizeof(bool));
	
	// Copy Eg vector (small_vector - need to handle size)
	const size_t eg_size = other.Eg.size();
	cell.Eg.resize(eg_size);
	if (eg_size > 0)
		std::memcpy(cell.Eg.data(), other.Eg.data(), eg_size * sizeof(double));
}



Slope3D::Slope3D(void) : xderivative(ComputationalCell3D()), yderivative(ComputationalCell3D()), zderivative(ComputationalCell3D()) {}

Slope3D::Slope3D(ComputationalCell3D const & x, ComputationalCell3D const & y,ComputationalCell3D const & z) : xderivative(x), yderivative(y),zderivative(z)
{}
