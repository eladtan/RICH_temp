#include "extensive.hpp"
#include "../../misc/utils.hpp"

Extensive::Extensive(void):
  mass(0),
  energy(0),
  momentum(0,0),
  tracers() {}

Extensive::Extensive(const Extensive& other):
  mass(other.mass),
  energy(other.energy),
  momentum(other.momentum),
  tracers(other.tracers) {}

Extensive::Extensive(tvector const& Tracers):mass(0),
energy(0),
momentum(0, 0),
tracers()
{
	size_t N = Tracers.size();
	for (size_t i = 0; i < N; ++i)
		tracers[i] = 0;
}

Extensive& Extensive::operator-=(const Extensive& diff)
{
  mass -= diff.mass;
  energy -= diff.energy;
  momentum -= diff.momentum;
  assert(diff.tracers.size() == this->tracers.size());
  size_t N = diff.tracers.size();
  for (size_t i = 0; i < N; ++i)
	  this->tracers[i] -= diff.tracers[i];
  return *this;
}

Extensive& Extensive::operator=(const Extensive& origin)
{
  mass = origin.mass;
  energy = origin.energy;
  momentum = origin.momentum;
  tracers = origin.tracers;
  return *this;
}

void ReplaceExtensive(Extensive &toreplace, Extensive const& other)
{
	toreplace.mass = other.mass;
	toreplace.energy = other.energy;
	toreplace.momentum = other.momentum;
	assert(other.tracers.size() == toreplace.tracers.size());
	size_t N = other.tracers.size();
	for (size_t i = 0; i < N; ++i)
		toreplace.tracers[i] = other.tracers[i];
}

Extensive& Extensive::operator+=(const Extensive& diff)
{
  mass += diff.mass;
  energy += diff.energy;
  momentum += diff.momentum;
  assert(diff.tracers.size() == this->tracers.size());
  size_t N = diff.tracers.size();
  for (size_t i = 0; i < N; ++i)
	  this->tracers[i] += diff.tracers[i];
  return *this;
}

Extensive operator*(const double s,
		    const Extensive& e)
{
  Extensive res;
  res.mass = s*e.mass;
  res.energy = s*e.energy;
  res.momentum = s*e.momentum;
  res.tracers = e.tracers;
  size_t N = res.tracers.size();
  for (size_t i = 0; i < N; ++i)
	  res.tracers[i] *= s;
  return res;
}

Extensive operator+(const Extensive& e1,
		    const Extensive& e2)
{
  Extensive res;
  res.mass = e1.mass + e2.mass;
  res.energy = e1.energy + e2.energy;
  res.momentum = e1.momentum + e2.momentum;
  res.tracers = e1.tracers;
  size_t N = res.tracers.size();
  for (size_t i = 0; i < N; ++i)
	  res.tracers[i] += e2.tracers[i];
  return res;
}

Extensive& Extensive::operator*=(const double scalar)
{
	mass *=scalar;
	energy *=scalar;
	momentum *=scalar;
	size_t N = tracers.size();
	for (size_t i = 0; i < N; ++i)
		tracers[i] *= scalar;
	return *this;
}

Extensive operator-(const Extensive& e1,
		    const Extensive& e2)
{
  return e1+(-1)*e2;
}
		   
#ifdef RICH_MPI
  size_t Extensive::dump(Serializer *serializer) const
  {
    size_t bytes = 0;
    bytes += serializer->insert(this->mass);
    bytes += serializer->insert(this->energy);
    bytes += serializer->insert(this->momentum);
    bytes += serializer->insert(this->tracers.size());
    for(size_t i = 0; i < this->tracers.size(); ++i)
    {
      bytes += serializer->insert(this->tracers[i]);
    }
    return bytes;
  }

  size_t Extensive::load(const Serializer *serializer, size_t byteOffset)
  {
    size_t bytesRead = 0;
    bytesRead += serializer->extract(this->mass, byteOffset);
    bytesRead += serializer->extract(this->energy, byteOffset + bytesRead);
    bytesRead += serializer->extract(this->momentum, byteOffset + bytesRead);
    size_t N;
    bytesRead += serializer->extract(N, byteOffset + bytesRead);
    for(size_t i = 0; i < N; ++i)
    {
      bytesRead += serializer->extract(this->tracers[i], byteOffset + bytesRead);
    }
    return bytesRead;
  }
#endif // RICH_MPI
