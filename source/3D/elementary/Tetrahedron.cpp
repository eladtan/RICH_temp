#include "Tetrahedron.hpp"

Tetrahedron::Tetrahedron():
  points(), neighbors(), toCheck(true)  {}

Tetrahedron::Tetrahedron(Tetrahedron const & other)
{
	this->toCheck = other.toCheck;
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
	for (int i = 0; i < 4; i++)
	{
		points[i] = other.points[i];
		neighbors[i] = other.neighbors[i];
	}
}

Tetrahedron::~Tetrahedron()
{}

Tetrahedron & Tetrahedron::operator=(Tetrahedron const & other)
{
	this->toCheck = other.toCheck;
	if (&other == this)
		return *this;
#ifdef __INTEL_COMPILER
#pragma omp simd
#endif
	for (int i = 0; i < 4; ++i)
	{
		points[i] = other.points[i];
		neighbors[i] = other.neighbors[i];
	}
	return *this;
}
