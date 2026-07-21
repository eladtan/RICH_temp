#include "Tetrahedron.hpp"

Tetrahedron::Tetrahedron(): points(), neighbors(), checkBig(true), newTetra(true)
{}

Tetrahedron::Tetrahedron(Tetrahedron const & other)
{
	this->checkBig = other.checkBig;
	this->newTetra = other.newTetra;
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

Tetrahedron &Tetrahedron::operator=(Tetrahedron const & other)
{
	this->checkBig = other.checkBig;
	this->newTetra = other.newTetra;
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
