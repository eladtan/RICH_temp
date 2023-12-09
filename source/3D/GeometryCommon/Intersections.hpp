#ifndef INTERSECTIONS_HPP
#define INTERSECTIONS_HPP 1

#include "3D/elementary/Face.hpp"

//! \brief Sphere data
struct Sphere
{
  //! \brief Centre of the sphere
	Vector3D center;
  //! \brief Sphere radius 
	double radius;

  //! \brief Default constructor
	Sphere(): center(Vector3D()), radius(0){}
	Sphere(const Vector3D &center, double radius): center(center), radius(radius){}
};

bool FaceSphereIntersections(Face const& face, Sphere const& sphere,Vector3D const& normal);

#endif //INTERSECTIONS_HPP
