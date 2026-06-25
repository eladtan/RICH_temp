/*! \file RoundGrid3D.hpp
  \brief Makes the initial cells rounder
  \author Elad Steinberg
 */

#ifndef ROUNDGRID3D
#define ROUNDGRID3D 1

#include "../tessellation/Voronoi3D.hpp"
#include <functional>

/*!
	\brief Makes the cells rounder
	\param points The initial points
	\param ll The lower left corner of the domain
	\param ur The upper right corner of the domain
	\param NumberIt The number of correction iterations
	\param tess The tessellation
	\return The points that give a rounder tessellation
*/
vector<Vector3D> RoundGrid3D(vector<Vector3D> const& points,Vector3D const& ll,Vector3D const& ur,
	size_t NumberIt=10,	Tessellation3D *tess=nullptr);

#ifdef RICH_MPI
/*!
\brief Makes the cells rounder
\param points The initial points
\param ll The lower left corner of the domain
\param ur The upper right corner of the domain
\param NumberIt The number of correction iterations
\return The points that give a rounder tessellation
*/
vector<Vector3D> RoundGrid3DSingle(vector<Vector3D> const& points, Vector3D const& ll, Vector3D const& ur,
	size_t NumberIt = 10);
#endif

/*!
    \brief Adjusts the points to create a rounder tessellation on a spherical surface
    \param points The initial set of points to be adjusted.
    \param ll The lower left corner of the bounding box of the domain.
    \param ur The upper right corner of the bounding box of the domain.
    \param center The center of the spherical domain. Defaults to the origin (0,0,0).
    \param NumberIt The number of iterations for the correction process. Defaults to 10.
    \param tess An optional pointer to a Tessellation3D object for additional tessellation data. Defaults to nullptr.
    \param criteria A function to determine if a point should be included in the adjustment process. Defaults to a function that returns true for all points.
    \param verbose Whether to print iteration progress to stdout. Defaults to true.
    \return A vector of Vector3D objects representing the adjusted points that result in a rounder tessellation.
*/
std::vector<Vector3D> RoundGridSphere3D(std::vector<Vector3D> const& points, Vector3D const& ll, Vector3D const& ur, Vector3D const center = Vector3D(0,0,0),
    size_t NumberIt = 10,	Tessellation3D *tess = nullptr, std::function<bool(Vector3D)> const& criteria = [](Vector3D){return true;}, bool verbose = true);
#endif //ROUNDGRID3D
