/*! \file RoundGrid3D.hpp
  \brief Makes the initial cells rounder
  \author Elad Steinberg
 */

#ifndef ROUNDGRID3D
#define ROUNDGRID3D 1

#include "../tesselation/voronoi/Voronoi3D.hpp"

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
	size_t NumberIt=10,	Tessellation3D *tess=nullptr, std::function<bool(Vector3D)> const& criteria = [](Vector3D v){return true;});

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

#endif //ROUNDGRID3D
