#ifndef MONTE_CARLO_3D_HPP
#define MONTE_CARLO_3D_HPP

#include "monte/MonteCarloParticle.hpp"
#include "3D/tessellation/Tessellation3D.hpp"

using MonteCarloParticle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

#endif // MONTE_CARLO_3D_HPP