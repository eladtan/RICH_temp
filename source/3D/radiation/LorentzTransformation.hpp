#ifndef LORENTZ_TRANSFORMATION_HPP
#define LORENTZ_TRANSFORMATION_HPP

#include "monte/MonteCarloParticle.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"

using Particle3D = MonteCarloParticle<Vector3D, Tessellation3D>;

void LorentzTransformation(Particle3D &particle, const Vector3D &velocity);

double DopplerShift(const Particle3D &particle, const Vector3D &velocity);

#endif // LORENTZ_TRANSFORMATION_HPP