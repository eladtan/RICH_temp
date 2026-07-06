#ifndef LORENTZ_TRANSFORMATION_HPP
#define LORENTZ_TRANSFORMATION_HPP

#include "monte/MonteCarloParticle.hpp"
#include "3D/elementary/Vector3D.hpp"
#include "3D/tessellation/Tessellation3D.hpp"
#include "Radiation/CMMC/src/units/units.hpp"
#include "3D/monte/MonteCarlo3D.hpp"

void LorentzTransformation(Particle3D &particle, const Vector3D &velocity);

void LabToComovingPacket(Particle3D &particle, const Vector3D &cellVelocity);

void ComovingToLabPacket(Particle3D &particle, const Vector3D &cellVelocity);

double DopplerShift(const Particle3D &particle, const Vector3D &velocity);

#endif // LORENTZ_TRANSFORMATION_HPP
