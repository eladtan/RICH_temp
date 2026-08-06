#ifndef FMM_KERNELS_HPP
#define FMM_KERNELS_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"

namespace FmmKernels
{
void accumulateP2M(const FmmNode& leaf,
                   const std::vector<Vector3D>& positions,
                   const std::vector<double>& masses,
                   const std::vector<std::size_t>& particleOrder,
                   const FmmTaylorExpansion& layout,
                   std::vector<double>& multipoles);

void translateM2M(const FmmNode& child,
                  const FmmNode& parent,
                  const FmmTaylorExpansion& layout,
                  std::vector<double>& multipoles);

void computeM2LOperator(const Vector3D& displacement,
                        const FmmTaylorExpansion& layout,
                        std::vector<double>& derivativeScratch,
                        std::vector<double>& translationOperator);

void translateM2L(const FmmNode& source,
                  const FmmNode& target,
                  const FmmTaylorExpansion& layout,
                  const std::vector<double>& multipoles,
                  std::vector<double>& locals,
                  const std::vector<double>& translationOperator,
                  double inverseDistanceScale = 1.0);

void translateM2LRaw(const FmmNode& source,
                     const FmmNode& target,
                     const FmmTaylorExpansion& layout,
                     const double* sourceCoefficients,
                     std::vector<double>& locals,
                     const std::vector<double>& translationOperator,
                     double inverseDistanceScale = 1.0);

void translateL2L(const FmmNode& parent,
                  const FmmNode& child,
                  const FmmTaylorExpansion& layout,
                  std::vector<double>& locals);

void evaluateL2P(const FmmNode& leaf,
                 const std::vector<Vector3D>& positions,
                 const std::vector<std::size_t>& particleOrder,
                 const FmmTaylorExpansion& layout,
                 const std::vector<double>& locals,
                 std::vector<Vector3D>& acceleration,
                 std::vector<double>* positiveKernelPotential);

// Evaluate the order-two Cartesian Taylor multipole directly at one point.
// The coefficient order is the canonical FmmTaylorExpansion(2) order.  This
// produces exactly the same truncated potential and gradient as M2L to a
// zero-radius target followed by L2P, without constructing the unused local
// coefficients.
void accumulateM2POrder2(const Vector3D& displacement,
                         const double* sourceCoefficients,
                         Vector3D& acceleration,
                         double* positiveKernelPotential);

// Closed-form order-three evaluation. This retains monopole, dipole, and
// quadrupole contributions to acceleration and all order-three contributions
// to the optional potential.
void accumulateM2POrder3(const Vector3D& displacement,
                         const double* sourceCoefficients,
                         Vector3D& acceleration,
                         double* positiveKernelPotential);

// Evaluate a general-order multipole directly at a zero-radius target.  Only
// the constant and first-degree local coefficients can contribute at that
// point, so avoid constructing the unused remainder of the local expansion.
void accumulateM2P(const Vector3D& displacement,
                   const FmmTaylorExpansion& layout,
                   const double* sourceCoefficients,
                   Vector3D& acceleration,
                   double* positiveKernelPotential,
                   std::vector<double>& derivativeScratch);

void accumulateP2P(const std::vector<Vector3D>& targetPositions,
                   const std::vector<Vector3D>& sourcePositions,
                   const std::vector<double>& sourceMasses,
                   const std::vector<std::size_t>& targetOrder,
                   const std::vector<std::size_t>& sourceOrder,
                   std::size_t targetBegin,
                   std::size_t targetEnd,
                   std::size_t sourceBegin,
                   std::size_t sourceEnd,
                   bool sameParticleSet,
                   std::vector<Vector3D>& acceleration,
                   std::vector<double>* positiveKernelPotential,
                   std::uint64_t& evaluatedPairs);
}

#endif // FMM_KERNELS_HPP
