#ifndef LAPLACE_SOLID_HARMONICS_HPP
#define LAPLACE_SOLID_HARMONICS_HPP

#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmExpansionLayout.hpp"

namespace LaplaceSolidHarmonics
{
// Compact real storage of C_n^m = r^n P_n^m(cos(theta)) exp(i m phi),
// using associated Legendre polynomials without the Condon-Shortley phase.
// m=0 is real; positive/negative layout slots hold real/imaginary parts.
void fillRegular(const Vector3D& displacement,
                 const FmmExpansionLayout& layout,
                 std::vector<double>& coefficients);

void fillSingular(const Vector3D& displacement,
                  const FmmExpansionLayout& layout,
                  std::vector<double>& coefficients);
}

#endif // LAPLACE_SOLID_HARMONICS_HPP
