#include "3D/gravity/fmm/LaplaceSolidHarmonics.hpp"

#include <cmath>
#include <complex>

#include "misc/universal_error.hpp"

namespace
{
bool finiteVector(const Vector3D& value)
{
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

void storeCoefficient(int n,
                      int m,
                      const std::complex<double>& value,
                      const FmmExpansionLayout& layout,
                      std::vector<double>& coefficients)
{
    if(m == 0)
        coefficients[layout.index(n, 0)] = value.real();
    else
    {
        coefficients[layout.indexReal(n, m)] = value.real();
        coefficients[layout.indexImag(n, m)] = value.imag();
    }
}
}

void LaplaceSolidHarmonics::fillRegular(const Vector3D& displacement,
                                        const FmmExpansionLayout& layout,
                                        std::vector<double>& coefficients)
{
    if(!finiteVector(displacement))
        throw UniversalError("LaplaceSolidHarmonics::fillRegular: non-finite displacement");

    coefficients.assign(layout.coefficientCount(), 0.0);
    const double r2 = displacement.x * displacement.x +
                      displacement.y * displacement.y +
                      displacement.z * displacement.z;
    const std::complex<double> xy(displacement.x, displacement.y);
    std::complex<double> sectoral(1.0, 0.0);

    for(int m = 0; m <= layout.order(); ++m)
    {
        if(m > 0)
            sectoral *= static_cast<double>(2 * m - 1) * xy;
        storeCoefficient(m, m, sectoral, layout, coefficients);
        if(m == layout.order())
            continue;

        std::complex<double> previous2 = sectoral;
        std::complex<double> previous1 =
            static_cast<double>(2 * m + 1) * displacement.z * sectoral;
        storeCoefficient(m + 1, m, previous1, layout, coefficients);
        for(int n = m + 2; n <= layout.order(); ++n)
        {
            const std::complex<double> current =
                (static_cast<double>(2 * n - 1) * displacement.z * previous1 -
                 static_cast<double>(n + m - 1) * r2 * previous2) /
                static_cast<double>(n - m);
            storeCoefficient(n, m, current, layout, coefficients);
            previous2 = previous1;
            previous1 = current;
        }
    }
}

void LaplaceSolidHarmonics::fillSingular(const Vector3D& displacement,
                                         const FmmExpansionLayout& layout,
                                         std::vector<double>& coefficients)
{
    const double r2 = displacement.x * displacement.x +
                      displacement.y * displacement.y +
                      displacement.z * displacement.z;
    if(!finiteVector(displacement) || !(r2 > 0.0) || !std::isfinite(r2))
        throw UniversalError("LaplaceSolidHarmonics::fillSingular: invalid displacement");

    fillRegular(displacement, layout, coefficients);
    const double invR = 1.0 / std::sqrt(r2);
    double inverseOddPower = invR;
    for(int n = 0; n <= layout.order(); ++n)
    {
        const double scale = (n & 1) == 0 ? inverseOddPower : -inverseOddPower;
        coefficients[layout.index(n, 0)] *= scale;
        for(int m = 1; m <= n; ++m)
        {
            coefficients[layout.indexReal(n, m)] *= scale;
            coefficients[layout.indexImag(n, m)] *= scale;
        }
        inverseOddPower *= invR * invR;
    }

    for(double value : coefficients)
    {
        if(!std::isfinite(value))
            throw UniversalError("LaplaceSolidHarmonics::fillSingular: coefficient overflow");
    }
}
