#include "3D/gravity/fmm/FmmKernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include "misc/universal_error.hpp"

namespace
{
void fillPowers(double value, int order, std::vector<double>& powers)
{
    powers.resize(static_cast<std::size_t>(order + 1));
    powers[0] = 1.0;
    for(int i = 1; i <= order; ++i)
        powers[static_cast<std::size_t>(i)] =
            powers[static_cast<std::size_t>(i - 1)] * value;
}

double monomial(const FmmMultiIndex& index,
                const std::vector<double>& x,
                const std::vector<double>& y,
                const std::vector<double>& z)
{
    return x[static_cast<std::size_t>(index.x)] *
           y[static_cast<std::size_t>(index.y)] *
           z[static_cast<std::size_t>(index.z)];
}

double binomial(int n, int k)
{
    if(k < 0 || k > n)
        return 0.0;
    k = std::min(k, n - k);
    double result = 1.0;
    for(int i = 1; i <= k; ++i)
        result *= static_cast<double>(n - k + i) / static_cast<double>(i);
    return result;
}

double convolution(const FmmMultiIndex& alpha,
                   const FmmTaylorExpansion& layout,
                   const std::vector<double>& coefficients)
{
    double result = 0.0;
    for(int x = 0; x <= alpha.x; ++x)
    {
        for(int y = 0; y <= alpha.y; ++y)
        {
            for(int z = 0; z <= alpha.z; ++z)
            {
                result += coefficients[layout.index(x, y, z)] *
                    coefficients[layout.index(alpha.x - x,
                                              alpha.y - y,
                                              alpha.z - z)];
            }
        }
    }
    return result;
}

void computeKernelDerivativesImpl(const Vector3D& displacement,
                                  const FmmTaylorExpansion& layout,
                                  std::vector<double>& derivatives)
{
    const double r2 = displacement.x * displacement.x +
                      displacement.y * displacement.y +
                      displacement.z * displacement.z;
    if(!(r2 > 0.0) || !std::isfinite(r2))
        throw UniversalError("FmmKernels::computeKernelDerivatives: invalid center separation");

    derivatives.assign(layout.coefficientCount(), 0.0);
    derivatives[layout.index(0, 0, 0)] = 1.0 / std::sqrt(r2);
    for(std::size_t i = 1; i < layout.coefficientCount(); ++i)
    {
        const FmmMultiIndex alpha = layout.multiIndex(i);
        double knownSquare = 0.0;
        for(int x = 0; x <= alpha.x; ++x)
        {
            for(int y = 0; y <= alpha.y; ++y)
            {
                for(int z = 0; z <= alpha.z; ++z)
                {
                    if((x == 0 && y == 0 && z == 0) ||
                       (x == alpha.x && y == alpha.y && z == alpha.z))
                        continue;
                    knownSquare += derivatives[layout.index(x, y, z)] *
                        derivatives[layout.index(alpha.x - x,
                                                 alpha.y - y,
                                                 alpha.z - z)];
                }
            }
        }

        double rightHandSide = r2 * knownSquare;
        const double linear[3] = {
            2.0 * displacement.x, 2.0 * displacement.y, 2.0 * displacement.z};
        for(int axis = 0; axis < 3; ++axis)
        {
            const int component = axis == 0 ? alpha.x :
                                  (axis == 1 ? alpha.y : alpha.z);
            if(component >= 1)
            {
                FmmMultiIndex lower = alpha;
                if(axis == 0)
                    --lower.x;
                else if(axis == 1)
                    --lower.y;
                else
                    --lower.z;
                rightHandSide += linear[axis] * convolution(lower, layout, derivatives);
            }
            if(component >= 2)
            {
                FmmMultiIndex lower = alpha;
                if(axis == 0)
                    lower.x -= 2;
                else if(axis == 1)
                    lower.y -= 2;
                else
                    lower.z -= 2;
                rightHandSide += convolution(lower, layout, derivatives);
            }
        }
        derivatives[i] = -rightHandSide /
            (2.0 * r2 * derivatives[layout.index(0, 0, 0)]);
    }

    for(std::size_t i = 0; i < layout.coefficientCount(); ++i)
        derivatives[i] /= layout.inverseFactorial(i);
}
}

void FmmKernels::computeM2LOperator(const Vector3D& displacement,
                                    const FmmTaylorExpansion& layout,
                                    std::vector<double>& derivativeScratch,
                                    std::vector<double>& translationOperator)
{
    computeKernelDerivativesImpl(displacement, layout, derivativeScratch);
    const std::vector<FmmM2LTerm>& terms = layout.m2lTerms();
    translationOperator.resize(terms.size());
    for(std::size_t i = 0; i < terms.size(); ++i)
    {
        const FmmM2LTerm& term = terms[i];
        translationOperator[i] =
            term.scale * derivativeScratch[term.derivativeIndex];
    }
}

void FmmKernels::accumulateP2M(const FmmNode& leaf,
                               const std::vector<Vector3D>& positions,
                               const std::vector<double>& masses,
                               const std::vector<std::size_t>& particleOrder,
                               const FmmTaylorExpansion& layout,
                               std::vector<double>& multipoles)
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    for(std::size_t k = leaf.particleBegin; k < leaf.particleEnd; ++k)
    {
        const std::size_t particle = particleOrder[k];
        const Vector3D displacement = positions[particle] - leaf.center;
        fillPowers(displacement.x, layout.order(), x);
        fillPowers(displacement.y, layout.order(), y);
        fillPowers(displacement.z, layout.order(), z);
        for(std::size_t i = 0; i < layout.coefficientCount(); ++i)
            multipoles[leaf.multipoleOffset + i] += masses[particle] *
                monomial(layout.multiIndex(i), x, y, z) * layout.inverseFactorial(i);
    }
}

void FmmKernels::translateM2M(const FmmNode& child,
                              const FmmNode& parent,
                              const FmmTaylorExpansion& layout,
                              std::vector<double>& multipoles)
{
    const Vector3D displacement = child.center - parent.center;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    fillPowers(displacement.x, layout.order(), x);
    fillPowers(displacement.y, layout.order(), y);
    fillPowers(displacement.z, layout.order(), z);

    for(std::size_t ai = 0; ai < layout.coefficientCount(); ++ai)
    {
        const FmmMultiIndex alpha = layout.multiIndex(ai);
        double translated = 0.0;
        for(std::size_t bi = 0; bi < layout.coefficientCount(); ++bi)
        {
            const FmmMultiIndex beta = layout.multiIndex(bi);
            if(beta.x > alpha.x || beta.y > alpha.y || beta.z > alpha.z)
                continue;
            const std::size_t difference = layout.index(alpha.x - beta.x,
                                                        alpha.y - beta.y,
                                                        alpha.z - beta.z);
            translated += multipoles[child.multipoleOffset + bi] *
                monomial(layout.multiIndex(difference), x, y, z) *
                layout.inverseFactorial(difference);
        }
        multipoles[parent.multipoleOffset + ai] += translated;
    }
}

void FmmKernels::translateM2L(const FmmNode& source,
                              const FmmNode& target,
                              const FmmTaylorExpansion& layout,
                              const std::vector<double>& multipoles,
                              std::vector<double>& locals,
                              const std::vector<double>& translationOperator,
                              double inverseDistanceScale)
{
    translateM2LRaw(source, target, layout,
                    multipoles.data() + source.multipoleOffset,
                    locals, translationOperator, inverseDistanceScale);
}

void FmmKernels::translateM2LRaw(
    const FmmNode& source,
    const FmmNode& target,
    const FmmTaylorExpansion& layout,
    const double* sourceCoefficients,
    std::vector<double>& locals,
    const std::vector<double>& translationOperator,
    double inverseDistanceScale)
{
    (void) source;
    const std::vector<std::size_t>& offsets = layout.m2lOffsets();
    const std::vector<FmmM2LTerm>& terms = layout.m2lTerms();
    if(translationOperator.size() != terms.size())
        throw UniversalError("FmmKernels::translateM2L: operator size mismatch");
    if(!(inverseDistanceScale > 0.0) || !std::isfinite(inverseDistanceScale))
        throw UniversalError("FmmKernels::translateM2L: invalid inverse distance scale");

    std::array<double, FMM_MAX_ORDER + 2> inversePowers{};
    inversePowers[0] = 1.0;
    for(int degree = 1; degree <= layout.order() + 1; ++degree)
        inversePowers[static_cast<std::size_t>(degree)] =
            inversePowers[static_cast<std::size_t>(degree - 1)] *
            inverseDistanceScale;

    if(layout.order() == 3)
    {
        // Order three has 20 coefficients grouped by total degree as
        // 1, 3, 6, 10. M2L terms retain source-index order within each
        // target coefficient, so factor the distance scale by source and
        // target degree instead of loading and multiplying it for all 84
        // terms independently.
        std::array<double, 20> scaledSource{};
        for(std::size_t sourceIndex = 0; sourceIndex < scaledSource.size();
            ++sourceIndex)
        {
            const int sourceDegree = sourceIndex == 0 ? 0 :
                (sourceIndex <= 3 ? 1 : (sourceIndex <= 9 ? 2 : 3));
            scaledSource[sourceIndex] = sourceCoefficients[sourceIndex] *
                inversePowers[static_cast<std::size_t>(sourceDegree)];
        }

        std::size_t termIndex = 0;
        for(std::size_t targetIndex = 0; targetIndex < scaledSource.size();
            ++targetIndex)
        {
            const int targetDegree = targetIndex == 0 ? 0 :
                (targetIndex <= 3 ? 1 : (targetIndex <= 9 ? 2 : 3));
            const std::size_t sourceCount = targetDegree == 0 ? 20 :
                (targetDegree == 1 ? 10 : (targetDegree == 2 ? 4 : 1));
            double translated = 0.0;
            for(std::size_t sourceIndex = 0; sourceIndex < sourceCount;
                ++sourceIndex, ++termIndex)
            {
                translated += translationOperator[termIndex] *
                    scaledSource[sourceIndex];
            }
            locals[target.localOffset + targetIndex] +=
                layout.inverseFactorial(targetIndex) *
                inversePowers[static_cast<std::size_t>(targetDegree + 1)] *
                translated;
        }
        return;
    }

    for(std::size_t ai = 0; ai < layout.coefficientCount(); ++ai)
    {
        double translated = 0.0;
        for(std::size_t termIndex = offsets[ai];
            termIndex < offsets[ai + 1]; ++termIndex)
        {
            const FmmM2LTerm& term = terms[termIndex];
            translated += translationOperator[termIndex] *
                inversePowers[term.inverseScalePower] *
                sourceCoefficients[term.sourceIndex];
        }
        locals[target.localOffset + ai] += layout.inverseFactorial(ai) * translated;
    }
}

void FmmKernels::translateL2L(const FmmNode& parent,
                              const FmmNode& child,
                              const FmmTaylorExpansion& layout,
                              std::vector<double>& locals)
{
    const Vector3D displacement = child.center - parent.center;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    fillPowers(displacement.x, layout.order(), x);
    fillPowers(displacement.y, layout.order(), y);
    fillPowers(displacement.z, layout.order(), z);

    for(std::size_t bi = 0; bi < layout.coefficientCount(); ++bi)
    {
        const FmmMultiIndex beta = layout.multiIndex(bi);
        double translated = 0.0;
        for(std::size_t ai = 0; ai < layout.coefficientCount(); ++ai)
        {
            const FmmMultiIndex alpha = layout.multiIndex(ai);
            if(alpha.x < beta.x || alpha.y < beta.y || alpha.z < beta.z)
                continue;
            const FmmMultiIndex difference = {
                alpha.x - beta.x, alpha.y - beta.y, alpha.z - beta.z};
            translated += locals[parent.localOffset + ai] *
                binomial(alpha.x, beta.x) * binomial(alpha.y, beta.y) *
                binomial(alpha.z, beta.z) *
                x[static_cast<std::size_t>(difference.x)] *
                y[static_cast<std::size_t>(difference.y)] *
                z[static_cast<std::size_t>(difference.z)];
        }
        locals[child.localOffset + bi] += translated;
    }
}

void FmmKernels::evaluateL2P(const FmmNode& leaf,
                             const std::vector<Vector3D>& positions,
                             const std::vector<std::size_t>& particleOrder,
                             const FmmTaylorExpansion& layout,
                             const std::vector<double>& locals,
                             std::vector<Vector3D>& acceleration,
                             std::vector<double>* positiveKernelPotential)
{
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    for(std::size_t k = leaf.particleBegin; k < leaf.particleEnd; ++k)
    {
        const std::size_t particle = particleOrder[k];
        const Vector3D displacement = positions[particle] - leaf.center;
        fillPowers(displacement.x, layout.order(), x);
        fillPowers(displacement.y, layout.order(), y);
        fillPowers(displacement.z, layout.order(), z);
        for(std::size_t i = 0; i < layout.coefficientCount(); ++i)
        {
            const FmmMultiIndex alpha = layout.multiIndex(i);
            const double coefficient = locals[leaf.localOffset + i];
            if(positiveKernelPotential != nullptr)
                (*positiveKernelPotential)[particle] += coefficient * monomial(alpha, x, y, z);
            if(alpha.x > 0)
                acceleration[particle].x += static_cast<double>(alpha.x) * coefficient *
                    x[static_cast<std::size_t>(alpha.x - 1)] *
                    y[static_cast<std::size_t>(alpha.y)] *
                    z[static_cast<std::size_t>(alpha.z)];
            if(alpha.y > 0)
                acceleration[particle].y += static_cast<double>(alpha.y) * coefficient *
                    x[static_cast<std::size_t>(alpha.x)] *
                    y[static_cast<std::size_t>(alpha.y - 1)] *
                    z[static_cast<std::size_t>(alpha.z)];
            if(alpha.z > 0)
                acceleration[particle].z += static_cast<double>(alpha.z) * coefficient *
                    x[static_cast<std::size_t>(alpha.x)] *
                    y[static_cast<std::size_t>(alpha.y)] *
                    z[static_cast<std::size_t>(alpha.z - 1)];
        }
    }
}

void FmmKernels::accumulateM2POrder2(
    const Vector3D& displacement,
    const double* sourceCoefficients,
    Vector3D& acceleration,
    double* positiveKernelPotential)
{
    const double x = displacement.x;
    const double y = displacement.y;
    const double z = displacement.z;
    const double r2 = x * x + y * y + z * z;
    if(!(r2 > 0.0) || !std::isfinite(r2))
        throw UniversalError(
            "FmmKernels::accumulateM2POrder2: invalid center separation");

    const double inverseR = 1.0 / std::sqrt(r2);
    const double inverseR3 = inverseR / r2;
    const double inverseR5 = inverseR3 / r2;

    const double dx = -x * inverseR3;
    const double dy = -y * inverseR3;
    const double dz = -z * inverseR3;
    const double dxx = (3.0 * x * x - r2) * inverseR5;
    const double dxy = 3.0 * x * y * inverseR5;
    const double dxz = 3.0 * x * z * inverseR5;
    const double dyy = (3.0 * y * y - r2) * inverseR5;
    const double dyz = 3.0 * y * z * inverseR5;
    const double dzz = (3.0 * z * z - r2) * inverseR5;

    // FmmTaylorExpansion orders equal total degree first, then x and y.  P2M
    // stores each Cartesian moment divided by its multi-index factorial.
    const double m0 = sourceCoefficients[0];
    const double mz = sourceCoefficients[1];
    const double my = sourceCoefficients[2];
    const double mx = sourceCoefficients[3];

    acceleration.x += m0 * dx - mx * dxx - my * dxy - mz * dxz;
    acceleration.y += m0 * dy - mx * dxy - my * dyy - mz * dyz;
    acceleration.z += m0 * dz - mx * dxz - my * dyz - mz * dzz;

    if(positiveKernelPotential != nullptr)
    {
        const double mzz = sourceCoefficients[4];
        const double myz = sourceCoefficients[5];
        const double myy = sourceCoefficients[6];
        const double mxz = sourceCoefficients[7];
        const double mxy = sourceCoefficients[8];
        const double mxx = sourceCoefficients[9];
        *positiveKernelPotential +=
            m0 * inverseR - mx * dx - my * dy - mz * dz +
            mxx * dxx + mxy * dxy + mxz * dxz +
            myy * dyy + myz * dyz + mzz * dzz;
    }
}

void FmmKernels::accumulateM2POrder3(
    const Vector3D& displacement,
    const double* sourceCoefficients,
    Vector3D& acceleration,
    double* positiveKernelPotential)
{
    const double x = displacement.x;
    const double y = displacement.y;
    const double z = displacement.z;
    const double xx = x * x;
    const double yy = y * y;
    const double zz = z * z;
    const double r2 = xx + yy + zz;
    if(!(r2 > 0.0) || !std::isfinite(r2))
        throw UniversalError(
            "FmmKernels::accumulateM2POrder3: invalid center separation");

    const double inverseR = 1.0 / std::sqrt(r2);
    const double inverseR3 = inverseR / r2;
    const double inverseR5 = inverseR3 / r2;
    const double inverseR7 = inverseR5 / r2;

    const double dx = -x * inverseR3;
    const double dy = -y * inverseR3;
    const double dz = -z * inverseR3;
    const double dxx = (3.0 * xx - r2) * inverseR5;
    const double dxy = 3.0 * x * y * inverseR5;
    const double dxz = 3.0 * x * z * inverseR5;
    const double dyy = (3.0 * yy - r2) * inverseR5;
    const double dyz = 3.0 * y * z * inverseR5;
    const double dzz = (3.0 * zz - r2) * inverseR5;
    const double dxxx = x * (9.0 * r2 - 15.0 * xx) * inverseR7;
    const double dxxy = y * (3.0 * r2 - 15.0 * xx) * inverseR7;
    const double dxxz = z * (3.0 * r2 - 15.0 * xx) * inverseR7;
    const double dxyy = x * (3.0 * r2 - 15.0 * yy) * inverseR7;
    const double dxyz = -15.0 * x * y * z * inverseR7;
    const double dxzz = x * (3.0 * r2 - 15.0 * zz) * inverseR7;
    const double dyyy = y * (9.0 * r2 - 15.0 * yy) * inverseR7;
    const double dyyz = z * (3.0 * r2 - 15.0 * yy) * inverseR7;
    const double dyzz = y * (3.0 * r2 - 15.0 * zz) * inverseR7;
    const double dzzz = z * (9.0 * r2 - 15.0 * zz) * inverseR7;

    // Canonical coefficient order: 0, z, y, x, zz, yz, yy, xz, xy, xx.
    const double m0 = sourceCoefficients[0];
    const double mz = sourceCoefficients[1];
    const double my = sourceCoefficients[2];
    const double mx = sourceCoefficients[3];
    const double mzz = sourceCoefficients[4];
    const double myz = sourceCoefficients[5];
    const double myy = sourceCoefficients[6];
    const double mxz = sourceCoefficients[7];
    const double mxy = sourceCoefficients[8];
    const double mxx = sourceCoefficients[9];

    acceleration.x +=
        m0 * dx - mx * dxx - my * dxy - mz * dxz +
        mxx * dxxx + mxy * dxxy + mxz * dxxz +
        myy * dxyy + myz * dxyz + mzz * dxzz;
    acceleration.y +=
        m0 * dy - mx * dxy - my * dyy - mz * dyz +
        mxx * dxxy + mxy * dxyy + mxz * dxyz +
        myy * dyyy + myz * dyyz + mzz * dyzz;
    acceleration.z +=
        m0 * dz - mx * dxz - my * dyz - mz * dzz +
        mxx * dxxz + mxy * dxyz + mxz * dxzz +
        myy * dyyz + myz * dyzz + mzz * dzzz;

    if(positiveKernelPotential != nullptr)
    {
        const double mzzz = sourceCoefficients[10];
        const double myzz = sourceCoefficients[11];
        const double myyz = sourceCoefficients[12];
        const double myyy = sourceCoefficients[13];
        const double mxzz = sourceCoefficients[14];
        const double mxyz = sourceCoefficients[15];
        const double mxyy = sourceCoefficients[16];
        const double mxxz = sourceCoefficients[17];
        const double mxxy = sourceCoefficients[18];
        const double mxxx = sourceCoefficients[19];
        *positiveKernelPotential +=
            m0 * inverseR - mx * dx - my * dy - mz * dz +
            mxx * dxx + mxy * dxy + mxz * dxz +
            myy * dyy + myz * dyz + mzz * dzz -
            (mxxx * dxxx + mxxy * dxxy + mxxz * dxxz +
             mxyy * dxyy + mxyz * dxyz + mxzz * dxzz +
             myyy * dyyy + myyz * dyyz + myzz * dyzz +
             mzzz * dzzz);
    }
}

void FmmKernels::accumulateM2P(
    const Vector3D& displacement,
    const FmmTaylorExpansion& layout,
    const double* sourceCoefficients,
    Vector3D& acceleration,
    double* positiveKernelPotential,
    std::vector<double>& derivativeScratch)
{
    computeKernelDerivativesImpl(displacement, layout, derivativeScratch);
    const std::vector<std::size_t>& offsets = layout.m2lOffsets();
    const std::vector<FmmM2LTerm>& terms = layout.m2lTerms();
    const auto localCoefficient = [&](std::size_t targetIndex) {
        double translated = 0.0;
        for(std::size_t termIndex = offsets[targetIndex];
            termIndex < offsets[targetIndex + 1]; ++termIndex)
        {
            const FmmM2LTerm& term = terms[termIndex];
            translated += term.scale *
                derivativeScratch[term.derivativeIndex] *
                sourceCoefficients[term.sourceIndex];
        }
        return layout.inverseFactorial(targetIndex) * translated;
    };

    if(positiveKernelPotential != nullptr)
        *positiveKernelPotential += localCoefficient(layout.index(0, 0, 0));
    acceleration.x += localCoefficient(layout.index(1, 0, 0));
    acceleration.y += localCoefficient(layout.index(0, 1, 0));
    acceleration.z += localCoefficient(layout.index(0, 0, 1));
}

void FmmKernels::accumulateP2P(const std::vector<Vector3D>& targetPositions,
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
                               std::uint64_t& evaluatedPairs)
{
    for(std::size_t ti = targetBegin; ti < targetEnd; ++ti)
    {
        const std::size_t target = targetOrder[ti];
        const Vector3D targetPosition = targetPositions[target];
        Vector3D targetAcceleration = acceleration[target];
        double targetPotential = positiveKernelPotential == nullptr ? 0.0 :
            (*positiveKernelPotential)[target];
        for(std::size_t sj = sourceBegin; sj < sourceEnd; ++sj)
        {
            const std::size_t source = sourceOrder[sj];
            if(sameParticleSet && target == source)
                continue;

            const Vector3D delta = targetPosition - sourcePositions[source];
            const double r2 = delta.x * delta.x + delta.y * delta.y + delta.z * delta.z;
            if(r2 == 0.0)
            {
                UniversalError eo("FmmKernels::accumulateP2P: coincident source and target positions");
                eo.addEntry("target_index", target);
                eo.addEntry("source_index", source);
                throw eo;
            }

            const double invR = 1.0 / std::sqrt(r2);
            const double invR3 = invR * invR * invR;
            targetAcceleration -= sourceMasses[source] * delta * invR3;
            if(positiveKernelPotential != nullptr)
                targetPotential += sourceMasses[source] * invR;
            ++evaluatedPairs;
        }
        acceleration[target] = targetAcceleration;
        if(positiveKernelPotential != nullptr)
            (*positiveKernelPotential)[target] = targetPotential;
    }
}
