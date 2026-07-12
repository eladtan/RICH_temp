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

    const double* sourceCoefficients =
        multipoles.data() + source.multipoleOffset;
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
        for(std::size_t sj = sourceBegin; sj < sourceEnd; ++sj)
        {
            const std::size_t source = sourceOrder[sj];
            if(sameParticleSet && target == source)
                continue;

            const Vector3D delta = targetPositions[target] - sourcePositions[source];
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
            acceleration[target] -= sourceMasses[source] * delta * invR3;
            if(positiveKernelPotential != nullptr)
                (*positiveKernelPotential)[target] += sourceMasses[source] * invR;
            ++evaluatedPairs;
        }
    }
}
