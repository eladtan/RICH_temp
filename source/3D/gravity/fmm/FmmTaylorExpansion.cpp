#include "3D/gravity/fmm/FmmTaylorExpansion.hpp"

#include <limits>

#include "3D/gravity/fmm/FmmConfig.hpp"
#include "misc/universal_error.hpp"

namespace
{
double factorial(int n)
{
    double result = 1.0;
    for(int i = 2; i <= n; ++i)
        result *= static_cast<double>(i);
    return result;
}
}

FmmTaylorExpansion::FmmTaylorExpansion(int order):
    order_(order)
{
    if(order < 0 || order > FMM_MAX_ORDER)
        throw UniversalError("FmmTaylorExpansion: order outside supported range");

    const std::size_t side = static_cast<std::size_t>(order + 1);
    lookup_.assign(side * side * side, std::numeric_limits<std::size_t>::max());
    indices_.reserve(fmmTaylorCoefficientCount(order));
    inverseFactorials_.reserve(fmmTaylorCoefficientCount(order));

    for(int degree = 0; degree <= order; ++degree)
    {
        for(int x = 0; x <= degree; ++x)
        {
            for(int y = 0; y <= degree - x; ++y)
            {
                const int z = degree - x - y;
                FmmMultiIndex value;
                value.x = x;
                value.y = y;
                value.z = z;
                const std::size_t flat =
                    (static_cast<std::size_t>(x) * side + static_cast<std::size_t>(y)) * side +
                    static_cast<std::size_t>(z);
                lookup_[flat] = indices_.size();
                indices_.push_back(value);
                inverseFactorials_.push_back(1.0 /
                    (factorial(x) * factorial(y) * factorial(z)));
            }
        }
    }

    m2lOffsets_.reserve(indices_.size() + 1);
    for(std::size_t targetIndex = 0; targetIndex < indices_.size(); ++targetIndex)
    {
        m2lOffsets_.push_back(m2lTerms_.size());
        const FmmMultiIndex& target = indices_[targetIndex];
        for(std::size_t sourceIndex = 0; sourceIndex < indices_.size(); ++sourceIndex)
        {
            const FmmMultiIndex& source = indices_[sourceIndex];
            if(target.degree() + source.degree() > order_)
                continue;
            const std::size_t derivativeIndex =
                index(target.x + source.x, target.y + source.y, target.z + source.z);
            const double sign = (source.degree() & 1) == 0 ? 1.0 : -1.0;
            m2lTerms_.push_back(
                FmmM2LTerm{sourceIndex, derivativeIndex, sign,
                    static_cast<std::uint8_t>(
                        target.degree() + source.degree() + 1)});
        }
    }
    m2lOffsets_.push_back(m2lTerms_.size());
}

std::size_t FmmTaylorExpansion::index(int x, int y, int z) const
{
    if(x < 0 || y < 0 || z < 0 || x + y + z > order_)
        throw UniversalError("FmmTaylorExpansion: invalid multi-index");
    const std::size_t side = static_cast<std::size_t>(order_ + 1);
    return lookup_[(static_cast<std::size_t>(x) * side + static_cast<std::size_t>(y)) * side +
                   static_cast<std::size_t>(z)];
}
