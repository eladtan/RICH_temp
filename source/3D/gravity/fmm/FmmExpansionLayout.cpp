#include "3D/gravity/fmm/FmmExpansionLayout.hpp"

#include "misc/universal_error.hpp"

FmmExpansionLayout::FmmExpansionLayout(int expansionOrder):
    order_(expansionOrder)
{
    if(expansionOrder < 0 || expansionOrder > FMM_MAX_ORDER)
        throw UniversalError("FmmExpansionLayout: expansion order outside supported range");
}

int FmmExpansionLayout::order() const
{
    return order_;
}

std::size_t FmmExpansionLayout::coefficientCount() const
{
    return fmmCoefficientCount(order_);
}

std::size_t FmmExpansionLayout::index(int n, int m) const
{
    if(n < 0 || n > order_ || m < -n || m > n)
        throw UniversalError("FmmExpansionLayout: invalid coefficient index");
    if(m == 0)
        return static_cast<std::size_t>(n * n);
    const int magnitude = m > 0 ? m : -m;
    return static_cast<std::size_t>(n * n + 2 * magnitude - (m > 0 ? 1 : 0));
}

std::size_t FmmExpansionLayout::indexReal(int n, int m) const
{
    if(m <= 0)
        throw UniversalError("FmmExpansionLayout: real index requires m > 0");
    return index(n, m);
}

std::size_t FmmExpansionLayout::indexImag(int n, int m) const
{
    if(m <= 0)
        throw UniversalError("FmmExpansionLayout: imaginary index requires m > 0");
    return index(n, -m);
}
