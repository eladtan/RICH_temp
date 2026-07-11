#ifndef FMM_EXPANSION_LAYOUT_HPP
#define FMM_EXPANSION_LAYOUT_HPP

#include <cstddef>

#include "3D/gravity/fmm/FmmConfig.hpp"

class FmmExpansionLayout
{
public:
    explicit FmmExpansionLayout(int expansionOrder);

    int order() const;
    std::size_t coefficientCount() const;
    // m > 0 addresses the real part; m < 0 addresses the imaginary part.
    std::size_t index(int n, int m) const;
    std::size_t indexReal(int n, int m) const;
    std::size_t indexImag(int n, int m) const;

private:
    int order_;
};

#endif // FMM_EXPANSION_LAYOUT_HPP
