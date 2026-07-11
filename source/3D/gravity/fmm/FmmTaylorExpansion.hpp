#ifndef FMM_TAYLOR_EXPANSION_HPP
#define FMM_TAYLOR_EXPANSION_HPP

#include <cstddef>
#include <vector>

struct FmmMultiIndex
{
    FmmMultiIndex(): x(0), y(0), z(0) {}
    FmmMultiIndex(int xValue, int yValue, int zValue):
        x(xValue), y(yValue), z(zValue) {}

    int x;
    int y;
    int z;

    int degree() const { return x + y + z; }
};

struct FmmM2LTerm
{
    std::size_t sourceIndex;
    std::size_t derivativeIndex;
    double scale;
};

class FmmTaylorExpansion
{
public:
    explicit FmmTaylorExpansion(int order);

    int order() const { return order_; }
    std::size_t coefficientCount() const { return indices_.size(); }
    const FmmMultiIndex& multiIndex(std::size_t index) const { return indices_[index]; }
    std::size_t index(int x, int y, int z) const;
    double inverseFactorial(std::size_t index) const { return inverseFactorials_[index]; }
    const std::vector<std::size_t>& m2lOffsets() const { return m2lOffsets_; }
    const std::vector<FmmM2LTerm>& m2lTerms() const { return m2lTerms_; }

private:
    int order_;
    std::vector<FmmMultiIndex> indices_;
    std::vector<std::size_t> lookup_;
    std::vector<double> inverseFactorials_;
    std::vector<std::size_t> m2lOffsets_;
    std::vector<FmmM2LTerm> m2lTerms_;
};

#endif // FMM_TAYLOR_EXPANSION_HPP
