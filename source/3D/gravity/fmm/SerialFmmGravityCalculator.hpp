#ifndef SERIAL_FMM_GRAVITY_CALCULATOR_HPP
#define SERIAL_FMM_GRAVITY_CALCULATOR_HPP

#include <vector>

#include "3D/elementary/Vector3D.hpp"
#include "3D/gravity/fmm/FmmConfig.hpp"
#include "3D/gravity/fmm/FmmDiagnostics.hpp"
#include "3D/gravity/fmm/FmmM2LOperatorCache.hpp"
#include "3D/gravity/fmm/FmmTree.hpp"

class SerialFmmGravityCalculator
{
public:
    explicit SerialFmmGravityCalculator(FmmGravityOptions options = FmmGravityOptions());

    void solve(const std::vector<Vector3D>& positions,
               const std::vector<double>& masses,
               const Vector3D& domainLower,
               const Vector3D& domainUpper,
               std::vector<Vector3D>& acceleration,
               std::vector<double>* positiveKernelPotential = nullptr);

    const FmmSolveStats& stats() const noexcept;

private:
    void validateOptions() const;
    void validateInputs(const std::vector<Vector3D>& positions,
                        const std::vector<double>& masses) const;

    FmmGravityOptions options_;
    FmmSolveStats stats_;
    FmmTree tree_;
    FmmM2LOperatorCache operatorCache_;
    std::vector<double> multipoles_;
    std::vector<double> locals_;
};

#endif // SERIAL_FMM_GRAVITY_CALCULATOR_HPP
