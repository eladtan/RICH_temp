#include "MovingProfileGravity3D.hpp"
#include <algorithm>
#include <cmath>
#include "misc/universal_error.hpp"
#include "misc/utils.hpp"

namespace {
    // Linear interpolation in a monotonic table, clamped at both ends
    double Interpolate(std::vector<double> const& X, std::vector<double> const& Y, double Value) {
        if (Value <= X.front())
            return Y.front();
        if (Value >= X.back())
            return Y.back();
        size_t const Upper = static_cast<size_t>(std::upper_bound(X.begin(), X.end(), Value) - X.begin());
        double const Weight = (Value - X[Upper - 1]) / (X[Upper] - X[Upper - 1]);
        return Y[Upper - 1] + Weight * (Y[Upper] - Y[Upper - 1]);
    }
}

MovingProfileGravity3D::MovingProfileGravity3D(std::vector<double> const& RTable, std::vector<double> const& MTable, double G, bool ZeroZ)
    : RTable_(RTable), MTable_(MTable), PhiTable_(RTable.size(), 0), G_(G), ZeroZ_(ZeroZ), Centre0_(), Velocity_(), StartTime_(0) {
    if (RTable_.size() < 2 || RTable_.size() != MTable_.size() || !(RTable_.front() > 0))
        throw UniversalError("MovingProfileGravity3D: need at least two radii, matching masses and RTable[0] > 0");
    for (size_t i = 1; i < RTable_.size(); ++i)
        if (!(RTable_[i] > RTable_[i - 1]))
            throw UniversalError("MovingProfileGravity3D: radii must increase");
    // Phi(r) = -G M / R_out - int_r^R_out G M(r') / r'^2 dr', trapezoid rule
    size_t const N = RTable_.size();
    PhiTable_[N - 1] = -G_ * MTable_[N - 1] / RTable_[N - 1];
    for (size_t i = N - 1; i > 0; --i) {
        double const Inner = MTable_[i - 1] / (RTable_[i - 1] * RTable_[i - 1]);
        double const Outer = MTable_[i] / (RTable_[i] * RTable_[i]);
        PhiTable_[i - 1] = PhiTable_[i] - 0.5 * G_ * (Inner + Outer) * (RTable_[i] - RTable_[i - 1]);
    }
}

void MovingProfileGravity3D::SetPath(Vector3D const& Centre0, Vector3D const& Velocity, double StartTime) {
    Centre0_ = Centre0;
    Velocity_ = Velocity;
    StartTime_ = StartTime;
}

Vector3D MovingProfileGravity3D::GetCentre(double Time) const {
    return Centre0_ + Velocity_ * std::max(0.0, Time - StartTime_);
}

Vector3D MovingProfileGravity3D::GetVelocity(double Time) const {
    // At StartTime the centre starts moving; the driver boosts the gas at the same instant
    return Time >= StartTime_ ? Velocity_ : Vector3D();
}

double MovingProfileGravity3D::EnclosedMass(double R) const {
    if (R >= RTable_.back())
        return MTable_.back();
    if (R <= RTable_.front())
        return MTable_.front() * std::pow(R / RTable_.front(), 3);
    return Interpolate(RTable_, MTable_, R);
}

double MovingProfileGravity3D::Potential(double R) const {
    if (R >= RTable_.back())
        return -G_ * MTable_.back() / R;
    if (R <= RTable_.front()) {
        // Uniform density inside RTable_[0], consistent with the acceleration used there
        double const X = R / RTable_.front();
        return PhiTable_.front() - 0.5 * G_ * MTable_.front() / RTable_.front() * (1 - X * X);
    }
    return Interpolate(RTable_, PhiTable_, R);
}

void MovingProfileGravity3D::operator()(const Tessellation3D& Tess, const vector<ComputationalCell3D>& /*Cells*/, const vector<Conserved3D>& /*Fluxes*/,
    const double Time, vector<Vector3D>& Acc) const {
    size_t const N = Tess.GetPointNo();
    Acc.assign(N, Vector3D());
    Vector3D const Centre = GetCentre(Time);
    // Inside RTable_[0] the enclosed mass is taken as uniform density, so M(r)/r^3 is constant there
    double const InnerMassOverR3 = MTable_.front() / std::pow(RTable_.front(), 3);
    for (size_t i = 0; i < N; ++i) {
        Vector3D const Offset = Tess.GetCellCM(i) - Centre;
        double const R = fastabs(Offset);
        double const MassOverR3 = R <= RTable_.front() ? InnerMassOverR3 : EnclosedMass(R) / (R * R * R);
        Acc[i] = -G_ * MassOverR3 * Offset;
        if (ZeroZ_)
            Acc[i].z = 0;
    }
}
